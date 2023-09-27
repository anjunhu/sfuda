import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader, WeightedRandomSampler
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, confusion_matrix
from evaluation import *
from pprint import pprint
import wandb

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (11 + gamma * iter_num / max_iter) ** (-power)
    # decay = (1 + gamma) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsize = len(txt_tar)
    tr_size = int(0.5*dsize)
    # Uae the latter half as target training and evaluation
    _, txt_tar_tr = torch.utils.data.random_split(txt_tar, [tr_size, dsize - tr_size], generator=torch.Generator().manual_seed(0))
    txt_tar_te = txt_tar_tr

    print("target_tr", len(txt_tar_tr))
    print("target_te", len(txt_tar_te))
    print("test", len(txt_tar_te))

    dsets["test"] = ImageList_idx(txt_test, root_dir=f'./data/{args.dset}/', transform=image_test())
    dsets["target_tr"] = ImageList_idx(txt_tar_tr, root_dir=f'./data/{args.dset}/', transform=image_train())
    dsets["target_te"] = ImageList(txt_tar_te, root_dir=f'./data/{args.dset}/', transform=image_test())

    train_sampler = None
    test_sampler = None
    g = torch.Generator()
    g.manual_seed(args.seed)
    if args.train_resampling:
        weights = dsets["target_tr"].get_weights(args.train_resampling, flag=args.flag)
        train_sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=g)
    if args.test_resampling:
        test_weights = dsets["test"].get_weights(args.test_resampling, flag=args.flag)
        test_sampler = WeightedRandomSampler(test_weights, len(test_weights), replacement=True, generator=g)


    dset_loaders["target_tr"] = DataLoader(dsets["target_tr"], batch_size=train_bs, shuffle=False, sampler=train_sampler, num_workers=args.worker, drop_last=False)
    dset_loaders["target_te"] = DataLoader(dsets["target_te"], batch_size=train_bs, shuffle=False, sampler=train_sampler, num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, sampler=test_sampler, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=True):
    start_test = True
    group_metrics = {}
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            sensitives = data[-1]
            inputs = inputs.cuda()
            if netB is None:
                outputs = netC(netF(inputs))
            else:
                outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float().cpu()
                all_sensitives = sensitives.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu()), 0)
                all_sensitives =  torch.cat((all_sensitives, sensitives.float().cpu()), 0)

    print('\nEval Y0/Y1', all_output[all_label.squeeze()==0].shape, all_output[all_label.squeeze()==1].shape)
    for sa in range(args.sens_classes):
        print(f'A{sa} Total/Y0/Y1:', all_output[all_sensitives.squeeze()==sa].shape,
                    all_output[torch.logical_and((all_sensitives.squeeze()==sa),(all_label.squeeze()==0))].shape,
                    all_output[torch.logical_and((all_sensitives.squeeze()==sa),(all_label.squeeze()==1))].shape)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    auc = calculate_auc(F.softmax(all_output, dim=1)[:, 1], all_label.cpu())
    for sa in range(args.sens_classes):
        output_sa = all_output[all_sensitives.squeeze()==sa]
        labels_sa = all_label[all_sensitives.squeeze()==sa]
        group_metrics[f'acc A{sa}'] = accuracy_score(labels_sa.to('cpu'), output_sa.to('cpu').max(1)[1])
        group_metrics[f'auc A{sa}'] = calculate_auc(F.softmax(output_sa, dim=1)[:, 1].detach().cpu(), labels_sa.to('cpu'))
        group_metrics[f'entropy A{sa}'] = -(output_sa.softmax(1) * output_sa.log_softmax(1)).sum(1).mean(0).item()
    group_metrics.update({'auc': auc, 'acc': accuracy, 'mean_ent': mean_ent})
    pprint(group_metrics)

    if args.wandb:
        wandb.log(group_metrics)

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc, predict, mean_ent
    else:
        return accuracy, mean_ent, predict, mean_ent

def train_target(args):
    dset_loaders = data_load(args)
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
        
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
   
    modelpath = args.output_dir + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))

    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["target_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.eval()
    netB.eval()
    netC.eval()
    acc_s_te, _, pry, mean_ent = cal_acc(dset_loaders['test'], netF, netB, netC, False)
    log_str = 'Task: {}, Iter:{}/{}; Accuracy={:.4f}, Ent={:.4f}'.format(args.name, iter_num, max_iter, acc_s_te, mean_ent)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')
    netF.train()
    netB.train()
    netC.train()

    old_pry = 0
    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test, _, tar_idx, sensitives = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target_tr"])
            inputs_test, _, tar_idx, sensitives = next(iter_test)

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter, power=0.75)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        softmax_out = nn.Softmax(dim=1)(outputs_test)
        entropy_loss = torch.mean(loss.Entropy(softmax_out))

        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = -torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        entropy_loss -= gentropy_loss
        entropy_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            acc_s_te, _, pry, mean_ent = cal_acc(dset_loaders['test'], netF, netB, netC, False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy={:.4f}, Ent={:.4f}'.format(args.name, iter_num, max_iter, acc_s_te, mean_ent)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
                torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
                torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
            netF.train()
            netB.train()
            netC.train()

            if torch.abs(pry - old_pry).sum() == 0:
                break
            else:
                old_pry = pry.clone()
 
    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DINE')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=30, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='cardiomegaly', choices=['cardiomegaly', 'VISDA-C', 'office', 'image-clef', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet18, resnet50, resnext50")
    parser.add_argument('--net_src', type=str, default='resnet50', help="alexnet, vgg16, resnet18, resnet34, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='ckps/DINE/target')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    
    parser.add_argument('--proj_name', type=str, default='DINE')
    parser.add_argument('--train_resampling', default='balanced', choices=['natural', 'class', 'group', 'balanced'], type=str, help='')
    parser.add_argument('--test_resampling', default='balanced', choices=['natural', 'class', 'group', 'balanced'], type=str, help='')
    parser.add_argument('--flag', default='', choices=['', 'Younger', 'Older'], type=str, help='')
    parser.add_argument('--wandb', action='store_true', help="Use wandb")
    parser.add_argument('--sens_classes', type=int, default=5, help="number of sensitive classes")

    args = parser.parse_args()
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'cardiomegaly':
        names = ['chexpert', 'mimic']
        args.class_num = 2
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12

    run_name = ''.join(['FT_', args.dset, '_', names[args.s][0].upper(), '2', args.flag, names[args.t][0].upper()])
    if args.wandb:
        wandb.init(project=args.proj_name, name=run_name, config=args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = './data/'
    for i in range(1): #range(len(names)):
        #if i == args.s:
        #    continue
        #args.t = i
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        args.output_dir = osp.join(args.output, args.net_src + '_' + args.net, str(args.seed), args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'finetune'
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        print_args(args)

        train_target(args)
