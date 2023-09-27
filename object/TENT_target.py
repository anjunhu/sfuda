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
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
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

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    if True: #args.s == args.t: # If doing intra-dataset demographic shfiting
        dsize = len(txt_tar)
        tr_size = int(0.75*dsize)
        print('intra-dataset demographic shfiting', dsize, tr_size, dsize - tr_size)
        _, txt_tar = torch.utils.data.random_split(txt_tar, [tr_size, dsize - tr_size], generator=torch.Generator().manual_seed(0))

    txt_test = txt_tar

    dsets["target"] = ImageList_idx(txt_tar, root_dir=f'./data/{args.dset}', transform=image_train())
    dsets["test"] = ImageList_idx(txt_test, root_dir=f'./data/{args.dset}', transform=image_test())

    train_sampler = None
    test_sampler = None
    g = torch.Generator()
    g.manual_seed(args.seed)
    if args.train_resampling:
        weights = dsets["target"].get_weights(args.train_resampling, flag=args.flag)
        train_sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=g)
    if args.test_resampling:
        test_weights = dsets["test"].get_weights(args.test_resampling, flag=args.flag)
        test_sampler = WeightedRandomSampler(test_weights, len(test_weights), replacement=True, generator=g)

    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=False, sampler=train_sampler, num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, sampler=test_sampler, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, args, flag=False, full=False):
    start_test = True
    group_metrics = {}

    with torch.no_grad():
        iter_test = iter(loader)
        batches = len(loader) if full else min(20, len(loader))
        for i in range(batches):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            tar_idx = data[2]
            sensitives = data[3]

            mem_label = obtain_label(loader, netF, netB, netC, args, False)
            mem_label = torch.from_numpy(mem_label).cuda()
            pseudolabels = mem_label[tar_idx]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))

            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss(reduction='none')(outputs, pred)

            softmax_out = nn.Softmax(dim=1)(outputs)
            entropy = softmax_out #torch.mean(loss.Entropy(softmax_out))
            gentropy_loss = 0.0
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
            im_loss = entropy - gentropy_loss

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float().cpu()
                all_classifier_losses = classifier_loss.float().cpu()
                all_im_losses = im_loss.float().cpu()
                all_entropies = entropy.float().cpu()
                all_sensitives = sensitives.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu()), 0)
                all_classifier_losses =  torch.cat((all_classifier_losses, classifier_loss.float().cpu()), 0)
                all_im_losses =  torch.cat((all_im_losses, im_loss.float().cpu()), 0)
                all_entropies = torch.cat((all_entropies, entropy.float().cpu()), 0)
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
        group_metrics[f'cls_loss A{sa}'] = (all_classifier_losses[all_sensitives.squeeze()==sa]).mean().item()
        group_metrics[f'entropy A{sa}'] = -(output_sa.softmax(1) * output_sa.log_softmax(1)).sum(1).mean(0).item()
        group_metrics[f'im_loss A{sa}'] = (all_im_losses[all_sensitives.squeeze()==sa]).mean().item()
        group_metrics[f'acc A{sa}'] = accuracy_score(labels_sa.to('cpu'), output_sa.to('cpu').max(1)[1])
        group_metrics[f'auc A{sa}'] = calculate_auc(F.softmax(output_sa, dim=1)[:, 1].detach().cpu(), labels_sa.to('cpu'))
    pprint(group_metrics)
    if args.wandb:
        wandb.log({'auc': auc, 'mean_ent': mean_ent})
        wandb.log(group_metrics)
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))

    param_group = []

    for nm, m in netF.named_modules():
        if isinstance(m, nn.BatchNorm2d) and args.lr_decay1 > 0:
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    param_group += [{'params': p, 'lr': args.lr * args.lr_decay1}]
                else:
                    p.requires_grad = False

    for nm, m in netC.named_modules():
        if isinstance(m, nn.BatchNorm2d) and args.lr_decay1 > 0:
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    param_group += [{'params': p, 'lr': args.lr * args.lr_decay1}]
                else:
                    p.requires_grad = False

    for k, v in netB.named_modules():
        if isinstance(m, nn.BatchNorm2d) and args.lr_decay2 > 0:
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    param_group += [{'params': p, 'lr': args.lr * args.lr_decay2}]
                else:
                    p.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = 1  #max_iter // args.interval
    iter_num = 0

    # ** Get epoch 0 metrics **
    netF.eval()
    netB.eval()
    if args.dset=='VISDA-C':
        acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, args, True, True)
        log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.4f}'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
    else:
        acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, args, False, True)
        log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.4f}'.format(args.name, iter_num, max_iter, acc_s_te)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')
    if args.wandb:
        wandb.log({'acc_s_te': acc_s_te})
    netF.train()
    netB.train()

    while iter_num < max_iter:
        try:
            inputs_test, labels, tar_idx, sensitives = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, labels, tar_idx, sensitives = next(iter_test)

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.wandb:
            wandb.log({'classifier_loss': classifier_loss})

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if args.wandb:
            wandb.log({'entropy_loss': entropy_loss})

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, args, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.4f}'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, args, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.4f}'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
                torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
                torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
            if args.wandb:
                wandb.log({'acc_s_te': acc_s_te})
            netF.train()
            netB.train()

    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def obtain_label(loader, netF, netB, netC, args, verbose=True):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            tar_idx = data[2]
            sensitives = data[3]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>args.threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.4f} -> {:.4f}'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    if verbose: print(log_str+'\n')

    return predict.astype('int')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TENT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=50, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['cardiomegaly', 'VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--gent', action='store_true')
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='ckps/TENT/target')
    parser.add_argument('--output_src', type=str, default='ckps/TENT/source')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--wandb', action='store_true', help="Use wandb")
    parser.add_argument('--proj_name', type=str, default='TENT')
    parser.add_argument('--train_resampling', default='balanced', choices=['natural', 'class', 'group', 'balanced'], type=str, help='')
    parser.add_argument('--test_resampling', default='balanced', choices=['natural', 'class', 'group', 'balanced'], type=str, help='')
    parser.add_argument('--flag', default='', choices=['', 'Younger', 'Older'], type=str, help='')
    parser.add_argument('--sens_classes', type=int, default=5, help="number of sensitive classes")

    args = parser.parse_args()

    args.cls_par = 0.0 # TENT!

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'cardiomegaly':
        names = ['chexpert', 'mimic']
        args.class_num = 2

    run_name = ''.join(['TENT_', args.dset, '_', names[args.s][0].upper(), '2', args.flag, names[args.t][0].upper()])
    if args.wandb:
        wandb.init(project=args.proj_name, name=run_name, config=args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    if True:
        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        print('Source', names[args.s], 'Target', args.flag, names[args.t])

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper(), args.net)
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_target(args)
