import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader, WeightedRandomSampler
from data_list import ImageList
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from evaluation import *
import wandb
from pprint import pprint

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

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

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

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i
        
        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                new_src.append(line)
        txt_src = new_src.copy()

        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_test = new_tar.copy()

    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.5*dsize)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size], generator=torch.Generator().manual_seed(0))
    else:
        dsize = len(txt_src)
        tr_size = int(0.5*dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size], generator=torch.Generator().manual_seed(0))
        tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, root_dir=f'./data/{args.dset}/', transform=image_train(), attribute=args.sens_name)
    dsets["source_te"] = ImageList(te_txt, root_dir=f'./data/{args.dset}/', transform=image_test(), attribute=args.sens_name)
    dsets["test"] = ImageList(txt_test, root_dir=f'./data/{args.dset}/', transform=image_test(), attribute=args.sens_name)

    for key in dsets:
        print('{} dataset size {}'.format(key, len(dsets[key]) ) )

    source_tr_sampler = None
    source_te_sampler = None
    test_sampler = None
    g = torch.Generator()
    g.manual_seed(0)
    if args.train_resampling and args.train_resampling != 'natural':
        weights = dsets["source_tr"].get_weights(args.train_resampling)
        source_tr_sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=g)
    if args.test_resampling and args.test_resampling != 'natural':
        weights = dsets["source_te"].get_weights(args.test_resampling)
        source_te_sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=g)
        test_weights = dsets["test"].get_weights(args.test_resampling)
        test_sampler = WeightedRandomSampler(test_weights, len(test_weights), replacement=True, generator=g)

    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, sampler=source_tr_sampler, num_workers=args.worker, drop_last=False)
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, sampler=source_te_sampler, num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, sampler=test_sampler, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    group_metrics = {}
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            sensitives = data[2]
            inputs = inputs.cuda()
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
    ece = expected_calibration_error(F.softmax(all_output, dim=1)[:, 1].cpu().numpy(), all_label.cpu())
    for sa in range(args.sens_classes):
        output_sa = all_output[all_sensitives.squeeze()==sa]
        labels_sa = all_label[all_sensitives.squeeze()==sa]
        group_metrics[f'acc A{sa}'] = accuracy_score(labels_sa.to('cpu'), output_sa.to('cpu').max(1)[1])
        group_metrics[f'auc A{sa}'] = calculate_auc(F.softmax(output_sa, dim=1)[:, 1].detach().cpu(), labels_sa.to('cpu'))
        group_metrics[f'entropy A{sa}'] = -(output_sa.softmax(1) * output_sa.log_softmax(1)).sum(1).mean(0).item()
        group_metrics[f'ece A{sa}'] = expected_calibration_error(output_sa.softmax(1)[:, 1].detach().cpu().numpy(),  labels_sa.cpu())
    group_metrics.update({'auc': auc, 'acc': accuracy, 'mean_ent': mean_ent, 'ece': ece})
    pprint(group_metrics)

    if args.wandb:
        wandb.log(group_metrics, commit=False)
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc, group_metrics
    else:
        return accuracy, mean_ent, group_metrics

def cal_acc_oda(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            sensitives = data[2]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()
    initc = np.array([[0], [1]])
    kmeans = KMeans(n_clusters=2, random_state=0, init=initc, n_init=1).fit(ent.reshape(-1,1))
    threshold = (kmeans.cluster_centers_).mean()

    predict[ent>threshold] = args.class_num
    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    matrix = matrix[np.unique(all_label).astype(int),:]

    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    unknown_acc = acc[-1:].item()

    return np.mean(acc[:-1]), np.mean(acc), unknown_acc
    # return np.mean(acc), np.mean(acc[:-1])

def train_source(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 100
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    acc_s_te, acc_list, best_metric = cal_acc(dset_loaders['source_te'], netF, netB, netC, True)
    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.4f}'.format(args.name_src, iter_num, max_iter, acc_s_te) + '\n' + acc_list

    while iter_num < max_iter:
        try:
            inputs_source, labels_source, sensitives_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source, sensitives_source = next(iter_source)
        #labels_source = torch.tensor(labels_source).to(inputs_source.device)
        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = netC(netB(netF(inputs_source)))
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if args.wandb:
            wandb.log({'classifier_loss': classifier_loss}, commit=False)
        if iter_num % interval_iter == 1 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            acc_s_te, acc_list, group_metrics = cal_acc(dset_loaders['source_te'], netF, netB, netC, True)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.4f}'.format(args.name_src, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            if args.wandb:
                wandb.log({'acc_s_te': acc_s_te})
            if  group_metrics['auc'] >= best_metric['auc']: #acc_s_te >= best_metric:
                best_metric = group_metrics
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()
                torch.save(best_netF, osp.join(args.output_dir_src, f"{args.train_resampling[0].upper()}{args.test_resampling[0].upper()}_source_F.pt"))
                torch.save(best_netB, osp.join(args.output_dir_src, f"{args.train_resampling[0].upper()}{args.test_resampling[0].upper()}_source_B.pt"))
                torch.save(best_netC, osp.join(args.output_dir_src, f"{args.train_resampling[0].upper()}{args.test_resampling[0].upper()}_source_C.pt"))

            netF.train()
            netB.train()
            netC.train()

    print('*'*20 + 'Source Domain Best Metrics' + '*'*20)
    pprint(best_metric)
    for key in best_metric:
        wandb.log({f'Best {key}' : best_metric[key]}, commit=False)
    return netF, netB, netC

def test_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir_src + f'/{args.train_resampling[0].upper()}{args.test_resampling[0].upper()}_source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + f'/{args.train_resampling[0].upper()}{args.test_resampling[0].upper()}_source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + f'/{args.train_resampling[0].upper()}{args.test_resampling[0].upper()}_source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    if args.da == 'oda':
        acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB, netC)
        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.4f} / {:.4f} / {:.4f}'.format(args.trte, args.name, acc_os2, acc_os1, acc_unknown)
    else:
        acc, acc_list, group_metrics = cal_acc(dset_loaders['test'], netF, netB, netC, True)
        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.4f}'.format(args.trte, args.name, acc) + '\n' + acc_list
    print('*'*20 + 'Target Domain Best Metrics' + '*'*20)
    pprint(group_metrics)
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=10, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['cardiomegaly', 'VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='ckps/SHOT/source')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--run_name', type=str, default='chexpert_source')
    parser.add_argument('--proj_name', type=str, default='SHOT_SexBaselines')
    parser.add_argument('--train_resampling', default='balanced', choices=['natural', 'class', 'group', 'balanced'], type=str, help='')
    parser.add_argument('--test_resampling', default='balanced', choices=['natural', 'class', 'group', 'balanced'], type=str, help='')
    parser.add_argument('--sens_classes', type=int, default=2, help="number of sensitive classes")
    parser.add_argument('--sens_name', type=str, default='sex', choices=['age', 'sex', 'race'], help="name of sensitive attribute")
    parser.add_argument('--wandb', action='store_true', help="Use wandb")
    args = parser.parse_args()

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

    run_name = '_'.join([args.train_resampling[0].upper(), args.test_resampling[0].upper(), 'SHOT', 'Source', args.dset, names[args.s][0].upper()])
    if args.wandb:
        print('Running', run_name)
        wandb.init(project=args.proj_name, name=run_name, config=args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = './data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'     
    print('Source', names[args.s], 'Target', names[args.t])

    if args.dset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]
        if args.da == 'oda':
            args.class_num = 25
            args.src_classes = [i for i in range(25)]
            args.tar_classes = [i for i in range(65)]

    args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper(), args.net, str(args.seed))
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_source(args)

    args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]
            if args.da == 'oda':
                args.class_num = 25
                args.src_classes = [i for i in range(25)]
                args.tar_classes = [i for i in range(65)]

        test_target(args)
