import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader, WeightedRandomSampler
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from sklearn.metrics import accuracy_score, confusion_matrix
import torch.nn.functional as F
from evaluation import *
from pprint import pprint
import wandb

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
        param_group["weight_decay"] = 1e-3
        param_group["momentum"] = 0.9
        param_group["nesterov"] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsize = len(txt_src)
    tr_size = int(0.5 * dsize)
    tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])

    dsets["source_tr"] = ImageList(tr_txt, root_dir=f'./data/{args.dset}/', transform=image_train(), attribute=args.sens_name)
    dsets["source_te"] = ImageList(te_txt, root_dir=f'./data/{args.dset}/', transform=image_test(), attribute=args.sens_name)
    dsets["target"] = ImageList_idx(txt_tar, root_dir=f'./data/{args.dset}/', transform=image_train(), attribute=args.sens_name)
    dsets["test"] = ImageList_idx(txt_test, root_dir=f'./data/{args.dset}/', transform=image_test(), attribute=args.sens_name)

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

    dset_loaders["source_tr"] = DataLoader(
        dsets["source_tr"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dset_loaders["source_te"] = DataLoader(
        dsets["source_te"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dset_loaders["target"] = DataLoader(
        dsets["target"],
        batch_size=train_bs,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.worker,
        drop_last=False,
    )
    dset_loaders["test"] = DataLoader(
        dsets["test"],
        batch_size=train_bs * 3,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.worker,
        drop_last=False,
    )

    return dset_loaders


def cal_acc(loader, fea_bank, score_bank, netF, netB, netC, args, flag=False):
    start_test = True
    group_metrics = {}
    num_sample = len(loader.dataset)
    label_bank = torch.randn(num_sample)  # .cuda()
    pred_bank = torch.randn(num_sample)
    nu=[]
    s=[]
    var_all=[]

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            indx = data[2]
            sensitives = data[-1]

            inputs = inputs.cuda()
            fea = netB(netF(inputs))
            """if args.var:
                var_batch=fea.var()
                var_all.append(var_batch)"""

            # if args.singular:
            _, ss, _ = torch.svd(fea)
            s10=ss[:10]/ss[0]
            s.append(s10)

            outputs = netC(fea)
            softmax_out = nn.Softmax()(outputs)
            nu.append(torch.mean(torch.svd(softmax_out)[1]))
            output_f_norm = F.normalize(fea)
            fea_bank[indx] = output_f_norm.detach().clone().cpu()
            label_bank[indx] = labels.float().detach().clone()  # .cpu()
            pred_bank[indx] = outputs.max(-1)[1].float().detach().clone().cpu()
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_sensitives = sensitives.float().cpu()
                all_fea = output_f_norm.cpu()
                all_indx = indx.cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_sensitives =  torch.cat((all_sensitives, sensitives.float().cpu()), 0)
                all_fea = torch.cat((all_fea, output_f_norm.cpu()), 0)
                all_indx = torch.cat((all_indx, indx.cpu()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0]
    )

    # Losses
    with torch.no_grad():
        output_f_norm = F.normalize(all_fea)
        output_f_ = output_f_norm.cpu().detach().clone()
        softmax_out = nn.Softmax()(all_output)
        pred_bs = softmax_out

        fea_bank[all_indx] = output_f_.detach().clone().cpu()
        sbc = score_bank.cpu()
        sbc[all_indx] = softmax_out.detach().clone().cpu()

        distance = output_f_ @ fea_bank.T
        _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
        idx_near = idx_near[:, 1:]  # batch x K
        score_near = sbc[idx_near]  # batch x K x C

        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(
            -1, args.K, -1
        )  # batch x K x C

        #loss = F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1).sum(1)
        loss = (F.kl_div(softmax_out_un.log(), score_near.log(), reduction="none").sum(-1)).sum(1)
        #print(loss.shape, loss)

    _, score_bank_ = torch.max(score_bank, 1)
    distance = fea_bank.cpu() @ fea_bank.cpu().T
    _, idx_near = torch.topk(distance, dim=-1, largest=True, k=4)
    score_near = score_bank_[idx_near[:, :]].float().cpu()  # N x 4

    """acc1 = (score_near.mean(
        dim=-1) == score_near[:, 0]).sum().float() / score_near.shape[0]"""
    acc1 = (
        (score_near.mean(dim=-1) == score_near[:, 0]) & (score_near[:, 0] == pred_bank)
    ).sum().float() / score_near.shape[0]
    acc2 = (
        (score_near.mean(dim=-1) == score_near[:, 0]) & (score_near[:, 0] == label_bank)
    ).sum().float() / score_near.shape[0]

    nu_mean=sum(nu)/len(nu)

    s10_avg=torch.stack(s).mean(0)
    print('nuclear mean: {:.4f}'.format(nu_mean))

    auc = calculate_auc(F.softmax(all_output, dim=1)[:, 1], all_label.cpu())
    print('\nEvaluating Y0/Y1', all_output[all_label.squeeze()==0].shape, all_output[all_label.squeeze()==1].shape)
    for sa in range(args.sens_classes):
        print(f'A{sa} Total/Y0/Y1:', all_output[all_sensitives.squeeze()==sa].shape,
                    all_output[torch.logical_and((all_sensitives.squeeze()==sa),(all_label.squeeze()==0))].shape,
                    all_output[torch.logical_and((all_sensitives.squeeze()==sa),(all_label.squeeze()==1))].shape)
    for sa in range(args.sens_classes):
        output_sa = all_output[all_sensitives.squeeze()==sa]
        labels_sa = all_label[all_sensitives.squeeze()==sa]
        group_metrics[f'entropy A{sa}'] = -(output_sa.softmax(1) * output_sa.log_softmax(1)).sum(1).mean(0).item()
        group_metrics[f'acc A{sa}'] = accuracy_score(labels_sa.to('cpu'), output_sa.to('cpu').max(1)[1])
        group_metrics[f'auc A{sa}'] = calculate_auc(F.softmax(output_sa, dim=1)[:, 1].detach().cpu(), labels_sa.to('cpu'))
        group_metrics[f'loss_kl A{sa}'] = loss[all_sensitives.squeeze()==sa].mean(0).item()
    pprint(group_metrics)
    if args.wandb:
       wandb.log(group_metrics, commit=False)

    if True:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1)
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = " ".join(aa)
        if args.wandb:
            metrics = {'auc': auc, 'acc': aacc, 'nuclear_mean': nu_mean, 's10_avg': s10_avg}
            pprint(metrics)
            wandb.log(metrics)
        if True:
            return aacc, acc  # , acc1, acc2#, nu_mean, s10_avg

    else:
        return accuracy, mean_ent


def hyper_decay(x, beta=-2, alpha=1):
    weight = (1 + 10 * x) ** (-beta) * alpha
    return weight


def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bottleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    netC = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()

    modelpath = osp.join(args.output_dir_src, f"{args.train_resampling[0].upper()}{args.test_resampling[0].upper()}_source_F.pt")
    netF.load_state_dict(torch.load(modelpath))
    modelpath = osp.join(args.output_dir_src, f"{args.train_resampling[0].upper()}{args.test_resampling[0].upper()}_source_B.pt")
    netB.load_state_dict(torch.load(modelpath))
    modelpath = osp.join(args.output_dir_src, f"{args.train_resampling[0].upper()}{args.test_resampling[0].upper()}_source_C.pt")
    netC.load_state_dict(torch.load(modelpath))

    param_group = []
    param_group_c = []
    for k, v in netF.named_parameters():
        # if k.find('bn')!=-1:
        if True:
            param_group += [{"params": v, "lr": args.lr * 0.1}]  # 0.1

    for k, v in netB.named_parameters():
        if True:
            param_group += [{"params": v, "lr": args.lr * 1}]  # 1
    for k, v in netC.named_parameters():
        param_group_c += [{"params": v, "lr": args.lr * 1}]  # 1

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    # building feature bank and score bank
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, 256)
    score_bank = torch.randn(num_sample, args.class_num).cuda()

    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            indx = data[2]
            sensitives = data[-1]
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            output_norm = F.normalize(output)
            outputs = netC(output)
            outputs = nn.Softmax(-1)(outputs)

            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  # .cpu()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()
    acc_log = 0

    real_max_iter = max_iter

    # ** Get epoch 0 metrics **
    acc, accc = cal_acc(
                    dset_loaders["test"],
                    fea_bank,
                    score_bank,
                    netF,
                    netB,
                    netC,
                    args,
                    flag=True,
                )
    log_str = (
                    "Task: {}, Iter:{}/{};  Acc on target: {:.4f}".format(
                        args.name, iter_num, max_iter, acc
                    )
                    + "\n"
                    + "T: "
                    + accc
                )
    print(log_str)

    while iter_num < real_max_iter:
        try:
            inputs_test, _, indx, sensitives = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, indx, sensitives = next(iter_test)

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        if True:
            alpha = (1 + 10 * iter_num / max_iter) ** (-args.beta) * args.alpha
        else:
            alpha = args.alpha

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        # output_re = softmax_out.unsqueeze(1)

        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            pred_bs = softmax_out

            fea_bank[indx] = output_f_.detach().clone().cpu()
            score_bank[indx] = softmax_out.detach().clone()

            distance = output_f_ @ fea_bank.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
            idx_near = idx_near[:, 1:]  # batch x K
            score_near = score_bank[idx_near]  # batch x K x C

        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(
            -1, args.K, -1
        )  # batch x K x C

        loss = (F.kl_div(softmax_out_un.log(), score_near.log(), reduction="none").sum(-1)).sum(1)
        if args.wandb:
            wandb.log({'loss_kl': loss.mean(0).item()}, commit=False)

        mask = torch.ones((inputs_test.shape[0], inputs_test.shape[0]))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        copy = softmax_out.T  # .detach().clone()#

        dot_neg = softmax_out @ copy  # batch x batch

        dot_neg = (dot_neg * mask.cuda()).sum(-1)  # batch
        if args.wandb:
            wandb.log({'loss_neg': dot_neg.mean(0).item()}, commit=False)
        loss += dot_neg * alpha

        optimizer.zero_grad()
        optimizer_c.zero_grad()
        (loss.mean(0)).backward()
        optimizer.step()
        optimizer_c.step()

        if iter_num % len(iter(dset_loaders["target"])) == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if True: #args.dset == "VISDA-C":
                acc, accc = cal_acc(
                    dset_loaders["test"],
                    fea_bank,
                    score_bank,
                    netF,
                    netB,
                    netC,
                    args,
                    flag=True,
                )
                log_str = (
                    "Task: {}, Iter:{}/{};  Acc on target: {:.4f}".format(
                        args.name, iter_num, max_iter, acc
                    )
                    + "\n"
                    + "T: "
                    + accc
                )

            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str + "\n")
            netF.train()
            netB.train()
            netC.train()
            if acc>acc_log:
                acc_log = acc
                torch.save(
                    netF.state_dict(),
                    osp.join(args.output_dir, "target_F_" + str(args.seed) +'_' + str(args.tag) + ".pt"))
                torch.save(
                    netB.state_dict(),
                    osp.join(args.output_dir,
                                "target_B_" + str(args.seed) + '_' + str(args.tag) + ".pt"))
                torch.save(
                    netC.state_dict(),
                    osp.join(args.output_dir,
                                "target_C_" + str(args.seed) + '_' + str(args.tag) + ".pt"))

    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPA")
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--max_epoch", type=int, default=15, help="max iterations")
    parser.add_argument("--interval", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='cardiomegaly', choices=['cardiomegaly', 'VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--net", type=str, default="resnet50")
    parser.add_argument("--seed", type=int, default=2021, help="random seed")

    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='ckps/AaD/target')
    parser.add_argument('--output_src', type=str, default='ckps/AaD/source')
    parser.add_argument("--tag", type=str, default="LPA")
    parser.add_argument("--da", type=str, default="uda")
    parser.add_argument("--issave", type=bool, default=True)
    parser.add_argument("--cc", default=False, action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--alpha_decay", default=True)
    parser.add_argument("--nuclear", default=False, action="store_true")
    parser.add_argument("--var", default=False, action="store_true")

    parser.add_argument('--wandb', action='store_true', help="Use wandb")
    parser.add_argument('--run_name', type=str, default='chexpert_source')
    parser.add_argument('--proj_name', type=str, default='AaD')
    parser.add_argument('--train_resampling', default='balanced', choices=['natural', 'class', 'group', 'balanced'], type=str, help='')
    parser.add_argument('--test_resampling', default='balanced', choices=['natural', 'class', 'group', 'balanced'], type=str, help='')
    parser.add_argument('--flag', default='', choices=['', 'Younger', 'Older'], type=str, help='')
    parser.add_argument('--sens_classes', type=int, default=5, help="number of sensitive classes")
    parser.add_argument('--sens_name', type=str, default='age', choices=['age', 'sex', 'race'], help="name of sensitive attribute")

    args = parser.parse_args()

    if args.wandb:
        wandb.init(project=args.proj_name, name=args.run_name, config=args)

    if args.dset == "office-home":
        names = ["Art", "Clipart", "Product", "RealWorld"]
        args.class_num = 65
    if args.dset == "VISDA-C":
        names = ["train", "validation"]
        args.class_num = 12
    if args.dset == 'cardiomegaly':
        names = ['chexpert', 'mimic']
        args.class_num = 2

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    if True: #for i in range(len(names)):
        folder = "./data/"
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_list.txt"
        args.t_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"
        args.test_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"

        args.output_dir_src = osp.join(
            args.output_src, args.da, args.dset, names[args.s][0].upper(), args.net, str(args.seed)
        )
        args.output_dir = osp.join(
            args.output,
            args.da,
            args.dset,
            names[args.s][0].upper() + names[args.t][0].upper(), args.net, str(args.seed)
        )
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        print_args(args)
        if not osp.exists(args.output_dir):
            os.system("mkdir -p " + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.out_file = open(
            osp.join(args.output_dir, "log_{}.txt".format(args.tag)), "w"
        )
        args.out_file.write(print_args(args) + "\n")
        args.out_file.flush()
        train_target(args)
