import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn.model_selection import KFold
from utils import losses, metrics, ramps, test_util, cube_losses, cube_utils
from config import get_config
from dataloaders.dataset import *
from networks.magicnet_2D import VNet_Magic_2D


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='ACDC', help='dataset_name')
parser.add_argument('--root_path', type=str, default='../ACDC', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='MagicNet_2D', help='exp_name')
parser.add_argument('--model', type=str, default='V-Net_2D', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,help='output channel of network')
parser.add_argument('--max_iterations', type=int, default=20000, help='maximum iteration to train')
parser.add_argument('--labeled_num', type=int, default=7, help='labeled trained samples')
parser.add_argument('--labeled_bs', type=int, default=12, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--patch_size', type=list,  default=[256, 256],help='patch size of network input')
parser.add_argument('--cube_size', type=int, default=32, help='size of each cube')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=5179, help='random seed')

parser.add_argument('--lamda', type=float, default=0.2, help='weight to balance all losses')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--T_dist', type=float, default=1.0, help='Temperature for organ-class distribution')
args = parser.parse_args()
#config = get_config(args)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def create_model(n_classes=14, cube_size=32, patchsize=96, ema=False):
    # Network definition
    net = VNet_Magic_2D(n_channels=1, n_classes=n_classes, cube_size=cube_size, patch_size=patchsize,has_dropout=True,has_residual=True)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def config_log(snapshot_path_tmp, typename):

    formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)

    handler = logging.FileHandler(snapshot_path_tmp + "/log_{}.txt".format(typename), mode="w")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
    return handler, sh


def train(args, snapshot_path):
    test_list=[]
    snapshot_path_tmp = snapshot_path

    model = create_model(n_classes=args.num_classes, 
                         cube_size=args.cube_size, patchsize=args.patch_size[0])
    #model1.load_from(config)
    ema_model = create_model(n_classes=args.num_classes, 
                             cube_size=args.cube_size, patchsize=args.patch_size[0], ema=True)
    #model1.load_from(config)
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, 
                            transform=transforms.Compose([RandomGeneratorv2(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, 
                                          args.batch_size, args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(os.path.join(snapshot_path_tmp,"log"))
    logging.info("{} itertations per epoch".format(len(trainloader)))

    dice_loss = losses.MagicDiceLoss_2D(n_classes=args.num_classes)

    iter_num = 0
    best_dice_avg = 0
    metric_all_cases = None
    max_epoch = args.max_iterations // len(trainloader) + 1
    lr_ = args.base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    loc_list = None
    dist_logger = cube_utils.OrganClassLogger(num_classes=args.num_classes)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda().to(torch.long)
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            labeled_volume_batch = volume_batch[:args.labeled_bs]

            model.train()
            outputs = model(volume_batch)[0] # Original Model Outputs

            # Cross-image Partition-and-Recovery
            bs, c, w, h = volume_batch.shape
            # cube_size: size of each small-cube
            # cube_size = 24
            # nb_cubes: (96, 96, 96) // (24, 24, 24) -> (4, 4, 4)
            # nb_cubes = 4
            nb_cubes = h // args.cube_size
            cube_part_ind, cube_rec_ind = cube_utils.get_part_and_rec_ind_2d(volume_shape=volume_batch.shape,
                                                                          nb_cubes=nb_cubes,
                                                                          nb_chnls=16)

            # volume_batch: 4, 1, 96, 96, 96
            # img_cross: 4, 1, 96, 96, 96
            img_cross_mix = volume_batch.view(bs, c, w, h)
            img_cross_mix = torch.gather(img_cross_mix, dim=0, index=cube_part_ind)
            img_cross_mix = img_cross_mix.view(bs, c, w, h)

            outputs_mix, embedding = model(img_cross_mix)
            c_ = embedding.shape[1]
            pred_rec = torch.gather(embedding, dim=0, index=cube_rec_ind)
            pred_rec = pred_rec.view(bs, c_, w, h)
            outputs_unmix = model.forward_prediction_head(pred_rec)

            # Get pseudo-label from teacher model
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)[0]
                unlab_pl_soft = F.softmax(ema_output, dim=1)
                pred_value_teacher, pred_class_teacher = torch.max(unlab_pl_soft, dim=1)

            # nt = 3, ts = 32
            # loc_list: 27 x [1, 1] (x + Wy + WHz)
            if iter_num == 0:
                loc_list = cube_utils.get_loc_mask_2d(volume_batch, args.cube_size)

            # calculate some losses
            loss_seg = F.cross_entropy(outputs[:args.labeled_bs], label_batch[:args.labeled_bs])
            outputs_soft = F.softmax(outputs, dim=1)
            outputs_unmix_soft = F.softmax(outputs_unmix, dim=1)
            loss_seg_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs])
            loss_unmix_dice = dice_loss(outputs_unmix_soft[:args.labeled_bs], label_batch[:args.labeled_bs])
            supervised_loss = (loss_seg + loss_seg_dice + loss_unmix_dice)
            count_ss = 3

            # Magic-cube Location Reasoning
            # patch_list: N=27 x [4, 1, 1, 32, 32, 32] (bs, pn, c, w, h, d)
            patch_list = cube_losses.get_patch_list_2d(volume_batch, cube_size=args.cube_size)
            # idx = 27
            idx = torch.randperm(len(patch_list)).cuda()
            # cube location loss
            loc_loss = 0
            feat_list = None
            if loc_list is not None:
                loc_loss, feat_list = cube_losses.cube_location_loss(model, loc_list, patch_list, idx, args.labeled_bs, cube_size=args.cube_size)

            consistency_loss = 0
            count_consist = 1

            # Within-image Partition-and-Recovery
            if feat_list is not None:
                embed_list = []
                for i in range(bs):
                    # pred_tmp: [f1-f5] -> 27x9x32x32x32
                    # embed_tmp: [f1-f5] -> 27x16x32x32x32
                    pred_tmp, embed_tmp = model.forward_decoder(feat_list[i])
                    # add batch_size dimension: 27x9x32x32x32 -> 1x27x9x32x32x32
                    embed_list.append(embed_tmp.unsqueeze(0))

                # embed_all: 2 x [1, 27, 16, 32, 32, 32] -> 2, 27, 16, 32, 32, 32 -> 2, 16, 96, 96, 96
                embed_all = torch.cat(embed_list, dim=0)
                embed_all_unmix = cube_losses.unmix_tensor_2d(embed_all, labeled_volume_batch.shape)
                pred_all_unmix = model.forward_prediction_head(embed_all_unmix)
                unmix_pred_soft = F.softmax(pred_all_unmix, dim=1)

                loss_lab_local_dice = dice_loss(unmix_pred_soft[:args.labeled_bs], label_batch[:args.labeled_bs])
                supervised_loss += loss_lab_local_dice
                count_ss += 1

            # Cube-wise Pseudo-label Blending
            pred_class_mix = None
            with torch.no_grad():
                # To store some class pixels at the beginning of training to calculate the organ-class dist
                if iter_num > 100 and feat_list is not None:
                    # Get organ-class distribution
                    current_organ_dist = dist_logger.get_class_dist().cuda()  # (1, C)
                    # Normalize
                    current_organ_dist = current_organ_dist ** (1. / args.T_dist)
                    current_organ_dist = current_organ_dist / current_organ_dist.sum()
                    current_organ_dist = current_organ_dist / current_organ_dist.max()

                    # weight_map(omega of R): 2x96x96x96 -> 2x1x96x96x96 -> 2x14x96x96x96
                    weight_map = current_organ_dist[pred_class_teacher].unsqueeze(1).repeat(1, args.num_classes, 1, 1)

                    # un_pl: 2x9x96x96x96(no softmax), ema_output: 2x9x96x96x96(no softmax)
                    unmix_pl = cube_losses.get_mix_pl_2d(model, feat_list, volume_batch.shape, bs - args.labeled_bs)
                    unlab_pl_mix = (1. - weight_map) * ema_output + weight_map * unmix_pl
                    unlab_pl_mix_soft = F.softmax(unlab_pl_mix, dim=1)
                    _, pred_class_mix = torch.max(unlab_pl_mix_soft, dim=1)

                    # pr_class: 2x96**3, 1
                    conf, pr_class = torch.max(unlab_pl_mix_soft.detach(), dim=1)
                    dist_logger.append_class_list(pr_class.view(-1, 1))

                elif feat_list is not None:
                    conf, pr_class = torch.max(unlab_pl_soft.detach(), dim=1)
                    dist_logger.append_class_list(pr_class.view(-1, 1))

                # if iter_num > 12000 and iter_num % 100 and len(dist_logger.pl_total_list_in):
            if iter_num % 20 == 0 and len(dist_logger.class_total_pixel_store):
                dist_logger.update_class_dist()

            consistency_weight = get_current_consistency_weight(iter_num // 350)
            # debiase the pseudo-label: blend ema and unmixed_within pseudo-label
            if pred_class_mix is None:
                consistency_loss_unmix = dice_loss(outputs_unmix_soft[args.labeled_bs:], pred_class_teacher)
            else:
                consistency_loss_unmix = dice_loss(outputs_unmix_soft[args.labeled_bs:], pred_class_mix)

            consistency_loss += consistency_loss_unmix

            supervised_loss /= count_ss
            consistency_loss /= count_consist

            # Final Loss
            loss = supervised_loss + 0.1 * loc_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1

            if iter_num % 20 == 0:
                logging.info('iteration {}: loss: {:.3f}, '
                             'cons_dist: {:.3f}, loss_weight: {:f}, '
                             'loss_loc: {:.3f}'.format(iter_num,
                                                       loss,
                                                       consistency_loss,
                                                       consistency_weight,
                                                       0.1 * loc_loss))
                # save checkpoint 
                # save_mode_path = os.path.join(snapshot_path_tmp, 'latest.pth')
                # torch.save(model.state_dict(), save_mode_path)
                # writer.add_scalar('train_loss/Total_loss', loss, iter_num)
                # logging.info("save model to {}".format(save_mode_path))

            lr_ = args.base_lr * (1.0 - iter_num / args.max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            # if iter_num >= 2500 and iter_num % 2500 == 0:
            if iter_num >= args.max_iterations:
                model.eval()
                dice_all, std_all, metric_all_cases = test_util.validation_all_case(model,
                                                                                    num_classes=args.num_classes,
                                                                                    base_dir=args.root_path,
                                                                                    image_list=test_list,
                                                                                    patch_size=args.patch_size,
                                                                                    stride_xy=16,
                                                                                    stride_z=16)
                dice_avg = dice_all.mean()

                logging.info('iteration {}, '
                             'average DSC: {:.3f}, '
                             'spleen: {:.3f}, '
                             'r.kidney: {:.3f}, '
                             'l.kidney: {:.3f}, '
                             'gallbladder: {:.3f}, '
                             'esophagus: {:.3f}, '
                             'liver: {:.3f}, '
                             'stomach: {:.3f}, '
                             'aorta: {:.3f}, '
                             'inferior vena cava: {:.3f}'
                             'portal vein and splenic vein: {:.3f}, '
                             'pancreas: {:.3f}, '
                             'right adrenal gland: {:.3f}, '
                             'left adrenal gland: {:.3f}'
                             .format(iter_num,
                                     dice_avg,
                                     dice_all[0],
                                     dice_all[1],
                                     dice_all[2],
                                     dice_all[3],
                                     dice_all[4],
                                     dice_all[5],
                                     dice_all[6],
                                     dice_all[7],
                                     dice_all[8],
                                     dice_all[9],
                                     dice_all[10],
                                     dice_all[11],
                                     dice_all[12]))

                # if dice_avg > best_dice_avg:
                #     best_dice_avg = dice_avg
                #     save_mode_path = os.path.join(snapshot_path_tmp, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice_avg))
                #     save_best_path = os.path.join(snapshot_path_tmp, '{}_best_model.pth'.format(args.model))
                #     torch.save(model.state_dict(), save_mode_path)
                #     torch.save(model.state_dict(), save_best_path)
                #     logging.info("save best model to {}".format(save_mode_path))
                # writer.add_scalar('Var_dice/Dice', dice_avg, iter_num)
                # writer.add_scalar('Var_dice/Best_dice', best_dice_avg, iter_num)
                model.train()

            if iter_num >= args.max_iterations:
                save_mode_path = os.path.join(snapshot_path_tmp, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                break
        if iter_num >= args.max_iterations:
            iterator.close()
            break
    writer.close()
    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)

    return metric_all_cases


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    metric_final = train(args, snapshot_path)

    # 12x4x13
    # 4x13, 4x13
    metric_mean, metric_std = np.mean(metric_final, axis=0), np.std(metric_final, axis=0)

    metric_save_path = os.path.join(snapshot_path, 'metric_final_{}_{}.npy'.format(args.dataset_name, args.exp))
    np.save(metric_save_path, metric_final)

    handler, sh = config_log(snapshot_path, 'total_metric')
    logging.info('Final Average DSC:{:.4f}, HD95: {:.4f}, NSD: {:.4f}, ASD: {:.4f}, '
                 'spleen: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'r.kidney: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'l.kidney: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'gallbladder: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'esophagus: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'liver: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'stomach: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'aorta: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'ivc: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'portal and splenic vein: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'pancreas: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'right adrenal gland: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, '
                 'Left adrenal gland: {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}, {:.4f}+-{:.4f}'
                 .format(metric_mean[0].mean(), metric_mean[1].mean(), metric_mean[2].mean(), metric_mean[3].mean(),
                         metric_mean[0][0], metric_std[0][0], metric_mean[1][0], metric_std[1][0], metric_mean[2][0], metric_std[2][0], metric_mean[3][0], metric_std[3][0],
                         metric_mean[0][1], metric_std[0][1], metric_mean[1][1], metric_std[1][1], metric_mean[2][1], metric_std[2][1], metric_mean[3][1], metric_std[3][1],
                         metric_mean[0][2], metric_std[0][2], metric_mean[1][2], metric_std[1][2], metric_mean[2][2], metric_std[2][2], metric_mean[3][2], metric_std[3][2],
                         metric_mean[0][3], metric_std[0][3], metric_mean[1][3], metric_std[1][3], metric_mean[2][3], metric_std[2][3], metric_mean[3][3], metric_std[3][3],
                         metric_mean[0][4], metric_std[0][4], metric_mean[1][4], metric_std[1][4], metric_mean[2][4], metric_std[2][4], metric_mean[3][4], metric_std[3][4],
                         metric_mean[0][5], metric_std[0][5], metric_mean[1][5], metric_std[1][5], metric_mean[2][5], metric_std[2][5], metric_mean[3][5], metric_std[3][5],
                         metric_mean[0][6], metric_std[0][6], metric_mean[1][6], metric_std[1][6], metric_mean[2][6], metric_std[2][6], metric_mean[3][6], metric_std[3][6],
                         metric_mean[0][7], metric_std[0][7], metric_mean[1][7], metric_std[1][7], metric_mean[2][7], metric_std[2][7], metric_mean[3][7], metric_std[3][7],
                         metric_mean[0][8], metric_std[0][8], metric_mean[1][8], metric_std[1][8], metric_mean[2][8], metric_std[2][8], metric_mean[3][8], metric_std[3][8],
                         metric_mean[0][9], metric_std[0][9], metric_mean[1][9], metric_std[1][9], metric_mean[2][9], metric_std[2][9], metric_mean[3][9], metric_std[3][9],
                         metric_mean[0][10], metric_std[0][10], metric_mean[1][10], metric_std[1][10], metric_mean[2][10], metric_std[2][10], metric_mean[3][10], metric_std[3][10],
                         metric_mean[0][11], metric_std[0][11], metric_mean[1][11], metric_std[1][11], metric_mean[2][11], metric_std[2][11], metric_mean[3][11], metric_std[3][11],
                         metric_mean[0][12], metric_std[0][12], metric_mean[1][12], metric_std[1][12], metric_mean[2][12], metric_std[2][12], metric_mean[3][12], metric_std[3][12]))

    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)
