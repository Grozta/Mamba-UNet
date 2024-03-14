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
torch.autograd.set_detect_anomaly(True)
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn.model_selection import KFold
from utils import losses, metrics, ramps, test_util, cube_losses, cube_utils, masked_loss
from config import get_config
from dataloaders.dataset import *
from networks.magicnet_2D_mask import VNet_Magic_2D_mask
from val_2D import test_single_volume_magic
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='ACDC', help='dataset_name')
parser.add_argument('--root_path', type=str, default='../ACDC', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='MagicNet_2D', help='exp_name')
parser.add_argument('--model', type=str, default='V-Net_2D', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7, help='labeled trained samples')
parser.add_argument('--labeled_bs', type=int, default=8, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--patch_size', type=list,  default=[256, 256],help='patch size of network input')
parser.add_argument('--cube_size', type=int, default=32, help='size of each cube,!! cube_size must be divisible by patch_size')
parser.add_argument('--masked_rate', type=int, default=0.25, help='size of each cube')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=5179, help='random seed')

parser.add_argument('--lamda', type=float, default=0.2, help='weight to balance all losses')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--T_dist', type=float, default=1.0, help='Temperature for organ-class distribution')

parser.add_argument('--resume', action='store_true',help='The default value is false, false means training is restart')
parser.add_argument('--is_save_more_log', action='store_true',help='The default value is false, true means adding more logs during training')
parser.add_argument('--save_log_interval',type=int, default=20, help='The iteration interval for log saving logs')
parser.add_argument('--is_save_checkpoint',action='store_true',help='The default value is false, true means save checkpoint during training')
parser.add_argument('--save_checkpoint_interval',type=int, default=400, help='The iteration interval for log saving logs')
parser.add_argument('--max_iterations', type=int, default=20000, help='maximum iteration to train')
args = parser.parse_args()
#config = get_config(args)
# 获取当前机器的核心数

def get_current_num_workers(rate=1.2):
    num_cpus = multiprocessing.cpu_count()
    # 设置 DataLoader 的 num_workers 为当前机器核心数的一半
    num_workers = int(max(num_cpus // rate, 1))  # 至少为1
    return num_workers

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
    net = VNet_Magic_2D_mask(n_channels=1, n_classes=n_classes, cube_size=cube_size, patch_size=patchsize,has_dropout=True,has_residual=True)
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

def config_log(snapshot_path, typename):

    formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)

    handler = logging.FileHandler(snapshot_path + "/log_{}.txt".format(typename), mode="w")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
    return handler, sh


def train(args, snapshot_path):

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    if not args.resume:
        logging.info(str(args))
    writer = SummaryWriter(os.path.join(snapshot_path,"log"))

    latest_checkpoint_path = os.path.join(snapshot_path,'{}_latest_checkpoint.pth').format(args.model)
    if not args.resume or not os.path.exists(latest_checkpoint_path):
        args.resume = False

    model = create_model(n_classes=args.num_classes, 
                         cube_size=args.cube_size, patchsize=args.patch_size[0])
    ema_model = create_model(n_classes=args.num_classes, 
                             cube_size=args.cube_size, patchsize=args.patch_size[0], ema=True)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    dice_loss = losses.MagicDiceLoss_2D(n_classes=args.num_classes)

    if args.resume:
        latest_checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(latest_checkpoint['model_state_dict'])
        ema_model.load_state_dict(latest_checkpoint['model_state_dict'])
        optimizer.load_state_dict(latest_checkpoint['optimizer_state_dict'])

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
    if args.resume:
        for i in range(latest_checkpoint['iteration']):
            batch_sampler.__iter__()
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=8, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    if not args.resume:
        logging.info("{} itertations per epoch".format(len(trainloader)))

    max_epoch = args.max_iterations // len(trainloader) + 1
    lr_ = args.base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    if args.resume:
        iterator.update(latest_checkpoint['epoch'])
        iter_num = latest_checkpoint['iteration']
        dist_logger = cube_utils.OrganClassLogger(
            num_classes=args.num_classes,
            class_dist=latest_checkpoint['dist_logger_class_dist'],
            class_total_pixel_store=latest_checkpoint['dist_logger_class_total_pixel_store'])

        best_performance =latest_checkpoint['best_performance']
        performance = latest_checkpoint['performance']
        logging.info("checkpoint has recovery.epoch_num:{},iter_num:{}".format(iterator.n,iter_num))
    else:

        iter_num = 0
        best_performance = 0.0
        performance = 0.0
        dist_logger = cube_utils.OrganClassLogger(num_classes=args.num_classes)
        logging.info("start.epoch_num:{},iter_num:{}".format(iterator.n,iter_num))
    
    loc_list = None
    model.train()
    ema_model.train()

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            # volume_batch[24,1,256,256] label_batch[24,256,256]
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda().to(torch.long)
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            labeled_volume_batch = volume_batch[:args.labeled_bs]

            model.train()
            # outputs[24,4,256,256]
            outputs = model(volume_batch)[0] # Original Model Outputs

            # Cross-image Partition-and-Recovery
            bs, c, w, h = volume_batch.shape
            nb_cubes = h // args.cube_size # 8
            # cube_part_ind[24,1,256,256] cube_rec_ind[24,16,256,256]
            cube_part_ind, cube_rec_ind = cube_utils.get_part_and_rec_ind_2d(volume_shape=volume_batch.shape,
                                                                          nb_cubes=nb_cubes,nb_chnls=16)
            # img_cross_mix[24,1,256,256]                                                              
            img_cross_mix = volume_batch.view(bs, c, w, h)
            img_cross_mix = torch.gather(img_cross_mix, dim=0, index=cube_part_ind)
            img_cross_mix = img_cross_mix.view(bs, c, w, h)
            # outputs_mix[24,4,256,256], embedding[24,16,256,256]
            outputs_mix, embedding = model(img_cross_mix)
            c_ = embedding.shape[1] # 16
            # pred_rec[24,16,256,256]
            pred_rec = torch.gather(embedding, dim=0, index=cube_rec_ind)
            pred_rec = pred_rec.view(bs, c_, w, h)
            # outputs_unmix[24,4,256,256]
            outputs_unmix = model.forward_prediction_head(pred_rec)

            # Get pseudo-label from teacher model
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)[0]
                unlab_pl_soft = F.softmax(ema_output, dim=1)
                pred_value_teacher, pred_class_teacher = torch.max(unlab_pl_soft, dim=1)

            if iter_num == 0 or args.resume:
                # loc_list[64,1]
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
            # patch_list: N=64 x [24, 1, 1, 32, 32] (bs, pn, c, w, h)
            patch_list = cube_losses.get_patch_list_2d(volume_batch, cube_size=args.cube_size)
            # idx = 64
            idx = torch.randperm(len(patch_list)).cuda()
            # cube location loss
            loc_loss = 0
            feat_list = None
            if loc_list is not None:
                # pathc inner # feat_list 24x[f1,f2,f3,f4,f5]
                loc_loss, feat_list = cube_losses.cube_location_loss(model, loc_list, patch_list, idx)

            # patch outer
            shuffled_loss, pos_embed_pre = masked_loss.get_shuffled_recovery_loss(model,volume_batch.clone(),args.cube_size)
            mask_recovery_loss, pos_embed_mask = masked_loss.get_mask_recovery_loss(model,volume_batch.clone(), args.masked_rate, args.cube_size)
            mask_recovery_shuffled_loss = shuffled_loss = F.mse_loss(pos_embed_pre, pos_embed_mask)
            loc_recv_loss = shuffled_loss + mask_recovery_loss + mask_recovery_shuffled_loss
            #loc_recv_loss = 0.0

            
            consistency_loss = 0
            count_consist = 1

            # Within-image Partition-and-Recovery
            if feat_list is not None:
                embed_list = []
                for i in range(bs):
                    # pred_tmp: [f1-f5] -> 64x4x32x32
                    # embed_tmp: [f1-f5] -> 64x16x32x32
                    pred_tmp, embed_tmp = model.forward_decoder(feat_list[i])
                    # add batch_size dimension: 27x9x32x32x32 -> 1x64x16x32x32
                    embed_list.append(embed_tmp.unsqueeze(0))

                #embed_all 24x64x16x32x32
                embed_all = torch.cat(embed_list, dim=0)
                #embed_all_unmix[24x16x256x256]
                embed_all_unmix = cube_losses.unmix_tensor_2d(embed_all, labeled_volume_batch.shape)
                #pred_all_unmix[24x4x256x256]
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
            loss = supervised_loss + 0.1 * loc_loss + consistency_weight * consistency_loss + loc_recv_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1

            if iter_num % args.save_log_interval == 0:
                logging.info('iteration {}: loss: {:.3f},cons_dist: {:.3f},loss_weight: {:f},loss_loc: {:.3f}'.
                             format(iter_num, loss, consistency_loss, consistency_weight, 0.1 * loc_loss))

            lr_ = args.base_lr * (1.0 - iter_num / args.max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            if iter_num % args.save_log_interval == 0 and args.is_save_more_log:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                    outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

                writer.add_scalar('consistency_weight/consistency_weight', consistency_weight, iter_num)
                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/total_loss', loss.item(), iter_num)
                writer.add_scalar('loss/supervised_loss', supervised_loss.item(), iter_num)
                writer.add_scalar('loss/loc_loss', 0.1 * (loc_loss.item()), iter_num)
                writer.add_scalar('loss/consistency_loss', consistency_loss.item(), iter_num)
                writer.add_scalar('loss/consistency_weight',consistency_weight, iter_num)

            if iter_num % args.save_checkpoint_interval == 0 and iter_num < args.max_iterations:
                # per 400 iter, validate case and save checkpoint
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_magic(
                        sampled_batch["image"], sampled_batch["label"], model, classes=args.num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                
                # save validation log
                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                logging.info('iter_num:{},model_val_mean_dice:{:.3f},model_val_mean_hd95: {:.3f}'.
                             format(iter_num, performance, mean_hd95))
                
                if args.is_save_more_log: 
                    writer.add_scalar('info/model_val_mean_dice',
                                    performance, iter_num)
                    writer.add_scalar('info/model_val_mean_hd95',
                                    mean_hd95, iter_num)
                    for class_i in range(args.num_classes-1):
                        writer.add_scalar('info/model_val_{}_dice'.format(class_i+1),
                                        metric_list[class_i, 0], iter_num)
                        writer.add_scalar('info/model_val_{}_hd95'.format(class_i+1),
                                        metric_list[class_i, 1], iter_num)
                # save resume checkpoint 
                latest_checkpoint = {
                'epoch': epoch_num,
                'iteration': iter_num,
                'best_performance' :best_performance,
                'performance':performance,
                'dist_logger_class_dist': dist_logger.class_dist,
                'dist_logger_class_total_pixel_store':dist_logger.class_total_pixel_store,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }
                
                torch.save(latest_checkpoint, latest_checkpoint_path)
                logging.info("save model to {}".format(latest_checkpoint_path))

            if performance > best_performance:
                best_performance = performance
                save_best = os.path.join(snapshot_path,
                                            '{}_best_model.pth'.format(args.model))
                torch.save(model.state_dict(), save_best)
                logging.info("save model to {}".format(save_best))

            if iter_num % 4000 == 0 or iter_num >= args.max_iterations:
                save_mode_path = os.path.join(snapshot_path,
                                                  'model_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            model.train()

            if iter_num >= args.max_iterations:
                break
            model.train()
        if iter_num >= args.max_iterations:
            iterator.close()
            break
    writer.close()


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
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git/*', '__pycache__/*']))

    train(args, snapshot_path)
