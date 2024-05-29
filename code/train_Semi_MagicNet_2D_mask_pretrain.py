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
parser.add_argument('--pretrain_path', type=str, default='../pretrained_ckpt/MagicNet_2D_mask_pretrain.pth', help='path of pretrain')
parser.add_argument('--exp', type=str, default='MagicNet_2D_mask_pretrain', help='exp_name')
parser.add_argument('--model', type=str, default='V-Net_2D_mask', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7, help='labeled trained samples')
parser.add_argument('--labeled_bs', type=int, default=12, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
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
parser.add_argument('--max_iterations', type=int, default=60000, help='maximum iteration to train')
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
    net = VNet_Magic_2D_mask(n_channels=1, n_classes=n_classes, cube_size=cube_size, patch_size=patchsize, has_dropout=True,has_residual=True)
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
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    dice_loss = losses.MagicDiceLoss_2D(n_classes=args.num_classes)

    if os.path.exists(args.pretrain_path) and not args.resume:
        pretrain = torch.load(args.pretrain_path)
        model.load_state_dict(pretrain)

    if args.resume:
        latest_checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(latest_checkpoint['model_state_dict'])
        optimizer.load_state_dict(latest_checkpoint['optimizer_state_dict'])

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, 
                            transform=transforms.Compose([RandomGeneratorv2(args.patch_size)]))

    print("Total silices is: {}".format(len(db_train)))
    trainloader = DataLoader(db_train, batch_size=args.batch_size,
                             num_workers=get_current_num_workers(), pin_memory=True)
    if not args.resume:
        logging.info("{} itertations per epoch".format(len(trainloader)))

    max_epoch = args.max_iterations // len(trainloader) + 1
    lr_ = args.base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    if args.resume:
        iterator.update(latest_checkpoint['epoch'])
        iter_num = latest_checkpoint['iteration']
        logging.info("checkpoint has recovery.epoch_num:{},iter_num:{}".format(iterator.n,iter_num))
    else:

        iter_num = 0
        logging.info("start.epoch_num:{},iter_num:{}".format(iterator.n,iter_num))
    
    loc_list = None
    model.train()
    best_loss = 1.0
    best_state_dict = None

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            # volume_batch[24,1,256,256] label_batch[24,256,256]
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda().to(torch.long)

            model.train()

            patch_list = cube_losses.get_patch_list_2d(volume_batch, cube_size=args.cube_size)
            # idx = 64
            idx = torch.randperm(len(patch_list)).cuda()
            if iter_num == 0 or args.resume:
                # loc_list[64,1]
                loc_list = cube_utils.get_loc_mask_2d(volume_batch, args.cube_size)
            if loc_list is not None:
                # pathc inner # feat_list 24x[f1,f2,f3,f4,f5]
                loc_loss, feat_list = cube_losses.cube_location_loss(model, loc_list, patch_list, idx)

            # patch outer
            shuffled_loss, pos_embed_pre = masked_loss.get_shuffled_recovery_loss(model,volume_batch,args.cube_size)
            mask_recovery_loss, pos_embed_mask = masked_loss.get_mask_recovery_loss(model,volume_batch, args.masked_rate, args.cube_size)
            mask_recovery_shuffled_loss = shuffled_loss = F.mse_loss(pos_embed_pre, pos_embed_mask)
            loss = shuffled_loss + mask_recovery_loss + mask_recovery_shuffled_loss + loc_loss * 0.1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            

            if iter_num % args.save_log_interval == 0:
                logging.info('iteration {}: loss: {:.3f},'
                             'shuffled_loss: {:.3f},mask_recovery_loss: {:.3f},'
                             'mask_recovery_shuffled_loss: {:.3f},'
                             'loc_loss: {:.3f}'
                             .format(iter_num, loss,
                                     shuffled_loss ,mask_recovery_loss ,
                                       mask_recovery_shuffled_loss , loc_loss * 0.1))

            lr_ = args.base_lr * (1.0 - iter_num / args.max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            if best_loss>loss.item():
                best_loss = loss.item()
                best_state_dict = model.state_dict()
                writer.add_scalar('loss/best_loss', best_loss, iter_num)

            if iter_num % args.save_log_interval == 0 and args.is_save_more_log:

                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/loss', loss.item(), iter_num)
                writer.add_scalar('loss/shuffled_loss', shuffled_loss.item(), iter_num)
                writer.add_scalar('loss/mask_recovery_loss', mask_recovery_loss.item(), iter_num)
                writer.add_scalar('loss/mask_recovery_shuffled_loss', mask_recovery_shuffled_loss.item(), iter_num)
                writer.add_scalar('loss/loc_loss', 0.1 * (loc_loss.item()), iter_num)

            if iter_num % 10000 == 0 or iter_num >= args.max_iterations:
                save_mode_path = os.path.join(snapshot_path,'MagicNet_2D_mask_pretrain_model_iter_{}.pth'.format(iter_num))
                torch.save(model.state_dict(), save_mode_path)

                logging.info("save model to {}".format(save_mode_path))
                                # save resume checkpoint 
                latest_checkpoint = {
                'epoch': epoch_num,
                'iteration': iter_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }
                latest_checkpoint_path = os.path.join(snapshot_path,'MagicNet_2D_mask_pretrain_model_latest_checkpoint.pth')
                torch.save(latest_checkpoint, latest_checkpoint_path)
                logging.info("save model to {}".format(latest_checkpoint_path))

            model.train()
            
            if iter_num >= args.max_iterations:
                break
            iter_num = iter_num + 1
            model.train()
        if iter_num >= args.max_iterations:
            save_mode_path = os.path.join(snapshot_path,'MagicNet_2D_mask_pretrain_model_best.pth')
            torch.save(best_state_dict, save_mode_path)
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
