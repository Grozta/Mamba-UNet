import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator, BaseDataSets4v1, RandomGeneratorv4
from networks.net_factory import net_factory
from networks.unet import initialize_module
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds, test_single_volume_for_trainLabel

parser = argparse.ArgumentParser()
parser.add_argument('--train_label',default=False, 
                    action="store_true", help="train label mode")
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised_TrainLabel', help='experiment_name')
parser.add_argument('--tag',type=str,
                    default='v99', help='tag of experiment')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--pretrain_path', type=str,
                    default='../model/ACDC/Fully_Supervised_140_labeled/unet/unet_best_model.pth', help='pretrain model path')
parser.add_argument('--mask_pretrain_path', type=str,
                    default='../model/ACDC/trainLabel_140_labeled/unet_v0.4/unet_best_model.pth', help='mask pretrain model path')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=140,
                    help='labeled data')
parser.add_argument('--num_workers', type=int, default=16,
                    help='numbers of workers in dataloader')
parser.add_argument('--ema_decay', type=float,  default=0.999, 
                    help='ema_decay')
args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136, "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        
def get_current_consistency_weight(consistency,epoch):  
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
        
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)  

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    if args.train_label:
        seg_model = net_factory(None,args,net_type=args.model, in_chns=1, class_num=num_classes)
        mad_model = net_factory(None,args,net_type=args.model, in_chns=num_classes, class_num=num_classes)
        ema_model = net_factory(None,args,net_type=args.model, in_chns=num_classes, class_num=num_classes)
    else:
        model = net_factory(None,args,net_type='unet', in_chns=num_classes, class_num=num_classes)
        
    if args.train_label:
        model_pretrained_dict = torch.load(args.pretrain_path)
        mask_model_pretrained_dict = torch.load(args.mask_pretrain_path)
        seg_model.load_state_dict(model_pretrained_dict, strict=False)
        mad_model.load_state_dict(mask_model_pretrained_dict, strict=False)
        initialize_module(ema_model)
        
    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=labeled_slice, transform=transforms.Compose([
        RandomGeneratorv4(args.patch_size, num_classes=num_classes)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True,worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,num_workers =1)

    seg_model.train()
    mad_model.train()
    ema_model.train()

    optimizer_seg = optim.SGD(seg_model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer_mad = optim.SGD(mad_model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer_ema = optim.SGD(ema_model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch, mask_label_batch = sampled_batch['image'], sampled_batch['label'],sampled_batch['mask_label']
            volume_batch, label_batch, mask_label_batch = volume_batch.cuda(), label_batch.cuda(), mask_label_batch.cuda()
            
            """struct
            label-----------|-----mad_model
                           /x/       |
                            |        | ema_update
            img------seg_model----ema_model------------output
            """            
            seg_outputs = seg_model(volume_batch)
            seg_outputs_soft = torch.softmax(seg_outputs, dim=1)
            
            # mask_input = seg_outputs_soft.detach()
            # blend_outputs = (mask_input+mask_label_batch)/2
            # blend_input = torch.softmax(blend_outputs, dim=1)
            mad_outputs = mad_model(mask_label_batch)
            mad_outputs_soft = torch.softmax(seg_outputs, dim=1)
            
            ema_outputs = ema_model(seg_outputs_soft)
            ema_outputs_soft = torch.softmax(ema_outputs, dim=1)
            
            #------------------------loss------------------------------
            seg_loss_ce = ce_loss(seg_outputs, label_batch[:].long())
            seg_loss_dice = dice_loss(seg_outputs_soft, label_batch.unsqueeze(1))
            seg_loss = 0.5 * (seg_loss_dice + seg_loss_ce)*0.8
            
            mad_loss_ce = ce_loss(mad_outputs, label_batch[:].long())
            mad_loss_dice = dice_loss(mad_outputs_soft, label_batch.unsqueeze(1))
            mad_loss = 0.5 * (mad_loss_dice + mad_loss_ce)*0.6
            
            ema_loss_ce = ce_loss(ema_outputs, label_batch[:].long())
            ema_loss_dice = dice_loss(ema_outputs_soft, label_batch.unsqueeze(1))
            ema_loss = 0.5 * (ema_loss_dice + ema_loss_ce)
            
            loss = seg_loss + mad_loss + ema_loss
            
            optimizer_seg.zero_grad()
            optimizer_mad.zero_grad()
            optimizer_ema.zero_grad()
            loss.backward()
            optimizer_seg.step()
            optimizer_mad.step()
            optimizer_ema.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group,param_group_ema,param_group_mad in zip(optimizer_seg.param_groups,optimizer_ema.param_groups,optimizer_mad.param_groups):
                param_group['lr'] = lr_
                param_group_ema['lr'] = lr_
                param_group_mad['lr'] = lr_
                
            update_ema_variables(mad_model, ema_model, args.ema_decay, iter_num)
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/seg_loss', seg_loss, iter_num)
            writer.add_scalar('info/mad_loss', mad_loss, iter_num)
            writer.add_scalar('info/ema_loss', ema_loss, iter_num)

            logging.info(
                'iteration %d : loss : %f, seg_loss: %f, mad_loss: %f, ema_loss: %f' %
                (iter_num, loss.item(), seg_loss.item(), mad_loss.item(), ema_loss.item()))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
                
                mask_labs = torch.argmax(torch.softmax(
                    mask_label_batch, dim=1), dim=1, keepdim=True)
                writer.add_image('train/mask_lable',
                                 mask_labs[1, ...] * 50, iter_num)
                
                seg_outputs = torch.argmax(torch.softmax(
                    seg_outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/segPrediction',
                                 seg_outputs[1, ...] * 50, iter_num)
                
                ema_outputs = torch.argmax(torch.softmax(
                    ema_outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/emaPrediction',
                                 ema_outputs[1, ...] * 50, iter_num)
                
                mad_outputs = torch.argmax(torch.softmax(
                    mad_outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/madPrediction',
                                 mad_outputs[1, ...] * 50, iter_num)
                
                
            if iter_num > 0 and iter_num % 200 == 0:
                seg_model.eval()
                mad_model.eval()
                ema_model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_for_trainLabel(
                        sampled_batch["image"], sampled_batch["label"], seg_model, ema_model,classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    check_point={"seg_model":seg_model.state_dict(),"ema_model":ema_model.state_dict()}
                    torch.save(check_point, save_mode_path)
                    torch.save(check_point, save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                seg_model.train()
                mad_model.train()
                ema_model.train()

            if iter_num % 2000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                check_point={"seg_model":seg_model.state_dict(),"ema_model":ema_model.state_dict()}
                torch.save(check_point, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


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
    
    snapshot_path = "../model/{}_{}_labeled/{}_{}".format(
        args.exp, args.labeled_num, args.model, args.tag)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git/*', '__pycache__/*','pretrained_ckpt/*']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
