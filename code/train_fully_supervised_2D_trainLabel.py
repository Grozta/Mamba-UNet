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
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import get_config
from dataloaders.dataset import BaseDataSets4pretrain,RandomGeneratorv_4_finetune
from networks.net_factory import net_factory
from utils import losses
from val_2D import test_single_volume_for_trainLabel
from utils.utils import label2color, worker_init_fn, get_model_struct_mode, update_ema_variables, get_pth_files, calculate_metric_percase
from utils.argparse_c import parser

def test_fine_tune(args, snapshot_path):
    writer = args.writer
    model = net_factory(None,args,net_type=args.model, in_chns=args.input_channels, class_num=args.num_classes)
    db_test = BaseDataSets4pretrain(args,mode = "test",
                                   transform=transforms.Compose([RandomGeneratorv_4_finetune(args)])) 
    testloader = DataLoader(db_test, batch_size=1, shuffle=True,pin_memory=True,num_workers =args.num_workers)
    pth_list = get_pth_files(snapshot_path)
    iterator = tqdm(range(len(pth_list)), ncols=70)
    metric_list = []
    for iter_num in iterator:
        pth = pth_list[iter_num]
        model_pretrained_dict = torch.load(os.path.join(snapshot_path,pth))
        model.load_state_dict(model_pretrained_dict)
        model.eval()
        metric_list = []
        for i_batch, sampled_batch in enumerate(testloader):
            test_image, test_label = sampled_batch['image'], sampled_batch['label']
            test_image, test_label = test_image.cuda(), test_label.numpy()
            outputs = model(test_image)
            pred = torch.argmax(torch.softmax(outputs, dim=1),dim=1).detach().cpu().numpy()
            if args.image_fusion_mode in [3,4,5]:
                image_origin  = test_image[0,...].cpu().numpy()
                writer.add_image(f'test_{pth}/Image_origin', image_origin[0], iter_num, dataformats='HW')
                writer.add_image(f'test_{pth}/Image_mix', label2color(np.argmax(image_origin[1:],axis=0)), iter_num,dataformats='HWC')
            elif args.image_fusion_mode in [1,2]:
                image_origin  = test_image[0,...].cpu().numpy()
                writer.add_image(f'test_{pth}/Image_origin', image_origin[0], iter_num, dataformats='HW')
                writer.add_image(f'test_{pth}/Image_pred', label2color(image_origin[1]), iter_num,dataformats='HWC')
            elif args.input_channels == 1:
                image = test_image[0,0, ...].cpu().numpy()
                writer.add_image(f'test_{pth}/Image', label2color(image), i_batch,dataformats='HWC')
            else:
                image = torch.argmax(test_image[0, ...], dim=0).cpu().numpy()
                writer.add_image(f'test_{pth}/Image', label2color(image), i_batch,dataformats='HWC')
                
            prediction = pred[0]
            writer.add_image(f'test_{pth}/Prediction', label2color(prediction), i_batch,dataformats='HWC')
            labs = test_label[0, ...]
            writer.add_image(f'test_{pth}/GroundTruth',label2color(labs), i_batch,dataformats='HWC')

            first_metric = calculate_metric_percase(prediction == 1, labs == 1)
            second_metric = calculate_metric_percase(prediction == 2, labs == 2)
            third_metric = calculate_metric_percase(prediction == 3, labs == 3)
            metric_i = np.array([first_metric,second_metric,third_metric])# 3x2
            metric_i[metric_i==None] = np.nan
            if not np.all(np.isnan(metric_i.astype(float))):
                metric_i = np.nanmean(metric_i,axis=0)#1x2
                metric_list.append(metric_i)
            
        metric_list = np.stack(metric_list)
        performance = np.nanmean(metric_list,axis=0)
        logging.info(f'{pth}_metric :{performance}' )
        writer.add_text(f'test/performance',f"{pth}:"+str(performance),iter_num)

def train(args, snapshot_path):
    writer = args.writer
    logging.info("Current model struction is : {}".format(get_model_struct_mode(args.train_struct_mode)))
    
    db_train = BaseDataSets4pretrain(args, mode="train", transform=transforms.Compose([
        RandomGeneratorv_4_finetune(args.patch_size, num_classes=args.num_classes)
    ]))
    db_val = BaseDataSets4pretrain(args, mode="val", transform=transforms.Compose([
        RandomGeneratorv_4_finetune(args.patch_size, num_classes=args.num_classes)
    ]))
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True,worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1,pin_memory=True, shuffle=True,num_workers =1)
    
    seg_model = net_factory(args.config, args, net_type=args.seg_model, in_chns=1, class_num=args.num_classes)
    ema_model = net_factory(args.config, args, net_type=args.mad_model, in_chns=args.input_channels, class_num=args.num_classes)
    seg_model_pretrained_dict = torch.load(args.pretrain_path_seg)
    seg_model.load_state_dict(seg_model_pretrained_dict, strict=False)
    mad_model_pretrained_dict = torch.load(args.pretrain_path_mad)
    ema_model.load_state_dict(mad_model_pretrained_dict, strict=False)
    mad_model = net_factory(args.config, args, net_type=args.mad_model, in_chns=args.input_channels, class_num=args.num_classes)
    mad_model.load_state_dict(mad_model_pretrained_dict, strict=False)
    
    optimizer_seg = optim.SGD(seg_model.parameters(), lr=args.base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer_ema = optim.SGD(ema_model.parameters(), lr=args.base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer_mad = optim.SGD(mad_model.parameters(), lr=args.base_lr,
                          momentum=0.9, weight_decay=0.0001)
    seg_model.train()
    ema_model.train()
    mad_model.train()
        
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(args.num_classes)
    
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = args.max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch, mask_label_batch = sampled_batch['image'], sampled_batch['label'],sampled_batch['mask_label']
            volume_batch, label_batch, mask_label_batch = volume_batch.cuda(), label_batch.cuda(), mask_label_batch.cuda()
                       
            seg_outputs = seg_model(volume_batch)
            seg_outputs_soft = torch.softmax(seg_outputs, dim=1)
            
            mask_input = seg_outputs_soft.detach()
            blend_outputs = (mask_input+mask_label_batch)/2
            blend_input = torch.softmax(blend_outputs, dim=1)
            mad_outputs = mad_model(blend_input)
            mad_outputs_soft = torch.softmax(seg_outputs, dim=1)
            
            ema_outputs = ema_model(seg_outputs_soft)
            ema_outputs_soft = torch.softmax(ema_outputs, dim=1)
            
            #------------------------loss------------------------------
            seg_loss_ce = ce_loss(seg_outputs, label_batch[:].long())
            seg_loss_dice = dice_loss(seg_outputs_soft, label_batch.unsqueeze(1))
            seg_loss = 0.5 * (seg_loss_dice + seg_loss_ce)
            
            mad_loss_ce = ce_loss(mad_outputs, label_batch[:].long())
            mad_loss_dice = dice_loss(mad_outputs_soft, label_batch.unsqueeze(1))
            mad_loss = 0.5 * (mad_loss_dice + mad_loss_ce)
            
            ema_loss_ce = ce_loss(ema_outputs, label_batch[:].long())
            ema_loss_dice = dice_loss(ema_outputs_soft, label_batch.unsqueeze(1))
            ema_loss = 0.5 * (ema_loss_dice + ema_loss_ce)
            
            if args.train_struct_mode == 0:
                loss = seg_loss + mad_loss + ema_loss
            elif args.train_struct_mode == 1:
                loss = seg_loss + mad_loss
            elif args.train_struct_mode == 2:
                loss = seg_loss + ema_loss
            
            optimizer_seg.zero_grad()
            optimizer_mad.zero_grad()
            optimizer_ema.zero_grad()
            loss.backward()
            optimizer_seg.step()
            optimizer_mad.step()
            optimizer_ema.step()

            lr_ = args.base_lr * (1.0 - iter_num / args.max_iterations) ** 0.9
            for param_group_seg,param_group_ema,param_group_mad in zip(optimizer_seg.param_groups,optimizer_ema.param_groups,optimizer_mad.param_groups):
                param_group_seg['lr'] = lr_
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
                        sampled_batch["image"], sampled_batch["label"], seg_model, ema_model,classes=args.num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(args.num_classes-1):
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
                                             '{}_best_model.pth'.format(args.mad_model))
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

            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    args = parser.parse_args()
    args.patch_size = [args.patch_size,args.patch_size]
    args.config = get_config(args)
    
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
    
    snapshot_path = "../model/{}_{}_labeled/{}-{}_{}".format(
        args.exp, args.labeled_num, args.seg_model, args.mad_model, args.tag)
    if args.tag == "v99" and os.path.exists(snapshot_path):
        shutil.rmtree(snapshot_path)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    ignore=shutil.ignore_patterns('.git', '__pycache__','pretrained_ckpt'))
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    args.writer = SummaryWriter(snapshot_path + '/log')
    train(args, snapshot_path)
    if args.end2Test:
        test_fine_tune(args, snapshot_path)
    args.writer.close()
