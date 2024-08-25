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
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import get_config
from dataloaders.dataset import BaseDataSets4TrainLabel,BaseDataSets,RandomGeneratorv_4_finetune, resize_data_list
from networks.net_factory import net_factory
from networks.unet import kaiming_initialize_weights
from utils import losses
from utils.utils import label2color, get_model_struct_mode, update_ema_variables, get_pth_files, update_train_loss_MA, get_train_test_mode,worker_init_fn,improvement_log,update_train_loss_MA,extract_iter_number
from utils.argparse_c import parser
from val_2D import test_single_volume_for_trainLabel

def test_fine_tune(args, snapshot_path):
    writer = args.writer
    seg_model = net_factory(args.config, args, net_type=args.seg_model, in_chns=1, class_num=args.num_classes)
    ema_model = net_factory(args.config, args, net_type=args.mad_model, in_chns=args.input_channels_mad, class_num=args.num_classes)
    
    db_test = BaseDataSets(base_dir=args.root_path, split="test") 
    testloader = DataLoader(db_test, batch_size=1, shuffle=True,pin_memory=True,num_workers =args.num_workers)
    pth_list = get_pth_files(snapshot_path)
    pth_list = sorted(pth_list, key=extract_iter_number)
    iterator = tqdm(range(len(pth_list)), ncols=70)
    metric_list = []
    for iter_num in iterator:
        pth = pth_list[iter_num]
        model_pretrained_dict = torch.load(os.path.join(snapshot_path,pth))
        seg_model.load_state_dict(model_pretrained_dict["seg_state_dict"])
        ema_model.load_state_dict(model_pretrained_dict["ema_state_dict"])
        seg_model.eval()
        ema_model.eval()
        
        metric_list = 0.0 # 3x2
        for i_batch, sampled_batch in enumerate(testloader):
            metric_i = test_single_volume_for_trainLabel(
                sampled_batch["image"], sampled_batch["label"], seg_model, ema_model,classes=args.num_classes, patch_size=args.patch_size)
            metric_list += np.array(metric_i)
        metric_list = metric_list / len(db_test)

        performance = np.mean(metric_list, axis=0) #1x2
        writer.add_scalars('info/test_dice',{"1":metric_list[0][0],
                                            "2":metric_list[1][0],
                                            "3":metric_list[2][0],
                                            "avg_dice":performance[0]}, iter_num)
        writer.add_scalars('info/test_hd95',{"1":metric_list[0][1],
                                            "2":metric_list[1][1],
                                            "3":metric_list[2][1],
                                            "avg_hd95":performance[1]}, iter_num)
        logging.info(f'{pth}_test_metric :{performance}' )
        writer.add_text(f'test/performance',f"{pth}:"+str(performance),iter_num)

def train(args, snapshot_path):
    writer = args.writer
    args.train_loss_MA = None
    logging.info("Current model struction is : {},".format(get_model_struct_mode(args.train_struct_mode)))
    db_train = BaseDataSets4TrainLabel(args, mode="train", transform=transforms.Compose([
        RandomGeneratorv_4_finetune(args,mode="train")
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True,worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1,pin_memory=True, shuffle=True,num_workers =1)
    
    seg_model = net_factory(args.config, args, net_type=args.seg_model, in_chns=1, class_num=args.num_classes)
    ema_model = net_factory(args.config, args, net_type=args.mad_model, in_chns=args.input_channels_mad, class_num=args.num_classes)
    mad_model = net_factory(args.config, args, net_type=args.mad_model, in_chns=args.input_channels_mad, class_num=args.num_classes)

    latest_check_point_path = os.path.join(snapshot_path,'latest_check_point.pth')
    if os.path.exists(latest_check_point_path):
        latest_check_point_dict = torch.load(args.pretrain_path_seg)
        seg_model.load_state_dict(latest_check_point_dict["seg_state_dict"])
        ema_model.load_state_dict(latest_check_point_dict["ema_state_dict"])
        mad_model.load_state_dict(latest_check_point_dict["mad_state_dict"])
        iter_num = latest_check_point_dict["iter_num"]
        best_performance = latest_check_point_dict["best_performance"]
        start_epoch = iter_num // len(trainloader)
    else:
        seg_model_pretrained_dict = torch.load(args.pretrain_path_seg)
        mad_model_pretrained_dict = torch.load(args.pretrain_path_mad)
        seg_model.load_state_dict(seg_model_pretrained_dict)
        ema_model.apply(kaiming_initialize_weights)
        mad_model.load_state_dict(mad_model_pretrained_dict)
        iter_num = 0
        best_performance = 0.0
        start_epoch = 0
        
    optimizer_seg = torch.optim.Adam(seg_model.parameters(), args.initial_lr, weight_decay=args.weight_decay,
                                          amsgrad=True)
    lr_scheduler_seg = lr_scheduler.ReduceLROnPlateau(optimizer_seg, mode='min', factor=args.lr_scheduler_factor,
                                                        patience=args.lr_scheduler_patience,
                                                        verbose=True, threshold=args.lr_threshold,
                                                        threshold_mode="abs")
    optimizer_mad = torch.optim.Adam(mad_model.parameters(), args.initial_lr*0.001, weight_decay=args.weight_decay,
                                          amsgrad=True)
    lr_scheduler_mad = lr_scheduler.ReduceLROnPlateau(optimizer_mad, mode='min', factor=args.lr_scheduler_factor,
                                                        patience=args.lr_scheduler_patience,
                                                        verbose=True, threshold=args.lr_threshold,
                                                        threshold_mode="abs")
    optimizer_ema = torch.optim.Adam(ema_model.parameters(), args.initial_lr, weight_decay=args.weight_decay,
                                          amsgrad=True)
    lr_scheduler_ema = lr_scheduler.ReduceLROnPlateau(optimizer_ema, mode='min', factor=args.lr_scheduler_factor,
                                                        patience=args.lr_scheduler_patience,
                                                        verbose=True, threshold=args.lr_threshold,
                                                        threshold_mode="abs")
    
    seg_model.train()
    ema_model.train()
    mad_model.train()
        
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(args.num_classes)
    logging.info("{} iterations per epoch".format(len(trainloader)))

    args.all_tr_losses = []
    max_epoch = args.max_iterations // len(trainloader) + 1
    iterator = tqdm(total=max_epoch, desc=f'Epoch {start_epoch}',  leave=False, ncols=120,initial=start_epoch)
    for epoch_num in range(start_epoch, max_epoch):
        train_losses_epoch = []
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch, mask_label_batch = sampled_batch['image'], sampled_batch['label'],sampled_batch['mask_label']
            volume_batch, label_batch, mask_label_batch = volume_batch.cuda(), label_batch.cuda(),mask_label_batch.cuda()
            
            update_ema_variables(mad_model, ema_model, args.ema_decay, iter_num)
            
            seg_outputs = seg_model(volume_batch)
            seg_outputs_soft = torch.softmax(seg_outputs, dim=1)
            
            seg_soft2mad = seg_outputs_soft.detach()
            blend_outputs = (seg_soft2mad+mask_label_batch)/2
            blend_input2mad = torch.softmax(blend_outputs, dim=1)
            blend_input2mad = torch.concat([volume_batch,blend_input2mad],dim=1)
            mad_outputs = mad_model(blend_input2mad)
            mad_outputs_soft = torch.softmax(seg_outputs, dim=1)
            
            soft_input2ema = torch.concat([volume_batch,seg_outputs_soft],dim=1)
            ema_outputs = ema_model(soft_input2ema)
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
            
            loss = seg_loss + mad_loss + ema_loss
            train_losses_epoch.append(loss.cpu().item())
            optimizer_seg.zero_grad()
            optimizer_mad.zero_grad()
            optimizer_ema.zero_grad()
            loss.backward()
            optimizer_seg.step()
            optimizer_mad.step()
            optimizer_ema.step()
            
            iter_num = iter_num + 1
            
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/seg_loss', seg_loss, iter_num)
            writer.add_scalar('info/mad_loss', mad_loss, iter_num)
            writer.add_scalar('info/ema_loss', ema_loss, iter_num)

            logging.info('iteration %d : loss : %f seg_loss : %f mad_loss : %f ema_loss : %f' % (iter_num, loss.item(), seg_loss.item(), mad_loss.item(), ema_loss.item()))

            if iter_num % 200 == 0:
                image = sampled_batch['image'][0, 0, ...]
                writer.add_image('train/Image', image, iter_num, dataformats='HW')
                
                labs = sampled_batch['label'][0, ...]
                writer.add_image('train/GroundTruth', 
                                 label2color(labs), iter_num,dataformats='HWC')
                
                mask_labs = torch.argmax(sampled_batch['mask_label'], dim=1)[0, ...]
                writer.add_image('train/mask_lable', 
                                label2color(mask_labs), iter_num,dataformats='HWC')
                
                seg_outputs = torch.argmax(seg_outputs_soft, dim=1).cpu().numpy()[0, ...]
                writer.add_image('train/segPrediction',
                                 label2color(seg_outputs), iter_num,dataformats='HWC')
                
                ema_outputs = torch.argmax(ema_outputs_soft, dim=1).cpu().numpy()[0, ...]
                writer.add_image('train/emaPrediction',
                                 label2color(ema_outputs), iter_num,dataformats='HWC')
                
                mad_outputs = torch.argmax(mad_outputs_soft, dim=1).cpu().numpy()[0, ...]
                writer.add_image('train/madPrediction',
                                label2color(mad_outputs), iter_num,dataformats='HWC')
            model_state = {"seg_state_dict":seg_model.state_dict(),
                           "ema_state_dict":ema_model.state_dict(),
                           "mad_state_dict":mad_model.state_dict(),
                           "iter_num": iter_num,
                           "best_performance":best_performance}
              
        epoch_num = epoch_num + 1      
        if epoch_num % 1 == 0:
            seg_model.eval()
            ema_model.eval()
            metric_list = 0.0 # 3x2
            for i_batch, sampled_batch in enumerate(valloader):
                metric_i = test_single_volume_for_trainLabel(
                    sampled_batch["image"], sampled_batch["label"], seg_model, 
                    ema_model,classes=args.num_classes, patch_size=args.patch_size)
                metric_list += np.array(metric_i)
            metric_list = metric_list / len(db_val)

            performance = np.mean(metric_list, axis=0)
            logging.info(f'iteration: {iter_num} performance_list :\n {metric_list}')
            logging.info(f'iteration: {iter_num} performance_mean :\n {performance}')
            writer.add_scalars('info/val_dice',{"1":metric_list[0][0],
                                                "2":metric_list[1][0],
                                                "3":metric_list[2][0],
                                                "avg_dice":performance[0]}, iter_num)
            writer.add_scalars('info/val_hd95',{"1":metric_list[0][1],
                                                "2":metric_list[1][1],
                                                "3":metric_list[2][1],
                                                "avg_hd95":performance[1]}, iter_num)
            
            torch.save(model_state, latest_check_point_path)
            logging.info("save latest_check_point {}".format(latest_check_point_path))
            
            if performance[0] > best_performance:
                best_performance = performance[0]
                model_state["performance"] = best_performance
                writer.add_text(f'val/best_performance',f"{iter_num}_best_performance:"+str(performance),iter_num)
                save_best = os.path.join(snapshot_path,
                                            'TrainLabel{}_best_model.pth'.format(args.train_struct_mode))
                torch.save(model_state, save_best)
                logging.info("save_best_model to {}".format(save_best))
                
                if iter_num >len(trainloader)*80:
                    save_mode_path = os.path.join(snapshot_path,
                                                'iter_{}_dice_{}.pth'.format(
                                                    iter_num, round(best_performance, 4)))
                    torch.save(model_state, save_mode_path)
                    logging.info("save_best_iter_model to {}".format(save_mode_path))

            seg_model.train()
            ema_model.train()
                
        
        if iter_num % (len(trainloader)*40) == 0:
            save_mode_path = os.path.join(
                snapshot_path, 'iter_' + str(iter_num) + '.pth')
            torch.save(model_state, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        
        args.all_tr_losses.append(np.mean(train_losses_epoch))
        update_train_loss_MA(args)
        lr_scheduler_seg.step(args.train_loss_MA)
        lr_scheduler_ema.step(args.train_loss_MA)
        lr_scheduler_mad.step(args.train_loss_MA)
        writer.add_scalar('info/lr', optimizer_seg.state_dict()['param_groups'][0]['lr'], epoch_num)
        writer.add_scalars('info/lr_scheduler',{"lr": optimizer_seg.state_dict()['param_groups'][0]['lr'],
                                                "loss_MA": args.train_loss_MA,
                                                "3":metric_list[2][1],
                                                "avg_hd95":performance[1]}, iter_num)
        
        if args.tag == 'v99' and iter_num >=args.test_iterations:
            iterator.close()
            return "Testing Finished!"
        if iter_num >= args.max_iterations or optimizer_seg.state_dict()['param_groups'][0]['lr'] <= args.lr_threshold:
            iterator.close()
            break
         
    return "Training Finished!"


if __name__ == "__main__":
    args = parser.parse_args()
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
    
    snapshot_path = "../model/{}/{}-{}_{}".format(args.exp, args.seg_model,args.mad_model, args.tag)
    if get_train_test_mode(args.train_test_mode) == "only_Testing":
        args.clean_before_run = False
    if args.clean_before_run and os.path.exists(snapshot_path):
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
    if get_train_test_mode(args.train_test_mode) == "only_Training":
        train(args, snapshot_path)
    elif get_train_test_mode(args.train_test_mode) == "Train2Test":
        train(args, snapshot_path)
        test_fine_tune(args, snapshot_path)
    elif get_train_test_mode(args.train_test_mode) == "only_Testing":
        args.writer = SummaryWriter(snapshot_path + '/testlog')
        test_fine_tune(args, snapshot_path)
    args.writer.close()
