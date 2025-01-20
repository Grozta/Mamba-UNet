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
from networks.utils import gen_eme_hot_mask
from utils.losses import weighted_cross_entropy_loss_with_mask,weighted_dice_loss_with_mask
from utils.utils import label2color, get_model_struct_mode, extract_iter_number, get_pth_files\
    ,get_train_test_mode,worker_init_fn,improvement_log,update_train_loss_MA,get_ablation_option_mode,get_VAE_option_mode
from val_2D import test_single_volume_for_VAE
from networks.Probabilistic_Unet.utils import l2_regularisation
from utils.argparse_c import parser

def test_fine_tune(args, snapshot_path):
    writer = args.writer
    seg_model = net_factory(args.config, args, net_type=args.seg_model, in_chns=1, class_num=args.num_classes)
    eme_model = net_factory(args.config, args, net_type=args.ema_model, in_chns=args.input_channels_ema, class_num=args.num_classes)
    
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
        eme_model.load_state_dict(model_pretrained_dict["eme_state_dict"])
        seg_model.eval()
        eme_model.eval()
        
        metric_list = 0.0 # 3x2
        for i_batch, sampled_batch in enumerate(testloader):
            metric_i = test_single_volume_for_VAE(
                sampled_batch["image"], sampled_batch["label"], seg_model, eme_model,classes=args.num_classes, patch_size=args.patch_size,args=args)
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
    logging.info("Current ablation_option is : {},".format(["["+get_ablation_option_mode(desc)+"]" for desc in args.ablation_option]))
    logging.info("Current VAE_option is : {},".format(["["+get_VAE_option_mode(desc)+"]" for desc in args.vae_option]))
    
    db_train = BaseDataSets4TrainLabel(args, mode="train", transform=transforms.Compose([
        RandomGeneratorv_4_finetune(args,mode="train")
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True,worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1,pin_memory=True, shuffle=True,num_workers =1)
    
    if 3 in args.ablation_option:
        args.input_channels_ema = 4
    seg_model = net_factory(args.config, args, net_type=args.seg_model, in_chns=1, class_num=args.num_classes) # 分割网络
    eme_model = net_factory(args.config, args, net_type=args.ema_model, in_chns=args.input_channels_ema, class_num=args.num_classes) # 修正网络
    if 4 in args.ablation_option:
        seg_model.apply(kaiming_initialize_weights)
    else:
        #检查args.pretrain_path_seg路径中文件是否存在
        if os.path.exists(args.pretrain_path_seg):
            seg_model_pretrained_dict = torch.load(args.pretrain_path_seg)
            seg_model.load_state_dict(seg_model_pretrained_dict)
    if 2 in args.ablation_option:
        eme_model.apply(kaiming_initialize_weights)
    else:
        if os.path.exists(args.pretrain_path_seg):
            eme_model_pretrained_dict = torch.load(args.pretrain_path_ema)
            eme_model.load_state_dict(eme_model_pretrained_dict)
    if 5 in args.ablation_option:
        for param in seg_model.parameters():
            param.requires_grad = False
            
    optimizer_seg = torch.optim.Adam(seg_model.parameters(), args.initial_lr, weight_decay=args.weight_decay)
    lr_scheduler_seg = lr_scheduler.StepLR(optimizer_seg, step_size=5,gamma=0.95)
    optimizer_eme = torch.optim.Adam(eme_model.parameters(), args.initial_lr_eme, weight_decay=args.weight_decay)
    lr_scheduler_eme = lr_scheduler.StepLR(optimizer_eme, step_size=5,gamma=0.95)
    
    seg_model.train()
    eme_model.train()
    ce_loss = weighted_cross_entropy_loss_with_mask
    dice_loss = weighted_dice_loss_with_mask
    
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = args.max_epochs
    best_performance = 0.0
    args.all_tr_losses = []
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        train_losses_epoch = []
        for step_num, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch, mask_label_batch = sampled_batch['image'], sampled_batch['label'],sampled_batch['mask_label']
            volume_batch, label_batch, mask_label_batch = volume_batch.cuda(), label_batch.cuda(), mask_label_batch.cuda()
            seg_label_batch = torch.unsqueeze(label_batch,1)
            
            seg_model.forward(volume_batch, seg_label_batch, training=True)
            elbo_loss = seg_model.elbo(label_batch)
            seg_outputs = seg_model.reconstruction
            # seg的输出和label输入到修正网络
            eme_hot_mask = gen_eme_hot_mask(seg_outputs, label_batch)
            eme_outputs,eme_hot_mask = eme_model(seg_outputs, eme_hot_mask)
            #------------------------loss------------------------------
            loss = 0.0
            
            vae_reg_loss = l2_regularisation(seg_model.posterior) + l2_regularisation(seg_model.prior) + l2_regularisation(seg_model.fcomb.layers)
            seg_loss = args.vae_reg_loss_weight * vae_reg_loss +(-elbo_loss * args.vae_elbo_loss_weight)
            
            eme_loss_ce = ce_loss(eme_outputs, label_batch[:].long(),eme_hot_mask)
            eme_loss_dice = dice_loss(eme_outputs, label_batch[:],eme_hot_mask)
            eme_loss = eme_loss_ce + eme_loss_dice
            
            loss = seg_loss + eme_loss
            #------------------------loss------------------------------
            
            optimizer_seg.zero_grad()
            optimizer_eme.zero_grad()
            loss.backward()
            optimizer_seg.step()
            optimizer_eme.step()
                    
            iter_num = iter_num + 1
            #------------------------Recording observer data------------------------------
            #获取optimizer_seg和optimizer_eme的学习率
            seg_lr = optimizer_seg.param_groups[0]['lr']
            eme_lr = optimizer_eme.param_groups[0]['lr']
            writer.add_scalars('info/lr', {"seg_lr":seg_lr, "eme_lr":eme_lr}, iter_num)
            
            if iter_num%5==0:
                logging.info(
                'iteration %d : loss : %f\n -elbo_loss: %f, vae_reg_loss: %f,seg_loss:%f,\n eme_loss: %f' %
                (iter_num,loss, -elbo_loss, vae_reg_loss,seg_loss, eme_loss))
                writer.add_scalars('info/loss', {'total_loss':loss,
                                                       '-elbo_loss':-elbo_loss,
                                                       'vae_reg_loss':vae_reg_loss,
                                                       'seg_loss':seg_loss,
                                                       'eme_loss':eme_loss}, iter_num)

            if iter_num % 20 == 0:
                image = sampled_batch['image'][0, 0, ...]
                writer.add_image('train/Image', image, iter_num, dataformats='HW')
                
                labs = sampled_batch['label'][0, ...]
                writer.add_image('train/GroundTruth', 
                                 label2color(labs), iter_num,dataformats='HWC')
                
                seg_outputs = torch.argmax(torch.softmax(seg_outputs, dim=1), dim=1).cpu().numpy()[0, ...]
                writer.add_image('train/segPrediction',
                                 label2color(seg_outputs), iter_num,dataformats='HWC')
                
                eme_outputs = torch.argmax(torch.softmax(eme_outputs, dim=1), dim=1).cpu().numpy()[0, ...]
                writer.add_image('train/emePrediction',
                                 label2color(eme_outputs), iter_num,dataformats='HWC')
            #------------------------Recording observer data------------------------------
            model_state = {"seg_state_dict":seg_model.state_dict(),
                           "eme_state_dict":eme_model.state_dict()}    
        lr_scheduler_seg.step()
        lr_scheduler_eme.step()
               
        if epoch_num % 1 == 0:
            seg_model.eval()
            eme_model.eval()
            metric_list = 0.0 # 3x2
            for i_batch, sampled_batch in enumerate(valloader):
                metric_i = test_single_volume_for_VAE(
                    sampled_batch["image"], sampled_batch["label"], seg_model, 
                    eme_model,classes=args.num_classes, patch_size=args.patch_size,
                    args=args)
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
            
            if performance[0] > best_performance:
                best_performance = performance[0]
                writer.add_text(f'val/best_performance',f"{iter_num}_best_performance:"+str(performance),iter_num)
                save_best = os.path.join(snapshot_path,
                                            'TrainVAE{}_best_model.pth'.format(args.train_struct_mode))
                torch.save(model_state, save_best)
                logging.info("save_best_model to {}".format(save_best))
                
                if iter_num >len(trainloader)*80:
                    save_mode_path = os.path.join(snapshot_path,
                                                'iter_{}_dice_{}.pth'.format(
                                                    iter_num, round(best_performance, 4)))
                    torch.save(model_state, save_mode_path)
                    logging.info("save_best_iter_model to {}".format(save_mode_path))
                    
            seg_model.train()
            eme_model.train()

        if iter_num % (len(trainloader)*40) == 0:
            save_mode_path = os.path.join(
                snapshot_path, 'iter_' + str(iter_num) + '.pth')
            torch.save(model_state, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        
        if args.tag == 'v99' and iter_num >=args.test_iterations:
            iterator.close()
            return "Testing Finished!"
        
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
    
    snapshot_path = "../model/{}/{}-{}_{}".format(args.exp, args.seg_model,args.ema_model, args.tag)
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
