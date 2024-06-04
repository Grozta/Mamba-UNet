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
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders.dataset import BaseDataSets, RandomGenerator, BaseDataSets4pretrain, RandomGeneratorv3
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from utils.utils import calculate_metric_percase, label2color, get_pth_files

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/TrainLabelPretrain', help='experiment_name')
parser.add_argument('--tag',type=str,
                    default='v99', help='tag of experiment')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--pretrain_path', type=str,
                    default='../data/pretrain/xxxx.pth', help='pretrain model path')
parser.add_argument('--input_channels', type=int,  default=4,
                    help='Number of input channels about network')
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
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=140,
                    help='labeled data')
parser.add_argument('--val_num', type=int, default=7,
                    help='valset patient num')
parser.add_argument('--num_workers', type=int, default=8,
                    help='numbers of workers in dataloader')
parser.add_argument('--image_source',type=str,
                    default='label', help='The field name of the image source.options:[label,pred_vim_224]')
parser.add_argument('--image_need_fusion',default=False, 
                    action="store_true", help="Need to fuse image and pred as input")
parser.add_argument('--image_need_trans',default=False, 
                    action="store_true", help="The image needs to be transformed")
parser.add_argument('--image_need_mask',default=False, 
                    action="store_true", help='input image need mask operation')
parser.add_argument('--image_noise',type=float,  
                    default=0.001, help='Noise added when converting to binary image')
parser.add_argument('--end2Test',default=False, 
                    action="store_true", help='Test at the end of training')
args = parser.parse_args()

def test_pretrain(args, snapshot_path):
    writer = args.writer
    model = net_factory(None,args,net_type=args.model, in_chns=args.input_channels, class_num=args.num_classes)
    db_test = BaseDataSets4pretrain(args,mode = "test",
                                   transform=transforms.Compose([RandomGeneratorv3(args)])) 
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
            
            if args.input_channels == 2:
                image_origin  = test_image[0,...].cpu().numpy()
                writer.add_image('val/Image_origin', image_origin[0], iter_num, dataformats='HW')
                writer.add_image('val/Image_pred', label2color(image_origin[1]), iter_num,dataformats='HWC')
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
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    writer = args.writer
 
    model = net_factory(None,args,net_type=args.model, in_chns=args.input_channels, class_num=args.num_classes) 
     
    db_train = BaseDataSets4pretrain(args,mode = "train", 
                                     transform=transforms.Compose([RandomGeneratorv3(args)]))
    
    db_val = BaseDataSets4pretrain(args,mode = "val",
                                   transform=transforms.Compose([RandomGeneratorv3(args)]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True,worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=True,pin_memory=True,num_workers =args.num_workers)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalars('info/loss',{"total_loss":loss,
                                            "loss_ce":loss_ce,
                                            "loss_dice":loss_dice}, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                if args.input_channels == 2:
                    image_origin  = volume_batch[0,...].cpu().numpy()
                    writer.add_image('train/Image_origin', image_origin[0], iter_num, dataformats='HW')
                    writer.add_image('train/Image_pred', label2color(image_origin[1]), iter_num,dataformats='HWC')
                elif args.input_channels == 1:
                    image = volume_batch[0,0,...].cpu().numpy()
                    writer.add_image('train/Image', label2color(image), iter_num,dataformats='HWC')
                else:
                    image = torch.argmax(volume_batch[0, ...], dim=0).cpu().numpy()
                    writer.add_image('train/Image', label2color(image), iter_num,dataformats='HWC')

                outputs = torch.argmax(torch.softmax(outputs[0, ...], dim=0), dim=0).cpu().numpy()
                writer.add_image('train/Prediction', label2color(outputs), iter_num,dataformats='HWC')
                labs = label_batch[0, ...].cpu().numpy()
                writer.add_image('train/GroundTruth',label2color(labs), iter_num,dataformats='HWC')

            if iter_num > 0 and iter_num % (len(trainloader)*4) == 0:
                model.eval()
                metric_list = []
                display_image = []
                random_number = np.random.randint(0, len(valloader))
                for i_batch, sampled_batch in enumerate(valloader):
                    test_image, test_label = sampled_batch['image'], sampled_batch['label']
                    test_image, test_label = test_image.cuda(), test_label.numpy()
                    outputs = model(test_image)
                    pred = torch.argmax(torch.softmax(outputs, dim=1),dim=1).detach().cpu().numpy()
                    if i_batch == random_number:
                        if args.input_channels == 2:
                            image_origin  = test_image[0,...].cpu().numpy()
                            writer.add_image('val/Image_origin', image_origin[0], iter_num, dataformats='HW')
                            writer.add_image('val/Image_pred', label2color(image_origin[1]), iter_num,dataformats='HWC')
                        elif args.input_channels == 1:
                            image = test_image[0,0, ...].cpu().numpy()
                            writer.add_image('val/Image', label2color(image), iter_num,dataformats='HWC')
                        else:
                            image = torch.argmax(test_image[0, ...], dim=0).cpu().numpy()
                            writer.add_image('val/Image', label2color(image), iter_num,dataformats='HWC')
                            
                        prediction = pred[0]
                        writer.add_image('val/Prediction', label2color(prediction), iter_num,dataformats='HWC')
                        labs = test_label[0, ...]
                        writer.add_image('val/GroundTruth',label2color(labs), iter_num,dataformats='HWC')
                    pred = pred[0,...]
                    test_label = test_label[0,...]
                    first_metric = calculate_metric_percase(pred == 1, test_label == 1)
                    second_metric = calculate_metric_percase(pred == 2, test_label == 2)
                    third_metric = calculate_metric_percase(pred == 3, test_label == 3)
                    metric_i = np.array([first_metric,second_metric,third_metric])
                    metric_list.append(metric_i)
                metric_list = np.stack(metric_list)
                metric_list[metric_list==None] = np.nan
                avg_metric = np.nanmean(metric_list,axis=0)
                performance = np.mean(avg_metric,axis=0)
                
                logging.info(f'iteration: {iter_num} performance_list :\n {avg_metric}')
                logging.info(f'iteration: {iter_num} mean_performance : {performance}')
                writer.add_scalars('info/val_dice',{"1":avg_metric[0][0],
                                                    "2":avg_metric[1][0],
                                                    "3":avg_metric[2][0],
                                                    "avg_dice":performance[0]}, iter_num)
                writer.add_scalars('info/val_hd95',{"1":avg_metric[0][1],
                                                    "2":avg_metric[1][1],
                                                    "3":avg_metric[2][1],
                                                    "avg_hd95":performance[1]}, iter_num)
                
                if performance[0] > best_performance:
                    best_performance = performance[0]
                    writer.add_text(f'val/best_performance',f"{iter_num}_best_performance:"+str(performance),iter_num)
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_best)
                    logging.info("save_best_model to {}".format(save_best))
                    
                    if iter_num >len(trainloader)*80:
                        save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                        torch.save(model.state_dict(), save_mode_path)
                        logging.info("save_best_iter_model to {}".format(save_mode_path))
                    
                model.train()

            if iter_num % (len(trainloader)*40) == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    

    return "Training Finished!"

"""
训练mad模型
输入情况有4种:
- 原始的label经过mask,作为模型输入 图像的通道数是4  关键参数--image_source label --image_need_mask True
- seg的推理结果作为label,不使用mask,作为模型输入 图像的通道数是4  关键参数--image_source pred_vim_224 --image_need_mask False
- seg的推理结果作为label,经过mask,作为模型输入 图像的通道数是4  关键参数--image_source pred_vim_224 --image_need_mask True
- 原始的label不经过变换 图像的通道数是1 --image_need_trans True
"""
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
    if args.tag == "v99" and os.path.exists(snapshot_path):
        shutil.rmtree(snapshot_path)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns('.git', '__pycache__','pretrained_ckpt'))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    args.writer = SummaryWriter(snapshot_path + '/log')
    train(args, snapshot_path)
    if args.end2Test:
        test_pretrain(args, snapshot_path)
    args.writer.close()
