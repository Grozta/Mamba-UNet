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

from dataloaders.dataset import BaseDataSets, RandomGenerator, BaseDataSets4v1, RandomGeneratorv3
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from utils.utils import calculate_metric_percase, label2color, patients_to_slices

parser = argparse.ArgumentParser()
parser.add_argument('--train_label',default=True, 
                    action="store_true", help="train label mode")
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
parser.add_argument('--used_pred_train',type=str,
                    default='label', help='this is a pred dict name')
parser.add_argument('--image_need_mask',type= bool,
                    default=False, help='input image need mask operation')
args = parser.parse_args()

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
 
    model = net_factory(None,args,net_type=args.model, in_chns=num_classes, class_num=num_classes) 
     
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    db_train = BaseDataSets4v1(base_dir=args.root_path,num=labeled_slice, transform=transforms.Compose([
        RandomGeneratorv3(args.patch_size,args.num_classes, is_train= True, is_mask= args.image_need_mask)]),args=args)
    
    val_num = patients_to_slices(args.root_path, args.val_num)
    db_val = BaseDataSets4v1(base_dir=args.root_path,num=val_num,transform=transforms.Compose([
        RandomGeneratorv3(args.patch_size,args.num_classes, is_train= True, is_mask= args.image_need_mask)]), args=args)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True,worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,num_workers =args.num_workers)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
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
                image = torch.argmax(volume_batch[1, ...], dim=0).cpu().numpy()
                writer.add_image('train/Image', label2color(image), iter_num,dataformats='HWC')
                outputs = torch.argmax(torch.softmax(outputs[1, ...], dim=0), dim=0).cpu().numpy()
                writer.add_image('train/Prediction', label2color(outputs), iter_num,dataformats='HWC')
                labs = label_batch[1, ...].cpu().numpy()
                writer.add_image('train/GroundTruth',label2color(labs), iter_num,dataformats='HWC')

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = []
                for i_batch, sampled_batch in enumerate(valloader):
                    val_image, val_label = sampled_batch['image'], sampled_batch['label']
                    val_image, val_label = val_image.cuda(), val_label.numpy()
                    outputs = model(val_image)
                    pred = torch.argmax(torch.softmax(outputs, dim=1),dim=1).detach().cpu().numpy()
                    first_metric = calculate_metric_percase(pred == 1, val_label == 1)
                    second_metric = calculate_metric_percase(pred == 2, val_label == 2)
                    third_metric = calculate_metric_percase(pred == 3, val_label == 3)
                    metric_i = np.array([first_metric,second_metric,third_metric])
                    metric_list.append(metric_i)
                metric_list = np.stack(metric_list)
                metric_list[metric_list==None] = np.nan
                avg_metric = np.nanmean(metric_list,axis=0)
                performance = np.mean(avg_metric)
                
                writer.add_scalars('info/val_dice',{"1":avg_metric[0],
                                                    "2":avg_metric[1],
                                                    "3":avg_metric[2],
                                                    "avg_dice":performance}, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num % 2000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
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
    train(args, snapshot_path)
