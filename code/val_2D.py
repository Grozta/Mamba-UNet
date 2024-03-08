import os
import sys
import numpy as np
import torch
import random
import logging
from medpy import metric
from scipy.ndimage import zoom
from torch.utils.data import DataLoader
import argparse
from dataloaders.dataset import *
from networks.magicnet_2D import VNet_Magic_2D

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume_magic(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input)[0], dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

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

def test_magic(args, snapshot_path):
    logging.basicConfig(filename=snapshot_path+"/test_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    latest_checkpoint_path = os.path.join(snapshot_path,'{}_latest_checkpoint.pth').format(args.model)
    model = create_model(n_classes=args.num_classes, 
                         cube_size=args.cube_size, patchsize=args.patch_size[0])
    latest_checkpoint = torch.load(latest_checkpoint_path)
    model.load_state_dict(latest_checkpoint['model_state_dict'])
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(valloader):
        metric_i = test_single_volume_magic(
            sampled_batch["image"], sampled_batch["label"], model, classes=args.num_classes, patch_size=args.patch_size)
        metric_list += np.array(metric_i)
        logging.info('iter_num:{},\n'
                     'dice{:.3f},hd95:{:.3f}\n'
                     'dice{:.3f},hd95:{:.3f}\n'
                     'dice{:.3f},hd95:{:.3f}\n'.
                    format(i_batch 
                           ,metric_i[0][0], metric_i[0][1]
                           ,metric_i[1][0], metric_i[1][1]
                           ,metric_i[2][0], metric_i[2][1]))
        
    mean_dice = (np.mean(metric_list, axis=0)[0])/len(valloader)
    mean_hd95 = (np.mean(metric_list, axis=0)[1])/len(valloader)
    logging.info('model_val_mean_dice:{:.3f},model_val_mean_hd95: {:.3f}'.
                    format(mean_dice, mean_hd95))

def test_2D(args, snapshot_path):
    logging.basicConfig(filename=snapshot_path+"/test_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    latest_checkpoint_path = os.path.join(snapshot_path,args.checkpoint_name)
    model = create_model(n_classes=args.num_classes, 
                         cube_size=args.cube_size, patchsize=args.patch_size[0])
    latest_checkpoint = torch.load(latest_checkpoint_path)
    model.load_state_dict(latest_checkpoint)
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(valloader):
        metric_i = test_single_volume_magic(
            sampled_batch["image"], sampled_batch["label"], model, classes=args.num_classes, patch_size=args.patch_size)
        metric_list += np.array(metric_i)
        logging.info('iter_num:{},\n'
                     'dice{:.3f},hd95:{:.3f}\n'
                     'dice{:.3f},hd95:{:.3f}\n'
                     'dice{:.3f},hd95:{:.3f}\n'.
                    format(i_batch 
                           ,metric_i[0][0], metric_i[0][1]
                           ,metric_i[1][0], metric_i[1][1]
                           ,metric_i[2][0], metric_i[2][1]))
        
    mean_dice = (np.mean(metric_list, axis=0)[0])/len(valloader)
    mean_hd95 = (np.mean(metric_list, axis=0)[1])/len(valloader)
    logging.info('model_val_mean_dice:{:.3f},model_val_mean_hd95: {:.3f}'.
                    format(mean_dice, mean_hd95))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='ACDC', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/media/icml/H4T/DATASET/ACDC', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='MagicNet_2D', help='exp_name')
parser.add_argument('--model', type=str, default='V-Net_2D', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7, help='labeled trained samples')
parser.add_argument('--labeled_bs', type=int, default=12, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--patch_size', type=list,  default=[256, 256],help='patch size of network input')
parser.add_argument('--cube_size', type=int, default=32, help='size of each cube')
parser.add_argument('--seed', type=int, default=5179, help='random seed')
parser.add_argument('--is_magic', type=bool, default=True, help='is_magic')
parser.add_argument('--checkpoint_name', type=str, default='mambaunet_best_model.pth', help='is_magic')

args = parser.parse_args()

if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/ACDC/train_Semi_MagicNet_2D_7/V-Net_2D"
    test_magic(args, snapshot_path, args.is_magic)