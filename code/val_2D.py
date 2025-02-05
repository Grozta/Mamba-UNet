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
from dataloaders.dataset import image2binary, np_soft_max
from dataloaders.dataset import *
from networks.net_factory import net_factory
from networks.magicnet_2D import VNet_Magic_2D
from networks.magicnet_2D_mask import VNet_Magic_2D_mask
from utils.utils import patients_to_slices
from utils.argparse_c import parser
from config import get_config

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

def test_single_volume_for_mad_model(image, label, net, classes, transforms, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    image = label.copy()
    prediction = np.zeros_like(label)
    sample = dict()
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        slice = transforms.mask_label_onle(slice)
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

def test_single_volume_for_trainLabel(image, label, net, ema_net, classes, patch_size=[256, 256]):
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
        ema_net.eval()
        with torch.no_grad():
            out = torch.softmax(net(input), dim=1)
            out = torch.argmax(torch.softmax(ema_net(out), dim=1), dim=1).squeeze(0)
            
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

def test_magic_mask(args, snapshot_path):
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
    
def test_mad_pretrain_model(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    

    mad_model = net_factory(args.config, args, net_type=args.mad_model, in_chns=num_classes, class_num=num_classes)
    if os.path.exists(args.pretrain_path_mad):
        mad_model_pretrained_dict = torch.load(args.pretrain_path_mad)
        mad_model.load_state_dict(mad_model_pretrained_dict, strict=False)
    else:
        print("pretrain_path_mad path is error")     
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    mad_model.eval()
    metric_list = 0.0
    transforms = RandomGeneratorv4(args.patch_size,num_classes=args.num_classes)
    for i_batch, sampled_batch in enumerate(valloader):
        metric_i = test_single_volume_for_mad_model(
            sampled_batch["image"], sampled_batch["label"], mad_model, classes=args.num_classes, patch_size=args.patch_size,transforms = transforms)
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

if __name__ == "__main__":
    args = parser.parse_args()
    args.patch_size = [args.patch_size,args.patch_size]
    args.config = get_config(args)
    args.config = args.config
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}-{}_{}".format(
        args.exp, args.labeled_num, args.seg_model, args.mad_model, args.tag)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    test_mad_pretrain_model(args, snapshot_path)