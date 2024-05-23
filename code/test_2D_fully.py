import argparse
import logging
from tensorboardX import SummaryWriter
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
from networks.net_factory import net_factory
from dataloaders.dataset import RandomGeneratorv4, label2color

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--tag',type=str,
                    default='v99', help='tag of experiment')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--model_weight_path', type=str,
                    default='../data/pretrain/mad_model_unet.pth', help='model weight path')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    # asd = metric.binary.asd(pred, gt)
    # hd95 = metric.binary.hd95(pred, gt)
    return dice
    # , hd95
    # , asd


def test_single_volume(case, net, test_save_path, args, writer= None):
    h5f = h5py.File(args.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        if args.test_mad:
            slice = label[ind, :, :]
        else:
            slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (args.patch_size[0] / x, args.patch_size[1] / y), order=0)
        if args.test_mad:
            slice = args.transforms.mask_label_onle(slice)
            recv = np.argmax(slice,axis=0)
            recv = zoom(recv, (x / args.patch_size[0], y / args.patch_size[0]), order=0)
            input = torch.from_numpy(slice).unsqueeze(0).float().cuda()
        else:
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if args.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            s_out = torch.softmax(out_main, dim=1)
            out = torch.argmax(s_out, dim=1).squeeze(0)
    
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / args.patch_size[0], y / args.patch_size[0]), order=0)
            prediction[ind] = pred
            name = f"{case}"
            
            writer.add_image(name+"/input",label2color(recv), ind,dataformats='HWC')
            writer.add_image(name+"/output",label2color(prediction[ind,:,:]), ind,dataformats='HWC')
            writer.add_image(name+"/GT",label2color(label[ind, :, :]), ind,dataformats='HWC')

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)
    
    if not args.test_mad:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        img_itk.SetSpacing((1, 1, 10))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        prd_itk.SetSpacing((1, 1, 10))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        lab_itk.SetSpacing((1, 1, 10))
        sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(args):
    with open(args.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    test_save_path = "../model/{}_{}_labeled/{}/predictions/".format(    
        args.exp, args.labeled_num, args.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    logging.basicConfig(filename=test_save_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.info(str(args))
    writer = SummaryWriter(test_save_path + '/log')
    net = net_factory(None,None,net_type=args.model, in_chns=1,
                      class_num=args.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(args.model))
    print(save_mode_path)
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, args, writer)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    
    writer.close()
    return avg_metric

def Inference_mad_model(args):
    args.test_mad = True
    args.transforms = RandomGeneratorv4(args.patch_size,num_classes=args.num_classes)
    with open(args.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    test_save_path = "../model/{}_{}_labeled/{}_{}".format(    
        args.exp, args.labeled_num, args.model, args.tag)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    logging.basicConfig(filename=test_save_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.info(str(args))
    writer = SummaryWriter(test_save_path + '/log')
    net = net_factory(None,None,net_type=args.model, in_chns=args.num_classes,
                      class_num=args.num_classes)
    print(args.model_weight_path)
    net.load_state_dict(torch.load(args.model_weight_path),strict=False)
    print("init weight from {}".format(args.model_weight_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, args, writer)
        logging.info(f"{case}:[{first_metric}_{second_metric}_{third_metric}]")
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    writer.close()
    logging.info(f"cls_dice:{avg_metric}")
    logging.info(f"dsc:{(avg_metric[0]+avg_metric[1]+avg_metric[2])/3}")
    return avg_metric


if __name__ == '__main__':
    args = parser.parse_args()
    metric = Inference_mad_model(args)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
