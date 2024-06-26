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
from scipy.ndimage import zoom
from tqdm import tqdm
from utils.argparse_c import parser
from utils.utils import calculate_metric_percase

from networks.net_factory import net_factory
from dataloaders.dataset import RandomGeneratorv4
from utils.utils import label2color
from config import get_config

parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--model_weight_path', type=str,
                    default='../data/pretrain/mad_model_unet.pth', help='model weight path')
parser.add_argument('--save_test',default=False, 
                    action="store_true", help="save test label to filesystem")

def inference_single_case(case, seg_model, test_save_path, args, writer= None):
    """直接将推理的结果保存在原图中
    h5f["pred_vim_224"][:] = pred
    """
    with h5py.File(args.root_path + "/data/slices/{}.h5".format(case), 'r+') as h5f:
        image = h5f['image'][:]
        label = h5f['label'][:]
        slice = image
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (args.patch_size[0] / x, args.patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        seg_model.eval()
        
        with torch.no_grad():
            out_main = seg_model(input)
            s_out = torch.softmax(out_main, dim=1)
            out = torch.argmax(s_out, dim=1).squeeze(0)

            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / args.patch_size[0], y / args.patch_size[0]), order=0).astype(np.uint8)
            if "pred_vim_224" in h5f:
                h5f["pred_vim_224"][:] = pred
            else:
                h5f.create_dataset('pred_vim_224', data=pred)
            
    with h5py.File(args.root_path + "/data/slices/{}.h5".format(case), 'r') as h5f_s:    
        writer.add_image(f"{case}"+"/input",h5f_s['image'][:], dataformats='HW')
        writer.add_image(f"{case}"+"/pred",label2color(h5f_s["pred_vim_224"][:]), dataformats='HWC')
        writer.add_image(f"{case}"+"/GT",label2color(h5f_s["label"][:]), dataformats='HWC')

    first_metric = calculate_metric_percase(pred == 1, label == 1)
    second_metric = calculate_metric_percase(pred == 2, label == 2)
    third_metric = calculate_metric_percase(pred == 3, label == 3)
    
    return first_metric, second_metric, third_metric

def test_single_volume(case, seg_model, test_save_path, args, writer= None,ema_model = None):
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
            recv = image[ind, :, :]
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        seg_model.eval()
        if ema_model:
            ema_model.eval()
        name = f"{case}"
        with torch.no_grad():
            if args.model == "unet_urds":
                out_main, _, _, _ = seg_model(input)
            else:
                out_main = seg_model(input)
            s_out = torch.softmax(out_main, dim=1)
            if ema_model:
                seg_out = torch.argmax(s_out, dim=1).squeeze(0).cpu().detach().numpy()
                seg_out = zoom(seg_out, (x / args.patch_size[0], y / args.patch_size[0]), order=0)
                writer.add_image(name+"/seg_output",label2color(seg_out), ind,dataformats='HWC')
                s_out = torch.softmax(ema_model(s_out), dim=1)
            out = torch.argmax(s_out, dim=1).squeeze(0)
    
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / args.patch_size[0], y / args.patch_size[0]), order=0)
            prediction[ind] = pred
            
            if ema_model:
                writer.add_image(name+"/input",recv, ind,dataformats='WC')
            else:
                writer.add_image(name+"/input",label2color(recv), ind,dataformats='HWC')
            writer.add_image(name+"/ema_output",label2color(prediction[ind,:,:]), ind,dataformats='HWC')
            writer.add_image(name+"/GT",label2color(label[ind, :, :]), ind,dataformats='HWC')

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)
    
    if args.save_test:
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

def Inference_seg_model(args):
    args.test_mad = False
    args.save_test = False
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
    for key in args.model_name_list.keys():
        if key == 'unet':
            args.patch_size = [256,256]
        else:
            args.patch_size = [224,224]
        args.model = key
        args.model_weight_path = args.model_name_list[key]
        net = net_factory(args.config,args ,net_type=args.model, in_chns=1, class_num=args.num_classes)
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
        
        logging.info(f"[{key}]_cls_dice:{avg_metric}")
        logging.info(f"[{key}]_dsc:{(avg_metric[0]+avg_metric[1]+avg_metric[2])/3}")
        writer.add_text(f"[{key}]",f"{str(avg_metric)}")
    return avg_metric


def Inference_seg_ema_model(args):
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
    
    seg_model = net_factory(args.config,args,net_type=args.seg_model)
    ema_model = net_factory(args.config,args,net_type=args.mad_model, in_chns=args.num_classes, class_num=args.num_classes)
    if os.path.exists(args.pretrain_path_seg):
        seg_model_pretrained_dict = torch.load(args.pretrain_path_seg)
        seg_model.load_state_dict(seg_model_pretrained_dict, strict=False)
    if os.path.exists(args.pretrain_path_mad):
        mad_model_pretrained_dict = torch.load(args.pretrain_path_mad)
        ema_model.load_state_dict(mad_model_pretrained_dict, strict=False)

    print("init seg weight from {}".format(args.pretrain_path_seg))
    print("init ema weight from {}".format(args.pretrain_path_mad))
    ema_model.eval()
    seg_model.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
            case, seg_model, test_save_path, args, writer,ema_model=ema_model)
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

def Inference_seg_model_genarate_new_dataset(args):
    only_train = False
    args.sample_list=[]
    if only_train:
        with open(args.root_path + "/train_slices.list", "r") as f1:
            args.sample_list = f1.readlines()
        args.sample_list = [item.replace("\n", "") for item in args.sample_list]
    else:
        with open(args.root_path + "/val_slices.list", "r") as f1:
            sample_list = f1.readlines()
            for item in sample_list:
                i = item.replace("\n", "")
                args.sample_list.append(i)
        with open(args.root_path + "/test_slices.list", "r") as f1:
            sample_list = f1.readlines()
            for item in sample_list:
                i = item.replace("\n", "")
                args.sample_list.append(i)
    print("total {} samples".format(len(args.sample_list)))
    test_save_path = "../model/{}_{}_labeled/{}_{}".format(    
        args.exp, args.labeled_num, args.model, args.tag)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    logging.basicConfig(filename=test_save_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.info(str(args))
    writer = SummaryWriter(test_save_path + '/log')
    
    seg_model = net_factory(args.config,args,net_type=args.seg_model)
    if os.path.exists(args.pretrain_path_seg):
        seg_model_pretrained_dict = torch.load(args.pretrain_path_seg)
        seg_model.load_state_dict(seg_model_pretrained_dict, strict=False)
        print("init seg weight from {}".format(args.pretrain_path_seg))   
    seg_model.eval()
    metric = []
    args.sample_list = sorted(args.sample_list)
    for case in tqdm(args.sample_list):
        first_metric, second_metric, third_metric = inference_single_case(
            case, seg_model, test_save_path, args, writer)
        logging.info(f"{case}:[{first_metric}_{second_metric}_{third_metric}]")
        case_metric = np.array([first_metric, second_metric, third_metric])
        metric.append(case_metric)
    metric = np.stack(metric)
    metric[metric==None] = np.nan
    avg_metric = np.nanmean(metric,axis=0)
    writer.close()
    logging.info(f"cls_dice:{avg_metric}")
    logging.info(f"dsc:{(avg_metric[0]+avg_metric[1]+avg_metric[2])/3}")
    return avg_metric

if __name__ == '__main__':
    args = parser.parse_args()
    args.patch_size = [args.patch_size,args.patch_size]
    args.config = get_config(args)
    args.test_mad = False
    args.root_path = "/media/grozta/SOYO/DATASET/ACDC"
    current_mode = "Inference_seg_model"
    
    # 测试mad的恢复效果
    if current_mode == "Inference_mad_model":
        args.test_mad = True
        args.exp = "test/mad_model"
        args.labeled_num = 140
        args.patch_size = [256,256]
        args.tag = "v1"
        metric = Inference_mad_model(args)
    
    # 测试seg的预测效果
    if current_mode == "Inference_seg_model":
        args.test_mad = True
        args.exp = "test/seg_model"
        args.model = 'Test_all_seg'
        args.labeled_num = 140
        args.model_name_list = {"ViT_seg":"../data/pretrain/seg_model_ViT.pth","ViM_seg":"../data/pretrain/seg_model_ViM.pth","unet":"../data/pretrain/seg_model_unet.pth"}
        args.patch_size = [256,256]
        args.tag = "v3"
        args.config = get_config(args)
        metric = Inference_seg_model(args)
        
    # 将seg的输出直接给ema，测试该过程的修复效果
    if current_mode == "Inference_seg_ema_model":
        args.exp = "test/seg_ema_model"
        args.labeled_num = 140
        args.patch_size = [224,224]
        args.seg_model = "ViM_seg"
        args.mad_model = "unet"
        args.pretrain_path_seg = "../data/pretrain/seg_model_ViM.pth"
        args.pretrain_path_mad = "../data/pretrain/mad_model_unet.pth"
        args.tag = "v1"
        metric = Inference_seg_ema_model(args)    
    # 通过seg产生预测结果，并将结果保存在数据集中
    if current_mode == "Inference_seg_model_genarate_new_dataset":
        args.exp = "gen_new_dataset/seg_model"
        args.patch_size = [224,224]
        args.seg_model = "ViM_seg"
        args.pretrain_path_seg = "../data/pretrain/seg_model_ViM.pth"
        args.tag = "v2"
        metric = Inference_seg_model_genarate_new_dataset(args)   
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
