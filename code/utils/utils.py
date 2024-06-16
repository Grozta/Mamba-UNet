import time
import os
import random
import re
import numpy as np
from torch.optim import lr_scheduler
from utils import ramps
from medpy import metric

# 定义一个装饰器函数
def calculate_function_run_time(func_name):
    """计算函数运行时间的装饰器"""
    def decorator_timer(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            res = func(*args, **kwargs)
            end_time = time.time()
            run_time = end_time - start_time
            print("[{}]运行时间：".format(func_name), run_time, "秒")
            return res
        return wrapper
    return decorator_timer

@calculate_function_run_time("looper")
def looper(init_val):
    for e in range(init_val):
        print("{}".format(e))
        time.sleep(1)

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136, "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def get_files_from_dir(directory,endswith='.pth'):
    """is_recursive"""
    # 创建一个空列表来存储.pth文件路径
    pth_files = []

    # 遍历指定目录下的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件是否以.pth结尾
            if file.endswith(endswith):
                # 将完整文件路径添加到列表中
                pth_files.append(os.path.join(root, file))
    
    return pth_files

def get_pth_files(directory ,endswith='.pth'):
    """bot_recursive"""
    pth_files = []
    # 获取指定目录下的所有文件和文件夹
    for file in os.listdir(directory):
        # 检查是否为文件且扩展名为 .pth
        if os.path.isfile(os.path.join(directory, file)) and file.endswith(endswith):
            # 构建文件的完整路径，并添加到 pth_files 列表中
            pth_files.append(os.path.join(file))
    return pth_files

def worker_init_fn(worker_id):
        random.seed(1337 + worker_id)
        
def get_current_consistency_weight(args,consistency,epoch):  
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
        
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)  
        
def calculate_metric_percase(pred, gt):
    if np.sum(gt)==0:
        return np.nan, np.nan
    if np.sum(pred) == 0:
        return 0.0,99.0
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    #asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice,hd95#,asd
    
def label2color(label,class_num=4):
    color_img = np.zeros((label.shape[0],label.shape[1],3))
    color_img[:,:,0][label==1]=1
    color_img[:,:,1][label==2]=1
    color_img[:,:,2][label==3]=1
    return color_img

def get_image_fusion_mode(mode):
    ref_dict = {"0": "不使用图像融合", 
                "1": "seg_pred + image作为输入,dim=2", 
                "2": "label + image作为输入,dim=2", 
                "3": "label进行二值化 + image作为输入,dim=5", 
                "4": "label进行二值化,mask + image作为输入,dim=5",
                "5": "pred进行二值化+ image作为输入,dim=5", 
                "6": "(pred进行二值化和label_mask)/2+ image作为输入,dim=5", 
                "7": "(lable-mask+ seg_pred进行二值化))/2 作为输入,dim=4",
                }
    return ref_dict[str(mode)]

def get_model_struct_mode(mode):
    """struct 1
            label-----------·-----mad_model------------loss
                           /x/       |
                            |        | ema_update
            img------seg_model----ema_model------------loss
                            |
                            ·------loss
    """   
    """struct 1
            label-----------·-----mad_model------------loss
                           /x/       |
                            |        | ema_update
            img------seg_model----ema_model
                            |
                            ·------loss
    """   
    """struct 3
            img_s---------------·
                                | 
                                |
            img------seg_model--·-ema_model------------loss
                            |
                            ·------loss
    """   
             
    ref_dict = {
        
        "0": "[结构1]seg_model输出给mad_model和ema_model,mad通过EMA更新ema_model,并且ema_model通过loss也更新", 
        "1": "[结构2]0的基础上移除ema_model的loss", 
        "2": "[结构3]移除mad_model,将seg_model直接输入给ema_model",
                }
    return ref_dict[str(mode)]

def get_train_test_mode(mode):
    ref_dict = {"0": "only_Training", 
                "1": "Train2Test", 
                "2": "only_Testing"
                }
    return ref_dict[str(mode)]

def improvement_log(struct_mode,log_mode):
    ref_struct1_dict = {
        "0": "原始的设计",
        "1": "通道上concat image",
        "2": "only_Testing"}
    ref_struct2_dict = {
        "0": "原始的设计",
        "1": "通道上concat image",
        "2": "only_Testing"}
    ref_struct3_dict = {
        "0": "only_Training",
        "1": "Train2Test",
        "2": "only_Testing"}
    ref_dict = {get_model_struct_mode(0): ref_struct1_dict, 
                get_model_struct_mode(1): ref_struct2_dict, 
                get_model_struct_mode(2): ref_struct3_dict}
    return ref_dict[str(struct_mode)][str(log_mode)]

def update_train_loss_MA(args):
        if args.train_loss_MA is None:
            args.train_loss_MA = args.all_tr_losses[-1]
        else:
            args.train_loss_MA = args.train_loss_MA_alpha * args.train_loss_MA + (1 - args.train_loss_MA_alpha) * \
                                 args.all_tr_losses[-1]
                                
def merge_volume_in_dict(input_dict):
    """res_dict[volume_name] = {index:case_res}"""
    volume_pred_list = []
    volume_label_list = []
    for context in input_dict.values():
        volume_pred = []
        volume_label = []
        sorted_keys = sorted(context.keys())
        for key in sorted_keys:
            volume_pred.append(context[key][0])
            volume_label.append(context[key][1])
        volume_pred = np.stack(volume_pred)
        volume_label = np.stack(volume_label)
        volume_pred_list.append(volume_pred)
        volume_label_list.append(volume_label)
  
    return volume_pred_list, volume_label_list

def extract_iter_number(s):
    match = re.search(r'iter_(\d+)', s)
    return int(match.group(1)) if match else float('inf')

def update_train_loss_MA(args):
    if args.train_loss_MA is None:
        args.train_loss_MA = args.all_tr_losses[-1]
    else:
        args.train_loss_MA = args.train_loss_MA_alpha * args.train_loss_MA + (1 - args.train_loss_MA_alpha) * \
                                args.all_tr_losses[-1]

                                         
if __name__ == "__main__":
    looper(5)