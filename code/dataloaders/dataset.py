import os
import cv2
import torch
import random
import numpy as np
from glob import glob
import logging
from torch.utils.data import Dataset
import h5py
from scipy.ndimage import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image
from utils.utils import get_image_fusion_mode


class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        # print(image.shape, label.shape)
        # print('*'*10)
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample

class BaseDataSets4pretrain(Dataset):
    def __init__(self,args ,transform=None,mode = "train"):
        self.sample_list = []
        self.transform = transform
        self.mode = mode
        self.args = args

        with open(self.args.root_path + f"/{self.mode}_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
        self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        
        random.shuffle(self.sample_list)
        logging.info(f"total {len(self.sample_list)} samples")
        logging.info(f"Dataset {mode} image source from {self.args.image_source}")
        self.fusion_mode = get_image_fusion_mode(self.args.image_fusion_mode)
        logging.info(f"Dataset {mode} input image fusion mode: {self.fusion_mode}")

    def __len__(self):
        return len(self.sample_list)
    """
    for label train
    """
    def __getitem__(self, idx):
        case = self.sample_list[idx]
        case_path = self.args.root_path + "/data/slices/{}.h5".format(case)
        h5f = h5py.File(case_path, "r")
            
        if len(self.args.image_source):
            image = h5f[self.args.image_source][:]
        else:
            image = h5f["label"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label, "origin_img":h5f["image"][:]}
        sample = self.transform(sample)    
        sample["idx"] = idx
        return sample

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}
        

def random_crop_2D(image, label, output_size=(256, 256)):

    # pad the sample if necessary
    if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] :
        pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
        ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
        
        image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
        label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)

    (w, h) = image.shape
    w1 = np.random.randint(0, w - output_size[0])
    h1 = np.random.randint(0, h - output_size[1])

    label = label[w1:w1 + output_size[0], h1:h1 + output_size[1]]
    image = image[w1:w1 + output_size[0], h1:h1 + output_size[1]]
    
    return  image, label

def random_crop_2D_list(images, output_size=(256, 256)):
    # pad the sample if necessary
    if images[0].shape[0] <= output_size[0] or images[0].shape[1] <= output_size[1] :
        pw = max((output_size[0] - images[0].shape[0]) // 2 + 3, 0)
        ph = max((output_size[1] - images[0].shape[1]) // 2 + 3, 0)
        
        for idx, image in enumerate(images):
            images[idx] = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
        
    (w, h) = images[0].shape
    w1 = np.random.randint(0, w - output_size[0])
    h1 = np.random.randint(0, h - output_size[1])
    res_imgs = []    
    for image in images:
        image = image[w1:w1 + output_size[0], h1:h1 + output_size[1]]
        res_imgs.append(image)
    return res_imgs

def random_crop_2D_mask(image, label, mask_label, output_size=(256, 256)):

    # pad the sample if necessary
    if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] :
        pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
        ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
        
        image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
        label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
        mask_label = np.pad(mask_label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)

    (w, h) = image.shape
    w1 = np.random.randint(0, w - output_size[0])
    h1 = np.random.randint(0, h - output_size[1])

    mask_label = mask_label[w1:w1 + output_size[0], h1:h1 + output_size[1]]
    label = label[w1:w1 + output_size[0], h1:h1 + output_size[1]]
    image = image[w1:w1 + output_size[0], h1:h1 + output_size[1]]
    
    return  image, label, mask_label

def random_scale_2D( image, label, scale_range=(0.8, 1.2)):
    random_scale = np.random.uniform(0.8, 1.2)
    x, y = image.shape
    image = zoom(image, random_scale, order=0)
    label = zoom(label, random_scale, order=0)

    return  image, label

def random_scale_2D_list(images):
    random_scale = np.random.uniform(0.8, 1.2)
    x, y = images[0].shape
    res_imgs = []
    for image in images:
        image = zoom(image, random_scale, order=0)
        res_imgs.append(image)
    return tuple(res_imgs)

def random_scale_2D_mask( image, label, mask_label, scale_range=(0.8, 1.2)):
    random_scale = np.random.uniform(0.8, 1.2)
    x, y = image.shape
    image = zoom(image, random_scale, order=0)
    label = zoom(label, random_scale, order=0)
    mask_label = zoom(mask_label, random_scale, order=0)

    return  image, label, mask_label

def resize_data(image, label,output_size=(256, 256)):
    
    x, y = image.shape[-2], image.shape[-1]
    
    if len(image.shape) == 2:
        image = zoom(image, (output_size[0] / x, output_size[1] / y), order=0)
        label = zoom(label, (output_size[0] / x, output_size[1] / y), order=0)      
        
    return image, label

def resize_data_list(images,output_size=(256, 256)):
    x, y = images[0].shape[-2], images[0].shape[-1]
    res_imgs = []
    for image in images:
        image = zoom(image, (output_size[0] / x, output_size[1] / y), order=0)   
        res_imgs.append(image)
    return tuple(res_imgs)

def random_mask(image, label, mask_rate=0.25):
    tensor_ = np.random.rand(image.shape)
    mask = tensor_[tensor_<mask_rate]
    image[mask] = 0
    return  image, label

def random_mask_puzzle(image, label, mask_rate=0.25,mask_size = (8,8)):
    """Puzzle style add mask

    Args:
        image (numpy): input image
        label (numpy): input label(no operate)
        mask_rate (float, optional): for whole iamge. Defaults to 0.25.
        mask_size (tuple, optional): mask size. Defaults to (7,7).

    Returns:
        numpy: image and label 
    """
    image = torch.tensor(image)
    x,y = image.shape
    grid_size = x//(mask_size[0]), y//(mask_size[0])
    grid_img =  image.view(grid_size[0], mask_size[0], grid_size[1], mask_size[1]).permute(0, 2, 1, 3).contiguous().view(-1, mask_size[0], mask_size[1])
    num_zeros = int(grid_img.shape[0] * mask_rate)
    indices = np.random.choice(range(grid_img.shape[0]), num_zeros, replace=False)
    grid_img[indices,:,:] = 0
    image = grid_img.view(grid_size[0],grid_size[1], mask_size[0], mask_size[1]).permute(0, 2, 1, 3).contiguous().view(x,y)
    return  image.numpy(), label

def random_mask_edge(image, label, mask_rate=0.03,mask_size = (4,4),mask_val = -1):
    """randomly add mask by edge in center position 

    Args:
        image (numpy): input image
        label (numpy): input label(no operate)
        mask_rate (float, optional): only in edge positions. Defaults to 0.03.
        mask_size (tuple, optional): mask size. Defaults to (4,4) x2.

    Returns:
        numpy: image and label 
    """
    # Detect edges of images
    edges = cv2.Canny(image.astype(np.uint8), 1, 2)
    num_rows,num_clo = np.where(edges == 255)
    
    num_selected_rows = int(len(num_rows) * mask_rate)
    num_pos = np.arange(0, len(num_rows))
    selected_indices = np.random.choice(num_pos, num_selected_rows, replace=False)
    mask_positions = [(num_rows[indice],num_clo[indice]) for indice in selected_indices]
    
    for mask_pos in mask_positions:
        top = max(0, mask_pos[0] - mask_size[1])
        bottom = min(image.shape[0], mask_pos[0] + mask_size[1])
        left = max(0, mask_pos[1] - mask_size[0])
        right = min(image.shape[1], mask_pos[1] + mask_size[0])
        # Pick a value at random in the neighborhood
        if mask_val < 0:
            il = np.unique(image[top:bottom, left:right])
            mask_val_real = np.random.choice(image[top:bottom, left:right].flatten())
            image[top:bottom, left:right] = mask_val_real
        # Set the mask area to mask_val
        else:
            image[top:bottom, left:right] = mask_val
        
    return image, label

def image2binary(img, error_val = 1e-3, num_classes = 4):
    # Binarize label images according to categories
    binary_images = []
    for i in range(num_classes):
        binary_image = np.full_like(img,error_val,dtype = np.float32)
        binary_image[img == i] = 1- error_val
        binary_images.append(binary_image)
    binary_images = np.stack(binary_images)
    return binary_images


def np_soft_max(img):
    tensor_a = torch.from_numpy(img).unsqueeze(0)

    # 对张量 a 在 dim=1 上进行 softmax
    softmax_result = torch.nn.functional.softmax(tensor_a, dim=1).squeeze(0)

    # 将结果转换为 NumPy 数组
    softmax_array = softmax_result.numpy()
    
    return softmax_array
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, is_2d=False):
        self.is_2d = is_2d

    def __call__(self, sample):
        # image, label: For AHNet 2D to 3D,
        # 3D: WxHxD -> 1xWxHxD, 96x96x96 -> 1x96x96x96
        # 2D: WxHxD -> CxWxh, 224x224x3 -> 3x224x224
        image, label = sample['image'], sample['label']

        if self.is_2d:
            image = image.transpose(2, 0, 1).astype(np.float32)
            label = label.transpose(2, 0, 1)[1, :, :]
        else:
            # image = image.transpose(1, 0, 2)
            # label = label.transpose(1, 0, 2)
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)

        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long()}


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)
    
    image = np.rot90(image, k)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image

def random_rot_flip_list(images):
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)
    res_imgs = []
    for img in images:
        img = np.rot90(img, k)
        img = np.flip(img, axis=axis).copy()
        res_imgs.append(img)
    return tuple(res_imgs)

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def random_rotate_list(images):
    angle = np.random.randint(-20, 20)
    res_imgs = []
    for image in images:
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        res_imgs.append(image)
    return tuple(res_imgs)


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class CTATransform(object):
    def __init__(self, output_size, cta,grid_shape = (4,4)):
        self.output_size = output_size
        self.cta = cta
        self.grid_shape = grid_shape

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()
        
        # shuffle aug
        shuffle_img_index,shuffle_grid_index = augmentations.get_grid_shuffle_index((image.shape[-2],image.shape[-1]),self.grid_shape)
        Jigsaw_img = augmentations.grid_shuffle_image(image,shuffle_img_index)
        
        sample["image"] = image
        sample["label"] = label
        sample["image_weak"] = to_tensor(image_weak)
        sample["image_strong"] = to_tensor(image_strong)
        sample["label_aug"] = label_aug
        sample["Jigsaw_img"] = Jigsaw_img
        sample["Jigsaw_index"] = shuffle_img_index
        sample["Jigsaw_grid_index"] = shuffle_grid_index
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample
    

class RandomGeneratorv2(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        image, label = random_scale_2D(image,label)

        image, label = random_crop_2D(image,label,self.output_size)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.long))
        sample = {"image": image, "label": label}
        return sample
    
class RandomGeneratorv3(object):
    """for label train 
    """
    def __init__(self, args):
        self.output_size = args.patch_size
        self.num_classes = args.num_classes
        
        self.puzzle_mask_exe_rate = 0.2
        self.puzzle_mask_mask_rate = 0.25
        self.puzzle_mask_mask_size = (8,8)
        self.puzzle_mask_mask_size_list = [1,1,1,1,2,2,2,4,4,8]
        self.puzzle_mask_mask_rate_list = [0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.30,0.35,0.40,0.45,0.55,0.65]
        
        self.edge_mask_exe_rate = 0.3
        self.edge_mask_mask_rate = 0.03
        self.edge_mask_mask_size = (4,4)
        self.edge_mask_mask_size_list = [1,2,3,4]
        self.edge_mask_total = (1,4)
        self.val = -1
        self.val_list=[-1,0]
        self.image_need_trans = args.image_need_trans
        self.image_need_mask = args.image_need_mask
        
        self.error_val = args.image_noise
        self.image_fusion_mode = args.image_fusion_mode
        
    def gen_mask_param(self):
        puzzle_mask_mask_size = random.choice(self.puzzle_mask_mask_size_list)
        self.puzzle_mask_mask_size = (puzzle_mask_mask_size,puzzle_mask_mask_size)
        self.puzzle_mask_mask_rate = random.choice(self.puzzle_mask_mask_rate_list)
        
        total_value = random.uniform(self.edge_mask_total[-2],self.edge_mask_total[-1])
        edge_mask_mask_size = random.choice(self.edge_mask_mask_size_list)
        self.edge_mask_mask_size = (edge_mask_mask_size,edge_mask_mask_size)
        self.edge_mask_mask_rate = total_value/4/edge_mask_mask_size/edge_mask_mask_size
        
        self.val = random.choice(self.val_list)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        
        if self.image_fusion_mode == 0:
            if self.image_need_trans:
                if random.random() > 0.5:
                    image, label = random_rot_flip(image, label)
                    
                if random.random() > 0.5:
                    image, label = random_rotate(image, label)
                    
                image, label = random_scale_2D(image,label)

                image, label = random_crop_2D(image,label,self.output_size)
                
            image, label = resize_data(image, label,self.output_size)
            

            if self.image_need_mask:    
                if random.random() > 0.3:
                    self.gen_mask_param()
                    
                    rand = random.random()
                    if rand < 0.20:
                        image, label = random_mask_puzzle(image,label,self.puzzle_mask_mask_rate,self.puzzle_mask_mask_size)
                    elif rand < 0.85:
                        image, label = random_mask_edge(image,label,self.edge_mask_mask_rate,self.edge_mask_mask_size,self.val)
                    else:
                        image, label = random_mask_edge(image,label,self.edge_mask_mask_rate,self.edge_mask_mask_size,self.val)
                        image, label = random_mask_puzzle(image,label,self.puzzle_mask_mask_rate,self.puzzle_mask_mask_size)

                image = image2binary(image,error_val=self.error_val, num_classes=self.num_classes)
            
                image = np_soft_max(image)
                
            else:
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=0)
        else:
            image, label, origin_img = image, label, sample["origin_img"]
            if self.image_need_trans:
                if random.random() > 0.5:
                    image, label, origin_img = random_rot_flip_list([image,label,origin_img])
                    
                if random.random() > 0.5:
                    image, label, origin_img = random_rotate_list([image, label, origin_img])
                    
                image, label, origin_img = random_scale_2D_list([image, label, origin_img])

                image, label, origin_img = random_crop_2D_list([image, label, origin_img],self.output_size)
                
            image, label, origin_img = resize_data_list([image, label, origin_img],self.output_size)
            
            if self.image_fusion_mode == 1:
                image = np.stack([origin_img,image],axis=0)
            if self.image_fusion_mode == 2:
                image = np.stack([origin_img,label],axis=0)
            if self.image_fusion_mode == 3:
                b_label = image2binary(label,error_val=0.0001, num_classes=self.num_classes)
                b_label = np_soft_max(b_label)
                origin_img = np.expand_dims(origin_img,axis=0)
                image = np.concatenate([origin_img,b_label])
            if self.image_fusion_mode == 5:
                b_image = image2binary(image,error_val=0.0001, num_classes=self.num_classes)
                b_image = np_soft_max(b_image)
                origin_img = np.expand_dims(origin_img,axis=0)
                image = np.concatenate([origin_img,b_image])
            if self.image_fusion_mode == 4:
                mask_label = label.copy()
                if random.random() > 0.3:
                    self.gen_mask_param()
                    
                    rand = random.random()
                    if rand < 0.20:
                        mask_label, _ = random_mask_puzzle(mask_label,None,self.puzzle_mask_mask_rate,self.puzzle_mask_mask_size)
                    elif rand < 0.85:
                        mask_label, _ = random_mask_edge(mask_label,None,self.edge_mask_mask_rate,self.edge_mask_mask_size,self.val)
                    else:
                        mask_label, _ = random_mask_edge(mask_label,None,self.edge_mask_mask_rate,self.edge_mask_mask_size,self.val)
                        mask_label, _ = random_mask_puzzle(mask_label,None,self.puzzle_mask_mask_rate,self.puzzle_mask_mask_size)
                mask_label = image2binary(mask_label,error_val=0.0001, num_classes=self.num_classes)
                mask_label = np_soft_max(mask_label)
                origin_img = np.expand_dims(origin_img,axis=0)
                image = np.concatenate([origin_img,mask_label]) 
    
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.long))
        sample = {"image": image, "label": label}
        return sample

class RandomGeneratorv_4_finetune(object):
    """for label train 
    """
    def __init__(self, args):
        self.output_size = args.patch_size
        self.num_classes = args.num_classes
        
        self.puzzle_mask_exe_rate = 0.2
        self.puzzle_mask_mask_rate = 0.25
        self.puzzle_mask_mask_size = (8,8)
        self.puzzle_mask_mask_size_list = [1,1,1,1,2,2,2,4,4,8]
        self.puzzle_mask_mask_rate_list = [0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.30,0.35,0.40,0.45,0.55,0.65]
        
        self.edge_mask_exe_rate = 0.3
        self.edge_mask_mask_rate = 0.03
        self.edge_mask_mask_size = (4,4)
        self.edge_mask_mask_size_list = [1,2,3,4]
        self.edge_mask_total = (1,4)
        self.val = -1
        self.val_list=[-1,0]
        self.image_need_trans = args.image_need_trans
        self.image_need_mask = args.image_need_mask
        
        self.error_val = args.image_noise
        self.image_fusion_mode = args.image_fusion_mode
        
    def gen_mask_param(self):
        puzzle_mask_mask_size = random.choice(self.puzzle_mask_mask_size_list)
        self.puzzle_mask_mask_size = (puzzle_mask_mask_size,puzzle_mask_mask_size)
        self.puzzle_mask_mask_rate = random.choice(self.puzzle_mask_mask_rate_list)
        
        total_value = random.uniform(self.edge_mask_total[-2],self.edge_mask_total[-1])
        edge_mask_mask_size = random.choice(self.edge_mask_mask_size_list)
        self.edge_mask_mask_size = (edge_mask_mask_size,edge_mask_mask_size)
        self.edge_mask_mask_rate = total_value/4/edge_mask_mask_size/edge_mask_mask_size
        
        self.val = random.choice(self.val_list)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        
        if self.image_fusion_mode == 0:
            if self.image_need_trans:
                if random.random() > 0.5:
                    image, label = random_rot_flip(image, label)
                    
                if random.random() > 0.5:
                    image, label = random_rotate(image, label)
                    
                image, label = random_scale_2D(image,label)

                image, label = random_crop_2D(image,label,self.output_size)
                
            image, label = resize_data(image, label,self.output_size)
            

            if self.image_need_mask:    
                if random.random() > 0.3:
                    self.gen_mask_param()
                    
                    rand = random.random()
                    if rand < 0.20:
                        image, label = random_mask_puzzle(image,label,self.puzzle_mask_mask_rate,self.puzzle_mask_mask_size)
                    elif rand < 0.85:
                        image, label = random_mask_edge(image,label,self.edge_mask_mask_rate,self.edge_mask_mask_size,self.val)
                    else:
                        image, label = random_mask_edge(image,label,self.edge_mask_mask_rate,self.edge_mask_mask_size,self.val)
                        image, label = random_mask_puzzle(image,label,self.puzzle_mask_mask_rate,self.puzzle_mask_mask_size)

                image = image2binary(image,error_val=self.error_val, num_classes=self.num_classes)
            
                image = np_soft_max(image)
                
            else:
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=0)
        else:
            image, label, origin_img = image, label, sample["origin_img"]
            if self.image_need_trans:
                if random.random() > 0.5:
                    image, label, origin_img = random_rot_flip_list([image,label,origin_img])
                    
                if random.random() > 0.5:
                    image, label, origin_img = random_rotate_list([image, label, origin_img])
                    
                image, label, origin_img = random_scale_2D_list([image, label, origin_img])

                image, label, origin_img = random_crop_2D_list([image, label, origin_img],self.output_size)
                
            image, label, origin_img = resize_data_list([image, label, origin_img],self.output_size)
            
            if self.image_fusion_mode == 1:
                image = np.stack([origin_img,image],axis=0)
            if self.image_fusion_mode == 2:
                image = np.stack([origin_img,label],axis=0)
            if self.image_fusion_mode == 3:
                b_label = image2binary(label,error_val=0.0001, num_classes=self.num_classes)
                b_label = np_soft_max(b_label)
                origin_img = np.expand_dims(origin_img,axis=0)
                image = np.concatenate([origin_img,b_label])
            if self.image_fusion_mode == 5:
                b_image = image2binary(image,error_val=0.0001, num_classes=self.num_classes)
                b_image = np_soft_max(b_image)
                origin_img = np.expand_dims(origin_img,axis=0)
                image = np.concatenate([origin_img,b_image])
            if self.image_fusion_mode == 4:
                mask_label = label.copy()
                if random.random() > 0.3:
                    self.gen_mask_param()
                    
                    rand = random.random()
                    if rand < 0.20:
                        mask_label, _ = random_mask_puzzle(mask_label,None,self.puzzle_mask_mask_rate,self.puzzle_mask_mask_size)
                    elif rand < 0.85:
                        mask_label, _ = random_mask_edge(mask_label,None,self.edge_mask_mask_rate,self.edge_mask_mask_size,self.val)
                    else:
                        mask_label, _ = random_mask_edge(mask_label,None,self.edge_mask_mask_rate,self.edge_mask_mask_size,self.val)
                        mask_label, _ = random_mask_puzzle(mask_label,None,self.puzzle_mask_mask_rate,self.puzzle_mask_mask_size)
                mask_label = image2binary(mask_label,error_val=0.0001, num_classes=self.num_classes)
                mask_label = np_soft_max(mask_label)
                origin_img = np.expand_dims(origin_img,axis=0)
                image = np.concatenate([origin_img,mask_label]) 
    
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.long))
        sample = {"image": image, "label": label}
        return sample
    
class RandomGeneratorv4(object):
    """for label train 
    """
    def __init__(self, output_size, num_classes = 4):
        self.output_size = output_size
        self.num_classes = num_classes
        self.puzzle_mask_exe_rate = 0.2
        self.puzzle_mask_mask_rate = 0.25
        self.puzzle_mask_mask_size = (8,8)
        self.puzzle_mask_mask_size_list = [1,1,1,1,2,2,2,4,4,8]
        self.puzzle_mask_mask_rate_list = [0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.30,0.35,0.40,0.45,0.55,0.65]
        
        self.edge_mask_exe_rate = 0.3
        self.edge_mask_mask_rate = 0.03
        self.edge_mask_mask_size = (4,4)
        self.edge_mask_mask_size_list = [1,2,3,4]
        self.edge_mask_total = (1,4)
        self.val = -1
        self.val_list=[-1,0]
        
    def gen_mask_param(self):
        puzzle_mask_mask_size = random.choice(self.puzzle_mask_mask_size_list)
        self.puzzle_mask_mask_size = (puzzle_mask_mask_size,puzzle_mask_mask_size)
        self.puzzle_mask_mask_rate = random.choice(self.puzzle_mask_mask_rate_list)
        
        total_value = random.uniform(self.edge_mask_total[-2],self.edge_mask_total[-1])
        edge_mask_mask_size = random.choice(self.edge_mask_mask_size_list)
        self.edge_mask_mask_size = (edge_mask_mask_size,edge_mask_mask_size)
        self.edge_mask_mask_rate = total_value/4/edge_mask_mask_size/edge_mask_mask_size
        
        self.val = random.choice(self.val_list)
        
    def mask_label_onle(self,mask_label,label = None):
        self.gen_mask_param()
        
        rand = random.random()
        if rand < 0.20:
            mask_label, label = random_mask_puzzle(mask_label,label,self.puzzle_mask_mask_rate,self.puzzle_mask_mask_size)
        elif rand < 0.85:
            mask_label, label = random_mask_edge(mask_label,label,self.edge_mask_mask_rate,self.edge_mask_mask_size,self.val)
        else:
            mask_label, label = random_mask_edge(mask_label,label,self.edge_mask_mask_rate,self.edge_mask_mask_size,self.val)
            mask_label, label = random_mask_puzzle(mask_label,label,self.puzzle_mask_mask_rate,self.puzzle_mask_mask_size)
            
        mask_label = image2binary(mask_label,num_classes=self.num_classes)
        mask_label = np_soft_max(mask_label)
        return mask_label

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
            
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        image, label = resize_data(image, label,self.output_size)
        
        mask_label = label.copy()
        
        self.gen_mask_param()
        
        rand = random.random()
        if rand < 0.20:
            mask_label, label = random_mask_puzzle(mask_label,label,self.puzzle_mask_mask_rate,self.puzzle_mask_mask_size)
        elif rand < 0.85:
            mask_label, label = random_mask_edge(mask_label,label,self.edge_mask_mask_rate,self.edge_mask_mask_size,self.val)
        else:
            mask_label, label = random_mask_edge(mask_label,label,self.edge_mask_mask_rate,self.edge_mask_mask_size,self.val)
            mask_label, label = random_mask_puzzle(mask_label,label,self.puzzle_mask_mask_rate,self.puzzle_mask_mask_size)
        
        image, label, mask_label = random_scale_2D_mask(image,label,mask_label)

        image, label, mask_label = random_crop_2D_mask(image,label,mask_label,self.output_size)

        mask_label = image2binary(mask_label,num_classes=self.num_classes)
        mask_label = np_soft_max(mask_label)
        
        mask_label = torch.from_numpy(mask_label.astype(np.float32))
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.long))
        sample = {"image": image, "label": label, 'mask_label': mask_label}
        return sample


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images
    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)