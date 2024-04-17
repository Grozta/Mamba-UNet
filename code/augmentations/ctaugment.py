# https://raw.githubusercontent.com/google-research/fixmatch/master/libml/ctaugment.py
#
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Control Theory based self-augmentation, modified from https://github.com/vfdev-5/FixMatch-pytorch"""
import random
import torch
from collections import namedtuple

import numpy as np
from scipy.ndimage import zoom
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from torchvision import transforms


OPS = {}
OP = namedtuple("OP", ("f", "bins"))
Sample = namedtuple("Sample", ("train", "probe"))


def register(*bins):
    def wrap(f):
        OPS[f.__name__] = OP(f, bins)
        return f

    return wrap


class CTAugment(object):
    def __init__(self, depth=2, th=0.85, decay=0.99):
        self.decay = decay
        self.depth = depth
        self.th = th
        self.rates = {}
        self.random_depth_weak = 2
        self.random_depth_strong = 2
        for k, op in OPS.items():
            self.rates[k] = tuple([np.ones(x, "f") for x in op.bins])

    def rate_to_p(self, rate):
        p = rate + (1 - self.decay)  # Avoid to have all zero.
        p = p / p.max()
        p[p < self.th] = 0
        return p

    def policy(self, probe, weak):
        num_strong_ops = 9
#         print('list(OPS.keys())',list(OPS.keys()))
        kl_weak = list(OPS.keys())[num_strong_ops:]
#         print('kl_weak',kl_weak)
        kl_strong = list(OPS.keys())[:num_strong_ops]
#         print('kl_strong',kl_strong)

        if weak:
            kl = kl_weak
            current_depth = self.random_depth_weak
        else:
            kl = kl_strong
            current_depth = self.random_depth_strong

        v = []
        if probe:
            for _ in range(current_depth):
                k = random.choice(kl)
                bins = self.rates[k]
                rnd = np.random.uniform(0, 1, len(bins))
                v.append(OP(k, rnd.tolist()))
            return v
        for _ in range(current_depth):
            vt = []
            k = random.choice(kl)
            bins = self.rates[k]
            rnd = np.random.uniform(0, 1, len(bins))
            for r, bin in zip(rnd, bins):
                p = self.rate_to_p(bin)
                value = np.random.choice(p.shape[0], p=p / p.sum())
                vt.append((value + r) / p.shape[0])
            v.append(OP(k, vt))
        return v

    def update_rates(self, policy, proximity):
        for k, bins in policy:
            for p, rate in zip(bins, self.rates[k]):
                p = int(p * len(rate) * 0.999)
                rate[p] = rate[p] * self.decay + proximity * (1 - self.decay)
            print(f"\t {k} weights updated")

    def stats(self):
        return "\n".join(
            "%-16s    %s"
            % (
                k,
                " / ".join(
                    " ".join("%.2f" % x for x in self.rate_to_p(rate))
                    for rate in self.rates[k]
                ),
            )
            for k in sorted(OPS.keys())
        )


def _enhance(x, op, level):
    return op(x).enhance(0.1 + 1.9 * level)


def _imageop(x, op, level):
    return Image.blend(x, op(x), level)


def _filter(x, op, level):
    return Image.blend(x, x.filter(op), level)


@register(17)
def autocontrast(x, level):
    return _imageop(x, ImageOps.autocontrast, level)


@register(17)
def brightness(x, brightness):
    return _enhance(x, ImageEnhance.Brightness, brightness)


@register(17)
def color(x, color):
    return _enhance(x, ImageEnhance.Color, color)


@register(17)
def contrast(x, contrast):
    return _enhance(x, ImageEnhance.Contrast, contrast)


@register(17)
def equalize(x, level):
    return _imageop(x, ImageOps.equalize, level)


# @register(17)
# def invert(x, level):
#     return _imageop(x, ImageOps.invert, level)


# @register(8)
# def posterize(x, level):
#     level = 1 + int(level * 7.999)
#     return ImageOps.posterize(x, level)


# @register(17)
# def solarize(x, th):
#     th = int(th * 255.999)
#     return ImageOps.solarize(x, th)


@register(17)
def smooth(x, level):
    return _filter(x, ImageFilter.SMOOTH, level)


@register(17)
def blur(x, level):
    return _filter(x, ImageFilter.BLUR, level)


@register(17)
def sharpness(x, sharpness):
    return _enhance(x, ImageEnhance.Sharpness, sharpness)




@register(17)
def cutout(x, level):
    """Apply cutout to pil_img at the specified level."""
    size = 1 + int(level * min(x.size) * 0.499)
    img_height, img_width = x.size
    height_loc = np.random.randint(low=img_height // 2, high=img_height)
    width_loc = np.random.randint(low=img_height // 2, high=img_width)
    upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
    lower_coord = (
        min(img_height, height_loc + size // 2),
        min(img_width, width_loc + size // 2),
    )
    pixels = x.load()  # create the pixel map
    for i in range(upper_coord[0], lower_coord[0]):  # for every col:
        for j in range(upper_coord[1], lower_coord[1]):  # For every row
            x.putpixel((i, j), 0)  # set the color accordingly
    return x

# weak after here

@register()
def identity(x):
    return x


@register(17, 6)
def rescale(x, scale, method):
    s = x.size
    scale *= 0.25
    crop = (scale * s[0], scale * s[1], s[0] * (1 - scale), s[1] * (1 - scale))
    methods = (
        Image.ANTIALIAS,
        Image.BICUBIC,
        Image.BILINEAR,
        Image.BOX,
        Image.HAMMING,
        Image.NEAREST,
    )
    method = methods[int(method * 5.99)]
    return x.crop(crop).resize(x.size, method)


@register(17)
def rotate(x, angle):
    angle = int(np.round((2 * angle - 1) * 45))
    return x.rotate(angle)


@register(17)
def shear_x(x, shear):
    shear = (2 * shear - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))


@register(17)
def shear_y(x, shear):
    shear = (2 * shear - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, 0, 0, shear, 1, 0))


@register(17)
def translate_x(x, delta):
    delta = (2 * delta - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, 0, delta, 0, 1, 0))


@register(17)
def translate_y(x, delta):
    delta = (2 * delta - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, 0, 0, 0, 1, delta))


def get_grid_shuffle_index(image_shape,grid_blocks=(4,4)):
    """Split the image into a 4x4 grid of blocks.
    grid_blocks is counts of image's block
    
    Args:
        image_shape (tensor): [x,y]
    """
    x,y = image_shape[-2], image_shape[-1]
    assert (image_shape[0])%(grid_blocks[0]) == 0 and (image_shape[1])%(grid_blocks[1])==0
    #get shuffle image
    block_size = x//(grid_blocks[0]), y//(grid_blocks[1])
    img_index = torch.arange(x*y).reshape(x,y)
    shuffle_grid_indices = torch.randperm(grid_blocks[0]*grid_blocks[1])
    img_index_grid = img_index.view(grid_blocks[0], block_size[0], grid_blocks[1], block_size[1]).permute(0, 2, 1, 3).contiguous().view(-1, block_size[0], block_size[1])
    img_index_grid_permuted = img_index_grid[shuffle_grid_indices]
    shuffle_index = img_index_grid_permuted.view(grid_blocks[0],grid_blocks[1],block_size[0], block_size[1]).permute(0, 2, 1, 3).contiguous().view(x,y)
    return shuffle_index, shuffle_grid_indices

def grid_shuffle_image(image,shuffle_index):
    """Desc: new grid image from shuffle indexs
    
    Note: Deal with single image
    
    Args:
        image (tensor): [b,x,y] or [x,y]
        shuffle_index (tensor): [x,y]
    
    """
    batch = 1
    if len(image.shape) >2:
        batch = image.shape[0]

    flattened_image = image.view(batch, -1)
    shuffled_image = torch.gather(flattened_image, 1, shuffle_index.view(1, -1).expand_as(flattened_image)).reshape(image.shape)
    return shuffled_image

def grid_recover_image(image,shuffle_index):
    """_summary_ 
    Args:
        image (tensor): [b,c,x,y]
        shuffle_index (tensor): [x,y]
    """
    b,c,x,y = image.shape
    flattened_image = image.view(b,c, -1)
    recover_image = torch.gather(flattened_image, 2, shuffle_index.view(1, -1).argsort().expand_as(flattened_image)).reshape(image.shape)
    return recover_image

def grid_shuffle_image_inter_index(image,grid_blocks=(4,4)):
    """
    Split the image into a 4x4 grid of blocks.
    grid_blocks is counts of image's block
    """
    x,y = image.shape[-2], image.shape[-1]
    assert (image.shape[0])%(grid_blocks[0]) == 0 and (image.shape[1])%(grid_blocks[1])==0
    #get shuffle image
    block_size = x//(grid_blocks[0]), y//(grid_blocks[1])
    image_grid = image.view(grid_blocks[0], block_size[0], grid_blocks[1], block_size[1]).permute(0, 2, 1, 3).contiguous().view(-1, block_size[0], block_size[1])
    #get shuffle image index
    shuffle_index, shuffle_grid_indices = get_grid_shuffle_index((x,y)),grid_blocks
    grid_permuted = image_grid[shuffle_grid_indices]
    shuffle_grid_img = grid_permuted.view(grid_blocks[0],grid_blocks[1],block_size[0], block_size[1]).permute(0, 2, 1, 3).contiguous().view(x,y)
    
    return shuffle_grid_img,shuffle_grid_indices.reshape(grid_blocks),shuffle_index

def grid_recover_image_inter_index(image,shuffle_index):
    grid_blocks = shuffle_index.shape
    assert (image.shape[0])%(grid_blocks[0]) == 0 and (image.shape[1])%(grid_blocks[1])==0
    block_size = (image.shape[0])//(grid_blocks[0]), (image.shape[1])//(grid_blocks[1])
    image_grid = image.view(grid_blocks[0], block_size[0], grid_blocks[1], block_size[1]).permute(0, 2, 1, 3).contiguous().view(-1, block_size[0], block_size[1])
    recover_indices = shuffle_index.reshape(-1,1).squeeze(1)
    grid_permuted = image_grid[recover_indices.argsort()]
    recover_img = grid_permuted.view(grid_blocks[0],grid_blocks[1],block_size[0], block_size[1]).permute(0, 2, 1, 3).contiguous().view(image.shape)
    return recover_img
    
    
if __name__ == "__main__":
    is_use_inter_index = False
    image_shape = (4,4)
    grid_shape = (2,2)
    x = torch.arange(image_shape[0]*image_shape[1]).reshape(image_shape)
    if  is_use_inter_index:
        print(f"x:{x}")
        img,grid_index,indexs = grid_shuffle_image_inter_index(x,grid_shape)
        print(f"indexs:{indexs}")
        print(f"img:{img}")
        y = grid_recover_image_inter_index(img,grid_index)
        print(f"y:{y}") 
    else:
        x1 = torch.arange(10,image_shape[0]*image_shape[1]+10).reshape(image_shape)
        x2 = x
        ori_img = torch.stack([x1.repeat(2,1,1),x2.repeat(2,1,1)])
        print(ori_img)
        s_indexs,_ = get_grid_shuffle_index(x.shape,(2,2))
        print(s_indexs)
        s_x1 = grid_shuffle_image(x1,s_indexs)
        s_x2 = grid_shuffle_image(x2,s_indexs)
        img1 = s_x1.repeat(2,1,1)
        img2 = s_x2.repeat(2,1,1)
        out_img = torch.stack([img1,img2])
        print(out_img)
        recover_img = grid_recover_image(out_img,s_indexs)
        print(recover_img)