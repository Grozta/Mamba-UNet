import torch
import random
import torch.nn.functional as F

def get_shuffled_recovery_loss(model,input,cube_size=32):
    normal_pre = model.forward_mix_pos_mask(input)
    n_cubes = input.shape[-1]//cube_size
    b_sub_images = input.unfold(2, cube_size, cube_size).unfold(3, cube_size, cube_size).reshape(input.shape[0],-1,cube_size, cube_size)
    shuffled_indices = torch.stack([torch.randperm(n_cubes**2) for _ in range(input.shape[0])]).to(input.device)
    input_set = torch.stack([sub_img[index] for sub_img, index in zip(b_sub_images,shuffled_indices)],dim=0).reshape(input.shape)

    pos_embed_pre = model.forward_mix_pos_mask(input_set, pos_embed = shuffled_indices)
    shuffled_loss = F.mse_loss(normal_pre, pos_embed_pre)
    return shuffled_loss, pos_embed_pre

def get_mask_recovery_loss(model,input,masked_rate = 0.25,cube_size=32):
    # normal_pre bx4096
    normal_pre = model.forward_mix_pos_mask(input)
    n_cubes = input.shape[-1]//cube_size
    b_sub_images = input.unfold(2, cube_size, cube_size).unfold(3, cube_size, cube_size).reshape(input.shape[0],-1,cube_size, cube_size)
    mask = torch.stack([torch.rand(n_cubes**2) for _ in range(input.shape[0])]).to(input.device)
    mask[mask > masked_rate] = 1.0
    mask[mask <= masked_rate] = 0.0
    mask = mask.to(torch.int)
    input_set=[]
    for sub_img, row_mask in zip(b_sub_images,mask):
        sub_img[row_mask]=1e-6
        input_set.append(sub_img)
    input_set = torch.stack(input_set,dim=0).reshape(input.shape)

    pos_embed_mask = model.forward_mix_pos_mask(input_set, mask = mask.to(torch.float))
    mask_recovery_loss = F.mse_loss(normal_pre, pos_embed_mask)
    return mask_recovery_loss, pos_embed_mask




    
