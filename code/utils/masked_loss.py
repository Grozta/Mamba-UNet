import torch
import torch.nn.functional as F

def get_shuffled_recovery_loss(model,input,cube_size=32):
    normal_pre = model.forward_mix_pos_mask(input)
    sub_images = input.unfold(0, cube_size, cube_size).unfold(1, cube_size, cube_size)
    index_map = torch.range(sub_images.shape[0] * sub_images.shape[1])
    shuffled_indices = torch.randperm(index_map.numel())
    shuffled_sub_images = sub_images.reshape(-1, cube_size, cube_size)[shuffled_indices].reshape(sub_images.shape)
    shuffled_sub_images = shuffled_sub_images.reshape(input.shape)
    pos_embed_pre = model.forward_mix_pos_mask(shuffled_sub_images.to(input.device), pos_embed = shuffled_indices.to(input.device))
    shuffled_loss = F.mse_loss(normal_pre, pos_embed_pre)
    return shuffled_loss, pos_embed_pre

def get_mask_recovery_loss(model,input,masked_rate = 0.25,cube_size=32):
    normal_pre = model.forward_mix_pos_mask(input)
    sub_images = input.unfold(0, cube_size, cube_size).unfold(1, cube_size, cube_size)
    mask = torch.rand(1, sub_images.shape[0] * sub_images.shape[1])
    mask[mask > masked_rate] = 1
    mask[mask <= masked_rate] = 0
    line_sub_image = sub_images.reshape(-1, cube_size, cube_size)
    for index in range(line_sub_image.shape[0]):
        if not mask[index]:
            line_sub_image[index].fill_(1e-6)
    line_sub_image = line_sub_image.reshape(sub_images.shape)
    pos_embed_mask = model.forward_mix_pos_mask(line_sub_image.to(input.device), mask= mask.to(input.device))
    mask_recovery_loss = F.mse_loss(normal_pre, pos_embed_mask)
    return mask_recovery_loss, pos_embed_mask




    
