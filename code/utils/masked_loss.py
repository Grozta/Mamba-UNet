import torch
import torch.nn.functional as F

def get_shuffled_recovery_loss(model,input,cube_size=32):
    normal_pre = model.forward_mix_pos_mask(input)
    input_set = []
    indices_list = []
    for slice in input:
        slice = slice.squeeze(dim=0)
        sub_images = slice.unfold(0, cube_size, cube_size).unfold(1, cube_size, cube_size)
        index_map = torch.range(0,sub_images.shape[0] * sub_images.shape[1]-1).reshape(1,-1).to(input.device)
        shuffled_indices = torch.randperm(index_map.numel()).reshape(1,-1).to(input.device)
        shuffled_sub_images = sub_images.reshape(-1, cube_size, cube_size)[shuffled_indices].reshape(slice.shape).unsqueeze(dim=0)
        input_set.append(shuffled_sub_images)
        indices_list.append(shuffled_indices)

    input_set = torch.stack(input_set, dim=0)
    indices_list = torch.stack(indices_list, dim=0)
    pos_embed_pre = model.forward_mix_pos_mask(input_set, pos_embed = indices_list)
    shuffled_loss = F.mse_loss(normal_pre, pos_embed_pre)
    return shuffled_loss, pos_embed_pre

def get_mask_recovery_loss(model,input,masked_rate = 0.25,cube_size=32):
    normal_pre = model.forward_mix_pos_mask(input)
    input_set = []
    mask_list = []
    for slice in input:
        slice = slice.squeeze(dim=0)
        sub_images = slice.unfold(0, cube_size, cube_size).unfold(1, cube_size, cube_size)
        mask = torch.rand(sub_images.shape[0] * sub_images.shape[1]).reshape(1,-1).to(input.device)
        mask[mask > masked_rate] = 1.0
        mask[mask <= masked_rate] = 0.0
        line_sub_image = sub_images.reshape(-1, cube_size, cube_size)
        for index in range(line_sub_image.shape[0]):
            if not mask[0][index]:
                line_sub_image[index].fill_(1e-6)
        line_sub_image = line_sub_image.reshape(slice.shape).unsqueeze(dim=0)
        input_set.append(line_sub_image)
        mask_list.append(mask)

    input_set = torch.stack(input_set, dim=0)
    mask_list = torch.stack(mask_list, dim=0)
    pos_embed_mask = model.forward_mix_pos_mask(input_set, mask = mask_list)
    mask_recovery_loss = F.mse_loss(normal_pre, pos_embed_mask)
    return mask_recovery_loss, pos_embed_mask




    
