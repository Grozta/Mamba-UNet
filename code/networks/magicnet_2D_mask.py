import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv2d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv2d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            else:
                assert False
        else:

            ops.append(nn.ConvTranspose2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False))
        ops.append(nn.Conv2d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm2d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm2d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout2d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res


class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        upsampling = UpsamplingDeconvBlock  ## using transposed convolution

        self.block_five_up = upsampling(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = upsampling(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = upsampling(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = upsampling(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv2d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout2d(p=0.5, inplace=False)
    def forword_up_two(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3
        return x6_up

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        return out_seg, x9


class FcLayer(nn.Module):
    def __init__(self, ts=32, patch_size=96, n_filters=16):
        super(FcLayer, self).__init__()
        nt = patch_size // ts
        self.fc_layer = nn.Sequential(
            nn.Linear((n_filters * 16) * ((ts // 16) ** 2), 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(256, nt ** 2)
        )

    def forward(self, x):
        return self.fc_layer(x)
    
class Pos_embed_layer(nn.Module):
    def __init__(self, cube_size=32, patch_size=96):
        self.cube_size = cube_size
        self.patch_size = patch_size
        super(Pos_embed_layer, self).__init__()
        self.ncube = patch_size // cube_size
        self.scale_factor= cube_size/patch_size
        self.pos_embed_layer = nn.Sequential(
            nn.Linear(2*self.ncube ** 2, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Linear(256, patch_size ** 2)
        )
        self.conv = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=patch_size//cube_size, stride=patch_size//cube_size, padding=0)

    def forward(self,input, pos_embed, mask):
        if not len(pos_embed):
            pos_embed = torch.stack(list({torch.range(0,(self.ncube)**2 -1).view(1,(self.ncube)**2) for bat in range(input.shape[0])}),dim=0).squeeze(dim=1)
            pos_embed = pos_embed.to(input.device)
        if not len(mask):
            mask = torch.stack(list({torch.ones((1,(self.ncube)**2)).reshape(1,(self.ncube)**2) for bat in range(input.shape[0])}),dim=0).squeeze(dim=1)
            mask = mask.to(input.device)
        pos_embed_mask = torch.cat([pos_embed, mask], dim=1)
        embed = self.pos_embed_layer(pos_embed_mask.to(torch.float).reshape(input.shape[0],-1)).reshape(-1,self.patch_size,self.patch_size)
        if self.patch_size != input.shape[-1]:
            embed = F.interpolate(embed.unsqueeze(0), scale_factor=self.scale_factor, mode='bilinear', align_corners=False).squeeze(0)
        embed = embed.unsqueeze(dim=1)
        out_embed = input * embed
        return out_embed
    
class Mix_out_layer(nn.Module):
    def __init__(self, up_stage=2, patch_size=96,):
        super(Mix_out_layer, self).__init__()
        self.featue_size = patch_size // (2**up_stage)
        self.conv = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=4, stride=4, padding=0)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.mix_out_layer = nn.Sequential(
            nn.Linear(self.featue_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(256, self.featue_size ** 2)
        )

    def forward(self,x):
        x = self.conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x).reshape(x.shape[0],-1)
        return self.mix_out_layer(x)
    
class VNet_Magic_2D_mask(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, cube_size=32, patch_size=96, n_filters=16, normalization='instancenorm',
                 has_dropout=False, has_residual=False):
        super(VNet_Magic_2D_mask, self).__init__()
        self.num_classes = n_classes
        self.patch_size = patch_size
        self.cube_size = cube_size
        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.fc_layer = FcLayer(cube_size, patch_size)
        self.pos_embed_layer = Pos_embed_layer(cube_size, patch_size)
        self.mix_out_layer = Mix_out_layer(up_stage=2,patch_size=patch_size)


    def forward_prediction_head(self, feat):
        return self.decoder.out_conv(feat)

    def forward_encoder(self, x, pos_embed=[], mask=[]):
        # pos_embed mask [b,64]
        # x[b,1,256,256]
        x = self.pos_embed_layer(x, pos_embed,mask)
        return self.encoder(x)

    def forward_decoder(self, feat_list):
        return self.decoder(feat_list)
    
    def forward_mix_pos_mask(self, x, pos_embed= [],mask=[]):
        # pos_embed mask [b,64]
        # x[b,1,256,256]
        x = self.pos_embed_layer(x, pos_embed, mask)
        features = self.encoder(x)
        x = self.decoder.forword_up_two(features)
        output = self.mix_out_layer(x)
        return output

    def forward(self, x, pos_embed=[],mask=[]):
        # pos_embed mask [b,64]
        # x[b,1,256,256]
        x = self.pos_embed_layer(x, pos_embed, mask)
        features = self.encoder(x)
        out_seg, embedding = self.decoder(features)
        return out_seg, embedding  # 4, 16, 96, 96, 96


if __name__ == '__main__':
    pass
