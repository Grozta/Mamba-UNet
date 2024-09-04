from networks.efficientunet import Effi_UNet
from networks.enet import ENet
from networks.pnet import PNet2D
from networks.VAE import VAE_2D
from networks.unet import UNet, UNet_DS, UNet_URPC, UNet_CCT, TLUNet
from networks.magicnet_2D import VNet_2D
import argparse
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.vision_mamba import MambaUnet as ViM_seg
from networks.config import get_config
from networks.nnunet import initialize_network
from networks.projector import projectors, classifier, Jigsaw_classifier

def net_factory(config,args,net_type="unet", in_chns=1, class_num=4, vnet_n_filters= 16):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    if net_type == "vae_2d":
        net = VAE_2D(n_channels=in_chns, n_class=class_num).cuda()
    elif net_type == "TLunet":
        net = TLUNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "vnet":
        net = VNet_2D(in_chns, class_num,n_filters=vnet_n_filters).cuda()
    elif net_type == "enet":
        net = ENet(in_channels=in_chns, num_classes=class_num).cuda()
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_urpc":
        net = UNet_URPC(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "efficient_unet":
        net = Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
                        in_channels=in_chns, classes=class_num).cuda()
    elif net_type == "ViT_seg":
        net = ViT_seg(config, img_size=args.patch_size,
                      num_classes=args.num_classes).cuda()
    elif net_type == "ViM_seg":
        net = ViM_seg(config, img_size=args.patch_size,
                      num_classes=args.num_classes).cuda()
    elif net_type == "pnet":
        net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
    elif net_type == "nnUNet":
        net = initialize_network(num_classes=class_num).cuda()
    elif net_type == "classifier":
        net = classifier().cuda()
    elif net_type == "projector":
        net = projectors().cuda()
    elif net_type == "Jigsaw_classifier":
        net = Jigsaw_classifier().cuda()
    else:
        net = None
    return net
