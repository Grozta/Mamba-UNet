CUDA_VISIBLE_DEVICES=0 python ./MAD_FineTuning.py --root_path /media/icml/H4T/DATASET/ACDC --tag v3.7.3-1 --batch_size 12 --patch_size 256 --input_channels_mad 5 --seg_model unet --pretrain_path_seg ../data/pretrain/seg_model_unet.pth --pretrain_path_mad ../model/ACDC/MAD_Pretrain/unet_v4.8.2/unet_best_model.pth --train_struct_mode 2 --update_log_mode 1 --ablation_option 2 --max_iterations 30000 --num_workers 12 --clean_before_run && \
CUDA_VISIBLE_DEVICES=0 python ./MAD_FineTuning.py --root_path /media/icml/H4T/DATASET/ACDC --tag v3.7.3-2 --batch_size 12 --patch_size 256 --input_channels_mad 5 --seg_model unet --pretrain_path_seg ../data/pretrain/seg_model_unet.pth --pretrain_path_mad ../model/ACDC/MAD_Pretrain/unet_v4.8.2/unet_best_model.pth --train_struct_mode 2 --update_log_mode 1 --ablation_option 1 --max_iterations 30000 --num_workers 12 --clean_before_run && \
CUDA_VISIBLE_DEVICES=0 python ./MAD_FineTuning.py --root_path /media/icml/H4T/DATASET/ACDC --tag v3.7.3-3 --batch_size 12 --patch_size 256 --input_channels_mad 5 --seg_model unet --pretrain_path_seg ../data/pretrain/seg_model_unet.pth --pretrain_path_mad ../model/ACDC/MAD_Pretrain/unet_v4.8.2/unet_best_model.pth --train_struct_mode 2 --update_log_mode 1 --ablation_option 1 2 --max_iterations 30000 --num_workers 12 --clean_before_run