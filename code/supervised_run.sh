
CUDA_VISIBLE_DEVICES=0 python train_fully_supervised_2D.py --root_path /home/grozta/Desktop/DATASET/ACDC --exp ACDC/unet --model unet  --max_iterations 10000 --batch_size 18 --seed 5179 && \
CUDA_VISIBLE_DEVICES=0 python train_fully_supervised_2D_ViT.py --root_path /home/grozta/Desktop/DATASET/ACDC --exp ACDC/swinunet --model swinunet --max_iterations 10000 --batch_size 18 --seed 5179 && \
CUDA_VISIBLE_DEVICES=0 python train_fully_supervised_2D_VIM.py --root_path /home/grozta/Desktop/DATASET/ACDC --exp ACDC/VIM --model mambaunet --max_iterations 10000 --batch_size 18