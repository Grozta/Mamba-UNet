import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/MAD_FineTuning', help='experiment_name')
parser.add_argument('--tag',type=str,
                    default='v99', help='tag of experiment')
parser.add_argument('--input_channels_mad', type=int,  default=4,
                    help='Number of input channels about mad network')
parser.add_argument('--mad_model', type=str,
                    default='unet', help='mad_model_name and ema_model name')
parser.add_argument('--seg_model', type=str,
                    default='unet', help='seg_model name')
parser.add_argument('--pretrain_path_seg', type=str,
                    default='../data/pretrain/seg_model_unet.pth', help='pretrain seg_model path')
parser.add_argument('--pretrain_path_mad', type=str,
                    default='../data/pretrain/mad_model_unet.pth', help='pretrain mad_model path')
parser.add_argument('--train_struct_mode',type=int, default=0,choices=[0,1,2],
                    help='The structure of training.')
parser.add_argument('--image_fusion_mode',type=int, default=0,choices=[0,1,2,3,4,5],
                    help='Image fusion mode.')
parser.add_argument('--sample_pred_source',type=str,choices=["label","pred_vim_224","pred_unet_256"],
                    default='pred_vim_224', help='The field name of the image source from dataset sample') #labeled_num
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--image_noise',type=float,  
                    default=0.001, help='Noise added when converting to binary image')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--test_iterations', type=int,
                    default=660, help='maximum epoch number to test')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--initial_lr', type=float,  default=1e-2,
                    help='initial segmentation network learning rate')
parser.add_argument('--lr_threshold', type=float,  default=1e-6,
                    help='mim learning rate')
parser.add_argument('--weight_decay', type=float,  default=3e-5,
                    help='Adam weight_decay')
parser.add_argument('--lr_scheduler_factor', type=float,  default=0.8,
                    help='lr_scheduler factor')
parser.add_argument('--lr_scheduler_eps', type=float,  default=1e-3,
                    help='Only when the change amount exceeds this value, it will be considered to have a significant change.')
parser.add_argument('--lr_scheduler_patience', type=float,  default=10,
                    help='lr_scheduler_patience')
parser.add_argument('--train_loss_MA_alpha', type=float,  default=0.83,
                    help='train loss Move Avarge alpha value')
parser.add_argument('--patch_size', type=int, nargs='+', default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, 
                    help='random seed')
parser.add_argument('--num_workers', type=int, default=8,
                    help='numbers of workers in dataloader')
parser.add_argument('--ema_decay', type=float,  default=0.999, 
                    help='ema_decay')
parser.add_argument('--cfg', type=str, 
                    default="../code/configs/vmamba_tiny.yaml", help='path to config file', )
parser.add_argument("--opts", default=None, nargs='+',
                    help="Modify config options by adding 'KEY VALUE' pairs. ")
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, ''full: cache all data, ''part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
parser.add_argument('--train_test_mode',type=int, default=1,choices=[0,1,2],
                    help='The mode of train or test.')
parser.add_argument('--clean_before_run',default=False, 
                    action="store_true", help='Clean target folder before running')