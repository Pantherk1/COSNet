# -*- coding: utf-8 -*-

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=str, default="0", help="device")
parser.add_argument("--dataset", type=str, default="rose", choices=["rose", "cria", "drive"], help="dataset")  # choices可扩展
parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="train or test")

# data settings
parser.add_argument("--data_dir", type=str, default="autodl-tmp/OCTAC/code/dataa/ROSE-1", help="path to folder for getting dataset")
parser.add_argument("--input_nc", type=int, default=3, choices=[1, 3], help="gray or rgb")
parser.add_argument("--crop_size", type=int, default=304, help="304 crop size")
parser.add_argument("--scale_size", type=int, default=304, help="304 scale size (applied in drive and cria or rose)")

# training
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
parser.add_argument("--num_workers", type=int, default=4, help="number of threads")
parser.add_argument("--val_epoch_freq", type=int, default=1, help="frequency of validation at the end of epochs")
parser.add_argument("--save_epoch_freq", type=int, default=100, help="frequency of saving models at the end of epochs")
parser.add_argument("--init_lr", type=float, default=0.0003, help="initial learning rate")
parser.add_argument("--power", type=float, default=0.9, help="power")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay")
# first stage
parser.add_argument("--first_epochs", type=int, default=300, help="train epochs of first stage")
# second stage (if necessary)
parser.add_argument("--second_epochs", type=int, default=300, help="train epochs of second stage")
parser.add_argument("--pn_size", type=int, default=3, help="size of propagation neighbors")
parser.add_argument("--base_channels", type=int, default=64, help="basic channels")

# results
parser.add_argument("--logs_dir", type=str, default="logs", help="path to folder for saving logs")
parser.add_argument("--models_dir", type=str, default="models", help="path to folder for saving models")
parser.add_argument("--results_dir", type=str, default="results", help="path to folder for saving results")
parser.add_argument("--first_suffix", type=str, default="best_thick.pth", help="front_model-[model_suffix].pth will be loaded in models_dir")
parser.add_argument("--first_suffix1", type=str, default="best_thin.pth", help="front_model-[model_suffix].pth will be loaded in models_dir")
parser.add_argument("--second_suffix", type=str, default="best_fusion.pth", help="fusion_model-[model_suffix].pth will be loaded in models_dir")

# 对比损失部分
parser.add_argument("--temperature", type=float, default=0.07)
parser.add_argument("--proj_dmi", type=int, default=256)
parser.add_argument("--base_temperature",  type=float, default=0.07)
parser.add_argument("--max_samples", type=int, default=1024)
parser.add_argument("--max_views", type=int, default=200)
parser.add_argument("--stride", type=int, default=8)
parser.add_argument("--loss_weight", type=float, default=0.05)
parser.add_argument("--use_rmi", type=bool, default=False)
parser.add_argument("--use_lovasz", type=bool, default=False)
parser.add_argument("--with_memory", type=bool, default=True)
parser.add_argument("--with_embed", type=bool, default=True)
parser.add_argument("--memory_size", type=int, default=400)
parser.add_argument("--pixel_update_freq", type=int, default=10)
parser.add_argument("--loss_weights", default={'seg_loss': 1.0, 'aux_loss': 0.4, 'corr_loss': 0.01})
parser.add_argument("--display_iter", type=int, default=10)
parser.add_argument("--test_interval", type=int, default=1000)


args = parser.parse_args()
