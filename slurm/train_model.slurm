#!/bin/bash

# DESCRIPTION: Train Efficient Net for Bacterial Classification.

#SBATCH --account=lp_cosubi
#SBATCH --clusters=genius
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=2
#SBATCH --mem=14g  # Total memory for node
#SBATCH --partition=gpu_p100
#SBATCH --job-name=TRAIN_EFFICIENT_NET_1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=18:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=theodoro.gasperinterracamargo@student.kuleuven.be

# Navigate to your project directory
cd /data/leuven/359/vsc35907/EfficientNet_feature_extraction

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

# Activate virtual environment
source activate omnipose

# Print job details
echo "Job started on $(hostname) at $(date)"

python ./src/train.py \
    --data-dir '/scratch/leuven/359/vsc35907/big_data_feature_extraction/dataset' \
    --output-dir '/data/leuven/359/vsc35907/EfficientNet_feature_extraction/models' \
    --save-name 'model_lr_1e5_adamw_wd_5e2' \
    --epochs 200 \
    --optimizer 'AdamW' \
    --learning-rate 1e-5 \
    --weight-decay 5e-2 \
    --patience 15 \
    
# python ./src/train.py \
#     --data-dir '/scratch/leuven/359/vsc35907/big_data_feature_extraction/dataset' \
#     --output-dir '/data/leuven/359/vsc35907/EfficientNet_feature_extraction/models' \
#     --save-name 'model_lr_3e4_rmsprop_wd_1e3' \
#     --epochs 200 \
#     --optimizer 'RMSprop' \
#     --learning-rate 3e-4 \
#     --weight-decay 1e-3 \
#     --patience 15
    
# python ./src/train.py \
#     --data-dir '/scratch/leuven/359/vsc35907/big_data_feature_extraction/dataset' \
#     --output-dir '/data/leuven/359/vsc35907/EfficientNet_feature_extraction/models' \
#     --save-name 'model_lr_1e4_sgd_wd_5e3' \
#     --epochs 200 \
#     --optimizer 'SGD' \
#     --learning-rate 1e-4 \
#     --weight-decay 5e-3 \
#     --patience 15
    
# python ./src/train.py \
#     --data-dir '/scratch/leuven/359/vsc35907/big_data_feature_extraction/dataset' \
#     --output-dir '/data/leuven/359/vsc35907/EfficientNet_feature_extraction/models' \
#     --save-name 'model_lr_5e4_radam_wd_1e4' \
#     --epochs 200 \
#     --optimizer 'RAdam' \
#     --learning-rate 5e-4 \
#     --weight-decay 1e-4 \
#     --patience 15

# python ./src/train.py \
#     --data-dir '/scratch/leuven/359/vsc35907/big_data_feature_extraction/dataset' \
#     --output-dir '/data/leuven/359/vsc35907/EfficientNet_feature_extraction/models' \
#     --save-name 'model_lr_5e3_sgd_wd_1e2' \
#     --epochs 200 \
#     --optimizer 'SGD' \
#     --learning-rate 5e-3 \
#     --weight-decay 1e-2 \
#     --patience 15
    
# python ./src/train.py \
#     --data-dir '/scratch/leuven/359/vsc35907/big_data_feature_extraction/dataset' \
#     --output-dir '/data/leuven/359/vsc35907/EfficientNet_feature_extraction/models' \
#     --save-name 'model_lr_3e3_adamw_wd_1e5' \
#     --epochs 200 \
#     --optimizer 'AdamW' \
#     --learning-rate 3e-3 \
#     --weight-decay 1e-5 \
#     --patience 15
    

# Print completion time
echo "Job finished at $(date)"