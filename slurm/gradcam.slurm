#!/bin/bash

# DESCRIPTION: Extract Features using Trained model.

#SBATCH --account=lp_cosubi
#SBATCH --clusters=genius
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --gpus-per-node=1
#SBATCH --mem=14g  # Total memory for node
#SBATCH --partition=gpu_p100
#SBATCH --job-name=FEATURE_EXTRACTION
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=theodoro.gasperinterracamargo@student.kuleuven.be

# Navigate to your project directory
cd /data/leuven/359/vsc35907/EfficientNet_feature_extraction

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

# Activate virtual environment
source activate omnipose

# Print job details
echo "Job started on $(hostname) at $(date)"

python src/gradcam.py \
  --checkpoint '/data/leuven/359/vsc35907/EfficientNet_feature_extraction/models/model_lr_3e3_adamw_wd_1e5.pth' \
  --data_dir '/scratch/leuven/359/vsc35907/big_data_feature_extraction/patches_dirs' \
  --output_dir '/scratch/leuven/359/vsc35907/big_data_feature_extraction/grad_cam' \
  --target_layer _conv_head \
  --batch_size 1 \
  --num_samples 100
  
# Print completion time
echo "Job finished at $(date)"