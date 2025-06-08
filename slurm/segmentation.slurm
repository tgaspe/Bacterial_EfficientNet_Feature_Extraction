#!/bin/bash

# DESCRIPTION: Segment phase contrast images to make ground truth masks.

#SBATCH --account=lp_cosubi
#SBATCH --clusters=genius
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-cpu=3g
#SBATCH --partition=gpu_p100
#SBATCH --job-name=MASKS
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=theodoro.gasperinterracamargo@student.kuleuven.be

# Navigate to your project directory
cd /data/leuven/359/vsc35907/EfficientNet_feature_extraction

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

# Activate virtual environment
source activate omnipose

# Print job details
echo "Job started on $(hostname) at $(date)"

# Phase Contrast Segmentation using Omnipose
omnipose --dir '/scratch/leuven/359/vsc35907/feature_extraction_data/stacked_images/mutants/' \
    --use_gpu --pretrained_model 'bact_phase_omni' \
    --save_tif --savedir '/scratch/leuven/359/vsc35907/feature_extraction_data/stacked_images/mutants/masks' \
    --chan 1 --nchan 2 --dim 2 --nclasses 3  \
    --no_npy --mask_threshold 0
    
    
# Phase Contrast Segmentation using Omnipose
omnipose --dir '/scratch/leuven/359/vsc35907/feature_extraction_data/stacked_images/wild_type/' \
    --use_gpu --pretrained_model 'bact_phase_omni' \
    --save_tif --savedir '/scratch/leuven/359/vsc35907/feature_extraction_data/stacked_images/wild_type/masks' \
    --chan 1 --nchan 2 --dim 2 --nclasses 3  \
    --no_npy --mask_threshold 0

# SEGMENTATION USING CPU:
# omnipose --dir '/scratch/leuven/359/vsc35907/feature_extraction_data/stacked_images/mutants/' \
#     --pretrained_model 'bact_phase_omni' \
#     --save_tif --savedir '/scratch/leuven/359/vsc35907/feature_extraction_data/stacked_images/mutants/masks' \
#     --chan 1 --nchan 2 --dim 2 --nclasses 3  \
#     --no_npy --mask_threshold 0
    
# omnipose --dir '/scratch/leuven/359/vsc35907/feature_extraction_data/stacked_images/wild_type/' \
#     --pretrained_model 'bact_phase_omni' \
#     --save_tif --savedir '/scratch/leuven/359/vsc35907/feature_extraction_data/stacked_images/wild_type/masks' \
#     --chan 1 --nchan 2 --dim 2 --nclasses 3  \
#     --no_npy --mask_threshold 0

# Print completion time
echo "Job finished at $(date)"