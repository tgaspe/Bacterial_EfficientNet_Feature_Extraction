#!/bin/bash

# DESCRIPTION: Extract Area from dataset images and save it in a CSV.

#SBATCH --account=lp_cosubi
#SBATCH --clusters=genius
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=24g  # Total memory for node
#SBATCH --partition=batch
#SBATCH --job-name=AREA_EXTRACTION
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

python ./src/area_extraction.py \
    --input-dir '/scratch/leuven/359/vsc35907/big_data_feature_extraction/patches_dirs' \
    --output-csv '/scratch/leuven/359/vsc35907/big_data_feature_extraction/patches_dirs/area_cells.csv'

# Print completion time
echo "Job finished at $(date)"