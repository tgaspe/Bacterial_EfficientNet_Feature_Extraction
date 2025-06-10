# Bacterial_EfficientNet_Feature_Extraction
This repository provides a pipeline for extracting high-level features from multichannel **E. coli** images using an EfficientNet model, aimed at distinguishing between wild-type and mutant deletion strain cells. The extracted features can be used for downstream analysis in microbiology, such as studying gene delition effect on bacterial morphology. The project includes scripts for image preprocessing, model training, feature extraction, and data analysis.

## Table of Contents:
- [Virtual Environment Installation](#virtual-environment-installation)
- [Project Instructions](#project-instructions)
- [Dataset](#dataset) (Extracted Features)

## Virtual Environment Installation
To set up the project, follow these steps. A GPU with CUDA 11.8 support is required for efficient training, though CPU training is possible but slower.

1. **Create a Virtual Environment**:
```bash
conda create -n EfficientNet 'python==3.10.12' pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

2. **Activate the Environment**:
```bash
conda activate EfficientNet
```

3. **Install Omnipose**:
To install the lastest version of Omnipose:
```bash
git clone https://github.com/kevinjohncutler/omnipose.git
cd omnipose
pip install -e .
```

4. **Install Remaining Packages**:
```bash
pip install -r requirements.txt
```

5. **Set Up Jupyter Notebook**:
```bash
ipython kernel install --user --name=EfficientNet
conda deactivate
```

## Project Instructions

This project is about using EfficientNet to distinguish control groups and different mutant groups in a supervised way. The main point is that it extracts high-level features from images, which is useful for downstream analysis. 

There are four main blocks of code, 

### 1. `src/pre_processing.ipynb` 
It contains five parts: 
* Compose single channel images to multi-channel images
* Instance segmentation (Omnipose) 
* Patches generation based on segmentation masks
* Dataset split with class balancing (train/validation/test)

### 2. `src/EfficientNet/train.py`
It contains the EfficientNet training program. 
There are five main hyperparameters: learning rate (`-lr`), number of epochs (`-e`), weight decay (`-wd`), optimizer, and patience (`-p`). You can either set them in the script or via the command line. 

For example, using `100` epochs, `RMSprop` optimizer, learning rate `3e-4`, weight decay `1e-3`, and patience `15`:
```
python ./src/EfficientNet/train.py \
    --data-dir 'path/to/train/dataset' \
    --output-dir 'path/to/output/directory' \
    --save-name 'your_model_name' \
    --epochs 100 \
    --optimizer 'RMSprop' \
    --learning-rate 3e-4 \
    --weight-decay 1e-3 \
    --patience 15
```

It is recommended that you use wandb https://wandb.ai/site/ to track your training and manage the version. See the quick tutorial here: https://docs.wandb.ai/quickstart/ .

### 3. `src/feature_extraction.py` 
Program to extract the features using the trained model.
There are three hyperparameters: (`--model-path`) specify where the .pth file is saved, (`--dataset-paht`) specify the path to the data, and (`--output-path`) where you want to save the .csv file.

Example usage:
```
python ./src/pre_processing/feature_extraction.py \
    --model-path  'path/to/your/saved/model' \
    --dataset-path 'path/to/your/dataset' \
    --output-path 'path/to/save/features.csv'
```

### 3.5 `src/area_extraction.py` 
Program to extract the pixel area of your patches dataset and save it in a csv file.
There are three hyperparameters: (`--input-dir`) specify where the patches dataset is located. (`--output-csv`) where to save the csv file. (`--num-processes`) specify how many paralell process you want to use.

Example usage:
```
python ./src/area_extraction.py \
    --input-dir 'path/to/your/dataset' \
    --output-csv 'path/to/save/area_cells.csv' \
    --num-processes 8
```

### 4. `src/data_analysis`

This directory contains multiple jupyter notebooks going over the data analysis part of the extracted features for our **E. coli** dataset:
- __cell_cycle_trajectory_analysis.ipynb__: Analysis using PHATE dimension reduction
- __differential_feature_means_analysis.ipynb__: wild-type vs mutants
- __enrichment_analysis.ipynb__: clustering and dimension reduction visualization 

### 4.5. (Extra) Grad Cam

```
python src/EfficientNet/gradcam.py \
  --checkpoint 'path/to/your/saved/model' \
  --data_dir 'path/to/your/dataset' \
  --output_dir 'path/to/save/images/from/grad_cam/' \
  --target_layer _conv_head \
  --batch_size 1 \
  --num_samples 20
```

## Dataset
The extracted embeddings from our **E. coli** images can be found here: 

### Acknowledgments

This project is a fork of the __Jiaanzhu/EfficientNet_feature_extraction__ GitHub repository, which focused on similar feature extraction tasks for yeast cells, and was used as a base for my EfficientNet model training. While some sections of the EfficientNet code retain the original logic and remain unchanged, other parts have been significantly modified or added to suit the specific requirements of this project and the dataset.
