# Bacterial_EfficientNet_Feature_Extraction

## Virtual Environment

The EfficientNet used here is from https://github.com/lukemelas/EfficientNet-PyTorch, the installation of `efficientnet_pytorch` is required.

First we need to create a virtual env and install omnipose (package that we will use for our instance segmentation)

To create an virtual environment that works with Omnipose run:
```
conda create -n EfficientNet 'python==3.10.12' pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda activate EfficientNet 
```

To install the lastest version of Omnipose

```
git clone https://github.com/kevinjohncutler/omnipose.git
cd omnipose
pip install -e .
```

To install all the remainnig required package (including the EfficientNet), run this command in your environment:

```
pip install -r requirements.txt
```

To use jupyter notebook, run the following command

```bash
ipython kernel install --user --name=EfficientNet
```

then 

```
conda deactivate
```

it should work when you activate your environment again

* if your cuda drive version is too old, you may need to use older version of torch/torchvision

## Project introduction

This project is about using EfficientNet to distinguish control groups and different mutant groups in a supervised way. The main point is that it extracts high-level features from images, which is useful for downstream analysis. 

There are four main blocks of code, 

### 1. `src/pre_processing/pre_processing.ipynb` 
It contains five parts: 
* Compose single channel images to multi-channel images
* Instance segmentation (Omnipose) 
* Patches generation based on segmentation masks
* Dataset pplit with class balancing (train/validation/test)

### 2. `src/EfficientNet/train.py`
It contains the EfficientNet training program. 
There are five main hyperparameters: learning rate (`-lr`), number of epochs (`-e`), weight decay (`-wd`), optimizer, and patience (`-p`). You can either set them in the script or via the command line. 

For example, using `200` epochs, `RMSprop` optimizer, learning rate `3e-4`, weight decay `1e-3`, and patience `15`:
```
python ./src/EfficientNet/train.py \
    --data-dir '/scratch/leuven/359/vsc35907/feature_extraction_data/data/' \
    --output-dir '/data/leuven/359/vsc35907/EfficientNet_feature_extraction/outputs' \
    --save-name 'model_patches3_lr_3e4_rmsprop_wd_1e3' \
    --epochs 200 \
    --optimizer 'RMSprop' \
    --learning-rate 3e-4 \
    --weight-decay 1e-3 \
    --patience 15
```

It is recommended that you use wandb https://wandb.ai/site/ to track your training and manage the version. See the quick tutorial here: https://docs.wandb.ai/quickstart/ .

### 3. `src/pre_processing/feature_extraction.py`
Program to extract the features using the trained model.
There are three hyperparameters: (`--model-path`) specify where the .pth file is saved, (`--dataset-paht`) specify the path to the data, and (`--output-path`) where you want to save the .csv file.


Example usage:
```
python ./src/pre_processing/feature_extraction.py \
    --model-path  '/data/leuven/359/vsc35907/EfficientNet_feature_extraction/outputs/model_patches3_lr_3e4_rmsprop_wd_1e3.pth' \
    --dataset-path '/scratch/leuven/359/vsc35907/feature_extraction_data/patches3/' \
    --output-path '/scratch/leuven/359/vsc35907/feature_extraction_data/patches3/patches3_rmsprop_features.csv'
```

### 4. `src/data_analysis`

This directory contains multiple jupyter notebooks going over the data analysis part of the extracted features for our **E. coli** dataset:
- cell_cycle 
- 
- 
- 

### Extra. Grad Cam

```
python src/EfficientNet/gradcam.py \
  --checkpoint '/data/leuven/359/vsc35907/EfficientNet_feature_extraction/outputs/model_patches3_lr_3e4_rmsprop_wd_1e3.pth' \
  --data_dir '/scratch/leuven/359/vsc35907/feature_extraction_data/patches3/' \
  --output_dir '/scratch/leuven/359/vsc35907/feature_extraction_data/grad_cam/' \
  --target_layer _conv_head \
  --batch_size 1 \
  --num_samples 20
```
### Acknowledgments

This project is a fork of the __Jiaanzhu/EfficientNet_feature_extraction__ GitHub repository, which focused on similar feature extraction tasks for yeast cells, and used as a base for my EfficientNet model training. While some sections of the EfficientNet code retain the original logic and remain unchanged, other parts have been significantly modified or added to suit the specific requirements of this project and the dataset.

