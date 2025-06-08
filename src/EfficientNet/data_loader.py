import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import FourChannelTiffDataset, compute_mean_std_for_channels
import os

# Required constants
IMAGE_SIZE = (224, 224)  # Upscale to match EfficientNet-B0 expectations
BATCH_SIZE = 64
NUM_WORKERS = min(os.cpu_count(), 16) 

# Training transforms (adapted for 4-channel tensors)
def get_train_transform(image_size, pretrained=True):
    train_transform = transforms.Compose([
        transforms.Resize(image_size),  # Upscale to (224, 224), output: (4, 224, 224)
        # DATA Augmentation:
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        
        # transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1),  # Light noise instead of heavy jitter
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Stronger Jitter
        # transforms.RandomAffine(degrees=(-45, 45)), # Add rotation
        # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        # transforms.RandomErasing(p=0.5), # Cutout augmentation
        
        # Normalization for 4 channels (adjust 4th channel stats if needed) PS: These values (mean/std) were computed using the whole dataset
        # transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
        # transforms.Normalize(mean=[0.02007498, 0.01879781, 0.00255546, 0.00284279], std=[0.06595125, 0.06487907, 0.01003648, 0.00976247])
        transforms.Normalize(mean=[0.02406202, 0.01604822, 0.00229301, 0.00259577], std=[0.07806426, 0.05529144, 0.00880695, 0.00874168])
        
    ])
    return train_transform

# Validation transforms
def get_valid_transform(image_size, pretrained=True):
    valid_transform = transforms.Compose([
        transforms.Resize(image_size),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
        transforms.Normalize(mean=[0.02406202, 0.01604822, 0.00229301, 0.00259577], std=[0.07806426, 0.05529144, 0.00880695, 0.00874168])
    ])
    return valid_transform

# Data loader for train and validation
def get_data_loader(TRAIN_PATH, VALID_PATH, pretrained=True):
    train_dataset = FourChannelTiffDataset(TRAIN_PATH, transform=get_train_transform(IMAGE_SIZE, pretrained))
    valid_dataset = FourChannelTiffDataset(VALID_PATH, transform=get_valid_transform(IMAGE_SIZE, pretrained))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"[INFO]: Using {NUM_WORKERS} workers for data loading")
    return train_loader, valid_loader

# Data loader for test
def get_test_loader(TEST_PATH, pretrained=True):
    test_dataset = FourChannelTiffDataset(TEST_PATH, transform=get_valid_transform(IMAGE_SIZE, pretrained))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return test_loader

# Example usage
if __name__ == "__main__":

    # Paths to your data
    DATASET_PATH = '/scratch/leuven/359/vsc35907/big_data_feature_extraction/patches_dirs/'

    # Temporary transform for computing stats (resize only)
    temp_transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    # Load dataset
    dataset = FourChannelTiffDataset(DATASET_PATH, transform=temp_transform)

    # Compute mean and std
    mean, std = compute_mean_std_for_channels(dataset)
    print(f"Computed Mean: {mean}")
    print(f"Computed Std: {std}")
    
    
#     TRAIN_PATH = '/scratch/leuven/359/vsc35907/feature_extraction_data/data/train'
#     VALID_PATH = '/scratch/leuven/359/vsc35907/feature_extraction_data/data/val'
#     TEST_PATH = '/scratch/leuven/359/vsc35907/feature_extraction_data/data/test'
    
#     train_loader, valid_loader = get_data_loader(TRAIN_PATH, VALID_PATH)
#     test_loader = get_test_loader(TEST_PATH)
    
#     print(f"Number of training images: {len(train_loader.dataset)}")
#     print(f"Number of validation images: {len(valid_loader.dataset)}")
#     print(f"Number of test images: {len(test_loader.dataset)}")
#     print(f"Classes: {train_loader.dataset.classes}")
    
#     for images, labels in train_loader:
#         print(f"Batch shape: {images.shape}")  # Should be (64, 4, 224, 224)
#         break