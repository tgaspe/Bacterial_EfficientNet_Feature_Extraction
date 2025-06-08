import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import tifffile
import os
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"./outputs/accuracy_pretrained_{pretrained}.png")
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"./outputs/loss_pretrained_{pretrained}.png")

def save_model(epochs, model, optimizer, criterion, pretrained, name):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"./outputs/{name}_pretrained_{pretrained}.pth")
    
def compute_mean_std_for_channels(dataset, batch_size=64, num_workers=min(os.cpu_count(), 12)):
    """
    Compute the mean and standard deviation for each channel of a 4-channel dataset.
    
    Args:
        dataset: Instance of FourChannelTiffDataset (or compatible dataset)
        batch_size: Batch size for DataLoader (default: 64)
        num_workers: Number of workers for DataLoader (default: min(cpu_count, 12))
    
    Returns:
        mean: Numpy array of shape (4,) containing mean for each channel
        std: Numpy array of shape (4,) containing standard deviation for each channel
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    mean = torch.zeros(4)
    std = torch.zeros(4)
    total_images = 0

    # Add tqdm progress bar
    for images, _ in tqdm(loader, total=len(loader), unit='batch', desc="Computing mean/std"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images
    return mean.numpy(), std.numpy()


class FourChannelTiffDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing 4-channel TIFF images from a directory.

    This dataset is designed for binary classification tasks with 4-channel TIFF images
    of size 76x76, supporting 'mutants' and 'wild_type' classes. It loads images from all
    subfolders within these class directories, normalizes pixel values, converts them to
    PyTorch tensors, and applies optional transformations. Invalid TIFF files are skipped
    during initialization using multiprocessing for faster validation.

    Args:
        root_dir (str): Path to the root directory (e.g., 'train', 'val', or 'test') containing
                        subdirectories 'mutants' and 'wild_type', each with further subfolders of TIFF images.
        transform (callable, optional): Optional transform to be applied to the images.

    Attributes:
        classes (list): List of class names ['mutants', 'wild_type'].
        class_to_idx (dict): Mapping of class names to indices (e.g., {'mutants': 0, 'wild_type': 1}).
        images (list): List of tuples containing (image_path, class_index) for all valid TIFF images.
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['mutants', 'wild_type']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        
        # Collect all potential image paths and their class indices with progress bar
        image_paths = []
        total_files = sum(len(files) for _, _, files in os.walk(self.root_dir) for f in files if f.endswith(('.tiff', '.tif')))
        print(f"[INFO]: Collecting TIFF files")
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for dirpath, dirnames, filenames in tqdm(os.walk(cls_dir), desc=f"Scanning {cls}", unit='dir', total=sum(1 for _ in os.walk(cls_dir))):
                for fname in filenames:
                    if fname.endswith('.tiff') or fname.endswith('.tif'):
                        img_path = os.path.join(dirpath, fname)
                        image_paths.append((img_path, self.class_to_idx[cls]))
        
        if not image_paths:
            raise ValueError(f"No TIFF images found in {root_dir}")
        
        # Validate files in parallel using multiprocessing
        num_workers = min(cpu_count(), 16)  # Cap at 12 to match DataLoader
        print(f"[INFO]: Validating {len(image_paths)} TIFF files using {num_workers} workers")
        with Pool(processes=num_workers) as pool:
            results = pool.map(self._is_valid_tiff_wrapper, image_paths)
        
        # Collect valid images with progress bar
        for (img_path, class_idx), is_valid in tqdm(zip(image_paths, results), total=len(image_paths), desc="Processing validation results"):
            if is_valid:
                self.images.append((img_path, class_idx))
            else:
                print(f"[WARNING]: Skipped invalid file: {img_path}")
        
        if not self.images:
            raise ValueError(f"No valid TIFF images found in {root_dir}")
        print(f"[INFO]: Found {len(self.images)} valid TIFF images in {root_dir}")
        
    
    def _is_valid_tiff_wrapper(self, args):
        """
        Wrapper for _is_valid_tiff to use with multiprocessing.

        Args:
            args: Tuple of (img_path, class_idx).

        Returns:
            bool: True if the file is valid, False otherwise.
        """
        img_path, _ = args
        return self._is_valid_tiff(img_path)
    
    def _is_valid_tiff(self, img_path):
        """
        Check if a file is a valid single-image TIFF with shape (76, 76, 4).

        Args:
            img_path (str): Path to the file.

        Returns:
            bool: True if the file is a valid TIFF image with shape (76, 76, 4), False otherwise.
        """
        try:
            # Check if file is non-empty
            if os.path.getsize(img_path) == 0:
                return False
            # Attempt to read the TIFF image directly
            img = tifffile.imread(img_path)
            # Ensure shape is (76, 76, 4)
            if img.shape == (76, 76, 4):
                return True
            print(f"[DEBUG]: Invalid shape for {img_path}: {img.shape}")
            return False
        except (tifffile.TiffFileError, OSError, ValueError, IndexError, KeyError) as e:
            print(f"[DEBUG]: Error validating {img_path}: {str(e)}")
            return False
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Retrieve an image and its label by index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: (image, label) where image is a PyTorch tensor and label is an integer.
        """
        img_path, label = self.images[idx]
        # Load the 4-channel TIFF image
        img = tifffile.imread(img_path)  # Shape: (76, 76, 4)
        # Normalize to [0, 1] assuming 16-bit images
        img = img.astype(np.float32) / 65535.0
        # Convert to PyTorch tensor with shape (C, H, W)
        img = torch.from_numpy(img.transpose(2, 0, 1))  # Shape: (4, 76, 76)
        
        if self.transform:
            img = self.transform(img)
        return img, label