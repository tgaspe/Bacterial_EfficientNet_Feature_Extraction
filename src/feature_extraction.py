import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import os
import numpy as np
import pandas as pd
from data_loader import get_test_loader
import argparse
from tqdm import tqdm


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract features from TIFF images using a trained EfficientNet model.")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained model checkpoint (.pth file)")
    parser.add_argument('--dataset-path', type=str, required=True, help="Path to the dataset directory containing test images")
    parser.add_argument('--output-path', type=str, required=True, help="Path to save the output CSV file with extracted features")
    args = parser.parse_args()

    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize the model with the same architecture as during training
    model = EfficientNet.from_name('efficientnet-b0', num_classes=2)

    # Modify the first conv layer to accept 4 channels (same as train.py)
    model._conv_stem = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model = model.to(device)

    # Load the checkpoint
    print(f"Loading trained model weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)

    # Handle potential DataParallel wrapping
    if 'module' in list(checkpoint.keys())[0]:  # Check if keys have 'module.' prefix
        print("Detected DataParallel checkpoint, stripping 'module.' prefix...")
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    # Load the state dict
    model.load_state_dict(checkpoint)
    model.eval()  # Set to evaluation mode for inference
    print("Model weights loaded successfully!")

    # Load test data
    test_loader = get_test_loader(args.dataset_path, pretrained=True)
    print(f"Number of images in dataset: {len(test_loader.dataset)}")
    print(f"Classes: {test_loader.dataset.classes}")

    # Extract features
    features_list = []
    image_paths = [img[0] for img in test_loader.dataset.images]  # Track file paths

    with torch.no_grad():
        # Add progress bar for batch processing
        for images, _ in tqdm(test_loader, desc="Extracting features", unit="batch"):
            images = images.to(device)
            features = model.extract_features(images)  # Shape: (batch_size, 1280)
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, start_dim=1)
            features = features.detach()  # Detach features to avoid tracking gradients
            features_list.append(features.cpu().numpy())

    # Concatenate all features
    all_features = np.concatenate(features_list, axis=0)
    print(f"Extracted features shape: {all_features.shape}")  # Should be (num_images, 1280)

    # Prepare data for CSV
    feature_columns = [f'feature_{i}' for i in range(all_features.shape[1])]  # e.g., feature_0, feature_1, ..., feature_1279
    data = pd.DataFrame(all_features, columns=feature_columns)
    data['image_path'] = image_paths  # Add image paths as a column

    # Reorder columns to have image_path first
    data = data[['image_path'] + feature_columns]

    # Save to CSV
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)  # Ensure output directory exists
    data.to_csv(args.output_path, index=False)
    print(f"Features saved to '{args.output_path}'")

if __name__ == "__main__":
    main()