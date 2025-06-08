import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from data_loader import get_data_loader, get_test_loader
from utils import save_model, save_plots
from efficientnet_pytorch import EfficientNet
import os
import numpy as np
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    
    # Load datasets
    TEST_PATH = os.path.join('/scratch/leuven/359/vsc35907/big_data_feature_extraction/dataset', 'test')
    best_model_path = '/vsc-hard-mounts/leuven-data/359/vsc35907/EfficientNet_feature_extraction/models/model_lr_1e4_sgd_wd_5e3.pth'

    # Parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Computation device: {device}")

    # Load and modify EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    model._conv_stem = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"[INFO]: Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model.to(device)
    
    # Load the model checkpoint
    print(f"Loading trained model weights from {best_model_path}...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)

    # Handle potential DataParallel wrapping
    if 'module' in list(checkpoint.keys())[0]:  # Check if keys have 'module.' prefix
        print("Detected DataParallel checkpoint, stripping 'module.' prefix...")
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    # Load the state dict
    model.load_state_dict(checkpoint)

    # Load Test set 
    test_loader = get_test_loader(TEST_PATH, pretrained=True)
    
    TP, FP, TN, FN = 0, 0, 0, 0
    total_positive, total_negative = 0, 0
    all_labels = []
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            TP += ((preds == 1) & (labels == 1)).sum().item()
            FP += ((preds == 1) & (labels == 0)).sum().item()
            TN += ((preds == 0) & (labels == 0)).sum().item()
            FN += ((preds == 0) & (labels == 1)).sum().item()
            total_positive += (labels == 1).sum().item()
            total_negative += (labels == 0).sum().item()
            # Collect probabilities for AUC
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    
    total = total_positive + total_negative
    test_acc = (TP + TN) / total * 100 if total > 0 else 0
    sensitivity = TP / total_positive * 100 if total_positive > 0 else 0
    specificity = TN / total_negative * 100 if total_negative > 0 else 0
    test_auc = roc_auc_score(all_labels, all_probs)
    
    print(f"Test accuracy: {test_acc:.3f}%")
    print(f"Sensitivity: {sensitivity:.3f}%")
    print(f"Specificity: {specificity:.3f}%")
    print(f"Test AUC: {test_auc:.3f}")
