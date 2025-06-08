import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
import wandb
from tqdm import tqdm
from data_loader import get_data_loader, get_test_loader
from utils import save_model, save_plots
from efficientnet_pytorch import EfficientNet
import os
import numpy as np
from sklearn.metrics import roc_auc_score

# Argument parser
parser = argparse.ArgumentParser(description="Train EfficientNet on 4-channel bacterial dataset")
parser.add_argument('-e', '--epochs', type=int, default=60, help='Number of epochs')
parser.add_argument('-lr', '--learning-rate', type=float, dest='learning_rate', default=3e-4, help='Learning rate')
parser.add_argument('-wd', '--weight-decay', type=float, dest='weight_decay', default=5e-5, help='Weight decay')
parser.add_argument('-p', '--patience', type=int, default=10, help='Early stopping patience')
parser.add_argument('-s', '--save-name', type=str, default='b0_bacterial', help='Name for saving the trained model')
parser.add_argument('-o', '--output-dir', type=str, default='./outputs', help='Output directory')
parser.add_argument('-d', '--data-dir', type=str, help='Base directory for train/val/test data')
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'RAdam', 'SGD', 'RMSprop'], help='Optimizer to use (AdamW, RAdam, SGD, or RMSprop)')
args = vars(parser.parse_args())

# Training function
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        loss.backward()
        optimizer.step()
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# Validation function
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
            # Collect probabilities for AUC
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probability for class 1
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    # Compute AUC
    auc_score = roc_auc_score(all_labels, all_probs)
    return epoch_loss, epoch_acc, auc_score

if __name__ == '__main__':
    # W&B setup
    name = args['save_name']
    wandb.init(
        project="efficient-net-224-bacterial",
        name=name,
        config={
            "learning_rate": args['learning_rate'],
            "architecture": 'efficientnet-b0',
            "dataset": "bacterial-4ch-03272025",
            "epochs": args['epochs'],
            "weight_decay": args['weight_decay'],
            "optimizer": args['optimizer'],
            "patience": args['patience'],
            "gpus": 2
        }
    )
    
    # Load datasets
    TRAIN_PATH = os.path.join(args['data_dir'], 'train')
    VALID_PATH = os.path.join(args['data_dir'], 'val')
    TEST_PATH = os.path.join(args['data_dir'], 'test')
    train_loader, valid_loader = get_data_loader(TRAIN_PATH, VALID_PATH)
    
    print(f"[INFO]: Number of training set images: {len(train_loader.dataset)}")
    print(f"[INFO]: Number of validation set images: {len(valid_loader.dataset)}")
    print(f"[INFO]: Class names: {train_loader.dataset.classes}\n")

    # Parameters
    lr = args['learning_rate']
    epochs = args['epochs']
    weight_decay = args['weight_decay']
    patience = args['patience']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}")
    print(f"Early stopping patience: {patience}")
    print(f"Save name: {args['save_name']}")
    print(f"Data directory: {args['data_dir']}")
    print(f"Optimizer: {args['optimizer']}\n")

    # Load and modify EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    model._conv_stem = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"[INFO]: Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    # Optimizer and scheduler
    if args['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif args['optimizer'] == 'RAdam':
        optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif args['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif args['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = nn.CrossEntropyLoss()

    # Tracking lists
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    valid_auc = []

    # Early stopping variables
    best_valid_loss = float('inf')
    best_valid_accuracy = -float('inf')
    patience_counter = 0
    name = f"{args['save_name']}.pth"
    best_model_path = os.path.join(args['output_dir'], name)

    # Training loop
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc, valid_epoch_auc = validate(model, valid_loader, criterion)
        
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step(valid_epoch_loss)
        
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        valid_auc.append(valid_epoch_auc)
        
        # Debug types
        print(f"Types - train_loss: {type(train_epoch_loss)}, valid_loss: {type(valid_epoch_loss)}, train_acc: {type(train_epoch_acc)}, valid_acc: {type(valid_acc)}, valid_auc: {type(valid_epoch_auc)}")
        
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}, validation AUC: {valid_epoch_auc:.3f}")

        wandb.log({
            "learning_rate": float(learning_rate),
            "train_acc": float(train_epoch_acc),
            "train_loss": float(train_epoch_loss),
            "valid_acc": float(valid_epoch_acc),
            "valid_loss": float(valid_epoch_loss),
            "valid_auc": float(valid_epoch_auc),
            "patience_counter": int(patience_counter)
        })

        # Early stopping logic
        if valid_epoch_loss < best_valid_loss:
            best_valid_loss = valid_epoch_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"[INFO]: Validation loss improved to {best_valid_loss:.3f}, saved best model to {best_model_path}")
        else:
            patience_counter += 1
            print(f"[INFO]: No improvement, patience counter: {patience_counter}/{patience}")
        
        print('-'*50)
        
        if patience_counter >= patience:
            print(f"[INFO]: Early stopping triggered after {epoch+1} epochs")
            break
        
        time.sleep(5)
    
    # After training
    print(f"[INFO]: Training complete. Best model saved at {best_model_path}")
    save_plots(train_acc, valid_acc, train_loss, valid_loss, True)
    
    # Test phase with sensitivity, specificity, and AUC
    test_loader = get_test_loader(TEST_PATH, pretrained=True)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"[INFO]: Loaded best model from {best_model_path} for testing")
    else:
        print("[INFO]: No improvement during training, using final model state for testing")
    
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
    wandb.log({
        "test_acc": test_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "test_auc": test_auc
    })
    
    wandb.finish()