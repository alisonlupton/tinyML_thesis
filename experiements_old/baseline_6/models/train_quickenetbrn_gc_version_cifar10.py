import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
# from baseline_6.models.quicknet import QuickNet
import torch.nn as nn
from pathlib import Path
import pandas as pd
from PIL import Image
from math import inf
from tqdm import tqdm
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import  numpy as np
import random


# Hyperparameters
batch_size = 64
learning_rate = 1e-3
epochs = 100

def main():

    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    random.seed(43)
    np.random.seed(43)
    
    
    cifar_classes = list(range(10))
    #selected_classes = random.sample(cifar_classes, 2)
    pretrained_classes = [0,5]
    print(f"Selected CIFAR-10 classes: {pretrained_classes}")

    def filter_cifar(dataset, pretrained_classes):
        idx_map = {c: i for i, c in enumerate(pretrained_classes)}  # map old → new label
        filtered_data = [(x, idx_map[y]) for x, y in dataset if y in pretrained_classes]
        images, labels = zip(*filtered_data)
        return list(images), list(labels)
    
    print('hello?')

    
    save_path = "/content/drive/MyDrive/quicknetFP32"
    os.makedirs(save_path, exist_ok=True)

    print('hello!')

    # Early stopping
    patience = 10 
    counter = 0
    best_val_acc = -inf


    # Data transforms (added more augmentations)
    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # Datasets & loaders for cifar10
    class FilteredCIFAR(Dataset):
      def __init__(self, dataset, pretrained_classes):
          idx_map = {c: i for i, c in enumerate(pretrained_classes)}
          self.data = [(x, idx_map[y]) for x, y in dataset if y in pretrained_classes]
      def __len__(self):
          return len(self.data)
      def __getitem__(self, idx):
          return self.data[idx]

    train_full = CIFAR10(root="/content", train=True, transform=transform_train, download=True)
    val_full = CIFAR10(root="/content", train=False, transform=transform_val, download=True)

    train_dataset = FilteredCIFAR(train_full, pretrained_classes)
    val_dataset = FilteredCIFAR(val_full, pretrained_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model, criterion, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BinarizedQuickNetBRN(
    num_classes=2,
    first_layer_fp32=True,  # or False if you want a fully binary model
    dropout=0.5).to(device)
    print(model)


   
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)



    # TO TRY: OneCycleLR with warmup
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1,
    #   steps_per_epoch=len(train_loader), epochs=epochs)



    # Training loop
    print("Starting Training!")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        for images, labels in tqdm(train_loader, desc= f"On epoch {epoch}/{epochs}", unit='batch'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = correct / len(train_loader.dataset)
        print(f"[Epoch {epoch}/{epochs}] - Training Loss: {avg_loss:.4f} | Training Accuracy: {accuracy:.4f}")
        
        
        # Validation
        model.eval()
        val_correct = 0
        val_loss = 0

        class_0_correct = 0
        class_0_total = 0
        class_1_correct = 0
        class_1_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                # Per-class tracking
                for pred, label in zip(preds, labels):
                    if label == 0:
                        class_0_total += 1
                        if pred == 0:
                            class_0_correct += 1
                    elif label == 1:
                        class_1_total += 1
                        if pred == 1:
                            class_1_correct += 1

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        print(f"[Epoch {epoch}/{epochs}] - Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")


        # Print per-class accuracy
        print(f"Class 0 accuracy: {class_0_correct/class_0_total:.4f}")
        print(f"Class 1 accuracy: {class_1_correct/class_1_total:.4f}")
                
                
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            counter = 0 
            model.eval()
            # Save with meaningful name and save selected classes info
            torch.save({
                'model_state_dict': model.state_dict(),
                'pretrained_classes': pretrained_classes,
                'val_accuracy': val_acc,
                'epoch': epoch
            }, f'{save_path}/quicknet_bnn_brn_cifar10_2class_best.pth')
            print(f"New best model saved (Val acc: {best_val_acc:.4f})")
        else:
            counter +=1 
            print(f"No model improvement for {counter} epochs!")
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        
        print(f"Validation Accuracy: {val_acc:.4f}\n")
        scheduler.step()

    
if __name__ == '__main__':
    main()
    
