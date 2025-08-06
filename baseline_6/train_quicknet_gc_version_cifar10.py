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
import  numpy as np
import random

class TinyValDataset(Dataset):
    def __init__(self, root, class_to_idx, transform=None):
        root = Path(root)
        ann_path = "tiny-imagenet-200/val/val_annotations.txt"
        # Six columns: image, class, x_min, y_min, x_max, y_max
        ann = pd.read_csv(
            ann_path,
            sep="\t",
            header=None,
            names=["img","cls","xmin","ymin","xmax","ymax"]
        )
        # Build a class→index map
        classes = sorted(set(ann.cls))
        if class_to_idx is None:
            # classes = sorted({entry[1] for entry in ann})
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
        # Store image file paths and integer labels
        self.imgs = [root/"images"/img_name for img_name in ann.img]
        self.targets = [self.class_to_idx[c] for c in ann.cls]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]

# Hyperparameters
batch_size = 64
learning_rate = 1e-3
epochs = 100

def main():

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    np.random.seed(42)
    
    
    cifar_classes = list(range(10))
    selected_classes = random.sample(cifar_classes, 2)
    print(f"Selected CIFAR-10 classes: {selected_classes}")

    def filter_cifar(dataset, selected_classes):
        idx_map = {c: i for i, c in enumerate(selected_classes)}  # map old → new label
        filtered_data = [(x, idx_map[y]) for x, y in dataset if y in selected_classes]
        images, labels = zip(*filtered_data)
        return list(images), list(labels)
        
    
    save_path = "/content/drive/MyDrive/quicknetFP32"
    os.makedirs(save_path, exist_ok=True)

    # Early stopping
    patience = 10 
    counter = 0
    best_val_acc = -inf


    # Data transforms (added more augmentations)
    transform_train = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])


    # Datasets & loaders for tinyimagenet
    train_full_dataset = CIFAR10(train = True, transform=transform_train, download= True)
    val_full_dataset = CIFAR10(train = False, transform=transform_val, download= True)
    
    train_imgs, train_lbs = filter_cifar(train_full_dataset, selected_classes)
    val_imgs, val_lbs = filter_cifar(val_full_dataset, selected_classes)

    
    class SimpleDataset(Dataset):
        def __init__(self, imgs, labels, transform=None):
            self.imgs = imgs
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            x = self.imgs[idx]
            y = self.labels[idx]
            if self.transform:
                x = self.transform(x)
            return x, y

    train_dataset = SimpleDataset(train_imgs, train_lbs, transform= None)
    val_dataset = SimpleDataset(val_imgs, val_lbs, transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model, criterion, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QuickNet(num_classes=200).to(device)
   
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
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        print(f"[Epoch {epoch}/{epochs}] - Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            counter = 0 
            model.eval()
            torch.save(model.state_dict(), f'/content/drive/MyDrive/quicknetFP32/best_tinyimagenet_quicknet_{epoch}.pth')
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
    
