import torch
from torchvision import transforms, datasets
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
    np.random.seed(420)
    
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
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),   # brightness, contrast, sat, hue
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(64),                   # ensure 64×64
        transforms.CenterCrop(64),    # no data aug
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4802, 0.4481, 0.3975],       # TinyImageNet stats
            std=[0.2302, 0.2265, 0.2262]
        ),
    ])


    # Datasets & loaders for tinyimagenet
    train_dataset = datasets.ImageFolder(root="tiny-imagenet-200/train", transform=transform_train)
    val_dataset = TinyValDataset(root="tiny-imagenet-200/val", class_to_idx=train_dataset.class_to_idx, transform=transform_val,)
  
    print(f"First image: {val_dataset.imgs[0]}")
    print(f"Class index: {val_dataset.targets[0]}")
    # test_dataset = TinyValDataset(root="../data/tiny-imagenet-200/test", class_to_idx=None, transform=transform_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model, criterion, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QuickNet(num_classes=200).to(device)
   
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


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
    
