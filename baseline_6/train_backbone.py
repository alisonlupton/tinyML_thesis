import os
import sys
import time
import logging
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

from utils import *
from birealnet import birealnet18

parser = argparse.ArgumentParser("birealnet")
parser.add_argument('--batch_size',   type=int,   default=512)
parser.add_argument('--epochs',       type=int,   default=256)
parser.add_argument('--learning_rate',type=float, default=0.001)
parser.add_argument('--momentum',     type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--save',         type=str,   default='./models')
parser.add_argument('--data',         type=str,   required=True,
                    help='root of tiny-imagenet-200 (contains train/ val/)')
parser.add_argument('--label_smooth', type=float, default=0.1)
parser.add_argument('-j', '--workers',default=40,   type=int)
args = parser.parse_args()

CLASSES = 200  # TinyImageNet has 200 classes

# logging
os.makedirs('log', exist_ok=True)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d %I:%M:%S %p'
)
fh = logging.FileHandler('log/log.txt')
fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logging.getLogger().addHandler(fh)

def main():
    # 1) Device + cudnn
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required.")
        sys.exit(1)
    device = torch.device("cuda")
    cudnn.benchmark = True
    cudnn.enabled   = True
    logging.info("Using device: %s", device)
    logging.info("args = %s", args)

    # 2) Model
    model = birealnet18(num_classes=CLASSES)
    model = nn.DataParallel(model).to(device)
    logging.info(model)

    # 3) Losses + optimizer + scheduler
    criterion        = nn.CrossEntropyLoss().to(device)
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth).to(device)

    # separate weight_decay parameters
    all_params = list(model.named_parameters())
    weight_p, other_p = [], []
    for name, p in all_params:
        if p.ndimension() == 4 or 'fc.' in name:
            weight_p.append(p)
        else:
            other_p.append(p)

    optimizer = torch.optim.Adam(
        [{'params': other_p},
         {'params': weight_p, 'weight_decay': args.weight_decay}],
        lr=args.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (1.0 - step / args.epochs),
        last_epoch=-1
    )

    start_epoch, best_acc = 0, 0.0
    ckpt = os.path.join(args.save, 'checkpoint.pth.tar')
    if os.path.exists(ckpt):
        logging.info("Loading checkpoint %s", ckpt)
        d = torch.load(ckpt, map_location=device)
        model.load_state_dict(d['state_dict'], strict=False)
        optimizer.load_state_dict(d['optimizer'])
        start_epoch = d['epoch']
        best_acc    = d['best_top1_acc']
        logging.info("Resumed from epoch %d (best_acc %.2f)", start_epoch, best_acc)

    # 4) Data loaders
    traindir = os.path.join(args.data, 'train')
    valdir   = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        Lighting(0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(traindir, transform=train_transforms)
    val_ds   = datasets.ImageFolder(valdir,   transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # 5) Training loop
    for epoch in range(start_epoch, args.epochs):
        train(epoch, train_loader, model, criterion_smooth, optimizer, scheduler, device)
        val_loss, val_acc = validate(epoch, val_loader, model, criterion, args, device)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save)

    logging.info("Done. Best validation acc: %.2f", best_acc)


def train(epoch, loader, model, criterion, optimizer, scheduler, device):
    model.train()
    scheduler.step()
    meters = [
        AverageMeter('Loss', ':.4e'),
        AverageMeter('Acc@1', ':6.2f'),
        AverageMeter('Acc@5', ':6.2f'),
    ]

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(logits, targets, topk=(1,5))
        meters[0].update(loss.item(), images.size(0))
        meters[1].update(prec1[0].item(), images.size(0))
        meters[2].update(prec5[0].item(), images.size(0))

    logging.info(f"Epoch {epoch}: " +
                 f"Train Loss {meters[0].avg:.4f} | " +
                 f"Acc@1 {meters[1].avg:.2f} | " +
                 f"Acc@5 {meters[2].avg:.2f}")


def validate(epoch, loader, model, criterion, args, device):
    model.eval()
    meters = [
        AverageMeter('Loss', ':.4e'),
        AverageMeter('Acc@1', ':6.2f'),
        AverageMeter('Acc@5', ':6.2f'),
    ]
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss   = criterion(logits, targets)

            prec1, prec5 = accuracy(logits, targets, topk=(1,5))
            meters[0].update(loss.item(), images.size(0))
            meters[1].update(prec1[0].item(), images.size(0))
            meters[2].update(prec5[0].item(), images.size(0))

    logging.info(f"Epoch {epoch}: " +
                 f"Val   Loss {meters[0].avg:.4f} | " +
                 f"Acc@1 {meters[1].avg:.2f} | " +
                 f"Acc@5 {meters[2].avg:.2f}")
    return meters[0].avg, meters[1].avg


if __name__ == '__main__':
    main()