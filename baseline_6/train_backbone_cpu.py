import os
import sys
import time
import logging
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

from utils import *            # your existing utilities
from birealnet import birealnet18

parser = argparse.ArgumentParser("birealnet")
parser.add_argument('--batch_size',   type=int,   default=128)
parser.add_argument('--epochs',       type=int,   default=256)
parser.add_argument('--learning_rate',type=float, default=0.001)
parser.add_argument('--momentum',     type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--save',         type=str,   default='./models')
parser.add_argument('--data',         type=str,   required=True,
                    help='path where MNIST will download/store')
parser.add_argument('--label_smooth', type=float, default=0.0)
parser.add_argument('-j', '--workers',default=0,    type=int)
args = parser.parse_args()

CLASSES = 10

# set up logging
os.makedirs('log', exist_ok=True)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d %I:%M:%S %p'
)
fh = logging.FileHandler(os.path.join('log/log.txt'))
fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logging.getLogger().addHandler(fh)

def main():
    # ** Force CPU only **
    device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    cudnn.benchmark = True
    cudnn.enabled = True
    # logging.info("args = %s", args)

    # 1) Build & patch the model for MNIST
    model = birealnet18()
    model.conv1   = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512 * model.layer4[0].expansion, CLASSES)
    model = model.to(device)
    # logging.info(model)

    # 2) Losses & optimizer
    criterion        = nn.CrossEntropyLoss().to(device)
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth).to(device)
    # split weights for weight_decay as you had before
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

    # optionally resume
    os.makedirs(args.save, exist_ok=True)
    ckpt = os.path.join(args.save, 'checkpoint.pth.tar')
    start_epoch, best_acc = 0, 0.0
    if os.path.exists(ckpt):
        logging.info("Loading checkpoint %s", ckpt)
        d = torch.load(ckpt, map_location=device)
        model.load_state_dict(d['state_dict'], strict=False)
        optimizer.load_state_dict(d['optimizer'])
        start_epoch = d['epoch']
        best_acc    = d['best_top1_acc']
        logging.info("Resumed from epoch %d (best_acc %.2f)", start_epoch, best_acc)

    # 3) Data loaders (MNIST)
    mnist_tf = transforms.Compose([
    transforms.Resize(32),        # 28×28 → 32×32
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
        ])
    
    val_tf = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
    train_ds = datasets.MNIST(
        root=args.data, train=True, download=True, transform=mnist_tf)
    test_ds  = datasets.MNIST(
        root=args.data, train=False,download=True, transform=val_tf)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
        test_ds,  batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # 4) Training loop
    print('starting training')
    for epoch in range(start_epoch, args.epochs):
        print(f"epoch number: {epoch}")
        train(epoch, train_loader, model, criterion_smooth, optimizer, scheduler, device)
        val_loss, val_acc = validate(epoch, val_loader, model, criterion, device)

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
    meters = [AverageMeter('Loss', ':.4e'),
              AverageMeter('Acc@1', ':6.2f')]
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        # record
        prec1, _ = accuracy(logits, targets, topk=(1,5))
        meters[0].update(loss.item(), images.size(0))
        meters[1].update(prec1[0].item(), images.size(0))

    logging.info(f"Epoch {epoch}: Train {meters[0]} | {meters[1]}")


def validate(epoch, loader, model, criterion, device):
    model.eval()
    losses, top1 = AverageMeter('Loss', ':.4e'), AverageMeter('Acc@1', ':6.2f')
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss   = criterion(logits, targets)
            prec1, _ = accuracy(logits, targets, topk=(1,5))
            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0].item(), images.size(0))

    logging.info(f"Epoch {epoch}: Val   {losses} | {top1}")
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()