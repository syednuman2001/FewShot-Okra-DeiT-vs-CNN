
# CNN FEW-SHOT (RESNET18) 


import os, random, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torch.optim import AdamW
from google.colab import drive

drive.mount('/content/drive')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# PATHS

TRAIN_DIR = "/content/drive/MyDrive/Okra leaf disease dataset/Training"
VAL_DIR   = "/content/drive/MyDrive/Okra leaf disease dataset/Validation"
TEST_DIR  = "/content/drive/MyDrive/Okra leaf disease dataset/Testing"

# FEW-SHOT SETTING

SHOTS = 10  # change: 1, 5, 10


# TRANSFORMS (SAME AS DEIT)

IMG_SIZE = 224

train_tfms = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(25),
    T.ColorJitter(0.4,0.4,0.4),
    T.RandomGrayscale(p=0.2),
    T.GaussianBlur(3),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_tfms = T.Compose([
    T.Resize((IMG_SIZE,IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# LOAD DATA

train_full = ImageFolder(TRAIN_DIR, transform=train_tfms)
val_ds     = ImageFolder(VAL_DIR, transform=test_tfms)
test_ds    = ImageFolder(TEST_DIR, transform=test_tfms)

NUM_CLASSES = len(train_full.classes)
print("Classes:", train_full.classes)


# FEW-SHOT SAMPLING

def create_few_shot(dataset, shots):
    targets = np.array(dataset.targets)
    indices = []

    for cls in np.unique(targets):
        cls_idx = np.where(targets == cls)[0]
        np.random.shuffle(cls_idx)
        indices.extend(cls_idx[:shots])

    return Subset(dataset, indices)

train_ds = create_few_shot(train_full, SHOTS)
print("Few-shot samples:", len(train_ds))

# LOADERS

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False)


# MODEL (RESNET18)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)


# PARTIAL FREEZING 

for name, param in model.named_parameters():
    param.requires_grad = False

# unfreeze last conv block + classifier
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True


# LOSS & OPTIMIZER

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-5,
    weight_decay=0.01
)

# TRAIN FUNCTION

def train_one_epoch():
    model.train()
    total, correct, loss_sum = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return loss_sum/total, correct/total


# EVAL FUNCTION

def evaluate(loader):
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct/total


# TRAIN LOOP
EPOCHS = 15

for epoch in range(EPOCHS):
    loss, acc = train_one_epoch()
    val_acc = evaluate(val_loader)

    print(f"[CNN-FEWSHOT] Epoch {epoch+1} | Loss: {loss:.4f} | Train Acc: {acc:.4f} | Val Acc: {val_acc:.4f}")


# FINAL TEST
test_acc = evaluate(test_loader)
print("\n CNN FEW-SHOT FINAL TEST ACC:", test_acc)
