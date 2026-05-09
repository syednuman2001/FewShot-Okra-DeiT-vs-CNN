# CNN (RESNET18) FEW-SHOT BASELINE

!pip install -q torchvision

import os, random, numpy as np
from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# PATHS 

TRAIN_DIR = "/content/drive/MyDrive/Okra leaf disease dataset/Training"
VAL_DIR   = "/content/drive/MyDrive/Okra leaf disease dataset/Validation"
TEST_DIR  = "/content/drive/MyDrive/Okra leaf disease dataset/Testing"


# FEW-SHOT 

SHOTS = 5   #Try: 1, 5, 10


# TRANSFORMS

IMAGE_SIZE = 224

train_tfms = T.Compose([
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
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
    T.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# LOAD DATA

train_full = ImageFolder(TRAIN_DIR, transform=train_tfms)
val_ds     = ImageFolder(VAL_DIR,   transform=test_tfms)
test_ds    = ImageFolder(TEST_DIR,  transform=test_tfms)

NUM_CLASSES = len(train_full.classes)

print("Classes:", train_full.classes)
print("Total Training Images:", len(train_full))


# CREATE FEW-SHOT DATASET

def create_few_shot(dataset, shots=5):
    targets = np.array(dataset.targets)
    indices = []

    for cls in np.unique(targets):
        cls_idx = np.where(targets == cls)[0]
        np.random.shuffle(cls_idx)
        indices.extend(cls_idx[:shots])

    return Subset(dataset, indices)

train_ds = create_few_shot(train_full, SHOTS)

print("Few-shot samples:", len(train_ds))


# DATALOADERS

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False)


# LOAD RESNET18 MODEL

model = models.resnet18(pretrained=True)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)


# FREEZE BACKBONE 

for name, param in model.named_parameters():
    if "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False


# OPTIMIZER & LOSS

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=1e-4, weight_decay=0.01)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


# TRAIN FUNCTION

def train_one_epoch():
    model.train()
    total, correct, loss_total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_total += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return loss_total/total, correct/total


# EVALUATION

def evaluate(loader):
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct/total


# TRAINING LOOP

EPOCHS = 15

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch()
    val_acc = evaluate(val_loader)

    print(f"[CNN] Epoch {epoch+1} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# FINAL TEST

test_acc = evaluate(test_loader)
print("\n CNN FEW-SHOT TEST ACCURACY:", test_acc)
