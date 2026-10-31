# ================================================================
# WEIGHTED L1 PRUNING USING DENSENET121
# ================================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune

# ================================================================
# DEVICE SETUP
# ================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

pin_memory_flag = device.type == "cuda"

# ================================================================
# PARAMETERS
# ================================================================

dataset_path = "path/to/MRI_Datasets"  # Change to your dataset folder
batch_size = 16
num_classes = 4
epochs = 10
learning_rate = 1e-4
prune_amount = 0.3  # 30% pruning

# ================================================================
# DATA TRANSFORMS
# ================================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ================================================================
# DATASETS AND LOADERS
# ================================================================

train_dataset = datasets.ImageFolder(
    os.path.join(dataset_path, "Train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(dataset_path, "Val"),
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=pin_memory_flag
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory_flag
)

class_names = train_dataset.classes
print("Classes:", class_names)

# ================================================================
# MODEL: DENSENET121
# ================================================================

model = models.densenet121(weights="IMAGENET1K_V1")

in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ================================================================
# TRAIN FUNCTION
# ================================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct = 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    return (
        total_loss / len(loader.dataset),
        correct / len(loader.dataset)
    )

# ================================================================
# VALIDATION FUNCTION
# ================================================================

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

    return (
        total_loss / len(loader.dataset),
        correct / len(loader.dataset)
    )

# ================================================================
# TRAINING LOOP BEFORE PRUNING
# ================================================================

train_acc_list, val_acc_list = [], []
train_loss_list, val_loss_list = [], []

print("\nTraining before pruning...")

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer
    )

    val_loss, val_acc = evaluate(
        model, val_loader, criterion
    )

    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

# ================================================================
# APPLY WEIGHTED L1 PRUNING
# ================================================================

print("\nApplying Weighted L1 Pruning (30%)...")

for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        prune.l1_unstructured(
            module,
            name="weight",
            amount=prune_amount
        )

print("Pruning complete!")

# ================================================================
# EVALUATE AFTER PRUNING
# ================================================================

print("\nEvaluating model after pruning...")

val_loss, val_acc = evaluate(
    model, val_loader, criterion
)

print(f"Validation Accuracy after pruning: {val_acc:.4f}")
print(f"Validation Loss after pruning: {val_loss:.4f}")

# ================================================================
# CONFUSION MATRIX AND CLASSIFICATION REPORT
# ================================================================

y_true, y_pred = [], []

model.eval()

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)

        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()

        y_true.extend(labels.numpy())
        y_pred.extend(preds)

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# ================================================================
# PLOT ACCURACY AND LOSS
# ================================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list, label="Val Acc")
plt.title("Accuracy Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list, label="Val Loss")
plt.title("Loss Curve")
plt.legend()

plt.show()

# ================================================================
# PRINT PRUNING STATISTICS
# ================================================================

total_params = sum(p.numel() for p in model.parameters())

zero_params = sum(
    torch.sum(module.weight_mask == 0).item()
    for module in model.modules()
    if hasattr(module, "weight_mask")
)

print(f"\nTotal Parameters: {total_params}")
print(f"Pruned (zero) Weights: {zero_params}")
print(f"Remaining Effective Weights: {total_params - zero_params}")
