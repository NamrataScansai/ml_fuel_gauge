import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --- 1. SETTINGS ---
DATA_DIR = r'D:\FuelGauge\Digital_Gauge Model\dataset_split_1' # Use your new split folder
MODEL_NAME = 'tf_efficientnetv2_s' 
BATCH_SIZE = 32 # Balanced data allows for slightly larger batches
EPOCHS =100  
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. DATA PREPARATION ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3), 
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), 
    transforms.ColorJitter(brightness=0.5, contrast=0.5), 
    transforms.RandomAffine(degrees=3, translate=(0.05, 0.05)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load balanced data
train_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
val_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class_names = train_set.classes
num_classes = len(class_names)
print(f"Solving for {num_classes} unique fuel patterns.")

# --- 3. BUILD CLEAN MODEL ---

model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# --- 4. TRAINING LOOP ---
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    val_acc = 100 * correct / total
    scheduler.step()
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

    if val_acc >= best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'fuel_expert_v2.pth')
        print("Best Model Saved!")

print(f"Best Accuracy: {best_acc}%")