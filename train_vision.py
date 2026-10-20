import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# 导入自己写的数据集类
from src.vision.dataset import PVDataset

# 自动判断设备，Mac没有NVIDIA GPU，会自动使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("训练设备：", device)

# ----------------图像预处理与数据增强----------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ----------------数据集路径（核对你的文件夹！）----------------
train_root = "/Users/liuyaru/Desktop/光伏项目-学生C/data/processed/02_方向B_图像识别与告警文本理解/ELPV_完整数据/split_dataset4class/train"
val_root = "/Users/liuyaru/Desktop/光伏项目-学生C/data/processed/02_方向B_图像识别与告警文本理解/ELPV_完整数据/split_dataset4class/val"

train_dataset = PVDataset(train_root, transform=train_transform)
val_dataset = PVDataset(val_root, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# ----------------搭建ResNet18二分类模型----------------
model = models.resnet18(pretrained=True)
in_feature = model.fc.in_features
model.fc = nn.Linear(in_feature, 2)
model = model.to(device)

# 损失函数、优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

save_model_dir = "models/vision"
os.makedirs(save_model_dir, exist_ok=True)

epoch_num = 10
best_acc = 0.0

# ----------------训练循环----------------
for epoch in range(epoch_num):
    model.train()
    train_loss = 0.0
    train_pred_all = []
    train_label_all = []

    for img, label in train_loader:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = torch.argmax(out, dim=1)
        train_pred_all.extend(pred.cpu().numpy())
        train_label_all.extend(label.cpu().numpy())

    train_acc = accuracy_score(train_label_all, train_pred_all)
    avg_train_loss = train_loss / len(train_loader)

    # 验证集
    model.eval()
    val_loss = 0.0
    val_pred_all = []
    val_label_all = []
    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(device), label.to(device)
            out = model(img)
            loss = criterion(out, label)
            val_loss += loss.item()
            pred = torch.argmax(out, dim=1)
            val_pred_all.extend(pred.cpu().numpy())
            val_label_all.extend(label.cpu().numpy())

    val_acc = accuracy_score(val_label_all, val_pred_all)
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{epoch_num}")
    print(f"Train Loss:{avg_train_loss:.4f} Acc:{train_acc:.4f}")
    print(f"Val Loss:{avg_val_loss:.4f} Acc:{val_acc:.4f}\n")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(save_model_dir, "best_resnet18.pth"))
        print(f"✅保存最优模型，验证集最高准确率：{best_acc:.4f}")

# 绘制混淆矩阵
cm = confusion_matrix(val_label_all, val_pred_all)
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Val Confusion Matrix (Binary PV Defect)")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Normal(0)", "Defect(1)"])
plt.yticks(tick_marks, ["Normal(0)", "Defect(1)"])
plt.xlabel("Predict")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.png")
plt.show()
print(f"训练全部结束，最佳验证准确率：{best_acc:.4f}")