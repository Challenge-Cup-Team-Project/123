import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

data_root = "/Users/liuyaru/Desktop/光伏项目-学生C/data/processed/02_方向B_图像识别与告警文本理解/ELPV_完整数据/split_dataset4class"
save_dir = "/Users/liuyaru/Desktop/光伏项目-学生C/models/vision"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cpu")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_transform)
val_set = datasets.ImageFolder(os.path.join(data_root, "val"), transform=val_transform)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_acc = 0.0
epoch_num = 10

for epoch in range(epoch_num):
    model.train()
    total_train = 0
    correct_train = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        pred = torch.argmax(out, dim=1)
        correct_train += (pred == labels).sum().item()
        total_train += imgs.size(0)
    train_acc = correct_train / total_train

    model.eval()
    total_val = 0
    correct_val = 0
    all_pred = []
    all_true = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            pred = torch.argmax(out, dim=1)
            correct_val += (pred == labels).sum().item()
            total_val += imgs.size(0)
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
    val_acc = correct_val / total_val

    print(f"Epoch:{epoch+1}/{epoch_num} | Train Acc:{train_acc:.4f} | Val Acc:{val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(save_dir, "best_resnet18_4class.pth"))
        print(f"✅保存最优4分类模型，当前最优val acc:{best_acc:.4f}")

classes_name = ["0_normal","1_light","2_mid","3_heavy"]
cm = confusion_matrix(all_true, all_pred)
plt.figure()
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks(range(4), classes_name, rotation=45)
plt.yticks(range(4), classes_name)
plt.xlabel("Predict")
plt.ylabel("True")
plt.title("Val Confusion Matrix 4‑class PV Defect")
plt.savefig("confusion_matrix_4class.png")
plt.show()

print(f"\n4分类训练全部结束，最佳验证集准确率：{best_acc:.4f}")