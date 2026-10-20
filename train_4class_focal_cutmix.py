import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

data_root = "/Users/liuyaru/Desktop/光伏项目-学生C/data/processed/02_方向B_图像识别与告警文本理解/ELPV_完整数据/split_dataset4class"
save_dir = "/Users/liuyaru/Desktop/光伏项目-学生C/models/vision"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cpu")

# ============FocalLoss损失函数============
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.view(-1,1))
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)

        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.mean()

# ============CutMix函数============
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    h, w = x.size()[2], x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    y_a = y
    y_b = y[index]
    return x, y_a, y_b, lam

# ✅修复：FocalLoss只接受输出和标签，不要传lam
def cutmix_criterion(criterion, out, y_a, y_b, lam):
    return lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)

# ============数据增强============
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(10),
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

model = models.resnet18(weights="IMAGENET1K_V1")
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 4)
)
model = model.to(device)

criterion = FocalLoss(gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)

best_acc = 0.0
epoch_num = 15

for epoch in range(epoch_num):
    model.train()
    total_train = 0
    correct_train = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        imgs, y_a, y_b, lam = cutmix_data(imgs, labels)
        out = model(imgs)
        loss = cutmix_criterion(criterion, out, y_a, y_b, lam)
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
        torch.save(model.state_dict(), os.path.join(save_dir, "best_4class_focal_cutmix.pth"))
        print(f"✅保存选做(Focal+CutMix)最优模型，当前最优val acc:{best_acc:.4f}")

classes_name = ["0_normal","1_light","2_mid","3_heavy"]
cm = confusion_matrix(all_true, all_pred)
plt.figure()
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks(range(len(classes_name)), classes_name, rotation=45)
plt.yticks(range(len(classes_name)), classes_name)
plt.xlabel("Predict")
plt.ylabel("True")
plt.title("Val Confusion Matrix FocalLoss+CutMix 4class")
plt.savefig("confusion_matrix_4class_focal_cutmix.png")
plt.show()

print(f"\n🎉选做实验 FocalLoss+CutMix训练结束｜最优验证准确率：{best_acc:.4f}")
print("策略：ResNet18 + FocalLoss(gamma=2) + CutMix + 基础增强 + Dropout0.4 + 权重衰减")