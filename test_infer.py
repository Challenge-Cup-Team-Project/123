import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

# 使用src里面的模型，交付之后别人直接可以跑
model_path = "/Users/liuyaru/Desktop/光伏项目-学生C/src/vision/best_resnet18.pth"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

model = resnet18(num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

class_names = ["0_normal【正常光伏板】", "1_defect【存在缺陷】"]

img_path ="/Users/liuyaru/Desktop/光伏项目-学生C/data/processed/02_方向B_图像识别与告警文本理解/ELPV_完整数据/split_dataset4class/test/0_normal/cell1230.png"

with torch.no_grad():
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    output = model(img_tensor)
    pred_idx = torch.argmax(output, dim=1).item()

print(f"✅图片预测输出结果： {class_names[pred_idx]}")