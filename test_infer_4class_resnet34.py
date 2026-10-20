import torch
import os
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet34

# -------------------------- 1.参数配置 --------------------------
MODEL_WEIGHT = "../models/vision/best_resnet34_4class_opt.pth"
CLASS_NAMES = ["0_normal", "1_light", "2_mid", "3_heavy"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cpu")

# -------------------------- 2.【重点！和训练脚本结构完全一致！】 --------------------------
model = resnet34(pretrained=False)
in_features = model.fc.in_features
# 和训练脚本保持一致：Dropout包一层Sequential！！
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(in_features, 4)
)

# 加载权重
model.load_state_dict(torch.load(MODEL_WEIGHT, map_location=device))
model.to(device)
model.eval()

# -------------------------- 3.推理函数 --------------------------
def predict_img(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        output = model(img)
        pred_idx = torch.argmax(output, dim=1).item()
    return CLASS_NAMES[pred_idx]


# -------------------------- 4.运行测试 --------------------------
if __name__ == "__main__":
    test_image_path = "/Users/liuyaru/Desktop/光伏项目-学生C/data/processed/02_方向B_图像识别与告警文本理解/ELPV_完整数据/split_dataset4class/test/3_heavy/cell0049.png"
    result = predict_img(test_image_path)
    print(f"图片：{test_image_path}")
    print(f"✅预测结果：{result}")