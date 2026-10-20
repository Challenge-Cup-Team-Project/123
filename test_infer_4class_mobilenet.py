import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image

device = torch.device("cpu")
num_classes = 4
class_names = ["0_normal","1_light","2_mid","3_heavy"]

model = models.mobilenet_v2(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 4)
)
# 绝对路径，现在文件真实存在
model.load_state_dict(torch.load("/Users/liuyaru/Desktop/光伏项目-学生C/models/vision/best_mobilenetv2_4class.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def predict_one(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_tensor)
        pred_idx = torch.argmax(out,dim=1).item()
    return class_names[pred_idx]


if __name__ == "__main__":
    test_img = "/Users/liuyaru/Desktop/光伏项目-学生C/data/processed/02_方向B_图像识别与告警文本理解/ELPV_完整数据/split_dataset4class/test/3_heavy/cell0026.png"
    res = predict_one(test_img)
    print(f"✅4分类预测结果：{res}")