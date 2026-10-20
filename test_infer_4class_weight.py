import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
import sys
sys.path.append("../src")

device = torch.device("cpu")
num_classes = 4
class_names = ["0_normal","1_light","2_mid","3_heavy"]

# 搭建网络，注意这里带Dropout结构，要和训练代码完全一致
model = resnet18(weights=None)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(model.fc.in_features, num_classes)
)
model.to(device)

# 加载第四周优化后的权重
model_path = "/Users/liuyaru/Desktop/光伏项目-学生C/models/vision/best_resnet18_4class_weight.pth"
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
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
    # 访达复制一张test测试集图片的完整绝对路径粘贴在这里
    test_img = "/Users/liuyaru/Desktop/光伏项目-学生C/data/processed/02_方向B_图像识别与告警文本理解/ELPV_完整数据/split_dataset4class/test/1_light/cell0176.png"
    print(f"输入路径：{test_img}")
    print(f"文件是否存在：{os.path.exists(test_img)}")
    if os.path.exists(test_img):
        res = predict_one(test_img)
        print(f"✅【第四周优化模型】预测结果：{res}")
    else:
        print("❌路径错误，文件不存在")