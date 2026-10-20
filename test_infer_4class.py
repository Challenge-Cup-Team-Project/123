import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
import sys
sys.path.append("../src")

device = torch.device("cpu")
num_classes = 4
class_names = ["0_normal","1_light","2_mid","3_heavy"]

# 构建4分类ResNet18
model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# 加载权重
model_path = "../src/vision/best_resnet18_4class.pth"
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
    # =========这里！粘贴复制得到的真实图片完整路径，不要手动敲！=========
    test_img = "/Users/liuyaru/Desktop/光伏项目-学生C/data/processed/02_方向B_图像识别与告警文本理解/ELPV_完整数据/split_dataset4class/test/3_heavy/cell0026.png"
    res = predict_one(test_img)
    print(f"✅4分类预测结果：{res}")