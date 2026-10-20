import os
from PIL import Image
from torch.utils.data import Dataset

# 自定义光伏红外数据集类，用来读取图片+对应故障标签
class PVDataset(Dataset):
    # 构造函数：打开数据集时自动执行，提前存好所有图片路径和标签
    def __init__(self, data_root, transform=None):
        # data_root：数据集路径，比如 data/processed/split_dataset4class/train
        self.data_root = data_root
        # transform：图像预处理、数据增强工具
        self.transform = transform
        # 两个列表，分别存图片完整路径、对应标签
        self.img_paths = []
        self.labels = []

        # 二分类映射规则：0_normal=正常(0)，剩下三类缺陷全部归为故障(1)
        class_map = {
            "0_normal": 0,
            "1_light": 1,
            "2_mid": 1,
            "3_heavy": 1
        }

        # 循环遍历train/val/test里的4个分类文件夹
        for cls_name in os.listdir(self.data_root):
            # 拼接当前分类文件夹完整路径
            cls_path = os.path.join(self.data_root, cls_name)
            # 如果不是文件夹，直接跳过（过滤掉txt、缓存文件）
            if not os.path.isdir(cls_path):
                continue
            # 根据文件夹名拿到对应0/1标签
            label = class_map[cls_name]
            # 遍历这个分类里所有图片
            for img_name in os.listdir(cls_path):
                # 拆分文件名和后缀
                suffix = os.path.splitext(img_name)[1]
                # 只保留png、jpg图片文件
                if suffix in [".png", ".jpg"]:
                    # 拼接单张图片完整路径
                    full_img_path = os.path.join(cls_path, img_name)
                    # 存入列表保存
                    self.img_paths.append(full_img_path)
                    self.labels.append(label)

    # 根据索引取出单张图片和标签，训练循环自动调用
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        # 打开图片，转RGB三通道格式
        img = Image.open(img_path).convert("RGB")
        # 如果有预处理工具，对图片做缩放、翻转、归一化
        if self.transform is not None:
            img = self.transform(img)
        # 返回图片、标签，给模型训练使用
        return img, label

    # 返回当前数据集一共有多少张图片，计算循环次数用
    def __len__(self):
        return len(self.img_paths)