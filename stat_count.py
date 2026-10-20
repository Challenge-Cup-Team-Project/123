import os

root_dir = "/Users/liuyaru/Desktop/光伏项目-学生C/data/processed/02_方向B_图像识别与告警文本理解/ELPV_完整数据/split_dataset4class"
splits = ["train", "val", "test"]
classes = ["0_normal", "1_light", "2_mid", "3_heavy"]

total_all = 0

for split in splits:
    print(f"\n===== {split} =====")
    sum_split = 0
    for cls in classes:
        folder = os.path.join(root_dir, split, cls)
        if not os.path.exists(folder):
            num = 0
        else:
            files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
            num = len(files)
        print(f"{cls}: {num} 张")
        sum_split += num
    print(f"{split} 合计：{sum_split} 张")
    total_all += sum_split

print(f"\n>>>数据集全部总图片：{total_all} 张")