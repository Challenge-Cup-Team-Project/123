import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

base_dir = "/Users/liuyaru/Desktop/光伏项目-学生C/data/processed/02_方向B_图像识别与告警文本理解/ELPV_完整数据"

csv_path = os.path.join(base_dir, "labels.csv")
img_dir = os.path.join(base_dir, "images")
out_root = os.path.join(base_dir, "split_dataset4class")

classes = ["0_normal", "1_light", "2_mid", "3_heavy"]

for phase in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(out_root, phase, cls), exist_ok=True)

# sep=r'\s+' 使用任意空格做分隔，不要用逗号
df = pd.read_csv(csv_path, sep=r'\s+', skip_blank_lines=True, header=None)

print("===读取后前5行===")
print(df.head())

def score_to_label(score):
    if abs(score - 0.0) < 0.01:
        return "0_normal"
    elif abs(score - 0.333333) < 0.01:
        return "1_light"
    elif abs(score - 0.666666) < 0.01:
        return "2_mid"
    elif abs(score - 1.0) < 0.01:
        return "3_heavy"
    else:
        return None

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=42)

def copy_data(sub_df, phase_name):
    cnt = 0
    skip_count = 0
    for _, row in sub_df.iterrows():
        if len(row) < 2:
            skip_count +=1
            continue
        img_full = row.iloc[0]
        score_val = row.iloc[1]
        try:
            score = float(score_val)
        except (ValueError, TypeError):
            skip_count +=1
            continue

        label_cls = score_to_label(score)
        if label_cls is None:
            skip_count +=1
            continue

        img_name = os.path.basename(img_full)
        src = os.path.join(img_dir, img_name)
        dst = os.path.join(out_root, phase_name, label_cls, img_name)
        if os.path.exists(src):
            shutil.copy(src, dst)
            cnt += 1
    print(f"{phase_name}:复制完成 {cnt}张，跳过异常行 {skip_count}")

copy_data(train_df,"train")
copy_data(val_df,"val")
copy_data(test_df,"test")

print("\n✅全部4分类数据集划分完毕！输出文件夹：split_dataset4class")
print("输出路径：", out_root)