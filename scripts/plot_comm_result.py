import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===================== 路径配置 =====================
# CSV文件路径
csv_trace_path = r"D:\vpp_elastic_control\src\communication\communication_trace.csv"
csv_msg_path = r"D:\vpp_elastic_control\src\communication\message_log.csv"

# 图片输出目录
fig_save_dir = r"D:\vpp_elastic_control\output\figures"
os.makedirs(fig_save_dir, exist_ok=True)

# ===================== 加载数据 =====================
df_trace = pd.read_csv(csv_trace_path)
df_msg = pd.read_csv(csv_msg_path)

# 信道状态编码映射，方便绘图
state_map = {"good": 0, "medium": 1, "bad": 2}
state_label = ["good", "medium", "bad"]
state_color = ["#2ca02c", "#ff7f0e", "#d62728"]

df_trace["state_code"] = df_trace["channel_state"].map(state_map)

# ===================== 图1：信道状态时序曲线 =====================
plt.figure(figsize=(12, 5), dpi=150)
plt.plot(df_trace["time_step"], df_trace["state_code"], color="#1f77b4", linewidth=1.2)
plt.yticks(ticks=[0, 1, 2], labels=state_label)
plt.xlabel("Time Step")
plt.ylabel("Channel State")
plt.title("Channel State Evolution Over Time")
plt.grid(alpha=0.3)
plt.tight_layout()
save_path1 = os.path.join(fig_save_dir, "comm_channel_timeline.png")
plt.savefig(save_path1, bbox_inches="tight")
plt.close()
print(f"信道时序图已保存：{save_path1}")

# ===================== 图2：消息统计（丢包柱状图 + 队列时延） =====================
# 统计丢包数量
drop_count = df_msg["dropped"].value_counts()

# 队列等待时延：消息入队到发起传输的间隔（当前传输发生时刻无直接字段，说明：
# 当前代码消息入队后立刻尝试发送，queue_in_step≈发送时刻；
# 后续队列长时间积压版本可以直接使用；
# 如需完整端到端时延，等后续补充deliver_step后再扩展）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

# 子图1：成功/丢包柱状图
x_labels = ["Delivered", "Dropped"]
values = [drop_count.get(False, 0), drop_count.get(True, 0)]
bars = ax1.bar(x_labels, values, color=["#2ca02c", "#d62728"])
ax1.set_title("Message Delivery / Drop Statistics")
ax1.set_ylabel("Message Count")
# 在柱子上方标注数值
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f"{int(height)}", ha="center", va="bottom", fontsize=8)

# 子图2：不同信道下消息数量分布
channel_msg_cnt = df_msg["channel_state"].value_counts()
bars2 = ax2.bar(channel_msg_cnt.index, channel_msg_cnt.values, color=state_color)
ax2.set_title("Message Count under Each Channel State")
ax2.set_ylabel("Message Count")
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f"{int(height)}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
save_path2 = os.path.join(fig_save_dir, "comm_msg_stat.png")
plt.savefig(save_path2, bbox_inches="tight")
plt.close()
print(f"消息统计图表已保存：{save_path2}")

print("\n===== 绘图全部完成 =====")
print(f"图片输出目录：{fig_save_dir}")