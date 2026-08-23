import numpy as np
import pandas as pd
import os
from collections import deque
from src.communication.channel_markov import MarkovChannel
from src.communication.message import Message


def main():
    # ========== 仿真配置（固定种子保证可复现） ==========
    SEED = 2000
    SIM_TOTAL_STEPS = 2000    # 总仿真时隙
    SEND_INTERVAL = 2         # 每隔几个时隙生成新消息并入队列
    link_id = "link_0"

    # 输出目标文件夹
    # 输出目录：项目根目录/output/csv
    output_dir = os.path.join(".", "output", "csv")
    os.makedirs(output_dir, exist_ok=True)
    trace_path = os.path.join(output_dir, "communication_trace.csv")
    msg_path = os.path.join(output_dir, "message_log.csv")

    # 初始化信道
    np.random.seed(SEED)
    channel = MarkovChannel(seed=SEED)

    # 存储日志
    communication_trace = []   # 信道状态时序
    message_log = []           # 数据包日志

    msg_seq = 0
    total_send = 0
    total_drop = 0
    state_drop_stat = {
        "good": {"send": 0, "drop": 0},
        "medium": {"send": 0, "drop": 0},
        "bad": {"send": 0, "drop": 0},
    }

    # ========== 新增：FIFO消息发送队列 ==========
    tx_queue = deque()  # 双向队列，FIFO先进先出
    # 队列元素：(message对象, queue_in_step)

    # ========== 主仿真循环 ==========
    for step in range(SIM_TOTAL_STEPS):
        # 信道状态更新
        channel.step()
        current_state = channel.get_state()
        obs = channel.get_observation(link_id, step)

        # 记录每一时隙信道状态时序（供给弹性控制F使用）
        communication_trace.append({
            "time_step": step,
            "channel_state": current_state,
            "observation": obs
        })

        # =========基线1：自适应发包间隔========
    curr_state = channel.get_state()
    if curr_state == "good":
        dyn_interval = 2
    elif curr_state == "medium":
        dyn_interval = 4
    else: # bad
        dyn_interval = 8

# 根据动态间隔产生新消息
        if step % dyn_interval == 0:
    # 这里复制你原来生成Message、加入tx_queue的整套代码
            tx_msg = Message(
                msg_id=f"msg_{msg_seq}",
                src="nodeA",
                dst="nodeB",
                generated_step=step,
                deliver_step=None,      # 必填参数
                payload={"data": 1},
                priority=1,
                dropped=False,
                sequence_id=msg_seq
            )
            # 消息进入队列，记录入队时刻
            tx_queue.append((tx_msg, step))
            msg_seq += 1

        # 2. 当前时隙：尝试处理【队列头部消息】（FIFO）
        if len(tx_queue) > 0:
            head_msg, queue_in_step = tx_queue[0]  # 取出队首，不出队
            state_drop_stat[current_state]["send"] += 1
            total_send += 1

            # 信道尝试传输
            tx_msg_after = channel.transmit(head_msg)
            dropped = tx_msg_after.dropped

            # 写入日志，新增 queue_in_step 字段
            message_log.append({
                "msg_seq": tx_msg_after.sequence_id,
                "generate_time": tx_msg_after.generated_step,
                "queue_in_step": queue_in_step,  # 【新增字段】消息入队时隙
                "channel_state": current_state,
                "dropped": dropped,
                "src": tx_msg_after.src,
                "dst": tx_msg_after.dst,
                "priority": tx_msg_after.priority
            })

            if dropped:
                total_drop += 1
                state_drop_stat[current_state]["drop"] += 1
                # 丢包：消息从队列弹出（丢弃）
                tx_queue.popleft()
            else:
                # 传输成功：消息从队列弹出
                tx_queue.popleft()
                # 传输失败不弹出，保留在下一个时隙继续尝试发送

    # ========== 输出CSV文件 ==========
    df_trace = pd.DataFrame(communication_trace)
    df_trace.to_csv(trace_path, index=False, encoding="utf-8-sig")

    df_msg = pd.DataFrame(message_log)
    df_msg.to_csv(msg_path, index=False, encoding="utf-8-sig")

    # ========== 控制台统计输出 ==========
    print("==================== 仿真结束 ====================")
    print(f"总仿真时隙：{SIM_TOTAL_STEPS}")
    print(f"总发送数据包：{total_send}")
    print(f"总丢包数量：{total_drop}")
    if total_send > 0:
        print(f"全局丢包率：{total_drop/total_send:.4f}")
    print(f"仿真结束时队列剩余消息数量：{len(tx_queue)}")
    print("\n各信道状态发包&丢包统计：")
    for s, data in state_drop_stat.items():
        send_cnt = data["send"]
        drop_cnt = data["drop"]
        rate = drop_cnt/send_cnt if send_cnt>0 else 0
        print(f"  {s:6s} | 发送:{send_cnt:4d} | 丢包:{drop_cnt:4d} | 状态内丢包率:{rate:.4f}")

    print("\n文件已生成：")
    print(f"  {trace_path} （信道时序，供弹性控制F使用）")
    print(f"  {msg_path}    （数据包完整传输日志，含队列入队时刻）")


if __name__ == "__main__":
    main()