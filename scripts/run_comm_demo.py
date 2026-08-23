import os
import csv
import random

from src.communication.channel_markov import MarkovChannel
from src.communication.message import Message

# ===================== 仿真配置区 =====================
SELECT_STRATEGY = "adaptive_rate"   # "fixed" / "adaptive_rate"
SEED = 42
SIM_STEPS = 600

FIXED_PER_STEP = 1

ADAPTIVE_GEN_CFG = {
    "good": 3,
    "medium": 1,
    "bad": 0
}

BASE_OUT_DIR = r"D:\vpp_elastic_control\src\communication"
TRACE_CSV = os.path.join(BASE_OUT_DIR, "communication_trace.csv")
MSG_CSV = os.path.join(BASE_OUT_DIR, "message_log.csv")
# ======================================================

def get_generate_msg_count(channel_state: str, strategy: str) -> int:
    if strategy == "fixed":
        return FIXED_PER_STEP
    elif strategy == "adaptive_rate":
        return ADAPTIVE_GEN_CFG.get(channel_state, 0)
    else:
        raise ValueError(f"不支持的策略：{strategy}")


def main():
    os.makedirs(BASE_OUT_DIR, exist_ok=True)

    channel = MarkovChannel(seed=SEED)
    msg_queue = []
    msg_id_counter = 0

    trace_list = []
    msg_log_list = []

    total_gen = 0
    total_send_ok = 0
    total_drop = 0
    delay_records = []

    for step in range(SIM_STEPS):
        cur_state = channel.get_state()

        gen_num = get_generate_msg_count(cur_state, SELECT_STRATEGY)
        for _ in range(gen_num):
            msg_id_counter += 1
            m = Message(
                msg_id=str(msg_id_counter),
                src="node_a",
                dst="node_b",
                generated_step=step,
                deliver_step=None,
                payload={"data": msg_id_counter},
                priority=random.randint(1, 5),
                dropped=False,
                sequence_id=msg_id_counter,
                queue_in_step=step
            )
            msg_queue.append(m)
            total_gen += 1

        # FIFO队列：取队首消息尝试传输
        if len(msg_queue) > 0:
            head_msg = msg_queue.pop(0)
            processed_msg = channel.transmit(head_msg)

            if processed_msg.dropped:
                # 丢包
                total_drop += 1
                msg_log_list.append({
                    "msg_id": processed_msg.msg_id,
                    "priority": processed_msg.priority,
                    "queue_in_step": processed_msg.queue_in_step,
                    "send_success": 0,
                    "drop": 1,
                    "delay_step": -1,
                    "channel_state": cur_state,
                    "queue_len_before_send": len(msg_queue)+1
                })
            else:
                # 发送成功
                total_send_ok +=1
                delay = step - processed_msg.queue_in_step
                delay_records.append(delay)
                msg_log_list.append({
                    "msg_id": processed_msg.msg_id,
                    "priority": processed_msg.priority,
                    "queue_in_step": processed_msg.queue_in_step,
                    "send_success": 1,
                    "drop": 0,
                    "delay_step": delay,
                    "channel_state": cur_state,
                    "queue_len_before_send": len(msg_queue)+1
                })

        trace_list.append({
            "step": step,
            "channel_state": cur_state,
            "queue_size": len(msg_queue)
        })
        channel.step()

    total_process = total_send_ok + total_drop
    drop_rate = total_drop / total_process if total_process > 0 else 0.0
    avg_delay = sum(delay_records)/len(delay_records) if len(delay_records) > 0 else 0.0

    print("="*65)
    print(f"策略 = {SELECT_STRATEGY} | seed={SEED} | 仿真总步数={SIM_STEPS}")
    print(f"总生成消息：{total_gen}")
    print(f"发送成功：{total_send_ok}")
    print(f"丢包数量：{total_drop}")
    print(f"丢包率  ：{drop_rate:.4f}")
    print(f"平均时延(时隙)：{avg_delay:.2f}")
    print("="*65)

    with open(TRACE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step","channel_state","queue_size"])
        w.writeheader()
        w.writerows(trace_list)

    with open(MSG_CSV, "w", newline="", encoding="utf-8") as f:
        fields = ["msg_id","priority","queue_in_step","send_success","drop","delay_step","channel_state","queue_len_before_send"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(msg_log_list)

    print(f"输出文件:\n{TRACE_CSV}\n{MSG_CSV}")


if __name__ == "__main__":
    main()