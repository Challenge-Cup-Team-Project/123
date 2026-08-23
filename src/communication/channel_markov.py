import numpy as np

class MarkovChannel:
    def __init__(self, seed=None):
        self.states = ["good", "medium", "bad"]
        self.loss_prob = {
            "good": 0.05,
            "medium": 0.25,
            "bad": 0.60
        }
        self.trans_matrix = np.array([
            [0.70, 0.26, 0.04],
            [0.20, 0.60, 0.20],
            [0.04, 0.26, 0.70]
        ])
        self.current_state = "good"
        # ✅独立随机生成器，不和丢包函数抢随机序列
        self.rng = np.random.default_rng(seed=seed)

    def step(self):
        idx = self.states.index(self.current_state)
        next_idx = self.rng.choice(3, p=self.trans_matrix[idx])
        self.current_state = self.states[next_idx]
        return self.current_state

    def is_drop(self):
        p = self.loss_prob[self.current_state]
        return self.rng.random() < p

    def get_state(self):
        return self.current_state

    def get_loss_prob(self):
        return self.loss_prob[self.current_state]

    def get_observation(self, link_id, step):
        return {
            "link_id": link_id,
            "step": step,
            "channel_state": self.current_state,
            "loss_prob": self.get_loss_prob()
        }

    def transmit(self, msg):
        drop_flag = self.is_drop()
        msg.dropped = drop_flag
        if not drop_flag:
            msg.deliver_step = msg.generated_step + 1
        return msg