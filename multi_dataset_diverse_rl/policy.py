from typing import Any, Dict, List
from collections import deque

import numpy as np


class BanditPolicy:
    def __init__(self, num_actions: int, lr: float = 0.2, baseline_momentum: float = 0.9):
        self.num_actions = num_actions
        self.lr = lr
        self.baseline_momentum = baseline_momentum
        self.preferences = np.zeros(num_actions, dtype=np.float64)
        self.baseline = 0.0

    def probs(self) -> np.ndarray:
        x = self.preferences - np.max(self.preferences)
        ex = np.exp(x)
        return ex / np.sum(ex)

    def sample(self):
        p = self.probs()
        a = int(np.random.choice(len(p), p=p))
        return a, p

    def update(self, action: int, reward: float):
        p = self.probs()
        adv = reward - self.baseline
        one_hot = np.zeros_like(p)
        one_hot[action] = 1.0
        self.preferences += self.lr * adv * (one_hot - p)
        self.baseline = self.baseline_momentum * self.baseline + (1 - self.baseline_momentum) * reward

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_actions": self.num_actions,
            "lr": self.lr,
            "baseline_momentum": self.baseline_momentum,
            "preferences": self.preferences.tolist(),
            "baseline": float(self.baseline),
        }


class AgentState:
    def __init__(
        self,
        initial_prompt: str,
        bandit_lr: float,
        baseline_momentum: float,
        homogeneity_window: int = 50,
    ):
        self.initial_prompt = initial_prompt
        self.current_prompt = initial_prompt
        self.history = [initial_prompt]
        self.gradient_history: List[str] = []
        self.bandit = BanditPolicy(num_actions=5, lr=bandit_lr, baseline_momentum=baseline_momentum)
        self.homogeneity_window = max(1, int(homogeneity_window))
        self.recent_homogeneity_flags = deque(maxlen=self.homogeneity_window)
        self.homogeneity_count = 0
        self.accept_count = 0
        self.reject_count = 0
        self.last_update_record: Dict[str, Any] = {}

    def observe_homogeneity_result(self, homogeneous_flag: int):
        flag = 1 if int(homogeneous_flag) > 0 else 0
        self.recent_homogeneity_flags.append(flag)
        self.homogeneity_count = int(sum(self.recent_homogeneity_flags))

    # Backward-compat helper: map "wrong" to "homogeneous" for old call sites.
    def observe_answer_result(self, correct: int):
        self.observe_homogeneity_result(0 if int(correct) == 1 else 1)
