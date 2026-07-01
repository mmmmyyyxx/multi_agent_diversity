from collections import deque
from typing import Any, Dict, List


class AgentState:
    def __init__(self, initial_prompt: str, homogeneity_window: int = 50):
        self.initial_prompt = initial_prompt
        self.current_prompt = initial_prompt
        self.history = [initial_prompt]
        self.prompt_beam: List[Dict[str, Any]] = [
            {
                "id": "",
                "prompt": initial_prompt,
                "score": None,
                "metrics": {},
                "parent_id": None,
                "generation": 0,
            }
        ]
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
