from __future__ import annotations

import math


def planned_updates(epochs: int, train_size: int, update_every: int) -> int:
    return epochs * max(1, math.ceil(train_size / update_every))


if __name__ == "__main__":
    assert planned_updates(8, 75, 25) == 24
    assert planned_updates(8, 75, 75) == 8
    print("deterministic high-frequency update smoke: PASS")
