import csv
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else ["id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_mean(xs: Iterable[Any]) -> float:
    vals = [safe_float(x) for x in xs]
    return float(sum(vals) / len(vals)) if vals else 0.0


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def prompt_hash(text: str) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()[:12]


def question_hash(question: str) -> str:
    return prompt_hash(normalize_spaces(question))


def find_prediction_file(run_dir: Path) -> Optional[Path]:
    files = sorted(run_dir.glob("test*_predictions.jsonl"))
    if not files:
        return None
    final_files = [p for p in files if "final" in p.name]
    return final_files[-1] if final_files else files[-1]


def tokenize_for_cosine(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower())


def cosine_sim_from_tokens(xs: List[str], ys: List[str]) -> float:
    if not xs and not ys:
        return 1.0
    if not xs or not ys:
        return 0.0
    cx = Counter(xs)
    cy = Counter(ys)
    keys = set(cx.keys()) | set(cy.keys())
    dot = sum(float(cx[k] * cy[k]) for k in keys)
    nx = math.sqrt(sum(float(v * v) for v in cx.values()))
    ny = math.sqrt(sum(float(v * v) for v in cy.values()))
    if nx <= 0.0 or ny <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, dot / (nx * ny))))


def pairwise_token_cosine_diversity(texts: Sequence[str]) -> Tuple[float, float]:
    if len(texts) < 2:
        return 0.0, 1.0
    tokenized = [tokenize_for_cosine(t) for t in texts]
    sims: List[float] = []
    for i in range(len(tokenized)):
        for j in range(i + 1, len(tokenized)):
            sims.append(cosine_sim_from_tokens(tokenized[i], tokenized[j]))
    mean_sim = safe_mean(sims)
    mean_sim = float(max(0.0, min(1.0, mean_sim)))
    return float(max(0.0, min(1.0, 1.0 - mean_sim))), mean_sim


def bootstrap_mean_ci(values: Sequence[float], iterations: int = 2000, seed: int = 42) -> Dict[str, Any]:
    vals = [safe_float(v) for v in values]
    if not vals:
        return {"n": 0, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    if len(vals) == 1 or iterations <= 0:
        return {"n": len(vals), "mean": vals[0], "ci_low": vals[0], "ci_high": vals[0]}
    rng = random.Random(seed)
    means: List[float] = []
    n = len(vals)
    for _ in range(iterations):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(float(sum(sample) / n))
    means.sort()
    lo_idx = int(0.025 * (len(means) - 1))
    hi_idx = int(0.975 * (len(means) - 1))
    return {
        "n": n,
        "mean": float(sum(vals) / n),
        "ci_low": float(means[lo_idx]),
        "ci_high": float(means[hi_idx]),
    }


def rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted((safe_float(v), i) for i, v in enumerate(values))
    ranks = [0.0 for _ in indexed]
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        while end < len(indexed) and indexed[end][0] == indexed[pos][0]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for k in range(pos, end):
            ranks[indexed[k][1]] = avg_rank
        pos = end
    return ranks


def spearman_corr(xs: Sequence[Any], ys: Sequence[Any]) -> Dict[str, Any]:
    pairs = [(safe_float(x), safe_float(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    n = len(pairs)
    if n < 2:
        return {"n": n, "rho": 0.0}
    rx = rankdata([p[0] for p in pairs])
    ry = rankdata([p[1] for p in pairs])
    mx = safe_mean(rx)
    my = safe_mean(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in rx))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ry))
    rho = num / (den_x * den_y) if den_x > 0.0 and den_y > 0.0 else 0.0
    return {"n": n, "rho": float(max(-1.0, min(1.0, rho)))}


def wilcoxon_signed_rank(deltas: Sequence[Any]) -> Dict[str, Any]:
    vals = [safe_float(x) for x in deltas if abs(safe_float(x)) > 1e-12]
    n = len(vals)
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "w_minus": 0.0, "z": 0.0, "p_approx": 1.0}
    abs_ranks = rankdata([abs(v) for v in vals])
    w_plus = sum(r for r, v in zip(abs_ranks, vals) if v > 0)
    w_minus = sum(r for r, v in zip(abs_ranks, vals) if v < 0)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    z = (w_plus - mean_w) / math.sqrt(var_w) if var_w > 0.0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {
        "n": n,
        "w_plus": float(w_plus),
        "w_minus": float(w_minus),
        "z": float(z),
        "p_approx": float(max(0.0, min(1.0, p))),
    }


def infer_probe_kind(*names: Any) -> str:
    text = " ".join(str(x or "") for x in names).lower()
    if "mixed" in text or "bank" in text:
        return "mixed"
    if "same" in text:
        return "same"
    return "other"

