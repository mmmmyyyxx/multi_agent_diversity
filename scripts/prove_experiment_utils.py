import csv
import hashlib
import json
import math
import os
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


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_CHUNK_WORDS = 320
DEFAULT_EMBEDDING_CHUNK_OVERLAP = 40


class SentenceEmbeddingEncoder:
    def __init__(self, model_name: str, enabled: bool = True):
        self.model_name = model_name
        self.enabled = enabled
        self.model: Any = None
        self.load_error = ""
        self.cache: dict[str, list[float]] = {}

    @property
    def status(self) -> str:
        if not self.enabled:
            return "disabled"
        if self.load_error:
            return "unavailable"
        if self.model is not None:
            return "ok"
        return "not_loaded"

    def _load(self) -> bool:
        if not self.enabled:
            return False
        if self.model is not None:
            return True
        if self.load_error:
            return False
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self._resolve_model_name())
            return True
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            return False

    def _resolve_model_name(self) -> str:
        model_name = str(self.model_name or "").strip()
        if not model_name:
            return model_name
        direct = Path(model_name)
        if direct.exists():
            return str(direct)

        candidates = [
            Path.cwd() / model_name,
            Path.cwd() / "models" / model_name,
            Path.cwd() / "models" / model_name.replace("/", "--"),
        ]
        for env_name in ["SENTENCE_TRANSFORMERS_HOME", "HF_HOME", "HUGGINGFACE_HUB_CACHE"]:
            root = os.getenv(env_name)
            if not root:
                continue
            candidates.extend(
                [
                    Path(root) / model_name,
                    Path(root) / model_name.replace("/", "--"),
                    Path(root) / "hub" / f"models--{model_name.replace('/', '--')}",
                ]
            )
        userprofile = os.getenv("USERPROFILE")
        if userprofile:
            candidates.append(Path(userprofile) / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}")

        for candidate in candidates:
            if not candidate.exists():
                continue
            snapshots = candidate / "snapshots"
            if snapshots.is_dir():
                snapshot_dirs = sorted([p for p in snapshots.iterdir() if p.is_dir()])
                if snapshot_dirs:
                    return str(snapshot_dirs[-1])
            return str(candidate)
        return model_name

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        normalized = [normalize_spaces(t) for t in texts if normalize_spaces(t)]
        if not normalized or not self._load():
            return []

        missing = [t for t in dict.fromkeys(normalized) if t not in self.cache]
        if missing:
            vectors = self.model.encode(
                missing,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, vec in zip(missing, vectors):
                self.cache[text] = [float(x) for x in vec]
        return [self.cache[t] for t in normalized if t in self.cache]

    def encode_documents(
        self,
        texts: Sequence[str],
        chunk_words: int = 0,
        chunk_overlap: int = 0,
    ) -> Tuple[List[List[float]], List[int]]:
        normalized = [normalize_spaces(t) for t in texts if normalize_spaces(t)]
        if not normalized or not self._load():
            return [], []

        vectors: List[List[float]] = []
        chunk_counts: List[int] = []
        for text in normalized:
            chunks = _split_text_for_embedding(text, chunk_words=chunk_words, chunk_overlap=chunk_overlap)
            if not chunks:
                continue
            chunk_vectors = self.encode(chunks)
            if not chunk_vectors:
                continue
            vectors.append(_mean_pool_vectors(chunk_vectors))
            chunk_counts.append(len(chunks))
        return vectors, chunk_counts


def _split_text_for_embedding(text: str, chunk_words: int = 0, chunk_overlap: int = 0) -> List[str]:
    text = normalize_spaces(text)
    if not text:
        return []
    if chunk_words <= 0:
        return [text]

    words = text.split()
    if len(words) <= chunk_words:
        return [text]

    overlap = max(0, min(int(chunk_overlap), int(chunk_words) - 1))
    step = max(1, int(chunk_words) - overlap)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + int(chunk_words)]).strip()
        if chunk:
            chunks.append(chunk)
        if start + int(chunk_words) >= len(words):
            break
    return chunks


def _mean_pool_vectors(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    usable = [v for v in vectors if len(v) == dim]
    if not usable:
        return []
    pooled = [sum(float(v[i]) for v in usable) / float(len(usable)) for i in range(dim)]
    norm = math.sqrt(sum(float(x * x) for x in pooled))
    if norm <= 0.0:
        return pooled
    return [float(x / norm) for x in pooled]


def cosine_sim_from_vectors(xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs and not ys:
        return 1.0
    if not xs or not ys or len(xs) != len(ys):
        return 0.0
    dot = sum(float(a * b) for a, b in zip(xs, ys))
    nx = math.sqrt(sum(float(a * a) for a in xs))
    ny = math.sqrt(sum(float(b * b) for b in ys))
    if nx <= 0.0 or ny <= 0.0:
        return 0.0
    return float(max(-1.0, min(1.0, dot / (nx * ny))))


def pairwise_embedding_cosine_diversity(
    texts: Sequence[str],
    encoder: Optional[SentenceEmbeddingEncoder],
) -> Tuple[Optional[float], Optional[float]]:
    if len(texts) < 2:
        return 0.0, 1.0
    if encoder is None or not encoder.enabled:
        return None, None

    embeddings = encoder.encode(list(texts))
    if len(embeddings) < 2:
        return None, None

    sims: List[float] = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(cosine_sim_from_vectors(embeddings[i], embeddings[j]))
    mean_sim = safe_mean(sims)
    mean_sim = float(max(-1.0, min(1.0, mean_sim)))
    return float(max(0.0, min(2.0, 1.0 - mean_sim))), mean_sim


def pairwise_document_embedding_cosine_diversity(
    texts: Sequence[str],
    encoder: Optional[SentenceEmbeddingEncoder],
    chunk_words: int = DEFAULT_EMBEDDING_CHUNK_WORDS,
    chunk_overlap: int = DEFAULT_EMBEDDING_CHUNK_OVERLAP,
) -> Tuple[Optional[float], Optional[float], int, float]:
    if len(texts) < 2:
        return 0.0, 1.0, len(texts), 1.0 if texts else 0.0
    if encoder is None or not encoder.enabled:
        return None, None, 0, 0.0

    embeddings, chunk_counts = encoder.encode_documents(
        list(texts),
        chunk_words=chunk_words,
        chunk_overlap=chunk_overlap,
    )
    if len(embeddings) < 2:
        return None, None, len(embeddings), safe_mean([float(x) for x in chunk_counts])

    sims: List[float] = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(cosine_sim_from_vectors(embeddings[i], embeddings[j]))
    mean_sim = safe_mean(sims)
    mean_sim = float(max(-1.0, min(1.0, mean_sim)))
    return (
        float(max(0.0, min(2.0, 1.0 - mean_sim))),
        mean_sim,
        len(embeddings),
        safe_mean([float(x) for x in chunk_counts]),
    )


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
