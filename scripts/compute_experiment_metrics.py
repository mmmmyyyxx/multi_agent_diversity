import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_SUMMARY_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_TRACE_EMBEDDING_CHUNK_WORDS = 320
DEFAULT_TRACE_EMBEDDING_CHUNK_OVERLAP = 40
KNOWN_EXPERIMENT_SETTINGS = ["shared_div", "bank_div", "shared_baseline", "bank_baseline"]
PUBLIC_METRIC_COLUMNS = [
    "run_dir",
    "run_name",
    "setting",
    "seed",
    "baseline_only",
    "init_mode",
    "diversity_reward_enabled",
    "agents",
    "epochs",
    "train_size",
    "test_size",
    "lambda_diversity",
    "latest_prompt_cosine_diversity",
    "latest_prompt_embedding_cosine_diversity",
    "latest_trace_cosine_diversity",
    "latest_trace_embedding_cosine_diversity",
    "latest_reasoning_summary_cosine_diversity",
    "latest_summary_embedding_cosine_diversity",
    "latest_test_mean_family_diversity",
    "latest_test_mean_family_homogeneity_rate",
    "latest_test_mean_llm_direct_diversity_score",
    "latest_test_vote_acc",
    "disagreement_rate",
    "prompt_drift_cosine_distance",
    "update_applied_rate",
    "all_same_pair_rate",
    "embedding_model",
    "embedding_status",
]


def _safe_mean(xs: List[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def _parse_run_name(run_name: str) -> Tuple[str, Optional[int]]:
    for setting in KNOWN_EXPERIMENT_SETTINGS:
        if run_name == setting:
            return setting, None
        prefix = f"{setting}_seed"
        if run_name.startswith(prefix):
            raw_seed = run_name[len(prefix) :]
            try:
                return setting, int(raw_seed)
            except ValueError:
                return setting, None
    return run_name, None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except json.JSONDecodeError:
                continue
    return out


def _find_latest_test_predictions(run_dir: Path) -> Optional[Path]:
    files = sorted(run_dir.glob("test_epoch*_predictions.jsonl"))
    if files:
        return files[-1]
    return None


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def _extract_agent_snapshots(records: List[Dict[str, Any]], split: str = "") -> List[List[Dict[str, Any]]]:
    out: List[List[Dict[str, Any]]] = []
    wanted_split = str(split or "").lower()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        rec_split = str(rec.get("split", "")).lower()
        if wanted_split == "test":
            if not rec_split.startswith("test"):
                continue
        elif wanted_split and rec_split != wanted_split:
            continue
        agents = rec.get("agents", [])
        if not isinstance(agents, list):
            continue
        out.append([x for x in agents if isinstance(x, dict)])
    return out


def _find_test_trace_snapshots(run_dir: Path) -> List[List[Dict[str, Any]]]:
    # Visualization-facing trace metrics should reflect the full test set, not a
    # small recent window that can be sensitive to sample order.
    records = _read_jsonl(run_dir / "test_trace_history.jsonl")
    return _extract_agent_snapshots(records, split="test")


def _find_test_summary_snapshots(run_dir: Path) -> List[List[Dict[str, Any]]]:
    records = _read_jsonl(run_dir / "reasoning_summary_history.jsonl")
    return _extract_agent_snapshots(records, split="test")


def _extract_latest_prompt_strings(prompt_history: Any) -> List[str]:
    prompts: List[str] = []
    if not isinstance(prompt_history, dict):
        return prompts

    for agent in prompt_history.values():
        if not isinstance(agent, dict):
            continue

        events = agent.get("events", [])
        if isinstance(events, list) and events:
            last_prompt = None
            for event in reversed(events):
                if isinstance(event, dict) and event.get("current_prompt"):
                    last_prompt = str(event.get("current_prompt", ""))
                    break
            if last_prompt:
                prompts.append(last_prompt)
                continue

        current_prompt = str(agent.get("current_prompt", ""))
        if current_prompt:
            prompts.append(current_prompt)

    return prompts


def _pairwise_mismatch_rate(items: List[str]) -> float:
    n = len(items)
    if n < 2:
        return 0.0

    total_pairs = n * (n - 1) / 2
    mismatched_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            mismatched_pairs += int(items[i] != items[j])
    return float(mismatched_pairs / total_pairs)


def _extract_latest_trace_strings(trace_agents: List[Dict[str, Any]]) -> List[str]:
    traces: List[str] = []
    for a in trace_agents:
        trace = str(a.get("trace", "")).strip()
        if not trace:
            trace = str(a.get("reasoning_summary", "")).strip()
        if trace:
            traces.append(trace)
    return traces


def _extract_latest_reasoning_summary_strings(trace_agents: List[Dict[str, Any]]) -> List[str]:
    summaries: List[str] = []
    for a in trace_agents:
        summary = str(a.get("reasoning_summary", "")).strip()
        if summary:
            summaries.append(summary)
    return summaries


def _extract_latest_summary_embedding_texts(trace_agents: List[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for a in trace_agents:
        text = str(a.get("summary_embedding_text", "")).strip()
        if not text:
            text = str(a.get("reasoning_summary", "")).strip()
        if text:
            texts.append(text)
    return texts


def _tokenize_for_cosine(text: str) -> List[str]:
    # 浣跨敤杞婚噺 bag-of-words锛岄伩鍏嶉澶栦緷璧栥€?
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _cosine_sim_from_tokens(xs: List[str], ys: List[str]) -> float:
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
    sim = float(dot / (nx * ny))
    return float(max(0.0, min(1.0, sim)))


def _pairwise_cosine_diversity(texts: List[str]) -> Tuple[float, float]:
    n = len(texts)
    if n < 2:
        return 0.0, 1.0

    tokenized = [_tokenize_for_cosine(t) for t in texts]
    sims: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(_cosine_sim_from_tokens(tokenized[i], tokenized[j]))

    mean_sim = _safe_mean(sims)
    mean_sim = float(max(0.0, min(1.0, mean_sim)))
    return float(max(0.0, min(1.0, 1.0 - mean_sim))), mean_sim


class SummaryEmbeddingEncoder:
    def __init__(self, model_name: str, enabled: bool = True):
        self.model_name = model_name
        self.enabled = enabled
        self.model: Any = None
        self.load_error = ""
        self.cache: Dict[str, List[float]] = {}

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
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"
            print(
                f"[WARN] Embedding disabled: failed to load {self.model_name}. "
                f"Install sentence-transformers and ensure the model is available. Error: {self.load_error}"
            )
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

    def encode(self, texts: List[str]) -> List[List[float]]:
        normalized = [_normalize_spaces(t) for t in texts if _normalize_spaces(t)]
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
        texts: List[str],
        chunk_words: int = 0,
        chunk_overlap: int = 0,
    ) -> Tuple[List[List[float]], List[int]]:
        normalized = [_normalize_spaces(t) for t in texts if _normalize_spaces(t)]
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


def _cosine_sim_from_vectors(xs: List[float], ys: List[float]) -> float:
    if not xs and not ys:
        return 1.0
    if not xs or not ys or len(xs) != len(ys):
        return 0.0
    dot = sum(float(a * b) for a, b in zip(xs, ys))
    nx = math.sqrt(sum(float(a * a) for a in xs))
    ny = math.sqrt(sum(float(b * b) for b in ys))
    if nx <= 0.0 or ny <= 0.0:
        return 0.0
    sim = float(dot / (nx * ny))
    return float(max(-1.0, min(1.0, sim)))


def _split_text_for_embedding(text: str, chunk_words: int = 0, chunk_overlap: int = 0) -> List[str]:
    text = _normalize_spaces(text)
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


def _mean_pool_vectors(vectors: List[List[float]]) -> List[float]:
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


def _pairwise_embedding_cosine_diversity(texts: List[str], encoder: Optional[SummaryEmbeddingEncoder]) -> Tuple[Optional[float], Optional[float]]:
    n = len(texts)
    if n < 2:
        return 0.0, 1.0
    if encoder is None or not encoder.enabled:
        return None, None

    embeddings = encoder.encode(texts)
    if len(embeddings) < 2:
        return None, None

    sims: List[float] = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(_cosine_sim_from_vectors(embeddings[i], embeddings[j]))

    mean_sim = _safe_mean(sims)
    mean_sim = float(max(-1.0, min(1.0, mean_sim)))
    return float(max(0.0, min(2.0, 1.0 - mean_sim))), mean_sim


def _pairwise_document_embedding_cosine_diversity(
    texts: List[str],
    encoder: Optional[SummaryEmbeddingEncoder],
    chunk_words: int,
    chunk_overlap: int,
) -> Tuple[Optional[float], Optional[float], int, float]:
    n = len(texts)
    if n < 2:
        return 0.0, 1.0, n, 1.0 if n else 0.0
    if encoder is None or not encoder.enabled:
        return None, None, 0, 0.0

    embeddings, chunk_counts = encoder.encode_documents(
        texts,
        chunk_words=chunk_words,
        chunk_overlap=chunk_overlap,
    )
    if len(embeddings) < 2:
        return None, None, len(embeddings), _safe_mean([float(x) for x in chunk_counts])

    sims: List[float] = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(_cosine_sim_from_vectors(embeddings[i], embeddings[j]))

    mean_sim = _safe_mean(sims)
    mean_sim = float(max(-1.0, min(1.0, mean_sim)))
    return (
        float(max(0.0, min(2.0, 1.0 - mean_sim))),
        mean_sim,
        len(embeddings),
        _safe_mean([float(x) for x in chunk_counts]),
    )


def _safe_mean_optional(xs: List[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if isinstance(x, (int, float))]
    if not vals:
        return None
    return _safe_mean(vals)


def _collect_eval_metrics(pred_records: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(pred_records)
    if n == 0:
        return {
            "eval_size": 0,
            "disagreement_rate": 0.0,
            "mean_family_diversity_from_preds": 0.0,
            "mean_family_homogeneity_rate_from_preds": 0.0,
            "mean_llm_direct_diversity_score_from_preds": 0.0,
            "mean_vote_acc_from_preds": 0.0,
            "all_same_pair_rate_from_preds": 0.0,
        }

    disagreement = []
    family_div_all = []
    family_homo_all = []
    direct_div_all = []
    vote_acc_all = []
    all_same_pair_all = []

    for r in pred_records:
        answers = r.get("answers", [])
        if not isinstance(answers, list):
            answers = []
        div = float(r.get("team_family_diversity", 0.0))
        family_homo_rate = float(r.get("team_family_homogeneity_rate", 0.0))
        direct_div = r.get("llm_direct_diversity_score", None)
        vote_correct = float(r.get("vote_correct", 0.0))
        disagreement.append(int(len(set(map(str, answers))) > 1) if answers else 0)
        family_div_all.append(div)
        family_homo_all.append(family_homo_rate)
        if direct_div is not None:
            direct_div_all.append(float(direct_div))
        vote_acc_all.append(vote_correct)
        all_same_pair_all.append(int(bool(r.get("all_same_pair", False))))

    return {
        "eval_size": n,
        "disagreement_rate": _safe_mean(disagreement),
        "mean_family_diversity_from_preds": _safe_mean(family_div_all),
        "mean_family_homogeneity_rate_from_preds": _safe_mean(family_homo_all),
        "mean_llm_direct_diversity_score_from_preds": _safe_mean(direct_div_all),
        "mean_vote_acc_from_preds": _safe_mean(vote_acc_all),
        "all_same_pair_rate_from_preds": _safe_mean(all_same_pair_all),
    }


def _collect_update_metrics(step_records: List[Dict[str, Any]]) -> Dict[str, float]:
    if not step_records:
        return {
            "train_steps": 0,
            "update_requested_rate": 0.0,
            "update_ready_rate": 0.0,
            "update_selected_rate": 0.0,
            "update_applied_rate": 0.0,
            "mean_selected_agents": 0.0,
            "mean_updated_agents": 0.0,
        }

    update_requested = []
    update_ready = []
    update_selected = []
    update_applied = []
    num_selected = []
    num_updated = []

    for r in step_records:
        u = r.get("update", {}) if isinstance(r.get("update", {}), dict) else {}
        req = int(bool(u.get("update_requested", False)))
        ready = int(bool(u.get("update_ready", False)))
        selected_ids = u.get("selected_agent_ids", [])
        updated_ids = u.get("updated_agent_ids", [])
        if not isinstance(selected_ids, list):
            selected_ids = []
        if not isinstance(updated_ids, list):
            updated_ids = []

        update_requested.append(req)
        update_ready.append(ready)
        update_selected.append(int(len(selected_ids) > 0))
        update_applied.append(int(len(updated_ids) > 0))
        num_selected.append(float(len(selected_ids)))
        num_updated.append(float(len(updated_ids)))

    return {
        "train_steps": len(step_records),
        "update_requested_rate": _safe_mean(update_requested),
        "update_ready_rate": _safe_mean(update_ready),
        "update_selected_rate": _safe_mean(update_selected),
        "update_applied_rate": _safe_mean(update_applied),
        "mean_selected_agents": _safe_mean(num_selected),
        "mean_updated_agents": _safe_mean(num_updated),
    }


def analyze_run(run_dir: Path, summary_encoder: Optional[SummaryEmbeddingEncoder] = None) -> Dict[str, Any]:
    run_meta = _read_json(run_dir / "run_meta.json") or {}
    history = _read_json(run_dir / "history.json") or []
    last_state = _read_json(run_dir / "last_state.json") or {}
    step_logs = _read_jsonl(run_dir / "train_step_logs.jsonl")
    trace_snapshots = _find_test_trace_snapshots(run_dir)
    summary_snapshots = _find_test_summary_snapshots(run_dir)
    pred_path = _find_latest_test_predictions(run_dir)
    pred_records = _read_jsonl(pred_path) if pred_path else []
    prompt_history = _read_json(run_dir / "prompt_history.json") or {}

    latest_train_metrics: Dict[str, Any] = {}
    latest_test_metrics: Dict[str, Any] = {}
    if isinstance(history, list) and history:
        latest = history[-1]
        if isinstance(latest, dict):
            if isinstance(latest.get("train", {}), dict) or isinstance(latest.get("test", {}), dict):
                latest_train_metrics = latest.get("train", {}) if isinstance(latest.get("train", {}), dict) else {}
                latest_test_metrics = latest.get("test", {}) if isinstance(latest.get("test", {}), dict) else {}

    cfg = run_meta.get("config", {}) if isinstance(run_meta.get("config", {}), dict) else {}
    lam_div = float(cfg.get("lambda_diversity", 0.0))
    diversity_reward_enabled = int(lam_div > 0.0)

    agents = last_state.get("agents", []) if isinstance(last_state.get("agents", []), list) else []
    latest_prompt_strings = _extract_latest_prompt_strings(prompt_history)
    prompt_diversity_rate = _pairwise_mismatch_rate(latest_prompt_strings)
    prompt_cos_div, prompt_cos_sim = _pairwise_cosine_diversity(latest_prompt_strings)
    prompt_emb_cos_div, prompt_emb_cos_sim = _pairwise_embedding_cosine_diversity(latest_prompt_strings, summary_encoder)

    trace_cos_div_all: List[float] = []
    trace_cos_sim_all: List[float] = []
    trace_emb_cos_div_all: List[Optional[float]] = []
    trace_emb_cos_sim_all: List[Optional[float]] = []
    trace_emb_counts_all: List[float] = []
    trace_emb_chunks_all: List[float] = []
    latest_trace_strings: List[str] = []
    latest_trace_embedding_count = 0
    for i, snapshot_agents in enumerate(trace_snapshots):
        trace_strings = _extract_latest_trace_strings(snapshot_agents)
        div_i, sim_i = _pairwise_cosine_diversity(trace_strings)
        trace_cos_div_all.append(div_i)
        trace_cos_sim_all.append(sim_i)
        emb_div_i, emb_sim_i, emb_count_i, emb_chunks_i = _pairwise_document_embedding_cosine_diversity(
            trace_strings,
            summary_encoder,
            chunk_words=DEFAULT_TRACE_EMBEDDING_CHUNK_WORDS,
            chunk_overlap=DEFAULT_TRACE_EMBEDDING_CHUNK_OVERLAP,
        )
        trace_emb_cos_div_all.append(emb_div_i)
        trace_emb_cos_sim_all.append(emb_sim_i)
        trace_emb_counts_all.append(float(emb_count_i))
        trace_emb_chunks_all.append(float(emb_chunks_i))
        if i == len(trace_snapshots) - 1:
            latest_trace_strings = trace_strings
            latest_trace_embedding_count = emb_count_i

    trace_cos_div = _safe_mean(trace_cos_div_all)
    trace_cos_sim = _safe_mean(trace_cos_sim_all)
    trace_emb_cos_div = _safe_mean_optional(trace_emb_cos_div_all)
    trace_emb_cos_sim = _safe_mean_optional(trace_emb_cos_sim_all)

    summary_cos_div_all: List[float] = []
    summary_cos_sim_all: List[float] = []
    summary_emb_cos_div_all: List[Optional[float]] = []
    summary_emb_cos_sim_all: List[Optional[float]] = []
    latest_summary_strings: List[str] = []
    latest_summary_embedding_texts: List[str] = []
    for i, snapshot_agents in enumerate(summary_snapshots):
        summary_strings = _extract_latest_reasoning_summary_strings(snapshot_agents)
        div_i, sim_i = _pairwise_cosine_diversity(summary_strings)
        summary_cos_div_all.append(div_i)
        summary_cos_sim_all.append(sim_i)
        summary_embedding_texts = _extract_latest_summary_embedding_texts(snapshot_agents)
        emb_div_i, emb_sim_i = _pairwise_embedding_cosine_diversity(summary_embedding_texts, summary_encoder)
        summary_emb_cos_div_all.append(emb_div_i)
        summary_emb_cos_sim_all.append(emb_sim_i)
        if i == len(summary_snapshots) - 1:
            latest_summary_strings = summary_strings
            latest_summary_embedding_texts = summary_embedding_texts

    summary_cos_div = _safe_mean(summary_cos_div_all)
    summary_cos_sim = _safe_mean(summary_cos_sim_all)
    summary_emb_cos_div = _safe_mean_optional(summary_emb_cos_div_all)
    summary_emb_cos_sim = _safe_mean_optional(summary_emb_cos_sim_all)

    run_name = run_dir.name
    setting, seed_from_name = _parse_run_name(run_name)
    cfg_baseline_only = bool(cfg.get("baseline_only", False))
    baseline_only = int(setting.endswith("_testonly") or cfg_baseline_only)

    eval_metrics = _collect_eval_metrics(pred_records)
    latest_test_mean_family_diversity = float(
        latest_test_metrics.get("mean_family_diversity", eval_metrics["mean_family_diversity_from_preds"]) or 0.0
    )
    latest_test_mean_family_homogeneity_rate = float(
        latest_test_metrics.get("mean_family_homogeneity_rate", eval_metrics["mean_family_homogeneity_rate_from_preds"]) or 0.0
    )
    latest_test_mean_llm_direct_diversity_score = float(
        latest_test_metrics.get("mean_llm_direct_diversity_score", eval_metrics["mean_llm_direct_diversity_score_from_preds"]) or 0.0
    )
    latest_train_mean_llm_direct_diversity_score = float(latest_train_metrics.get("mean_llm_direct_diversity_score", 0.0) or 0.0) if latest_train_metrics else 0.0
    latest_test_vote_acc = float(
        latest_test_metrics.get("vote_acc", eval_metrics["mean_vote_acc_from_preds"]) or 0.0
    )

    prompt_drift_flags = []
    prompt_drift_cos_distances = []
    for a in agents:
        if not isinstance(a, dict):
            continue
        init_prompt = _normalize_spaces(str(a.get("initial_prompt", "")))
        cur_prompt = _normalize_spaces(str(a.get("current_prompt", "")))
        if init_prompt and cur_prompt:
            prompt_drift_flags.append(int(init_prompt != cur_prompt))
            sim = _cosine_sim_from_tokens(_tokenize_for_cosine(init_prompt), _tokenize_for_cosine(cur_prompt))
            prompt_drift_cos_distances.append(float(max(0.0, min(1.0, 1.0 - sim))))
            continue
        init_h = str(a.get("initial_prompt_hash", ""))
        cur_h = str(a.get("current_prompt_hash", ""))
        prompt_drift_flags.append(int(init_h != "" and cur_h != "" and init_h != cur_h))

    out = {
        "run_dir": str(run_dir),
        "run_name": run_name,
        "setting": setting,
        "seed": cfg.get("seed", seed_from_name),
        "baseline_only": baseline_only,
        "init_mode": str(run_meta.get("init_mode", "unknown")),
        "all_agents_shared_origin": int(bool(run_meta.get("all_agents_shared_origin", False))),
        "agents": int(run_meta.get("agents", cfg.get("agents", 0) or 0)),
        "update_every": int(run_meta.get("update_every", cfg.get("update_every", 0) or 0)),
        "epochs": int(cfg.get("epochs", 0) or 0),
        "train_size": int(cfg.get("train_size", 0) or 0),
        "test_size": int(cfg.get("test_size", 0) or 0),
        "lambda_diversity": lam_div,
        "diversity_reward_enabled": diversity_reward_enabled,
        "latest_test_mean_family_diversity": latest_test_mean_family_diversity,
        "latest_test_mean_family_homogeneity_rate": latest_test_mean_family_homogeneity_rate,
        "latest_train_mean_llm_direct_diversity_score": latest_train_mean_llm_direct_diversity_score,
        "latest_test_mean_llm_direct_diversity_score": latest_test_mean_llm_direct_diversity_score,
        "latest_train_vote_acc": float(latest_train_metrics.get("vote_acc", 0.0) or 0.0) if latest_train_metrics else None,
        "latest_test_vote_acc": latest_test_vote_acc,
        "disagreement_rate": eval_metrics.get("disagreement_rate", 0.0),
        "prompt_drift_cosine_distance": _safe_mean(prompt_drift_cos_distances),
        "all_same_pair_rate": eval_metrics.get("all_same_pair_rate_from_preds", 0.0),
    }
    out.update(eval_metrics)
    out.update(_collect_update_metrics(step_logs))

    # For test-only baseline runs, these training/update metrics are not applicable.
    if baseline_only and not step_logs:
        out["train_steps"] = None
        out["update_requested_rate"] = None
        out["update_ready_rate"] = None
        out["update_selected_rate"] = None
        out["update_applied_rate"] = None
        out["mean_selected_agents"] = None
        out["mean_updated_agents"] = None

    if out.get("eval_size", 0) == 0 and isinstance(latest_test_metrics.get("size", None), (int, float)):
        out["eval_size"] = int(latest_test_metrics.get("size", 0) or 0)

    out["latest_test_prediction_file"] = str(pred_path) if pred_path else ""
    out["latest_prompt_diversity_rate"] = prompt_diversity_rate
    out["latest_prompt_exact_diversity_rate"] = prompt_diversity_rate
    out["latest_prompt_count"] = len(latest_prompt_strings)
    out["latest_prompt_cosine_diversity"] = prompt_cos_div
    out["latest_prompt_cosine_similarity"] = prompt_cos_sim
    out["latest_prompt_embedding_text_count"] = len(latest_prompt_strings)
    out["latest_prompt_embedding_cosine_diversity"] = prompt_emb_cos_div
    out["latest_prompt_embedding_cosine_similarity"] = prompt_emb_cos_sim
    out["prompt_embedding_status"] = summary_encoder.status if summary_encoder else "disabled"
    out["latest_trace_count"] = len(latest_trace_strings)
    out["latest_trace_cosine_diversity"] = trace_cos_div
    out["latest_trace_cosine_similarity"] = trace_cos_sim
    out["trace_cosine_window_used"] = len(trace_snapshots)
    out["test_trace_snapshot_count"] = len(trace_snapshots)
    out["latest_trace_embedding_text_count"] = latest_trace_embedding_count
    out["latest_trace_embedding_cosine_diversity"] = trace_emb_cos_div
    out["latest_trace_embedding_cosine_similarity"] = trace_emb_cos_sim
    out["trace_embedding_cosine_window_used"] = len(trace_snapshots) if summary_encoder and summary_encoder.enabled else 0
    out["test_trace_embedding_snapshot_count"] = len(trace_snapshots) if summary_encoder and summary_encoder.enabled else 0
    out["trace_embedding_chunk_words"] = DEFAULT_TRACE_EMBEDDING_CHUNK_WORDS
    out["trace_embedding_chunk_overlap"] = DEFAULT_TRACE_EMBEDDING_CHUNK_OVERLAP
    out["mean_trace_embedding_chunks"] = _safe_mean(trace_emb_chunks_all)
    out["latest_reasoning_summary_count"] = len(latest_summary_strings)
    out["latest_reasoning_summary_cosine_diversity"] = summary_cos_div
    out["latest_reasoning_summary_cosine_similarity"] = summary_cos_sim
    out["reasoning_summary_cosine_window_used"] = len(summary_snapshots)
    out["test_reasoning_summary_snapshot_count"] = len(summary_snapshots)
    out["latest_summary_embedding_text_count"] = len(latest_summary_embedding_texts)
    out["latest_summary_embedding_cosine_diversity"] = summary_emb_cos_div
    out["latest_summary_embedding_cosine_similarity"] = summary_emb_cos_sim
    out["summary_embedding_cosine_window_used"] = len(summary_snapshots) if summary_encoder and summary_encoder.enabled else 0
    out["test_summary_embedding_snapshot_count"] = len(summary_snapshots) if summary_encoder and summary_encoder.enabled else 0
    out["embedding_model"] = summary_encoder.model_name if summary_encoder else ""
    out["embedding_status"] = summary_encoder.status if summary_encoder else "disabled"
    out["summary_embedding_model"] = summary_encoder.model_name if summary_encoder else ""
    out["summary_embedding_status"] = summary_encoder.status if summary_encoder else "disabled"
    out["trace_embedding_model"] = summary_encoder.model_name if summary_encoder else ""
    out["trace_embedding_status"] = summary_encoder.status if summary_encoder else "disabled"
    return out


def _to_float_str(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_markdown(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        path.write_text("# Experiment Summary\n\nNo valid runs found.\n", encoding="utf-8")
        return

    columns = [
        *PUBLIC_METRIC_COLUMNS,
    ]
    lines = ["# Experiment Summary", "", "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_to_float_str(r.get(c, "")) for c in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment outputs.")
    parser.add_argument("--runs", nargs="*", default=[], help="Explicit run directories to analyze.")
    parser.add_argument("--runs_root", type=str, default="", help="Root directory containing run sub-directories.")
    parser.add_argument("--out_csv", type=str, default="", help="Output CSV path.")
    parser.add_argument("--out_md", type=str, default="", help="Output Markdown summary path.")
    parser.add_argument("--summary_embedding_model", type=str, default=DEFAULT_SUMMARY_EMBEDDING_MODEL)
    parser.add_argument("--disable_summary_embedding", action="store_true")
    args = parser.parse_args()

    run_dirs: List[Path] = []
    for x in args.runs:
        p = Path(x)
        if p.exists() and p.is_dir():
            run_dirs.append(p)
    if args.runs_root:
        root = Path(args.runs_root)
        if root.exists() and root.is_dir():
            for p in sorted(root.iterdir()):
                if p.is_dir() and (p / "run_meta.json").exists():
                    run_dirs.append(p)

    dedup = []
    seen = set()
    for p in run_dirs:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            dedup.append(p)
    run_dirs = dedup

    summary_encoder = SummaryEmbeddingEncoder(args.summary_embedding_model, enabled=not args.disable_summary_embedding)
    rows = [analyze_run(p, summary_encoder=summary_encoder) for p in run_dirs]
    rows = sorted(rows, key=lambda r: (r.get("init_mode", ""), int(r.get("diversity_reward_enabled", 0)), r.get("run_dir", "")))

    if args.out_csv:
        out_csv = Path(args.out_csv)
    elif args.runs_root:
        out_csv = Path(args.runs_root) / "experiment_metrics.csv"
    else:
        out_csv = Path("experiment_metrics.csv")

    if args.out_md:
        out_md = Path(args.out_md)
    elif args.runs_root:
        out_md = Path(args.runs_root) / "experiment_metrics.md"
    else:
        out_md = Path("experiment_metrics.md")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        public = [c for c in PUBLIC_METRIC_COLUMNS if any(c in r for r in rows)]
        extra = sorted({k for r in rows for k in r.keys() if k not in public})
        fieldnames = public + extra
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    else:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            f.write("run_dir\n")

    write_markdown(rows, out_md)

    print(f"Analyzed runs: {len(rows)}")
    print(f"CSV: {out_csv}")
    print(f"Markdown: {out_md}")


if __name__ == "__main__":
    main()


