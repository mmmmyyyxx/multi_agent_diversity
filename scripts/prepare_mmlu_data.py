import argparse
import importlib
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


CHOICE_LABELS = ["A", "B", "C", "D"]


def _to_choice_letter(x: Any) -> str:
    s = str(x).strip().upper()
    if s in CHOICE_LABELS:
        return s
    if s.isdigit():
        idx = int(s)
        if 0 <= idx < len(CHOICE_LABELS):
            return CHOICE_LABELS[idx]
    return s


def _extract_choices(ex: Dict[str, Any]) -> Optional[List[str]]:
    if isinstance(ex.get("choices"), list) and len(ex["choices"]) >= 4:
        return [str(c) for c in ex["choices"][:4]]

    # fallback common field names
    keys = ["A", "B", "C", "D"]
    if all(k in ex for k in keys):
        return [str(ex[k]) for k in keys]
    return None


def _extract_question(ex: Dict[str, Any]) -> Optional[str]:
    for k in ["question", "input", "query", "problem"]:
        if ex.get(k) is not None:
            return str(ex[k])
    return None


def _extract_answer_letter(ex: Dict[str, Any]) -> Optional[str]:
    for k in ["answer", "target", "label", "output"]:
        if ex.get(k) is not None:
            return _to_choice_letter(ex[k])
    return None


def _format_mmlu_question(question: str, choices: List[str]) -> str:
    lines = [f"Question: {question}", "", "Options:"]
    for i, c in enumerate(choices[:4]):
        lines.append(f"{CHOICE_LABELS[i]}. {c}")
    lines.append("")
    lines.append("Select the best option and output FINAL_ANSWER: <A/B/C/D>.")
    return "\n".join(lines)


def convert_split(ds_split, limit: int = -1, default_subject: str = "") -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for ex in ds_split:
        q = _extract_question(ex)
        choices = _extract_choices(ex)
        ans = _extract_answer_letter(ex)
        if not q or not choices or not ans:
            continue
        if ans not in CHOICE_LABELS:
            continue
        rows.append(
            {
                "question": _format_mmlu_question(q, choices),
                "answer": ans,
                "subject": str(ex.get("subject", default_subject)),
            }
        )
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def dedupe_rows(rows: Iterable[Dict[str, str]], seen_questions: Optional[Set[str]] = None) -> Tuple[List[Dict[str, str]], Set[str]]:
    seen = set(seen_questions or set())
    out: List[Dict[str, str]] = []
    for row in rows:
        key = str(row.get("question", "")).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out, seen


def balanced_sample(rows: List[Dict[str, str]], limit: int, seed: int) -> List[Dict[str, str]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)

    by_subject: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row.get("subject", ""))].append(row)

    rng = random.Random(seed)
    for bucket in by_subject.values():
        rng.shuffle(bucket)

    selected: List[Dict[str, str]] = []
    subject_keys = sorted(by_subject)
    while len(selected) < limit and subject_keys:
        progressed = False
        for subject in list(subject_keys):
            bucket = by_subject[subject]
            if not bucket:
                subject_keys.remove(subject)
                continue
            selected.append(bucket.pop())
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def write_jsonl(path: str, rows: List[Dict[str, str]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _hf_cache_dir() -> str:
    return os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")


def _dataset_cache_repo_dir(dataset_name: str) -> str:
    return os.path.join(_hf_cache_dir(), "hub", "datasets--" + dataset_name.replace("/", "--"))


def _find_dataset_infos_path(dataset_name: str) -> Optional[str]:
    repo_dir = _dataset_cache_repo_dir(dataset_name)
    snapshots_dir = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None

    candidates: List[str] = []
    for root, _, files in os.walk(snapshots_dir):
        if "dataset_infos.json" in files:
            candidates.append(os.path.join(root, "dataset_infos.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _subject_configs_from_cache(dataset_name: str, dataset_infos_path: str = "") -> List[str]:
    path = dataset_infos_path or _find_dataset_infos_path(dataset_name)
    if not path:
        raise FileNotFoundError(
            f"Could not find cached dataset_infos.json for {dataset_name}. "
            "Pass --dataset_infos_path or download the dataset metadata first."
        )

    with open(path, "r", encoding="utf-8") as f:
        infos = json.load(f)

    configs = sorted(k for k in infos.keys() if k not in {"all", "auxiliary_train"})
    if not configs:
        raise ValueError(f"No subject configs found in {path}")
    return configs


def _load_many_configs(load_dataset, args) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[str]]:
    if args.subject_configs == "auto":
        configs = _subject_configs_from_cache(args.dataset_name, args.dataset_infos_path)
    else:
        configs = [c.strip() for c in args.subject_configs.split(",") if c.strip()]

    train_rows: List[Dict[str, str]] = []
    val_rows: List[Dict[str, str]] = []
    test_rows: List[Dict[str, str]] = []
    for config in configs:
        ds = load_dataset(args.dataset_name, config)
        if args.train_split not in ds:
            raise ValueError(f"{config}: train_split={args.train_split} not found. available={list(ds.keys())}")
        if args.val_split and args.val_split not in ds:
            raise ValueError(f"{config}: val_split={args.val_split} not found. available={list(ds.keys())}")
        if not args.skip_test and args.test_split not in ds:
            raise ValueError(f"{config}: test_split={args.test_split} not found. available={list(ds.keys())}")

        train_rows.extend(convert_split(ds[args.train_split], -1, default_subject=config))
        if args.out_val and args.val_split:
            val_rows.extend(convert_split(ds[args.val_split], -1, default_subject=config))
        if not args.skip_test:
            test_rows.extend(convert_split(ds[args.test_split], -1, default_subject=config))
        print(
            f"Loaded {config}: train_total={len(train_rows)} "
            f"val_total={len(val_rows)} test_total={len(test_rows)}"
        )
    return train_rows, val_rows, test_rows, configs


def main():
    parser = argparse.ArgumentParser(description="Prepare MMLU jsonl files for this project.")
    parser.add_argument("--dataset_name", type=str, default="cais/mmlu")
    parser.add_argument("--dataset_config", type=str, default="all")
    parser.add_argument(
        "--subject_configs",
        type=str,
        default="",
        help="Comma-separated subject configs, or 'auto' to read all subject configs from cached dataset_infos.json.",
    )
    parser.add_argument(
        "--dataset_infos_path",
        type=str,
        default="",
        help="Optional path to a cached HuggingFace dataset_infos.json used by --subject_configs auto.",
    )
    parser.add_argument("--train_split", type=str, default="validation")
    parser.add_argument("--val_split", type=str, default="dev")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--out_train", type=str, default="mmlu_train.jsonl")
    parser.add_argument("--out_val", type=str, default="")
    parser.add_argument("--out_test", type=str, default="mmlu_test.jsonl")
    parser.add_argument("--train_limit", type=int, default=-1)
    parser.add_argument("--val_limit", type=int, default=-1)
    parser.add_argument("--test_limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced", type=int, default=1, choices=[0, 1])
    parser.add_argument("--skip_test", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    try:
        datasets_mod = importlib.import_module("datasets")
        load_dataset = datasets_mod.load_dataset
    except Exception as e:
        raise RuntimeError("Missing dependency 'datasets'. Please install requirements first.") from e

    loaded_configs = [args.dataset_config]
    if args.subject_configs:
        train_rows, val_rows, test_rows, loaded_configs = _load_many_configs(load_dataset, args)
    else:
        ds = load_dataset(args.dataset_name, args.dataset_config)
        if args.train_split not in ds:
            raise ValueError(f"train_split={args.train_split} not found. available={list(ds.keys())}")
        if args.val_split and args.val_split not in ds:
            raise ValueError(f"val_split={args.val_split} not found. available={list(ds.keys())}")
        if not args.skip_test and args.test_split not in ds:
            raise ValueError(f"test_split={args.test_split} not found. available={list(ds.keys())}")

        train_rows = convert_split(ds[args.train_split], -1)
        val_rows = convert_split(ds[args.val_split], -1) if args.out_val and args.val_split else []
        test_rows = convert_split(ds[args.test_split], args.test_limit) if not args.skip_test else []

    train_rows, seen = dedupe_rows(train_rows)
    val_rows, seen = dedupe_rows(val_rows, seen)

    if args.balanced:
        train_rows = balanced_sample(train_rows, args.train_limit, seed=args.seed)
        val_rows = balanced_sample(val_rows, args.val_limit, seed=args.seed + 1)
    else:
        if args.train_limit > 0:
            train_rows = train_rows[: args.train_limit]
        if args.val_limit > 0:
            val_rows = val_rows[: args.val_limit]

    if not args.skip_test:
        test_rows, _ = dedupe_rows(test_rows, seen)
        if args.test_limit > 0:
            test_rows = test_rows[: args.test_limit]

    write_jsonl(args.out_train, train_rows)
    if args.out_val:
        write_jsonl(args.out_val, val_rows)
    if not args.skip_test:
        write_jsonl(args.out_test, test_rows)

    print(
        f"Done. dataset={args.dataset_name}; configs={len(loaded_configs)}; "
        f"train={len(train_rows)} -> {args.out_train}; "
        f"val={len(val_rows)} -> {args.out_val or '<not_written>'}; "
        f"test={len(test_rows)} -> {args.out_test if not args.skip_test else '<skipped>'}"
    )


if __name__ == "__main__":
    main()
