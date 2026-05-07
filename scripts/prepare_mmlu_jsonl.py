import argparse
import importlib
import json
from typing import Any, Dict, List, Optional


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


def convert_split(ds_split, limit: int = -1) -> List[Dict[str, str]]:
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
                "subject": str(ex.get("subject", "")),
            }
        )
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def write_jsonl(path: str, rows: List[Dict[str, str]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare MMLU jsonl files for this project.")
    parser.add_argument("--dataset_name", type=str, default="cais/mmlu")
    parser.add_argument("--dataset_config", type=str, default="all")
    parser.add_argument("--train_split", type=str, default="dev")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--out_train", type=str, default="mmlu_train.jsonl")
    parser.add_argument("--out_test", type=str, default="mmlu_test.jsonl")
    parser.add_argument("--train_limit", type=int, default=-1)
    parser.add_argument("--test_limit", type=int, default=-1)
    args = parser.parse_args()

    try:
        datasets_mod = importlib.import_module("datasets")
        load_dataset = datasets_mod.load_dataset
    except Exception as e:
        raise RuntimeError("Missing dependency 'datasets'. Please install requirements first.") from e

    ds = load_dataset(args.dataset_name, args.dataset_config)
    if args.train_split not in ds:
        raise ValueError(f"train_split={args.train_split} not found. available={list(ds.keys())}")
    if args.test_split not in ds:
        raise ValueError(f"test_split={args.test_split} not found. available={list(ds.keys())}")

    train_rows = convert_split(ds[args.train_split], args.train_limit)
    test_rows = convert_split(ds[args.test_split], args.test_limit)

    write_jsonl(args.out_train, train_rows)
    write_jsonl(args.out_test, test_rows)

    print(
        f"Done. dataset={args.dataset_name}/{args.dataset_config}; "
        f"train={len(train_rows)} -> {args.out_train}; "
        f"test={len(test_rows)} -> {args.out_test}"
    )


if __name__ == "__main__":
    main()
