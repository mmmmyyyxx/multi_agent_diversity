import argparse
import importlib
import json
import random
import os
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


def _format_mmlu_question(question: str, choices: List[str]) -> str:
    lines = [f"Question: {question}", "", "Options:"]
    for i, choice in enumerate(choices[:4]):
        lines.append(f"{CHOICE_LABELS[i]}. {choice}")
    lines.append("")
    lines.append("Select the best option and output FINAL_ANSWER: <A/B/C/D>.")
    return "\n".join(lines)


def _convert_mmlu_row(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    question = str(row.get("question", "")).strip()
    choices = row.get("choices")
    answer = _to_choice_letter(row.get("answer", ""))
    if not question or not isinstance(choices, list) or len(choices) < 4 or answer not in CHOICE_LABELS:
        return None
    return {
        "question": _format_mmlu_question(question, [str(c) for c in choices[:4]]),
        "answer": answer,
        "subject": str(row.get("subject", "")),
    }


def load_rows(path: str) -> List[Dict[str, str]]:
    if path.lower().endswith(".arrow"):
        datasets_mod = importlib.import_module("datasets")
        ds = datasets_mod.Dataset.from_file(path)
        rows: List[Dict[str, str]] = []
        for row in ds:
            converted = _convert_mmlu_row(row)
            if converted:
                rows.append(converted)
        return rows

    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dedupe_rows(rows: Iterable[Dict[str, str]], seen: Set[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        key = str(row.get("question", "")).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def balanced_sample(rows: List[Dict[str, str]], size: int, seed: int) -> List[Dict[str, str]]:
    if size <= 0 or len(rows) <= size:
        return list(rows)

    by_subject: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row.get("subject", ""))].append(row)

    rng = random.Random(seed)
    for bucket in by_subject.values():
        rng.shuffle(bucket)

    selected: List[Dict[str, str]] = []
    subjects = sorted(by_subject)
    while len(selected) < size and subjects:
        progressed = False
        for subject in list(subjects):
            bucket = by_subject[subject]
            if not bucket:
                subjects.remove(subject)
                continue
            selected.append(bucket.pop())
            progressed = True
            if len(selected) >= size:
                break
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def describe(rows: List[Dict[str, str]]) -> Tuple[int, int, Dict[str, int]]:
    subjects: Dict[str, int] = defaultdict(int)
    for row in rows:
        subjects[str(row.get("subject", ""))] += 1
    return len(rows), len(subjects), dict(sorted(subjects.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample compact, disjoint, subject-balanced JSONL train/val/test splits.")
    parser.add_argument("--train_source", type=str, required=True)
    parser.add_argument("--val_source", type=str, required=True)
    parser.add_argument("--test_source", type=str, required=True)
    parser.add_argument("--out_train", type=str, default="mmlu_train_small.jsonl")
    parser.add_argument("--out_val", type=str, default="mmlu_val_small.jsonl")
    parser.add_argument("--out_test", type=str, default="mmlu_test_small.jsonl")
    parser.add_argument("--train_size", type=int, default=200)
    parser.add_argument("--val_size", type=int, default=150)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seen: Set[str] = set()
    for source in [args.train_source, args.val_source, args.test_source]:
        if not os.path.exists(source):
            raise FileNotFoundError(source)

    train_rows = balanced_sample(dedupe_rows(load_rows(args.train_source), seen), args.train_size, args.seed)
    val_rows = balanced_sample(dedupe_rows(load_rows(args.val_source), seen), args.val_size, args.seed + 1)
    test_rows = balanced_sample(dedupe_rows(load_rows(args.test_source), seen), args.test_size, args.seed + 2)

    write_jsonl(args.out_train, train_rows)
    write_jsonl(args.out_val, val_rows)
    write_jsonl(args.out_test, test_rows)

    for name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        count, subject_count, subjects = describe(rows)
        print(f"{name}: rows={count} subjects={subject_count} -> {getattr(args, 'out_' + name)}")
        print(f"{name}_subjects={subjects}")


if __name__ == "__main__":
    main()
