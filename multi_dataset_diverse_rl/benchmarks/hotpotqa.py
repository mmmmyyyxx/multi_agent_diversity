"""HotpotQA answer-only metric component; project task/split remain unselected."""

from __future__ import annotations

import re
import string

from ..search.benchmark import BenchmarkInput
from ..search.schemas import BenchmarkCapabilities, ParsedOutput


_ARTICLE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_ANSWER = re.compile(r"^FINAL_ANSWER:\s*(\S.*)$")


def normalize_answer(text: str) -> str:
    """Official HotpotQA answer normalization: lowercase, punctuation, articles, spaces."""
    lowered = text.lower()
    no_punctuation = "".join(ch for ch in lowered if ch not in string.punctuation)
    return " ".join(_ARTICLE.sub(" ", no_punctuation).split())


class HotpotQAAnswerAdapter:
    """Answer-only EM/F1; not an assertion that this project chose that task."""

    capabilities = BenchmarkCapabilities(True, False, True)
    benchmark_id = "hotpotqa"
    preferred_aggregation = "plurality"
    output_contract = "Return exactly one final line: FINAL_ANSWER: <short answer>"

    def format_input(self, item: BenchmarkInput) -> str:
        return item.problem

    def parse_member_output(self, raw: str, item: BenchmarkInput) -> ParsedOutput:
        if item.benchmark_id != "hotpotqa":
            return ParsedOutput("", False)
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        matches = [_ANSWER.fullmatch(line) for line in lines
                   if line.startswith("FINAL_ANSWER:")]
        if (len(matches) != 1 or matches[0] is None
                or not lines or not lines[-1].startswith("FINAL_ANSWER:")):
            return ParsedOutput("", False)
        answer = normalize_answer(matches[0].group(1))
        return ParsedOutput(answer, bool(answer))

    parse_output = parse_member_output

    def score_member_output(self, parsed: ParsedOutput, gold: object) -> float:
        return float(parsed.valid and parsed.answer == normalize_answer(str(gold)))

    def answer_f1(self, parsed: ParsedOutput, gold: object) -> float:
        if not parsed.valid:
            return 0.0
        normalized_gold = normalize_answer(str(gold))
        special = {"yes", "no", "noanswer"}
        if ((parsed.answer in special or normalized_gold in special)
                and parsed.answer != normalized_gold):
            return 0.0
        prediction, truth = parsed.answer.split(), normalized_gold.split()
        if not prediction or not truth:
            return float(prediction == truth)
        from collections import Counter
        common = sum((Counter(prediction) & Counter(truth)).values())
        return 2 * common / (len(prediction) + len(truth)) if common else 0.0

    def build_task_feedback(self, parsed: ParsedOutput, gold: object) -> str:
        return "correct" if self.score_member_output(parsed, gold) else "incorrect"
