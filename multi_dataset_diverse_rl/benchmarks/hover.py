"""GEPA HoVer retrieval coverage contract; no supported/refuted verdict task."""
from dataclasses import dataclass
import json
import unicodedata

from ..search.benchmark import BenchmarkInput
from ..search.schemas import BenchmarkCapabilities, ParsedOutput
from .hotpotqa import normalize_answer


@dataclass(frozen=True)
class RetrievedEvidence:
    titles: tuple[str, ...]
    passages: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceCoverage:
    normalized_gold_titles: tuple[str, ...]
    normalized_predicted_titles: tuple[str, ...]
    missing_gold_titles: tuple[str, ...]
    coverage_success: bool


def normalize_title(title: str) -> str:
    # DSPy's normalize_text used by the pinned GEPA metric follows SQuAD
    # lowercase -> punctuation -> articles -> whitespace normalization.
    return normalize_answer(unicodedata.normalize("NFD", title))


class HoVerBenchmarkAdapter:
    benchmark_id = "hover"
    preferred_aggregation = "llm_evidence"
    capabilities = BenchmarkCapabilities(False, False, True, False, False)
    output_contract = 'Return JSON {"titles": ["document title", ...]}; optional "passages" list. No claim verdict.'

    def format_input(self, item: BenchmarkInput) -> str:
        return item.problem

    def evidence(self, parsed: ParsedOutput) -> RetrievedEvidence:
        row = json.loads(parsed.answer)
        return RetrievedEvidence(tuple(row["titles"]), tuple(row.get("passages", ())))

    def parse_member_output(self, raw: str, item: BenchmarkInput) -> ParsedOutput:
        if item.benchmark_id != self.benchmark_id:
            return ParsedOutput("", False)
        try:
            row = json.loads(raw)
            if not isinstance(row, dict) or set(row) - {"titles", "passages"}:
                return ParsedOutput("", False)
            titles, passages = row.get("titles"), row.get("passages", [])
            if (not isinstance(titles, list) or not titles or not isinstance(passages, list)
                    or not all(isinstance(s, str) and bool(normalize_title(s.split(" | ")[0])) for s in titles)
                    or not all(isinstance(s, str) for s in passages)):
                return ParsedOutput("", False)
            # Document title is the portion preceding the upstream passage separator.
            normalized = sorted({normalize_title(s.split(" | ")[0]) for s in titles})
            return ParsedOutput(json.dumps({"titles": normalized, "passages": passages},
                                            ensure_ascii=False, sort_keys=True), True)
        except (ValueError, TypeError):
            return ParsedOutput("", False)

    parse_output = parse_member_output

    def evaluate(self, parsed: ParsedOutput, gold: RetrievedEvidence) -> EvidenceCoverage:
        expected = tuple(sorted({normalize_title(s) for s in gold.titles}))
        predicted = self.evidence(parsed).titles if parsed.valid else ()
        missing = tuple(sorted(set(expected) - set(predicted)))
        return EvidenceCoverage(expected, predicted, missing, bool(parsed.valid and not missing))

    def score_member_output(self, parsed: ParsedOutput, gold: RetrievedEvidence) -> float:
        return float(self.evaluate(parsed, gold).coverage_success)

    def build_task_feedback(self, parsed: ParsedOutput, gold: RetrievedEvidence) -> str:
        return json.dumps(self.evaluate(parsed, gold).__dict__, sort_keys=True)
