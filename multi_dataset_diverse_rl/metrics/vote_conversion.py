"""Pure Oracle-to-Vote conversion diagnostics built from canonical vote results."""

from typing import Any, Dict, Mapping, Sequence


def question_vote_conversion_diagnostics(row: Mapping[str, Any]) -> Dict[str, Any]:
    individual_correct = [int(value) for value in row.get("individual_correct", [])]
    correct_agent_count = sum(individual_correct)
    vote_counts = row.get("vote_counts", {})
    vote_counts = vote_counts if isinstance(vote_counts, Mapping) else {}
    counts = [int(value) for value in vote_counts.values()]
    top_count = max(counts, default=0)
    top_tie_size = sum(int(value == top_count) for value in counts) if top_count else 0
    gold_vote_count = int(row.get("gold_vote_count", 0) or 0)
    max_wrong_vote_count = int(
        row.get("max_wrong_vote_count", row.get("largest_wrong_vote_count", 0)) or 0
    )
    oracle_correct = int(correct_agent_count >= 1)
    vote_correct = int(row.get("vote_correct", 0) or 0)
    gold_in_top_tie = bool(
        gold_vote_count > 0 and gold_vote_count == top_count and top_tie_size > 1
    )
    normalization_anomaly = bool(
        vote_correct > oracle_correct or (correct_agent_count >= 3 and not vote_correct)
    )
    return {
        "correct_agent_count": int(correct_agent_count),
        "gold_vote_count": gold_vote_count,
        "max_wrong_vote_count": max_wrong_vote_count,
        "gold_plurality_margin": int(gold_vote_count - max_wrong_vote_count),
        "oracle_correct": oracle_correct,
        "vote_correct": vote_correct,
        "gold_in_top_tie": gold_in_top_tie,
        "top_tie_size": int(top_tie_size),
        "invalid_agent_count": sum(int(value) for value in row.get("invalid_flags", [])),
        "vote_normalization_anomaly": normalization_anomaly,
    }


def summarize_vote_conversion(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    diagnostics = [question_vote_conversion_diagnostics(row) for row in rows]
    question_count = len(diagnostics)
    oracle_correct_count = sum(row["oracle_correct"] for row in diagnostics)
    vote_correct_count = sum(row["vote_correct"] for row in diagnostics)
    oracle_vote_gap_count = oracle_correct_count - vote_correct_count
    depths = [row["correct_agent_count"] for row in diagnostics]
    gap_rows = [
        (source, diagnostic)
        for source, diagnostic in zip(rows, diagnostics)
        if diagnostic["oracle_correct"] and not diagnostic["vote_correct"]
    ]
    tie_rows = [row for row in diagnostics if row["gold_in_top_tie"]]
    anomaly_hashes = [
        str(source.get("question_hash", ""))
        for source, diagnostic in zip(rows, diagnostics)
        if diagnostic["vote_normalization_anomaly"]
    ]

    def mean(values: Sequence[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    result = {
        "oracle_correct_count": int(oracle_correct_count),
        "vote_correct_count": int(vote_correct_count),
        "oracle_vote_gap_count": int(oracle_vote_gap_count),
        "oracle_to_vote_conversion_rate": (
            float(vote_correct_count / oracle_correct_count) if oracle_correct_count else 0.0
        ),
        "c0_count": int(sum(depth == 0 for depth in depths)),
        "c1_count": int(sum(depth == 1 for depth in depths)),
        "c2_count": int(sum(depth == 2 for depth in depths)),
        "c3plus_count": int(sum(depth >= 3 for depth in depths)),
        "c1_vote_correct_count": int(sum(row["correct_agent_count"] == 1 and row["vote_correct"] for row in diagnostics)),
        "c1_vote_fail_count": int(sum(row["correct_agent_count"] == 1 and not row["vote_correct"] for row in diagnostics)),
        "c2_vote_correct_count": int(sum(row["correct_agent_count"] == 2 and row["vote_correct"] for row in diagnostics)),
        "c2_vote_fail_count": int(sum(row["correct_agent_count"] == 2 and not row["vote_correct"] for row in diagnostics)),
        "c3plus_vote_fail_count": int(sum(row["correct_agent_count"] >= 3 and not row["vote_correct"] for row in diagnostics)),
        "gold_top_tie_count": int(len(tie_rows)),
        "gold_top_tie_win_count": int(sum(row["vote_correct"] for row in tie_rows)),
        "gold_top_tie_loss_count": int(sum(not row["vote_correct"] for row in tie_rows)),
        "mean_gold_plurality_margin": mean([row["gold_plurality_margin"] for row in diagnostics]),
        "mean_gold_margin_on_oracle_vote_gap": mean([
            diagnostic["gold_plurality_margin"] for _, diagnostic in gap_rows
        ]),
        "mean_max_wrong_vote_on_oracle_vote_gap": mean([
            diagnostic["max_wrong_vote_count"] for _, diagnostic in gap_rows
        ]),
        "dominant_wrong_concentration": mean([
            diagnostic["max_wrong_vote_count"]
            / max(1, sum(int(value) for value in source.get("vote_counts", {}).values()) - diagnostic["gold_vote_count"])
            for source, diagnostic in gap_rows
        ]),
        "vote_normalization_anomaly_count": int(len(anomaly_hashes)),
        "vote_normalization_anomaly_question_hashes": anomaly_hashes,
    }
    assert result["c0_count"] + result["c1_count"] + result["c2_count"] + result["c3plus_count"] == question_count
    assert result["oracle_correct_count"] == result["c1_count"] + result["c2_count"] + result["c3plus_count"]
    assert result["vote_correct_count"] <= result["oracle_correct_count"]
    assert result["oracle_vote_gap_count"] == result["oracle_correct_count"] - result["vote_correct_count"]
    return result
