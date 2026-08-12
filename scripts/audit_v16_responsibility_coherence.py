from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_SPEC = (
    ROOT
    / "runs"
    / "v16_responsibility_coherence_generic_m20_prep"
    / "analysis_spec.json"
)
DEFAULT_OUT = ROOT / "runs" / "v16_responsibility_coherence_audit"
REQUIRED_SEEDS = (48, 49, 50, 51)
REQUIRED_SETTING_BY_SEED = {
    48: "shared_responsibility_conditioned_dual_target",
    49: "shared_responsibility_conditioned_dual_target",
    50: "shared_responsibility_conditioned_dual_target",
    51: "experimental_v16_c0_current_v15",
}
EXPECTED_TRAJECTORY_ROOTS = {
    "48": "runs/v15f48/disambiguation_qa/shared_responsibility_conditioned_dual_target_seed48",
    "49": "runs/v15f49/disambiguation_qa/shared_responsibility_conditioned_dual_target_seed49",
    "50": "runs/v15f50/disambiguation_qa/shared_responsibility_conditioned_dual_target_seed50",
    "51": "runs/v16p51/disambiguation_qa/experimental_v16_c0_current_v15_seed51",
}
ALLOWED_INPUT_FILES = frozenset(
    {
        "run_meta.json",
        "peer_state_history.jsonl",
        "responsibility_assignments.jsonl",
        "tcs_context_history.jsonl",
        "candidate_decisions.jsonl",
        "dual_target_branch_decisions.jsonl",
        "dual_target_commit_decisions.jsonl",
        "frozen_initialization_match.json",
        "_solver_cache.sqlite",
    }
)
FORBIDDEN_PATH_TOKENS = (
    "/test/",
    "\\test\\",
    "/val/",
    "\\val\\",
    "final_test",
    "validation",
)
PUBLISHABLE_SECRET_TOKENS = (
    "question",
    "gold_answer",
    "team_answers",
    "model_answer",
    "raw_response",
    "raw_prompt",
    "endpoint",
    "sqlite",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    names = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        if names:
            writer.writeheader()
            writer.writerows(rows)


def validate_analysis_spec(spec: Mapping[str, Any]) -> None:
    expected = {
        "spec_version": "v16_responsibility_coherence_analysis_v1",
        "frozen_before_candidate_outcome_analysis": True,
        "historical_seeds": list(REQUIRED_SEEDS),
        "trajectory_roots": EXPECTED_TRAJECTORY_ROOTS,
        "allowed_split": "train_optimization_probe_only",
        "primary_unit": "historical_m20_target_branch",
        "primary_portfolio": "visible_module2_evidence",
        "secondary_portfolio": (
            "full_module1_service_portfolio_when_exactly_reconstructible"
        ),
        "coarse_signature_fields": ["failure_class", "repair_distance_bin"],
        "fine_signature_fields": [
            "failure_class",
            "repair_distance_bin",
            "target_wrong_cluster_role",
            "target_wrong_cluster_size_bin",
        ],
        "repair_distance": "exact_subset_enumeration_through_real_aggregator",
        "repair_distance_bins": {"1": [1], "2": [2], "3+": [3, 4, 5]},
        "coherence_bins": {
            "HIGH": {"min_inclusive": 0.75},
            "MEDIUM": {"min_inclusive": 0.5, "max_exclusive": 0.75},
            "LOW": {"max_exclusive": 0.5},
        },
        "entropy_bins": {
            "LOW": {"max_inclusive": 0.33},
            "MEDIUM": {"min_exclusive": 0.33, "max_inclusive": 0.67},
            "HIGH": {"min_exclusive": 0.67},
        },
        "branch_outcomes": [
            "any_valid_candidate",
            "any_repair_gain_candidate",
            "any_common_safe_feasible_candidate",
            "any_F_candidate",
            "any_target_regression_candidate",
            "best_feasible_candidate_count",
        ],
        "candidate_focus_strata": [
            "all_repair_gain",
            "common_safe_feasible_repair_gain",
            "committed_repair_gain",
        ],
        "coherence_classifier": {
            "outcome_association": (
                "any_available_predeclared_directional_check_strictly_favorable"
            ),
            "directional_checks": [
                "spearman_cluster_share_feasible_gt_0",
                "spearman_entropy_F_gt_0",
                "spearman_entropy_target_regression_gt_0",
                "high_coherence_feasible_rate_gt_low",
                "high_entropy_F_rate_gt_low",
                "high_entropy_regression_rate_gt_low",
            ],
            "repair_concentration_min_candidates": 2,
            "repair_focus_mean_min": 0.75,
            "single_fine_signature_share_min": 0.5,
            "SUPPORTED": "outcome_association_and_repair_concentration",
            "PARTIAL": "exactly_one",
            "NOT_SUPPORTED": "neither",
        },
        "forbidden_path_tokens": list(FORBIDDEN_PATH_TOKENS),
        "no_p_values": True,
        "api_calls": 0,
        "validation_calls": 0,
        "test_calls": 0,
    }
    mismatches = [
        key for key, value in expected.items() if spec.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "analysis spec differs from frozen Study-A definitions: "
            + ", ".join(mismatches)
        )


def _path_key(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").lower()


class TrainOnlyReader:
    """Narrow, audited reader for the four preregistered train trajectories."""

    def __init__(self, roots: Mapping[int, Path]):
        self.roots = {int(seed): path.resolve() for seed, path in roots.items()}
        if tuple(sorted(self.roots)) != REQUIRED_SEEDS:
            raise ValueError("exactly Seeds48-51 are required")
        self.files_read: list[Path] = []
        self.test_files_read = 0
        self.validation_files_read = 0

    def _authorize(self, seed: int, name: str) -> Path:
        if int(seed) not in self.roots:
            raise ValueError(f"unregistered historical seed: {seed}")
        lowered_name = str(name).replace("\\", "/").lower()
        if any(
            token.replace("\\", "/").strip("/") in lowered_name
            for token in FORBIDDEN_PATH_TOKENS
        ):
            if "test" in lowered_name:
                self.test_files_read += 1
            if "val" in lowered_name:
                self.validation_files_read += 1
            raise PermissionError(f"test/validation access is prohibited: {name}")
        if name not in ALLOWED_INPUT_FILES:
            raise PermissionError(f"historical artifact is not allowlisted: {name}")
        path = (self.roots[int(seed)] / name).resolve()
        if path.parent != self.roots[int(seed)]:
            raise PermissionError("historical artifact escaped its registered train root")
        key = _path_key(path)
        if any(token.replace("\\", "/") in key for token in FORBIDDEN_PATH_TOKENS):
            if "test" in key:
                self.test_files_read += 1
            if "val" in key:
                self.validation_files_read += 1
            raise PermissionError(f"test/validation access is prohibited: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        self.files_read.append(path)
        return path

    def json(self, seed: int, name: str) -> Any:
        return read_json(self._authorize(seed, name))

    def jsonl(self, seed: int, name: str) -> list[dict[str, Any]]:
        return read_jsonl(self._authorize(seed, name))

    def candidate_answers(
        self,
        seed: int,
        prompt_hashes: set[str],
        *,
        namespace: Mapping[str, Any],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        path = self._authorize(seed, "_solver_cache.sqlite")
        uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
        answers: dict[tuple[str, str], dict[str, Any]] = {}
        connection = sqlite3.connect(uri, uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            schema_rows = connection.execute(
                "SELECT key, value FROM cache_metadata ORDER BY key"
            ).fetchall()
            if schema_rows != [("schema_version", "shared_solver_cache_v1")]:
                raise ValueError("historical solver cache metadata mismatch")
            endpoint_identities: set[str] = set()
            for prompt_hash in sorted(prompt_hashes):
                rows = connection.execute(
                    "SELECT cache_key, prompt_hash, question_hash, answer_json, "
                    "endpoint_identity FROM solver_cache "
                    "WHERE state = 'ready' AND prompt_hash = ? "
                    "AND model_request_identity = ? AND parser_version = ? "
                    "AND temperature = ? AND evaluation_replica_seed = ? "
                    "AND solver_model = ? AND max_tokens = ? "
                    "AND output_contract_version = ? ORDER BY cache_key",
                    (
                        prompt_hash,
                        str(namespace["model_request_identity"]),
                        str(namespace["parser_version"]),
                        float(namespace["temperature"]),
                        int(namespace["evaluation_replica_seed"]),
                        str(namespace["solver_model"]),
                        int(namespace["max_tokens"]),
                        str(namespace["output_contract_version"]),
                    ),
                ).fetchall()
                for _, row_prompt, question_hash, raw, endpoint_identity in rows:
                    key = (str(row_prompt), str(question_hash))
                    if key in answers:
                        raise ValueError(
                            "duplicate solver observations in the frozen cache namespace"
                        )
                    parsed = json.loads(str(raw))
                    if (
                        not isinstance(parsed, dict)
                        or str(parsed.get("request_identity", ""))
                        != str(namespace["model_request_identity"])
                    ):
                        raise ValueError("cached answer request identity mismatch")
                    answers[key] = parsed
                    endpoint_identities.add(str(endpoint_identity))
            if answers and (
                len(endpoint_identities) != 1
                or not next(iter(endpoint_identities))
            ):
                raise ValueError(
                    "frozen cache namespace has ambiguous endpoint identity"
                )
        finally:
            connection.close()
        return answers


def cache_namespace(
    *, seed: int, meta: Mapping[str, Any], frozen: Mapping[str, Any]
) -> dict[str, Any]:
    if not bool(frozen.get("matched")):
        raise ValueError("historical frozen initialization did not match")
    snapshot = frozen.get("initialization_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("historical initialization snapshot is missing")
    frozen_identity = tuple(snapshot.get("solver_identity", ()))
    meta_identity = tuple(meta.get("prompt_question_evaluator_identity", ()))
    if len(frozen_identity) != 5 or frozen_identity != meta_identity:
        raise ValueError("historical solver identity provenance mismatch")
    version, request_identity, parser, temperature, replica_seed = frozen_identity
    if (
        str(snapshot.get("solver_request_identity", ""))
        != str(request_identity)
        or int(replica_seed) != int(seed)
    ):
        raise ValueError("historical solver request namespace mismatch")
    config = meta.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("historical run config is missing")
    output_contract = str(meta.get("solver_output_contract_version", ""))
    if output_contract != str(config.get("solver_output_contract_version", "")):
        raise ValueError("historical solver output contract mismatch")
    namespace = {
        "evaluator_version": str(version),
        "model_request_identity": str(request_identity),
        "parser_version": str(parser),
        "temperature": float(temperature),
        "evaluation_replica_seed": int(replica_seed),
        "solver_model": str(config.get("agent_model", "")),
        "max_tokens": int(config.get("solver_max_tokens", 0)),
        "output_contract_version": output_contract,
    }
    if (
        namespace["evaluator_version"]
        != str(meta.get("prompt_question_evaluator_version", ""))
        or not namespace["model_request_identity"]
        or not namespace["parser_version"]
        or not namespace["solver_model"]
        or namespace["max_tokens"] <= 0
        or not namespace["output_contract_version"]
    ):
        raise ValueError("historical solver cache namespace is incomplete")
    return namespace


def repair_distance_bin(value: int) -> str:
    if value <= 0:
        raise ValueError("responsibility residual repair distance must be positive")
    return str(value) if value <= 2 else "3+"


def count_bin(value: int) -> str:
    if value <= 0:
        raise ValueError("wrong-cluster size must be positive")
    return str(value) if value <= 2 else "3+"


def exact_repair_distance(
    state: Any,
    *,
    normalize_answer: Any,
    match_answer: Any,
    seed: int,
) -> int:
    from multi_dataset_diverse_rl.module2_context import exact_repair_distance as real

    return real(
        state,
        normalize_answer=normalize_answer,
        match_answer=match_answer,
        tie_break="abstain",
        seed=int(seed),
    )


def residual_signature(
    row: Mapping[str, Any],
    *,
    target_agent_id: int,
    seed: int,
) -> dict[str, Any]:
    from multi_dataset_diverse_rl.peer_state import build_team_vote_state
    from multi_dataset_diverse_rl.tasks import match_bbh_answer, normalize_bbh_answer

    state = build_team_vote_state(
        question_hash=str(row["question_hash"]),
        gold_answer=str(row["gold_answer"]),
        answers=[str(value) for value in row["team_answers"]],
        valid_vector=[bool(value) for value in row["team_validity"]],
        normalize_answer=normalize_bbh_answer,
        match_answer=match_bbh_answer,
        tie_break="abstain",
        seed=int(seed),
    )
    recorded_checks = (
        tuple(state.team_correctness)
        == tuple(bool(value) for value in row["team_correctness"]),
        state.gold_vote_count == int(row["gold_vote_count"]),
        state.largest_wrong_vote_count == int(row["largest_wrong_vote_count"]),
        state.plurality_margin == int(row["plurality_margin"]),
        state.vote_correct == bool(row["vote_correct"]),
    )
    if not all(recorded_checks):
        raise AssertionError("repository BBH plurality replay differs from train state")
    target = int(target_agent_id)
    if state.vote_correct or state.team_correctness[target]:
        raise ValueError("responsibility set includes an ineligible residual")
    target_answer = state.team_answers[target]
    wrong_counts = dict(state.wrong_vote_histogram)
    cluster_size = int(wrong_counts.get(target_answer, 0))
    if cluster_size <= 0:
        raise ValueError("wrong target must belong to a valid wrong-answer cluster")
    distance = exact_repair_distance(
        state,
        normalize_answer=normalize_bbh_answer,
        match_answer=match_bbh_answer,
        seed=seed,
    )
    failure_class = "coverage" if state.gold_vote_count == 0 else "conversion"
    role = (
        "dominant"
        if target_answer in state.dominant_wrong_answers
        else "non_dominant"
    )
    coarse = (failure_class, repair_distance_bin(distance))
    fine = coarse + (role, count_bin(cluster_size))
    return {
        "question_hash": state.question_hash,
        "G": int(state.gold_vote_count),
        "H": int(state.largest_wrong_vote_count),
        "M": int(state.plurality_margin),
        "failure_class": failure_class,
        "repair_distance": int(distance),
        "repair_distance_bin": repair_distance_bin(distance),
        "target_wrong_cluster_role": role,
        "target_wrong_cluster_size": cluster_size,
        "target_wrong_cluster_size_bin": count_bin(cluster_size),
        "coarse_signature": coarse,
        "fine_signature": fine,
    }


def normalized_entropy(counts: Iterable[int]) -> float:
    values = [int(value) for value in counts if int(value) > 0]
    if len(values) <= 1:
        return 0.0
    total = sum(values)
    entropy = -sum((value / total) * math.log(value / total) for value in values)
    return entropy / math.log(len(values))


def mean_pairwise_signature_distance(signatures: Sequence[Sequence[str]]) -> float:
    if len(signatures) <= 1:
        return 0.0
    pairs = list(itertools.combinations(signatures, 2))
    return statistics.mean(
        sum(left[index] != right[index] for index in range(4)) / 4
        for left, right in pairs
    )


def coherence_metrics(signatures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not signatures:
        raise ValueError("coherence metrics require a nonempty responsibility set")
    coarse = Counter(tuple(row["coarse_signature"]) for row in signatures)
    fine = Counter(tuple(row["fine_signature"]) for row in signatures)
    size = len(signatures)
    coarse_share = max(coarse.values()) / size
    fine_share = max(fine.values()) / size
    return {
        "portfolio_size": size,
        "coarse_unique_signature_count": len(coarse),
        "fine_unique_signature_count": len(fine),
        "coarse_largest_cluster_share": coarse_share,
        "fine_largest_cluster_share": fine_share,
        "coarse_entropy_norm": normalized_entropy(coarse.values()),
        "fine_entropy_norm": normalized_entropy(fine.values()),
        "mean_pairwise_signature_distance": mean_pairwise_signature_distance(
            [tuple(row["fine_signature"]) for row in signatures]
        ),
        "fine_coherence_bin": coherence_bin(fine_share),
        "fine_entropy_bin": entropy_bin(normalized_entropy(fine.values())),
    }


def coherence_bin(share: float) -> str:
    if share >= 0.75:
        return "HIGH"
    if share >= 0.50:
        return "MEDIUM"
    return "LOW"


def entropy_bin(value: float) -> str:
    if value <= 0.33:
        return "LOW"
    if value <= 0.67:
        return "MEDIUM"
    return "HIGH"


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = ((start + 1) + stop) / 2
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = _average_ranks([float(value) for value in left])
    y = _average_ranks([float(value) for value in right])
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in x)
        * sum((b - mean_y) ** 2 for b in y)
    )
    return numerator / denominator if denominator else None


def visible_repair_hashes(
    context: Mapping[str, Any], *, update_index: int, target_agent_id: int
) -> set[str]:
    if (
        int(context.get("update_index", -1)) != int(update_index)
        or int(context.get("target_agent_id", -1)) != int(target_agent_id)
    ):
        raise ValueError("M20 context update/target provenance mismatch")
    if (
        context.get("context_type") != "SingleLaneDiagnosisContext"
        or context.get("context_class") != "SingleLaneDiagnosisContext"
        or context.get("context_mode")
        != "member_aware_responsibility_conditioned"
        or context.get("diagnosis_aggregation_version")
        != "single_lane_pattern_aggregation_v1"
        or int(context.get("selected_pattern_count", 0)) != 1
    ):
        raise ValueError("historical context is not current-v15 single-lane M20")
    selected_ids = tuple(map(str, context.get("selected_pattern_ids", ())))
    payload = context.get("selected_context_pattern_question_hashes")
    if len(selected_ids) != 1 or not isinstance(payload, Mapping):
        raise ValueError("M20 context lacks one selected dominant pattern")
    if set(map(str, payload)) != {selected_ids[0]}:
        raise ValueError("visible evidence includes a non-dominant pattern")
    hashes = payload[selected_ids[0]]
    if not isinstance(hashes, list) or not hashes:
        raise ValueError("M20 dominant pattern has no visible repair hashes")
    result = set(map(str, hashes))
    if len(result) != len(hashes):
        raise ValueError("M20 visible repair hashes contain duplicates")
    return result


def _state_blocks(rows: Sequence[dict[str, Any]], size: int = 75) -> list[list[dict[str, Any]]]:
    if len(rows) % size:
        raise ValueError("peer-state history does not contain complete train states")
    return [list(rows[index : index + size]) for index in range(0, len(rows), size)]


def _full_service_hashes(
    snapshot: Mapping[str, Any], target_agent_id: int
) -> set[str]:
    target = int(target_agent_id)
    return {
        str(question_hash)
        for question_hash, row in dict(
            snapshot.get("service_assignment_by_question", {})
        ).items()
        if int(row.get("service_agent_id", -1)) == target
    }


def active_service_hashes(
    snapshot: Mapping[str, Any], *, target_agent_id: int, active_lane: str
) -> set[str]:
    target = int(target_agent_id)
    lane = str(active_lane)
    if not lane:
        raise ValueError("historical branch has no active lane")
    return {
        str(question_hash)
        for question_hash, row in dict(
            snapshot.get("service_assignment_by_question", {})
        ).items()
        if int(row.get("service_agent_id", -1)) == target
        and str(row.get("repair_lane", "")) == lane
    }


def exact_branch_active_hashes(
    snapshot: Mapping[str, Any], branch: Mapping[str, Any], *, target_agent_id: int
) -> set[str]:
    if "assigned_question_hashes" not in branch or "active_lane" not in branch:
        raise ValueError("historical branch lacks active-slice fields")
    recorded = set(map(str, branch["assigned_question_hashes"]))
    reconstructed = active_service_hashes(
        snapshot,
        target_agent_id=target_agent_id,
        active_lane=str(branch["active_lane"]),
    )
    if reconstructed != recorded:
        raise AssertionError(
            "branch assigned hashes differ from exact lane-filtered service slice"
        )
    return reconstructed


def validate_branch_provenance(
    *,
    branch_audit: Mapping[str, Any],
    decision: Mapping[str, Any],
    branch: Mapping[str, Any],
    update_index: int,
    target_agent_id: int,
    team_state_version: int,
) -> None:
    required = {
        "update_index",
        "target_agent_id",
        "team_state_version",
        "parent_team_hash",
        "active_lane",
    }
    if not required.issubset(branch_audit):
        raise ValueError("branch audit lacks required provenance fields")
    if "parent_team_hash" not in decision or "active_lane" not in branch:
        raise ValueError("candidate branch lacks required provenance fields")
    if (
        int(branch_audit["update_index"]) != int(update_index)
        or int(branch_audit["target_agent_id"]) != int(target_agent_id)
        or int(branch_audit["team_state_version"]) != int(team_state_version)
        or str(branch_audit["parent_team_hash"])
        != str(decision["parent_team_hash"])
        or str(branch_audit["active_lane"]) != str(branch["active_lane"])
    ):
        raise AssertionError("branch audit identity/provenance mismatch")


def _branch_candidates(
    decision: Mapping[str, Any], target_agent_id: int
) -> list[dict[str, Any]]:
    target = int(target_agent_id)
    return [
        row
        for row in decision.get("candidates", [])
        if int(row["target_agent_id"]) == target
    ]


def _candidate_geometry(candidate: Mapping[str, Any]) -> str:
    constraint = candidate.get("constraint")
    if not isinstance(constraint, Mapping):
        raise ValueError("evaluated candidate lacks constraint geometry")
    if "target_gain" not in constraint or "vote_net_gain" not in constraint:
        raise ValueError("candidate constraint lacks canonical geometry fields")
    target_gain = int(constraint["target_gain"])
    vote_net = int(constraint["vote_net_gain"])
    canonical = (
        "A" if target_gain > 0 and vote_net > 0 else
        "B" if target_gain > 0 and vote_net == 0 else
        "C" if target_gain == 0 and vote_net > 0 else
        "D" if target_gain > 0 and vote_net < 0 else
        "E" if target_gain < 0 and vote_net > 0 else "F"
    )
    diagnostics = candidate.get("module2_diagnostics")
    if isinstance(diagnostics, Mapping) and "candidate_geometry" in diagnostics:
        persisted = str(diagnostics["candidate_geometry"])
        if persisted not in set("ABCDEF") or persisted != canonical:
            raise AssertionError("persisted Module2 geometry differs from canonical")
        if (
            "target_gain" in diagnostics
            and int(diagnostics["target_gain"]) != target_gain
        ) or (
            "vote_net_gain" in diagnostics
            and int(diagnostics["vote_net_gain"]) != vote_net
        ):
            raise AssertionError(
                "persisted Module2 geometry inputs differ from constraints"
            )
        return persisted
    return canonical


def _candidate_fixed_hashes(
    *,
    candidate: Mapping[str, Any],
    visible_hashes: set[str],
    parent_by_hash: Mapping[str, Mapping[str, Any]],
    answer_cache: Mapping[tuple[str, str], Mapping[str, Any]],
    target_agent_id: int,
) -> set[str]:
    from multi_dataset_diverse_rl.tasks import match_bbh_answer

    prompt_hash = str(candidate["prompt_hash"])
    fixed: set[str] = set()
    for question_hash in visible_hashes:
        parent = parent_by_hash[question_hash]
        if bool(parent["team_correctness"][int(target_agent_id)]):
            raise ValueError("visible responsibility contains target-correct item")
        cached = answer_cache.get((prompt_hash, question_hash))
        if cached is None:
            raise ValueError(
                f"read-only solver cache lacks evaluated candidate observation: "
                f"{prompt_hash[:12]}/{question_hash[:12]}"
            )
        if bool(cached.get("valid")) and match_bbh_answer(
            str(cached.get("answer", "")), str(parent["gold_answer"])
        ):
            fixed.add(question_hash)
    return fixed


def _validate_active_repair_count(
    *,
    candidate: Mapping[str, Any],
    active_hashes: set[str],
    parent_by_hash: Mapping[str, Mapping[str, Any]],
    answer_cache: Mapping[tuple[str, str], Mapping[str, Any]],
    target_agent_id: int,
) -> int:
    fixed = _candidate_fixed_hashes(
        candidate=candidate,
        visible_hashes=active_hashes,
        parent_by_hash=parent_by_hash,
        answer_cache=answer_cache,
        target_agent_id=target_agent_id,
    )
    recorded = int(
        ((candidate.get("evaluation") or {}).get("marginal") or {}).get(
            "assigned_residual_repair_count", 0
        )
    )
    if recorded != len(fixed):
        raise AssertionError(
            "active-slice repair reconstruction differs from persisted evaluation"
        )
    return recorded


def _mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return statistics.mean(rows) if rows else None


def _median(values: Iterable[float]) -> float | None:
    rows = list(values)
    return statistics.median(rows) if rows else None


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    return _mean(float(bool(row[field])) for row in rows)


def _outcome_strata(branches: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for dimension, bins in (
        ("fine_coherence_bin", ("HIGH", "MEDIUM", "LOW")),
        ("fine_entropy_bin", ("LOW", "MEDIUM", "HIGH")),
    ):
        for value in bins:
            rows = [row for row in branches if row[dimension] == value]
            result.append(
                {
                    "stratum_dimension": dimension,
                    "stratum": value,
                    "branch_count": len(rows),
                    "any_valid_rate": _rate(rows, "any_valid_candidate"),
                    "any_repair_gain_rate": _rate(
                        rows, "any_repair_gain_candidate"
                    ),
                    "any_feasible_rate": _rate(
                        rows, "any_common_safe_feasible_candidate"
                    ),
                    "any_F_rate": _rate(rows, "any_F_candidate"),
                    "any_target_regression_rate": _rate(
                        rows, "any_target_regression_candidate"
                    ),
                    "mean_best_feasible_candidate_count": _mean(
                        float(row["best_feasible_candidate_count"]) for row in rows
                    ),
                    "median_best_feasible_candidate_count": _median(
                        float(row["best_feasible_candidate_count"]) for row in rows
                    ),
                }
            )
    return result


def _directional_association(
    branches: Sequence[dict[str, Any]], strata: Sequence[dict[str, Any]]
) -> tuple[bool, dict[str, Any]]:
    cluster_feasible = spearman(
        [float(row["fine_largest_cluster_share"]) for row in branches],
        [float(row["any_common_safe_feasible_candidate"]) for row in branches],
    )
    entropy_f = spearman(
        [float(row["fine_entropy_norm"]) for row in branches],
        [float(row["any_F_candidate"]) for row in branches],
    )
    entropy_regression = spearman(
        [float(row["fine_entropy_norm"]) for row in branches],
        [float(row["any_target_regression_candidate"]) for row in branches],
    )
    indexed = {
        (row["stratum_dimension"], row["stratum"]): row for row in strata
    }

    def greater(
        left_key: tuple[str, str], right_key: tuple[str, str], field: str
    ) -> bool:
        left = indexed[left_key][field]
        right = indexed[right_key][field]
        return left is not None and right is not None and float(left) > float(right)

    checks = {
        "spearman_cluster_share_feasible_gt_0": (
            cluster_feasible is not None and cluster_feasible > 0
        ),
        "spearman_entropy_F_gt_0": entropy_f is not None and entropy_f > 0,
        "spearman_entropy_target_regression_gt_0": (
            entropy_regression is not None and entropy_regression > 0
        ),
        "high_coherence_feasible_rate_gt_low": greater(
            ("fine_coherence_bin", "HIGH"),
            ("fine_coherence_bin", "LOW"),
            "any_feasible_rate",
        ),
        "high_entropy_F_rate_gt_low": greater(
            ("fine_entropy_bin", "HIGH"),
            ("fine_entropy_bin", "LOW"),
            "any_F_rate",
        ),
        "high_entropy_regression_rate_gt_low": greater(
            ("fine_entropy_bin", "HIGH"),
            ("fine_entropy_bin", "LOW"),
            "any_target_regression_rate",
        ),
    }
    return any(checks.values()), {
        "spearman_fine_cluster_share_vs_any_feasible": cluster_feasible,
        "spearman_fine_entropy_vs_any_F": entropy_f,
        "spearman_fine_entropy_vs_any_target_regression": entropy_regression,
        "directional_checks": checks,
    }


def _repair_concentration(
    focus: Sequence[dict[str, Any]], *, stratum: str
) -> tuple[bool, dict[str, Any]]:
    rows = [row for row in focus if row[stratum]]
    mean_focus = _mean(float(row["repair_focus_share"]) for row in rows)
    single_share = _mean(
        float(int(row["fixed_fine_signature_count"]) == 1) for row in rows
    )
    supported = bool(
        len(rows) >= 2
        and mean_focus is not None
        and mean_focus >= 0.75
        and single_share is not None
        and single_share >= 0.50
    )
    return supported, {
        "candidate_count": len(rows),
        "repair_focus_share_mean": mean_focus,
        "repair_focus_share_median": _median(
            float(row["repair_focus_share"]) for row in rows
        ),
        "single_fine_signature_count": sum(
            int(row["fixed_fine_signature_count"]) == 1 for row in rows
        ),
        "multi_fine_signature_count": sum(
            int(row["fixed_fine_signature_count"]) > 1 for row in rows
        ),
        "single_fine_signature_share": single_share,
    }


def _sanitize_value(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in PUBLISHABLE_SECRET_TOKENS):
                raise AssertionError(f"publishable key is sensitive: {path}/{key}")
            _sanitize_value(child, f"{path}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _sanitize_value(child, f"{path}/{index}")
    elif isinstance(value, str):
        lowered = value.lower()
        if str(ROOT).lower() in lowered or "d:\\" in lowered:
            raise AssertionError(f"publishable value contains an absolute path: {path}")


def audit(
    *, spec_path: Path = DEFAULT_SPEC, out_dir: Path = DEFAULT_OUT
) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    if ROOT.resolve() not in spec_path.parents:
        raise PermissionError("analysis spec must remain under repository root")
    spec = read_json(spec_path)
    validate_analysis_spec(spec)
    roots = {
        int(seed): ROOT / relative
        for seed, relative in dict(spec["trajectory_roots"]).items()
    }
    reader = TrainOnlyReader(roots)
    out_dir = out_dir.resolve()
    if ROOT.resolve() not in out_dir.parents:
        raise PermissionError("audit output must remain under repository root")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("audit output directory must be fresh")
    out_dir.mkdir(parents=True, exist_ok=True)

    branch_rows: list[dict[str, Any]] = []
    focus_rows: list[dict[str, Any]] = []
    source_assertions: list[dict[str, Any]] = []
    for seed in REQUIRED_SEEDS:
        meta = reader.json(seed, "run_meta.json")
        frozen_initialization = reader.json(
            seed, "frozen_initialization_match.json"
        )
        namespace = cache_namespace(
            seed=seed, meta=meta, frozen=frozen_initialization
        )
        expected_setting = REQUIRED_SETTING_BY_SEED[seed]
        assertions = {
            "seed": seed,
            "method_version_v15": meta.get("method_version")
            == "member_aware_peer_state_v15",
            "canonical_setting_matches": meta.get("canonical_experiment_setting")
            == expected_setting,
            "train_only": not bool(meta.get("validation_used"))
            and int(meta.get("validation_evaluation_count", 0)) == 0
            and int(meta.get("test_evaluation_count", 0)) == 0,
            "test_isolation": (
                not bool(meta.get("test_used_for_training"))
                and not bool(meta.get("test_used_for_selection"))
                and not bool(meta.get("test_called_before_training_complete"))
                and isinstance(meta.get("final_state_selection"), Mapping)
                and not bool(
                    meta["final_state_selection"].get("final_test_enabled")
                )
                and not bool(
                    meta["final_state_selection"].get("validation_used")
                )
                and not bool(
                    meta["final_state_selection"].get("selected_by_validation")
                )
                and int(
                    meta["final_state_selection"].get(
                        "validation_evaluation_count", 0
                    )
                ) == 0
                and int(
                    meta["final_state_selection"].get(
                        "test_evaluation_count", 0
                    )
                ) == 0
            ),
            "candidate_generator_m20": meta.get("candidate_generator")
            == "member_aware_responsibility_conditioned",
            "module1_and_common_safe_semantics": (
                meta.get("tie_policy") == "abstain"
                and int(meta.get("target_branch_count", 0)) == 2
                and int(meta.get("candidates_per_target_branch", 0)) == 2
                and meta.get("candidate_acceptance_policy")
                == "fixed_peer_monotone_target_or_vote"
                and meta.get("candidate_ranking_policy") == "common_monotone_safe"
                and meta.get("stage_a_policy") == "matched_all_generated"
                and meta.get("responsibility_version")
                == "counterfactual_vote_margin_responsibility_v1"
                and meta.get("service_routing_version")
                == "single_service_anchor_routing_no_freeze_v2"
                and meta.get("target_selection_version")
                == "repairability_adjusted_expected_update_value_wait_coupled_v2"
                and meta.get("tcs_context_version")
                == "compact_single_lane_responsibility_context_v1"
            ),
        }
        if not all(value for key, value in assertions.items() if key != "seed"):
            raise AssertionError(f"Seed{seed} historical provenance mismatch")
        source_assertions.append(assertions)

        peer_rows = reader.jsonl(seed, "peer_state_history.jsonl")
        snapshots = reader.jsonl(seed, "responsibility_assignments.jsonl")
        contexts = reader.jsonl(seed, "tcs_context_history.jsonl")
        decisions = reader.jsonl(seed, "candidate_decisions.jsonl")
        branch_artifacts = reader.jsonl(seed, "dual_target_branch_decisions.jsonl")
        commits = reader.jsonl(seed, "dual_target_commit_decisions.jsonl")
        states = _state_blocks(peer_rows)
        snapshot_by_version = {
            int(row["team_state_version"]): row for row in snapshots
        }
        if len(snapshot_by_version) != len(snapshots):
            raise ValueError(f"Seed{seed}: duplicate responsibility state version")
        context_by_key = {
            (int(row["update_index"]), int(row["target_agent_id"])): row
            for row in contexts
        }
        if len(context_by_key) != len(contexts):
            raise ValueError(f"Seed{seed}: duplicate M20 context key")
        branch_artifact_by_key = {
            (int(row["update_index"]), int(row["target_agent_id"])): row
            for row in branch_artifacts
        }
        if len(branch_artifact_by_key) != len(branch_artifacts):
            raise ValueError(f"Seed{seed}: duplicate branch audit key")
        commit_by_update = {int(row["update_index"]): row for row in commits}
        if len(commit_by_update) != len(commits):
            raise ValueError(f"Seed{seed}: duplicate commit update")
        updates = [int(row["update_index"]) for row in decisions]
        if updates != list(range(len(updates))):
            raise ValueError(f"Seed{seed}: decision updates are not contiguous")
        expected_branch_keys = {
            (int(decision["update_index"]), int(branch["target_agent_id"]))
            for decision in decisions
            for branch in decision.get("branches", [])
        }
        if (
            set(context_by_key) != expected_branch_keys
            or set(branch_artifact_by_key) != expected_branch_keys
            or set(commit_by_update) != set(updates)
        ):
            raise ValueError(f"Seed{seed}: historical branch artifact inventory mismatch")
        candidate_hashes = {
            str(candidate["prompt_hash"])
            for decision in decisions
            for candidate in decision.get("candidates", [])
            if candidate.get("evaluation") is not None
        }
        answer_cache = reader.candidate_answers(
            seed, candidate_hashes, namespace=namespace
        )

        state_version = 0
        for decision in decisions:
            update = int(decision["update_index"])
            if state_version >= len(states):
                raise ValueError(f"Seed{seed}: missing parent state for update {update}")
            state_rows = states[state_version]
            parent_by_hash = {
                str(row["question_hash"]): row for row in state_rows
            }
            snapshot = snapshot_by_version.get(state_version)
            if snapshot is None:
                raise ValueError(f"Seed{seed}: missing responsibility state {state_version}")
            committed_hash = str(decision.get("accepted_prompt_hash") or "")
            commit = commit_by_update.get(update)
            if commit is None or str(commit.get("committed_prompt_hash") or "") != committed_hash:
                raise AssertionError(f"Seed{seed}: commit artifact mismatch at update {update}")
            for branch in decision.get("branches", []):
                target = int(branch["target_agent_id"])
                key = (update, target)
                context = context_by_key.get(key)
                branch_audit = branch_artifact_by_key.get(key)
                if context is None or branch_audit is None:
                    raise ValueError(f"Seed{seed}: incomplete branch provenance {key}")
                validate_branch_provenance(
                    branch_audit=branch_audit,
                    decision=decision,
                    branch=branch,
                    update_index=update,
                    target_agent_id=target,
                    team_state_version=state_version,
                )
                active_hashes = exact_branch_active_hashes(
                    snapshot, branch, target_agent_id=target
                )
                visible_hashes = visible_repair_hashes(
                    context,
                    update_index=update,
                    target_agent_id=target,
                )
                if not visible_hashes.issubset(active_hashes):
                    raise AssertionError("visible context escaped the active responsibility slice")
                full_hashes = _full_service_hashes(snapshot, target)
                if not active_hashes or not active_hashes.issubset(full_hashes):
                    raise AssertionError("active responsibility slice escaped full service portfolio")

                signatures = {
                    question_hash: residual_signature(
                        parent_by_hash[question_hash],
                        target_agent_id=target,
                        seed=seed,
                    )
                    for question_hash in full_hashes
                }
                visible_metrics = coherence_metrics(
                    [signatures[value] for value in sorted(visible_hashes)]
                )
                full_metrics = coherence_metrics(
                    [signatures[value] for value in sorted(full_hashes)]
                )
                candidates = _branch_candidates(decision, target)
                if int(branch_audit["candidate_count"]) != len(candidates):
                    raise AssertionError("branch candidate count mismatch")
                valid = [row for row in candidates if row.get("evaluation") is not None]
                funnel = branch.get("funnel") or {}
                if (
                    int(funnel.get("stage_b_evaluated", -1)) != len(valid)
                    or int(funnel.get("valid_candidate_count", -1))
                    < len(valid)
                ):
                    raise AssertionError("branch funnel/evaluated candidate mismatch")
                feasible = [
                    row for row in valid if bool((row.get("constraint") or {}).get("passed"))
                ]
                if int(branch_audit.get("passed_candidate_count", -1)) != len(feasible):
                    raise AssertionError("branch feasible candidate count mismatch")
                fixed_visible_by_prompt = {
                    str(row["prompt_hash"]): _candidate_fixed_hashes(
                        candidate=row,
                        visible_hashes=visible_hashes,
                        parent_by_hash=parent_by_hash,
                        answer_cache=answer_cache,
                        target_agent_id=target,
                    )
                    for row in valid
                }
                active_repair_count_by_prompt = {
                    str(row["prompt_hash"]): _validate_active_repair_count(
                        candidate=row,
                        active_hashes=active_hashes,
                        parent_by_hash=parent_by_hash,
                        answer_cache=answer_cache,
                        target_agent_id=target,
                    )
                    for row in valid
                }
                active_repair_gain = [
                    row
                    for row in valid
                    if active_repair_count_by_prompt[str(row["prompt_hash"])] > 0
                ]
                visible_repaired = [
                    row
                    for row in valid
                    if fixed_visible_by_prompt[str(row["prompt_hash"])]
                ]
                outcome = {
                    "any_valid_candidate": bool(
                        int(funnel.get("valid_candidate_count", 0))
                    ),
                    "any_repair_gain_candidate": bool(active_repair_gain),
                    "any_common_safe_feasible_candidate": bool(feasible),
                    "any_F_candidate": any(
                        _candidate_geometry(row) == "F" for row in valid
                    ),
                    "any_target_regression_candidate": any(
                        int((row.get("constraint") or {}).get("target_gain", 0)) < 0
                        for row in valid
                    ),
                    "best_feasible_candidate_count": len(feasible),
                }
                branch_id = f"seed{seed}_update{update}_target{target}"
                branch_rows.append(
                    {
                        "branch_id": branch_id,
                        "seed": seed,
                        "update_index": update,
                        "team_state_version": state_version,
                        "target_agent_id": target,
                        **visible_metrics,
                        "full_portfolio_size": full_metrics["portfolio_size"],
                        "full_coarse_unique_signature_count": full_metrics[
                            "coarse_unique_signature_count"
                        ],
                        "full_fine_unique_signature_count": full_metrics[
                            "fine_unique_signature_count"
                        ],
                        "full_coarse_largest_cluster_share": full_metrics[
                            "coarse_largest_cluster_share"
                        ],
                        "full_fine_largest_cluster_share": full_metrics[
                            "fine_largest_cluster_share"
                        ],
                        "full_coarse_entropy_norm": full_metrics[
                            "coarse_entropy_norm"
                        ],
                        "full_fine_entropy_norm": full_metrics["fine_entropy_norm"],
                        "full_mean_pairwise_signature_distance": full_metrics[
                            "mean_pairwise_signature_distance"
                        ],
                        "visible_minus_full_coarse_cluster_share": visible_metrics[
                            "coarse_largest_cluster_share"
                        ]
                        - full_metrics["coarse_largest_cluster_share"],
                        "visible_minus_full_fine_cluster_share": visible_metrics[
                            "fine_largest_cluster_share"
                        ]
                        - full_metrics["fine_largest_cluster_share"],
                        "visible_minus_full_coarse_entropy": visible_metrics[
                            "coarse_entropy_norm"
                        ]
                        - full_metrics["coarse_entropy_norm"],
                        "visible_minus_full_fine_entropy": visible_metrics[
                            "fine_entropy_norm"
                        ]
                        - full_metrics["fine_entropy_norm"],
                        **outcome,
                    }
                )

                for candidate in visible_repaired:
                    fixed = fixed_visible_by_prompt[str(candidate["prompt_hash"])]
                    fixed_signatures = [signatures[value] for value in sorted(fixed)]
                    coarse = Counter(
                        tuple(row["coarse_signature"]) for row in fixed_signatures
                    )
                    fine = Counter(
                        tuple(row["fine_signature"]) for row in fixed_signatures
                    )
                    prompt_hash = str(candidate["prompt_hash"])
                    is_feasible = bool(
                        (candidate.get("constraint") or {}).get("passed")
                    )
                    is_committed = bool(committed_hash and prompt_hash == committed_hash)
                    focus_rows.append(
                        {
                            "branch_id": branch_id,
                            "candidate_hash": prompt_hash,
                            "seed": seed,
                            "update_index": update,
                            "target_agent_id": target,
                            "fixed_residual_count": len(fixed),
                            "fixed_coarse_signature_count": len(coarse),
                            "fixed_fine_signature_count": len(fine),
                            "repair_focus_share": max(fine.values()) / len(fixed),
                            "all_repair_gain": True,
                            "common_safe_feasible_repair_gain": is_feasible,
                            "committed_repair_gain": is_committed,
                        }
                    )
            if committed_hash:
                state_version += 1
        if state_version + 1 != len(states):
            raise AssertionError(f"Seed{seed}: accepted-state history mismatch")

    strata = _outcome_strata(branch_rows)
    outcome_supported, association = _directional_association(branch_rows, strata)
    repair_supported, repair_summary = _repair_concentration(
        focus_rows, stratum="common_safe_feasible_repair_gain"
    )
    coherence_label = (
        "SUPPORTED"
        if outcome_supported and repair_supported
        else "PARTIAL"
        if outcome_supported or repair_supported
        else "NOT_SUPPORTED"
    )
    seed_rows = []
    for seed in REQUIRED_SEEDS:
        branches = [row for row in branch_rows if int(row["seed"]) == seed]
        candidates = [row for row in focus_rows if int(row["seed"]) == seed]
        seed_rows.append(
            {
                "seed": seed,
                "branch_count": len(branches),
                "visible_portfolio_mean_size": _mean(
                    float(row["portfolio_size"]) for row in branches
                ),
                "visible_fine_cluster_share_mean": _mean(
                    float(row["fine_largest_cluster_share"]) for row in branches
                ),
                "visible_fine_entropy_mean": _mean(
                    float(row["fine_entropy_norm"]) for row in branches
                ),
                "any_feasible_rate": _rate(
                    branches, "any_common_safe_feasible_candidate"
                ),
                "any_F_rate": _rate(branches, "any_F_candidate"),
                "repair_gain_candidate_count": len(candidates),
            }
        )

    audit_summary = {
        "audit_version": "v16_responsibility_coherence_audit_v1",
        "status": "PASS",
        "historical_seeds": list(REQUIRED_SEEDS),
        "branch_count": len(branch_rows),
        "visible_portfolio_mean_size": _mean(
            float(row["portfolio_size"]) for row in branch_rows
        ),
        "visible_fine_cluster_share_mean": _mean(
            float(row["fine_largest_cluster_share"]) for row in branch_rows
        ),
        "visible_fine_entropy_mean": _mean(
            float(row["fine_entropy_norm"]) for row in branch_rows
        ),
        "high_coherence_branch_count": sum(
            row["fine_coherence_bin"] == "HIGH" for row in branch_rows
        ),
        "medium_coherence_branch_count": sum(
            row["fine_coherence_bin"] == "MEDIUM" for row in branch_rows
        ),
        "low_coherence_branch_count": sum(
            row["fine_coherence_bin"] == "LOW" for row in branch_rows
        ),
        "outcome_association_supported": outcome_supported,
        "repair_concentration_supported": repair_supported,
        "coherence_bottleneck": coherence_label,
        "association": association,
        "feasible_repair_focus": repair_summary,
        "api_calls": 0,
        "model_calls": 0,
        "test_files_read": reader.test_files_read,
        "validation_files_read": reader.validation_files_read,
    }
    assertions = {
        "status": "PASS",
        "definitions_frozen": True,
        "exact_historical_seeds": list(REQUIRED_SEEDS),
        "source_assertions": source_assertions,
        "test_files_read": reader.test_files_read,
        "validation_files_read": reader.validation_files_read,
        "api_calls": 0,
        "model_calls": 0,
        "read_only_solver_cache": True,
        "source_artifact_count": len(reader.files_read),
        "source_artifact_set_sha256": hashlib.sha256(
            json.dumps(
                [
                    (seed, path.name, sha256_file(path), path.stat().st_size)
                    for seed, root in roots.items()
                    for path in reader.files_read
                    if path.parent == root.resolve()
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    _sanitize_value(audit_summary)
    _sanitize_value(branch_rows)
    _sanitize_value(focus_rows)
    _sanitize_value(strata)
    _sanitize_value(seed_rows)
    _sanitize_value(assertions)
    write_json(out_dir / "audit_summary.json", audit_summary)
    write_csv(out_dir / "branch_coherence_metrics.csv", branch_rows)
    write_csv(out_dir / "candidate_repair_focus.csv", focus_rows)
    write_csv(out_dir / "coherence_outcome_strata.csv", strata)
    write_csv(out_dir / "seed_summary.csv", seed_rows)
    write_json(out_dir / "audit_assertions.json", assertions)
    return audit_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = audit(spec_path=args.spec, out_dir=args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
