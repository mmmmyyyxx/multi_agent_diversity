"""Frozen prospective pilot for sequential plurality symmetry breaking.

Preparation, preflight, audit, and analysis are zero-API operations. Execution
is separately gated by the manifest lifecycle, explicit authorization metadata,
an environment handoff flag, and the complete source/data freeze.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from infrastructure.common_solver_contract_v1.contract import (  # noqa: E402
    COMMON_SOLVER_CONTRACT_ID,
    CONTRACT_SPEC,
    contract_identity,
)
from multi_dataset_diverse_rl.config import Config  # noqa: E402
from multi_dataset_diverse_rl.evaluation.categorical_profiles import (  # noqa: E402
    endpoint_identifiability_snapshot,
)
from multi_dataset_diverse_rl.evaluation.output_contract import (  # noqa: E402
    SOLVER_OUTPUT_CONTRACT_VERSION,
)
from multi_dataset_diverse_rl.evaluation.prompt_question import PromptAnswer  # noqa: E402
from multi_dataset_diverse_rl.governance.authorization import (  # noqa: E402
    require_api_authorization,
)
from multi_dataset_diverse_rl.local_optimizers.gepa_optimizer import (  # noqa: E402
    GEPAOptimizerConfig,
    GEPALocalPromptOptimizer,
    local_gepa_budget_capacity,
    verify_frozen_gepa_engine_contract,
)
from multi_dataset_diverse_rl.team_search.execution_runtime import (  # noqa: E402
    CommonContractExecutionSystem as Seed78System,
    ContextualLocalPromptOptimizer as ContextualOptimizer,
    DurableLedger,
    ReflectionLM,
    execution_context_from_system,
    profile_identity as _profile_identity,
)
from multi_dataset_diverse_rl.persistence.identity import build_run_identity  # noqa: E402
from multi_dataset_diverse_rl.team_search.candidate_selector import (  # noqa: E402
    CommonSafeTeamCandidateSelector,
)
from multi_dataset_diverse_rl.team_search.controller import TeamSearchController  # noqa: E402
from multi_dataset_diverse_rl.team_search.primary_responsibility_binding import (  # noqa: E402
    PrimaryResponsibilityOnlineBinding,
)
from multi_dataset_diverse_rl.team_search.primary_responsibility_scheduler import (  # noqa: E402
    PrimaryResponsibilityPersistentRealizabilityScheduler,
)
from multi_dataset_diverse_rl.team_search.schemas import TeamSearchRequest  # noqa: E402
from multi_dataset_diverse_rl.team_search.symmetry_breaking import (  # noqa: E402
    SymmetryBreakingTrajectory,
    classify_symmetry_breaking_trajectory,
)
from multi_dataset_diverse_rl.team_search.system_runtime import (  # noqa: E402
    FrozenResponsibilitySnapshot,
    SystemLocalSolverEvaluator,
    SystemResponsibilityAssignmentFactory,
    SystemTeamCandidateEvaluator,
    SystemTeamCommitter,
    freeze_current_responsibility,
)
from multi_dataset_diverse_rl.team_search.task_builder import LocalTaskBuilder  # noqa: E402
from multi_dataset_diverse_rl.versions import (  # noqa: E402
    SEQUENTIAL_SYMMETRY_BREAKING_ONLINE_PILOT_VERSION,
)
from scripts.anti_overfitting_shadow_support import (  # noqa: E402
    construct_assignment,
    metadata,
    sha256_file,
    sha256_json,
    write_json,
)


EXPERIMENT_ID = SEQUENTIAL_SYMMETRY_BREAKING_ONLINE_PILOT_VERSION
SEED = 78
ARM = "PRIMARY_RESPONSIBILITY_REALIZABILITY"
SOLVER_MODEL = "qwen3-8b"
ROLE_MODEL = "qwen3.7-flash"
MAX_OPPORTUNITIES = 8
MAX_SAFE_COMMITS = 4
NO_COMMIT_PATIENCE = 6
LOCAL_GEPA_METRIC_BUDGET = 36
SUCCESSFUL_PROVIDER_CALL_CEILING = 4_832
TRANSPORT_ATTEMPT_CEILING = 19_328
REFLECTION_SUCCESSFUL_CALL_CEILING = 64
AUTH_ENV = "SEQUENTIAL_SYMMETRY_BREAKING_V1_AUTHORIZED"
MANIFEST = ROOT / "experiments/manifests/sequential_symmetry_breaking_online_pilot_v1.yaml"
DESIGN = ROOT / "experiments/sequential_symmetry_breaking_online_pilot_v1"
PARENT_TASKS = (
    ROOT
    / "runs/local_gepa_acceptance_rate_pilot_phase_b_v2_freeze"
    / "selected_parent_tasks_private.json"
)
PARENT_TASKS_SHA256 = "1b27cce0731acfd0ee2fb9f34fe454caf97d3d1721be8b80d432a9bace94de27"
PHASE_B_MANIFEST = ROOT / "experiments/manifests/local_gepa_acceptance_rate_pilot_phase_b_v2.yaml"
PHASE_B_MANIFEST_SHA256 = "32933872b0dc2ed2d7fcdaf23b5c368e5320b0861204ad908f7a20cf732a1147"
BASELINE_REDUNDANCY_AUDIT = (
    ROOT
    / "reports/accepted_local_mutation_team_transfer_v2_execution_20260919"
    / "baseline_redundancy_audit.json"
)
BASELINE_REDUNDANCY_AUDIT_SHA256 = (
    "bc9e6f43202d5f296c328cb49226d6a6f3573c4d2f973e835d0eb888643b2f05"
)
SOURCE_STATE_HASH = "faa0fc81ebe71a686554355b9c5946a1765fa3913e026ab65e59f07ae01052dd"
PARENT_PROMPT_HASH = "549bc93c03f703faf5aa1bd56b557135fb6e65d0cf6c055b8fad6a15e7c87a63"
DEFAULT_PREP = ROOT / "runs/sequential_symmetry_breaking_online_pilot_v1_prep"
DEFAULT_RUN = ROOT / "runs/sequential_symmetry_breaking_online_pilot_v1_attempt1"
DEFAULT_REPORT = ROOT / "reports/sequential_symmetry_breaking_online_pilot_v1"


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class BoundedLedger(DurableLedger):
    """Durably reserve each provider attempt before it can leave the process."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.attempts = 0
        self.successes = 0
        self.reservation_path = path.with_name("provider_attempt_reservations.jsonl")

    def reserve_provider_attempt(self, role: str) -> None:
        if self.attempts >= TRANSPORT_ATTEMPT_CEILING:
            raise RuntimeError("transport_attempt_ceiling_exhausted")
        if self.successes >= SUCCESSFUL_PROVIDER_CALL_CEILING:
            raise RuntimeError("successful_provider_call_ceiling_exhausted")
        self.attempts += 1
        with self.reservation_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"attempt": self.attempts, "role": role}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def append(self, row: Mapping[str, Any]) -> None:
        successes = int(row.get("successful_provider_calls", 0))
        if self.successes + successes > SUCCESSFUL_PROVIDER_CALL_CEILING:
            raise RuntimeError("successful_provider_call_ceiling_exhausted")
        super().append(row)
        self.successes += successes


def _config(out: Path, *, optimize_path: Path, shadow_path: Path) -> Config:
    return Config.from_flat(
        task_type="bbh",
        dataset_format="mars",
        comparison_task_id="disambiguation_qa",
        benchmark="BBH",
        answer_format="option_letter",
        train_path=str(optimize_path.resolve()),
        val_path=str(shadow_path.resolve()),
        test_path="VALIDATION50_AND_TEST50_BLOCKED",
        manifest_sha256=sha256_file(MANIFEST),
        train_size=100,
        val_size=50,
        test_size=0,
        agent_model=SOLVER_MODEL,
        optimizer_model=ROLE_MODEL,
        evaluator_model=ROLE_MODEL,
        temperature=0.0,
        solver_max_tokens=1800,
        solver_invalid_max_retries=0,
        solver_contract_id=COMMON_SOLVER_CONTRACT_ID,
        experiment_setting="experimental_diversity_d2_rr_generic",
        target_scheduler="round_robin",
        agents=5,
        epochs=1,
        update_every=1,
        seed=SEED,
        proposal_memory_mode="off",
        num_candidates_per_parent=2,
        candidate_eval_pool_size=100,
        eval_solver_call_concurrency=8,
        stage_b_candidate_budget=2,
        out_dir=str(out),
        shared_solver_cache_path="",
        provider_call_budget=SUCCESSFUL_PROVIDER_CALL_CEILING,
        total_token_budget=100_000_000,
        max_retries=0,
        max_transient_retries=4,
        final_test_enabled=False,
        preserve_final_checkpoint=True,
    )


def _load_frozen_parent() -> tuple[dict[str, Any], list[dict[str, str]], str]:
    if not PARENT_TASKS.is_file() or sha256_file(PARENT_TASKS) != PARENT_TASKS_SHA256:
        raise RuntimeError("frozen Phase-B private parent task file is missing or changed")
    tasks = _read_json(PARENT_TASKS)
    names = [f"seed78_update0_member{member}" for member in range(1, 5)]
    if set(tasks) != set(names):
        raise RuntimeError("frozen parent task identities changed")
    if not PHASE_B_MANIFEST.is_file() or sha256_file(PHASE_B_MANIFEST) != PHASE_B_MANIFEST_SHA256:
        raise RuntimeError("frozen Phase-B manifest changed")
    phase_b = yaml.safe_load(PHASE_B_MANIFEST.read_text(encoding="utf-8"))
    selected = phase_b.get("design", {}).get("selected_parents", [])
    if (
        phase_b.get("design", {}).get("shared_baseline_state_hash") != SOURCE_STATE_HASH
        or [row.get("target_member") for row in selected] != [1, 2, 3, 4]
        or any(row.get("parent_prompt_hash") != PARENT_PROMPT_HASH for row in selected)
        or [row.get("parent_task_id") for row in selected] != names
    ):
        raise RuntimeError("Phase-B manifest no longer binds the selected baseline parents")
    if (
        not BASELINE_REDUNDANCY_AUDIT.is_file()
        or sha256_file(BASELINE_REDUNDANCY_AUDIT)
        != BASELINE_REDUNDANCY_AUDIT_SHA256
    ):
        raise RuntimeError("frozen five-member baseline redundancy audit changed")
    redundancy = _read_json(BASELINE_REDUNDANCY_AUDIT)
    if (
        redundancy.get("all_five_optimize_profiles_identical") is not True
        or redundancy.get("all_baseline_outputs_valid") is not True
        or redundancy.get("baseline_prompt_hash") != PARENT_PROMPT_HASH
    ):
        raise RuntimeError("five-member homogeneous baseline evidence is incomplete")
    reference = tasks[names[0]]
    prompt = str(reference["parent_prompt"])
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != PARENT_PROMPT_HASH:
        raise RuntimeError("frozen parent prompt hash mismatch")
    comparable = lambda task: [
        (row["example_id"], row["input_payload"], row["gold"], row["parent_output"])
        for row in task["search_examples"]
    ]
    reference_rows = comparable(reference)
    if len(reference_rows) != 100 or any(comparable(tasks[name]) != reference_rows for name in names):
        raise RuntimeError("Phase-B parent profiles are not one common 100-row baseline")
    optimize_rows: list[dict[str, str]] = []
    for example_id, question, gold, output in reference_rows:
        if hashlib.sha256(str(question).encode("utf-8")).hexdigest() != example_id:
            raise RuntimeError("frozen example identity mismatch")
        optimize_rows.append(
            {"question": str(question), "answer": str(gold), "parent_output": str(output)}
        )
    return reference, optimize_rows, prompt


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def protocol_document() -> dict[str, Any]:
    capacity = asdict(
        local_gepa_budget_capacity(
            metric_budget=LOCAL_GEPA_METRIC_BUDGET,
            validation_size=12,
            reflection_minibatch_size=GEPAOptimizerConfig().reflection_minibatch_size,
        )
    )
    return {
        "schema_version": "sequential_symmetry_breaking_online_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "evidence_type": "single_state_prospective_online_mechanism_pilot",
        "baseline": {
            "source_state_hash": SOURCE_STATE_HASH,
            "parent_tasks_sha256": PARENT_TASKS_SHA256,
            "phase_b_manifest_sha256": PHASE_B_MANIFEST_SHA256,
            "parent_prompt_sha256": PARENT_PROMPT_HASH,
            "five_member_redundancy_audit_sha256": BASELINE_REDUNDANCY_AUDIT_SHA256,
            "members": 5,
            "required_initial_total_p_i": 0,
            "initialization_provider_calls": 0,
        },
        "pipeline": {
            "scheduler": "max(4D,2N,C)/(1+persistent_member_failure_count)",
            "target_branches_per_opportunity": 2,
            "official_gepa": asdict(GEPAOptimizerConfig()),
            "local_metric_budget_per_target": LOCAL_GEPA_METRIC_BUDGET,
            "local_gepa_budget_capacity": capacity,
            "team_minibatch": "TEAM_MINIBATCH primary-lane strict 4/4/4",
            "common_safe": True,
            "winner_only_shadow50": True,
            "max_one_commit_per_opportunity": True,
        },
        "endpoints": {
            "T_commit": "first opportunity with a safe committed mutation",
            "T_pivotal": "first realized state with sum_i P_i > 0, reported in opportunity and commit clocks",
            "T_vote": "first realized state with Optimize100 VoteCorrect above baseline, reported in opportunity and commit clocks",
        },
        "stopping": {
            "first_vote_improvement": True,
            "max_safe_commits": MAX_SAFE_COMMITS,
            "max_opportunities": MAX_OPPORTUNITIES,
            "consecutive_no_commit_patience": NO_COMMIT_PATIENCE,
            "efficacy_adaptation": False,
        },
        "access": {
            "optimize100": "online search and Full evaluation",
            "shadow50": "winner-only safety gate",
            "validation50": "zero access",
            "test50": "zero access",
        },
        "hard_ceilings": {
            "successful_provider_calls_all_roles": SUCCESSFUL_PROVIDER_CALL_CEILING,
            "transport_attempts_all_roles": TRANSPORT_ATTEMPT_CEILING,
            "successful_reflection_calls": REFLECTION_SUCCESSFUL_CALL_CEILING,
        },
    }


def _source_paths() -> list[Path]:
    paths = [
        path.relative_to(ROOT)
        for path in (ROOT / "multi_dataset_diverse_rl").rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    paths.extend(
        [
            Path("infrastructure/common_solver_contract_v1/contract.py"),
            Path("infrastructure/common_solver_contract_v1/evaluator.py"),
            Path("scripts/anti_overfitting_shadow_support.py"),
            Path("scripts/run_seed78_primary_responsibility_ab.py"),
            Path("scripts/run_sequential_symmetry_breaking_online_pilot_v1.py"),
            PHASE_B_MANIFEST.relative_to(ROOT),
            MANIFEST.relative_to(ROOT),
            (DESIGN / "PROTOCOL.md").relative_to(ROOT),
            (DESIGN / "classifier_definition.json").relative_to(ROOT),
            (DESIGN / "EXPERIMENT_HANDOFF.md").relative_to(ROOT),
            Path("experiments/anti_overfitting_split_v1/split_manifest.json"),
            Path("experiments/anti_overfitting_split_v1/fold_assignment.json"),
            BASELINE_REDUNDANCY_AUDIT.relative_to(ROOT),
        ]
    )
    return sorted(set(paths), key=lambda value: value.as_posix())


def prepare(prep: Path) -> dict[str, Any]:
    if prep.exists():
        raise FileExistsError("fresh prep root required")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked worktree must be clean before source freeze")
    _reference, optimize_rows, _prompt = _load_frozen_parent()
    items, raw = metadata()
    assignment = construct_assignment(items)
    optimize_ids = {
        hashlib.sha256(str(row["question"]).encode("utf-8")).hexdigest()
        for row in optimize_rows
    }
    if optimize_ids != set(assignment["fold_a"] + assignment["fold_b"]):
        raise RuntimeError("frozen parent rows do not equal Optimize100")
    shadow_rows = [raw[digest] for digest in assignment["fold_c"]]
    prep.mkdir(parents=True)
    _write_csv(
        prep / "splits_private/optimize100.csv",
        optimize_rows,
        ("question", "answer", "parent_output"),
    )
    _write_csv(
        prep / "splits_private/shadow50.csv",
        shadow_rows,
        ("question", "answer"),
    )
    protocol = protocol_document()
    protocol_hash = sha256_json(protocol)
    write_json(prep / "protocol_freeze.json", protocol)
    write_json(
        prep / "evaluation_access_registry.json",
        {
            "events": [],
            "optimize100_initialization_calls": 0,
            "validation50_calls": 0,
            "test50_calls": 0,
        },
    )
    freeze = {
        "execution_commit": _git("rev-parse", "HEAD"),
        "protocol_sha256": protocol_hash,
        "source_files": [
            {"path": path.as_posix(), "sha256": sha256_file(ROOT / path)}
            for path in _source_paths()
        ],
        "private_inputs": {
            "phase_b_parent_tasks": {
                "path": PARENT_TASKS.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(PARENT_TASKS),
            },
            "five_member_redundancy_audit": {
                "path": BASELINE_REDUNDANCY_AUDIT.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(BASELINE_REDUNDANCY_AUDIT),
            },
            "optimize100.csv": sha256_file(prep / "splits_private/optimize100.csv"),
            "shadow50.csv": sha256_file(prep / "splits_private/shadow50.csv"),
        },
    }
    write_json(prep / "source_freeze.json", freeze)
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    ready = (
        manifest.get("status") == "RUNNING"
        and manifest.get("api_authorization", {}).get("authorized") is True
    )
    if ready:
        handoff_tail = (
            "READY_TO_RUN: `true`\n\n"
            "Run exactly once with the frozen command:\n\n"
            "```powershell\n"
            f"$env:{AUTH_ENV}='1'\n"
            "python scripts\\run_sequential_symmetry_breaking_online_pilot_v1.py "
            f"--execute --prep {prep.relative_to(ROOT)} --run {DEFAULT_RUN.relative_to(ROOT)}\n"
            "```\n"
        )
    else:
        handoff_tail = (
            "READY_TO_RUN: `false`\n\n"
            "Execution remains blocked pending explicit authorization and an authorization-only "
            "manifest/handoff refreeze. Validation50 and Test50 remain prohibited.\n"
        )
    handoff = (
        "# Sequential symmetry-breaking pilot handoff (generated)\n\n"
        f"Execution commit: `{freeze['execution_commit']}`  \n"
        f"Protocol SHA256: `{protocol_hash}`  \n"
        "API calls during preparation: `0`  \n"
        + handoff_tail
    )
    (prep / "EXPERIMENT_HANDOFF.md").write_text(handoff, encoding="utf-8")
    result = {
        "gate": "PASS",
        "ready_to_run": ready,
        "hold_reason": None if ready else "API_AUTHORIZATION_PENDING",
        "execution_commit": freeze["execution_commit"],
        "protocol_sha256": protocol_hash,
        "api_calls": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
    }
    write_json(prep / "preflight_gate.json", result)
    return result


def _verify_freeze(prep: Path) -> None:
    freeze = _read_json(prep / "source_freeze.json")
    if sha256_json(_read_json(prep / "protocol_freeze.json")) != freeze["protocol_sha256"]:
        raise RuntimeError("protocol freeze mismatch")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", freeze["execution_commit"], "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise RuntimeError("execution commit is not an ancestor of HEAD")
    for row in freeze["source_files"]:
        if sha256_file(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"source freeze mismatch: {row['path']}")
    if sha256_file(PARENT_TASKS) != freeze["private_inputs"]["phase_b_parent_tasks"]["sha256"]:
        raise RuntimeError("private parent task freeze mismatch")
    if (
        sha256_file(BASELINE_REDUNDANCY_AUDIT)
        != freeze["private_inputs"]["five_member_redundancy_audit"]["sha256"]
    ):
        raise RuntimeError("five-member redundancy audit freeze mismatch")
    for name in ("optimize100.csv", "shadow50.csv"):
        if sha256_file(prep / "splits_private" / name) != freeze["private_inputs"][name]:
            raise RuntimeError(f"private split freeze mismatch: {name}")


def _authorize() -> None:
    if os.environ.get(AUTH_ENV) != "1":
        raise PermissionError(f"{AUTH_ENV}=1 is required")
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for role in ("solver", "reflection"):
        require_api_authorization(
            manifest,
            phase="online_trajectory",
            role=role,
            explicit_user_authorized=True,
        )


def _materialize_frozen_baseline(
    system: Seed78System,
    *,
    optimize_rows: Sequence[Mapping[str, Any]],
    parent_prompt: str,
) -> dict[str, Any]:
    if any(agent.current_prompt != parent_prompt for agent in system.agents):
        raise RuntimeError("runtime shared prompt differs from frozen parent")
    system.validate_active_prompts()
    system.fixed_probe = system.build_probe(optimize_rows)
    answers = tuple(
        PromptAnswer(
            answer=str(row["parent_output"]),
            trace="",
            valid=True,
            validity_status="frozen_phase_b_parent_output",
            raw_final_answer_payload=str(row["parent_output"]),
            final_answer_line_count=1,
            first_attempt_valid=True,
        )
        for row in optimize_rows
    )
    system.active_profiles = [answers for _ in range(5)]
    system.initial_profiles = list(system.active_profiles)
    system.accepted_state_count = 1
    system.stable_correct_question_hashes_by_agent = {
        member: {
            example.question_hash
            for answer, example in zip(answers, system.fixed_probe.examples, strict=True)
            if system.match_answer(answer.answer, example.gold_answer)
        }
        for member in range(5)
    }
    prompt_hash = system.prompt_hash(parent_prompt)
    for example, answer in zip(system.fixed_probe.examples, answers, strict=True):
        key = system.prompt_question_evaluator.key(prompt_hash, example.question_hash)
        system.prompt_question_evaluator.cache[key] = answer
    for member in range(5):
        system.persist_team_full_categorical_profile(
            profile=answers,
            update_index=-1,
            target_member=member,
            candidate_hash=prompt_hash,
            candidate_id=f"frozen_baseline_member_{member}",
            evaluation_stage="frozen_phase_b_baseline_materialization",
        )
    snapshot = system.persist_endpoint_identifiability_state(
        update_index=-1,
        trigger="frozen_phase_b_baseline_materialization",
    )
    if snapshot["total_member_row_opportunities"] != 0:
        raise RuntimeError("frozen baseline is not the preregistered homogeneous lock")
    return snapshot


def _trajectory_event(outcome: Any, snapshot: Mapping[str, Any], metrics: Any) -> dict[str, Any]:
    return {
        "update_index": int(snapshot["update_index"]),
        "selected_target_ids": list(outcome.decision.selected_member_ids),
        "primary_lanes": {
            str(row.member_id): row.primary_lane for row in outcome.decision.summaries
        },
        "committed_member_id": outcome.team_outcome.audit_metadata["committed_member_id"],
        "committed_candidate_id": outcome.team_outcome.committed_candidate_id,
        "termination_reason": outcome.team_outcome.termination_reason,
        "funnel": dict(outcome.team_outcome.funnel),
        "cost": asdict(outcome.team_outcome.cost),
        "p_i": dict(snapshot["p_i"]),
        "total_member_row_opportunities": snapshot["total_member_row_opportunities"],
        "vote_correct": metrics.vote_correct_count,
        "vote_accuracy": metrics.plurality_vote_acc,
        "member_correct": list(metrics.per_agent_correct_counts),
        "realizability_transitions": [
            asdict(row) for row in outcome.realizability_transitions
        ],
    }


async def execute(prep: Path, run_root: Path) -> dict[str, Any]:
    _authorize()
    _verify_freeze(prep)
    if run_root.exists():
        raise FileExistsError("fresh run root required; resume and automatic retry are forbidden")
    run_root.mkdir(parents=True)
    optimize_rows = _rows(prep / "splits_private/optimize100.csv")
    shadow_rows = _rows(prep / "splits_private/shadow50.csv")
    _reference, frozen_rows, parent_prompt = _load_frozen_parent()
    if optimize_rows != frozen_rows:
        raise RuntimeError("prepared Optimize100 differs from frozen Phase-B parent")
    ledger = BoundedLedger(run_root / "ledger.jsonl")
    cfg = _config(
        run_root,
        optimize_path=prep / "splits_private/optimize100.csv",
        shadow_path=prep / "splits_private/shadow50.csv",
    )
    system = Seed78System(cfg, arm=ARM, ledger=ledger, raw_cache={})
    system.set_run_identity(
        build_run_identity(
            cfg,
            train_rows=optimize_rows,
            val_rows=shadow_rows,
            test_rows=[],
            workspace=ROOT,
        )
    )
    baseline_snapshot = _materialize_frozen_baseline(
        system, optimize_rows=optimize_rows, parent_prompt=parent_prompt
    )
    baseline_metrics = system.active_probe_metrics()
    tracker = SymmetryBreakingTrajectory(
        baseline_vote_correct=baseline_metrics.vote_correct_count,
        baseline_p_i=baseline_snapshot["p_i"],
    )

    loop = asyncio.get_running_loop()
    task_builder = LocalTaskBuilder()
    holder: dict[str, FrozenResponsibilitySnapshot] = {}
    assignment_factory = SystemResponsibilityAssignmentFactory(
        system=system,
        snapshot_reader=lambda: holder["value"],
        task_builder=task_builder,
    )
    local_solver = SystemLocalSolverEvaluator(
        system=system,
        loop=loop,
        stage=system.set_stage,
        accounting=system.common.accounting,
        solver_contract_id=COMMON_SOLVER_CONTRACT_ID,
        output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION,
    )
    official = GEPALocalPromptOptimizer(
        evaluator=local_solver,
        reflection_lm=ReflectionLM(system),
        accounting_reader=system.optimizer_accounting,
        run_root=run_root / "local_gepa",
    )
    optimizer = ContextualOptimizer(
        official,
        local_solver,
        execution_context_from_system(
            system,
            local_no_update_patience=3,
            team_no_update_patience=NO_COMMIT_PATIENCE,
            saturation_mode="sequential_symmetry_breaking_bounded_pilot",
        ),
    )
    update = {"value": -1}
    evaluator = SystemTeamCandidateEvaluator(
        system=system,
        shadow_probe=system.build_probe(shadow_rows),
        loop=loop,
        stage=system.set_stage,
        accounting=system.common.accounting,
        update_index_reader=lambda: update["value"],
    )
    controller = TeamSearchController(
        responsibility=assignment_factory,
        task_builder=task_builder,
        local_optimizer=optimizer,
        evaluator=evaluator,
        selector=CommonSafeTeamCandidateSelector(),
        committer=SystemTeamCommitter(
            system=system,
            evaluator=evaluator,
            update_index_reader=lambda: update["value"],
        ),
    )
    scheduler = PrimaryResponsibilityPersistentRealizabilityScheduler()
    binding = PrimaryResponsibilityOnlineBinding(
        scheduler=scheduler,
        assignment_factory=assignment_factory,
        controller=controller,
    )

    events: list[dict[str, Any]] = []
    stop_reason = "max_opportunities"
    for update_index in range(MAX_OPPORTUNITIES):
        update["value"] = update_index
        frozen = freeze_current_responsibility(system, update_index=update_index)
        holder["value"] = frozen
        outcome = await binding.run_opportunity(
            TeamSearchRequest(
                seed=SEED,
                update_index=update_index,
                team_state_hash=system.team_prompt_state_hash(),
                local_metric_budget=LOCAL_GEPA_METRIC_BUDGET,
                solver_contract_id=COMMON_SOLVER_CONTRACT_ID,
                output_contract_id=SOLVER_OUTPUT_CONTRACT_VERSION,
            ),
            assigned=frozen.assigned,
            current_margin_by_question=frozen.current_margin_by_question,
        )
        state = system.persist_endpoint_identifiability_state(
            update_index=update_index,
            trigger="post_online_opportunity",
            committed_target_member=outcome.team_outcome.audit_metadata["committed_member_id"],
        )
        metrics = system.active_probe_metrics()
        tracker.observe(
            committed=outcome.team_outcome.committed_candidate_id is not None,
            vote_correct=metrics.vote_correct_count,
            p_i=state["p_i"],
        )
        event = _trajectory_event(outcome, state, metrics)
        events.append(event)
        write_json(run_root / f"opportunity_{update_index:02d}.json", event)
        write_json(
            run_root / "checkpoint_private.json",
            {
                "completed_opportunities": len(events),
                "team_identity": _profile_identity(system),
                "prompts": [agent.current_prompt for agent in system.agents],
                "scheduler_state": scheduler.state.checkpoint_payload(),
                "trajectory": tracker.payload(),
                "provider_attempts": ledger.attempts,
                "successful_provider_calls": ledger.successes,
            },
        )
        reason = tracker.stop_reason(
            max_opportunities=MAX_OPPORTUNITIES,
            max_commits=MAX_SAFE_COMMITS,
            no_commit_patience=NO_COMMIT_PATIENCE,
        )
        if reason is not None:
            stop_reason = reason
            break

    trajectory = tracker.payload()
    reflection_attempt_rows = [
        row for row in system.llm.calls if row.get("client_role") == "optimizer"
    ]
    reflection_successes = sum(bool(row.get("success")) for row in reflection_attempt_rows)
    if reflection_successes > REFLECTION_SUCCESSFUL_CALL_CEILING:
        raise RuntimeError("successful reflection call ceiling exceeded")
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "execution_gate": "PASS",
        "baseline": {
            "source_state_hash": SOURCE_STATE_HASH,
            "parent_prompt_sha256": PARENT_PROMPT_HASH,
            "materialized_team_prompt_state_hash": baseline_snapshot[
                "team_prompt_state_hash"
            ],
            "materialized_profile_state_sha256": baseline_snapshot[
                "profile_state_sha256"
            ],
            "vote_correct": baseline_metrics.vote_correct_count,
            "p_i": baseline_snapshot["p_i"],
        },
        "stop_reason": stop_reason,
        "trajectory": trajectory,
        "events": events,
        "scheduler_telemetry": scheduler.telemetry_summary(),
        "final_team_identity": _profile_identity(system),
        "provider_attempts": ledger.attempts,
        "successful_provider_calls": ledger.successes,
        "successful_reflection_calls": reflection_successes,
        "reflection_attempt_records": len(reflection_attempt_rows),
        "validation50_calls": 0,
        "test50_calls": 0,
    }
    write_json(run_root / "execution_summary_private.json", summary)
    return summary


def audit(prep: Path, run_root: Path) -> dict[str, Any]:
    _verify_freeze(prep)
    summary = _read_json(run_root / "execution_summary_private.json")
    checks = {
        "execution_gate": summary["execution_gate"] == "PASS",
        "homogeneous_lock": sum(summary["baseline"]["p_i"].values()) == 0,
        "opportunity_ceiling": summary["trajectory"]["opportunities"] <= MAX_OPPORTUNITIES,
        "commit_ceiling": summary["trajectory"]["safe_commits"] <= MAX_SAFE_COMMITS,
        "provider_ceiling": summary["successful_provider_calls"] <= SUCCESSFUL_PROVIDER_CALL_CEILING,
        "transport_ceiling": summary["provider_attempts"] <= TRANSPORT_ATTEMPT_CEILING,
        "reflection_ceiling": summary["successful_reflection_calls"]
        <= REFLECTION_SUCCESSFUL_CALL_CEILING,
        "validation50_zero": summary["validation50_calls"] == 0,
        "test50_zero": summary["test50_calls"] == 0,
    }
    result = {
        "gate": "PASS" if all(checks.values()) else "HOLD",
        "checks": checks,
        "api_calls": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
    }
    write_json(run_root / "audit.json", result)
    return result


def analyze(prep: Path, run_root: Path, report: Path) -> dict[str, Any]:
    gate = audit(prep, run_root)
    if gate["gate"] != "PASS":
        raise RuntimeError("integrity audit must pass before interpretation")
    if report.exists():
        raise FileExistsError("fresh report root required")
    summary = _read_json(run_root / "execution_summary_private.json")
    result = {
        "experiment_id": EXPERIMENT_ID,
        "evidence_type": "single_state_prospective_online_mechanism_pilot",
        "classifier": classify_symmetry_breaking_trajectory(
            summary["trajectory"], integrity_passed=True
        ),
        "stop_reason": summary["stop_reason"],
        "trajectory": summary["trajectory"],
        "events": summary["events"],
        "provider_attempts": summary["provider_attempts"],
        "successful_provider_calls": summary["successful_provider_calls"],
        "validation50_calls": 0,
        "test50_calls": 0,
    }
    report.mkdir(parents=True)
    write_json(report / "summary.json", result)
    write_json(report / "audit.json", gate)
    write_json(report / "protocol_freeze.json", _read_json(prep / "protocol_freeze.json"))
    return result


def preflight() -> dict[str, Any]:
    verify_frozen_gepa_engine_contract()
    _reference, rows, prompt = _load_frozen_parent()
    items, _raw = metadata()
    assignment = construct_assignment(items)
    optimize_ids = {
        hashlib.sha256(str(row["question"]).encode("utf-8")).hexdigest()
        for row in rows
    }
    protocol = protocol_document()
    capacity = protocol["pipeline"]["local_gepa_budget_capacity"]
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    authorized = manifest["api_authorization"]["authorized"] is True
    authorization_state_valid = (
        authorized and manifest["status"] == "RUNNING"
    ) or (not authorized and manifest["status"] == "PREFLIGHT_PASS")
    checks = {
        "frozen_parent_rows": len(rows) == 100,
        "optimize100_identity": optimize_ids
        == set(assignment["fold_a"] + assignment["fold_b"]),
        "frozen_parent_prompt": hashlib.sha256(prompt.encode("utf-8")).hexdigest() == PARENT_PROMPT_HASH,
        "zero_call_materialization": protocol["baseline"]["initialization_provider_calls"] == 0,
        "official_gepa": protocol["pipeline"]["official_gepa"] == asdict(GEPAOptimizerConfig()),
        "local_budget": capacity["max_accepted_children"] == 1,
        "two_targets": protocol["pipeline"]["target_branches_per_opportunity"] == 2,
        "max_one_commit": protocol["pipeline"]["max_one_commit_per_opportunity"] is True,
        "fixed_stops": protocol["stopping"] == {
            "first_vote_improvement": True,
            "max_safe_commits": 4,
            "max_opportunities": 8,
            "consecutive_no_commit_patience": 6,
            "efficacy_adaptation": False,
        },
        "heldout_zero": protocol["access"]["validation50"] == "zero access"
        and protocol["access"]["test50"] == "zero access",
        "authorization_state_valid": authorization_state_valid,
    }
    return {
        "gate": "PASS" if all(checks.values()) else "HOLD",
        "checks": checks,
        "ready_to_run": authorized and manifest["status"] == "RUNNING",
        "hold_reason": None if authorized else "API_AUTHORIZATION_PENDING",
        "protocol_sha256": sha256_json(protocol),
        "api_calls": 0,
        "validation50_calls": 0,
        "test50_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.prep)
    elif args.preflight:
        result = preflight()
    elif args.execute:
        result = asyncio.run(execute(args.prep, args.run))
    elif args.audit:
        result = audit(args.prep, args.run)
    else:
        result = analyze(args.prep, args.run, args.report)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
