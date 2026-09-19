"""Read-only progressive team replay of five frozen accepted local mutations."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from infrastructure.common_solver_contract_v1.contract import (
    CONTRACT_SPEC, canonical_json_bytes,
)
from infrastructure.common_solver_contract_v1.evaluator import CommonSolverEvaluator, TransportResponse
from multi_dataset_diverse_rl.answer_formats import canonical_answer, match_answer
from multi_dataset_diverse_rl.candidate_selection import CandidateEvaluation
from multi_dataset_diverse_rl.evaluation.fixed_probe import ProbeExample, evaluate_candidate_profile, subset_profiles
from multi_dataset_diverse_rl.evaluation.prompt_question import PromptAnswer
from multi_dataset_diverse_rl.governance.authorization import require_api_authorization
from multi_dataset_diverse_rl.governance.manifest import validate_manifest
from multi_dataset_diverse_rl.governance.run_lifecycle import run_with_lifecycle
from multi_dataset_diverse_rl.shadow_gate import ShadowGateMetrics, evaluate_shadow_gate
from multi_dataset_diverse_rl.team_search.accepted_mutation_replay import (
    classify_transfer, common_safe, full_positive, minibatch_pass, transfer_decomposition,
)
from multi_dataset_diverse_rl.team_search.schemas import TeamMiniBatchMetrics
from multi_dataset_diverse_rl.provider_credentials import resolve_api_key, resolve_base_url


IDENTITY = "accepted_local_mutation_team_transfer_v1"
AUTH_ENV = "ACCEPTED_MUTATION_TEAM_TRANSFER_V1_AUTHORIZED"
MANIFEST = ROOT / "experiments/manifests/accepted_local_mutation_team_transfer_v1.yaml"
REPORT = ROOT / "reports/accepted_local_mutation_team_transfer_v1_prep_20260919"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_frozen_bundle(bundle: Path) -> tuple[dict, dict, dict]:
    freeze = read(bundle / "freeze.json")
    handoff = read(bundle / "HANDOFF.json")
    private = read(bundle / "private_bundle.json")
    if not freeze["READY_TO_RUN"] or not handoff["READY_TO_RUN"]:
        raise ValueError("READY_TO_RUN_required")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != freeze["commit"] or head != handoff["source"]["repository_commit"]:
        raise ValueError("frozen_commit_mismatch")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise ValueError("tracked_worktree_dirty")
    for relative, expected in freeze["source_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"source_hash_mismatch:{relative}")
    if sha(bundle / "private_bundle.json") != freeze["private_bundle_sha256"]:
        raise ValueError("private_bundle_hash_mismatch")
    if sha(REPORT / "accepted_mutation_freeze.json") != freeze["candidate_freeze_sha256"]:
        raise ValueError("candidate_freeze_hash_mismatch")
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    schema = read(ROOT / "infrastructure/experiment_manifest.schema.json")
    errors = validate_manifest(manifest, schema)
    if errors or sha(MANIFEST) != freeze["manifest_sha256"]:
        raise ValueError("manifest_invalid_or_changed")
    if private["baseline_team_hash"] != freeze["baseline_team_hash"]:
        raise ValueError("baseline_team_hash_mismatch")
    mutations = private["accepted_mutations"]
    if len(mutations) != 5 or any(
        hashlib.sha256(row["accepted_candidate_prompt"].encode()).hexdigest()
        != row["accepted_candidate_hash"] for row in mutations
    ):
        raise ValueError("accepted_mutation_identity_mismatch")
    return manifest, private, freeze


def _answer(value: str | None) -> PromptAnswer:
    valid = value is not None
    return PromptAnswer(
        answer=value or "", trace="", valid=valid,
        validity_status="valid" if valid else "invalid_baseline",
        terminal_invalid=not valid, first_attempt_valid=valid,
        raw_invalid_attempt_count=int(not valid),
    )


def _optimize_state(private: dict):
    tasks = private["phase_a_tasks"]
    ids = [f"seed78_update0_member{i}" for i in range(5)]
    rows0 = tasks[ids[0]]["search_examples"]
    # Keyword construction keeps the source schema explicit and avoids relying
    # on positional order if ProbeExample grows.
    examples = tuple(ProbeExample(
        question=row["input_payload"], question_hash=row["example_id"], gold_answer=row["gold"]
    ) for row in rows0)
    profiles = []
    for task_id in ids:
        rows = tasks[task_id]["search_examples"]
        if [row["example_id"] for row in rows] != [row.question_hash for row in examples]:
            raise ValueError("Optimize profile ordering mismatch")
        profiles.append(tuple(_answer(row["parent_output"]) for row in rows))
    return examples, profiles


def _evaluate(
    *, examples: Sequence[ProbeExample], active_profiles, candidate_profile,
    target: int, prompt: str, prompt_hash: str, assigned: set[str],
) -> CandidateEvaluation:
    return evaluate_candidate_profile(
        prompt=prompt, prompt_hash=prompt_hash, examples=examples,
        active_profiles=active_profiles, initial_profiles=active_profiles,
        candidate_profile=candidate_profile, target_agent_id=target,
        assigned_question_hashes=assigned,
        normalize_answer=lambda value: canonical_answer(value, "option_letter"),
        match_answer=lambda a, b: match_answer(a, b, "option_letter"),
        tie_break="abstain", seed=78, tau=1.0,
    )


class Runtime:
    def __init__(self, *, client, ledger_path: Path):
        self.client = client
        self.ledger_path = ledger_path
        self.sequence = self.attempts = self.successes = self.failures = 0
        self.stage: dict[str, Any] | None = None
        self.semaphore = asyncio.Semaphore(8)
        self.evaluator = CommonSolverEvaluator(transport=self._transport, retryable=self._retryable)

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        return (isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError))
                or getattr(exc, "status_code", None) in CONTRACT_SPEC.retry_status_codes)

    def _event(self, row: dict[str, Any]) -> None:
        self.sequence += 1
        with self.ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"sequence": self.sequence, **row}, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    async def _transport(self, request: dict) -> TransportResponse:
        if self.stage is None:
            raise RuntimeError("team replay Solver call lacks stage attribution")
        identity = hashlib.sha256(canonical_json_bytes(request)).hexdigest()
        self.attempts += 1
        attempt = self.attempts
        base = {"attempt": attempt, "request_identity": identity, **self.stage}
        self._event({"event": "provider_attempt_start", **base})
        try:
            response = await self.client.chat.completions.create(**request)
            if response.usage is None or len(response.choices) != 1:
                raise ValueError("provider_usage_or_choice_missing")
            choice = response.choices[0]
            result = TransportResponse(
                choice.message.content or "", int(response.usage.prompt_tokens),
                int(response.usage.completion_tokens), str(choice.finish_reason or ""),
            )
            if result.finish_reason != "stop":
                raise ValueError("provider_finish_reason_not_stop")
            self.successes += 1
            self._event({"event": "provider_attempt_success", **base,
                         "prompt_tokens": result.prompt_tokens,
                         "completion_tokens": result.completion_tokens,
                         "finish_reason": result.finish_reason})
            return result
        except Exception as exc:
            self.failures += 1
            self._event({"event": "provider_attempt_failure", **base,
                         "error_type": type(exc).__name__})
            raise

    async def profile(self, *, prompt: str, examples: Sequence[ProbeExample],
                      stage: str, mutation_id: str) -> tuple[PromptAnswer, ...]:
        self.stage = {"phase": stage, "mutation_id": mutation_id, "role": "solver"}
        try:
            async def one(example: ProbeExample) -> PromptAnswer:
                async with self.semaphore:
                    result = await self.evaluator.evaluate(
                        decision_procedure=prompt, question=example.question,
                    )
                parsed = result.response
                return PromptAnswer(
                    answer=parsed.answer, trace="", valid=parsed.valid,
                    validity_status=parsed.status,
                    raw_final_answer_payload=parsed.answer,
                    final_answer_line_count=parsed.final_answer_line_count,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.prompt_tokens + result.completion_tokens,
                    request_identity=result.request_identity,
                    terminal_invalid=not parsed.valid, first_attempt_valid=parsed.valid,
                    raw_invalid_attempt_count=int(not parsed.valid),
                )
            return tuple(await asyncio.gather(*(one(row) for row in examples)))
        finally:
            self.stage = None


async def replay(private: dict, runtime: Runtime) -> dict:
    examples, active_profiles = _optimize_state(private)
    index_by_id = {row.question_hash: i for i, row in enumerate(examples)}
    tasks = private["phase_a_tasks"]
    shadow_examples = tuple(ProbeExample(
        question=row["question"], question_hash=hashlib.sha256(row["question"].encode()).hexdigest(),
        gold_answer=canonical_answer(row["answer"], "option_letter"),
    ) for row in private["shadow50"])
    shadow_active_profiles = None
    results = []
    for mutation in private["accepted_mutations"]:
        target = mutation["target_member"]
        prompt = mutation["accepted_candidate_prompt"]
        candidate_hash = mutation["accepted_candidate_hash"]
        mutation_id = mutation["mutation_id"]
        assigned = {
            row["example_id"] for row in tasks[mutation["source_parent_task_id"]]["search_examples"]
            if "responsibility" in row["tags"]
        }
        mini_ids = mutation["team_minibatch_example_ids"]
        indices = tuple(index_by_id[value] for value in mini_ids)
        mini_examples, mini_active = subset_profiles(examples, active_profiles, indices)
        mini_candidate = await runtime.profile(
            prompt=prompt, examples=mini_examples,
            stage="team_minibatch_eval", mutation_id=mutation_id,
        )
        mini_eval = _evaluate(
            examples=mini_examples, active_profiles=mini_active,
            candidate_profile=mini_candidate, target=target, prompt=prompt,
            prompt_hash=candidate_hash, assigned=assigned,
        )
        mini_incumbent = _evaluate(
            examples=mini_examples, active_profiles=mini_active,
            candidate_profile=mini_active[target], target=target,
            prompt=tasks[mutation["source_parent_task_id"]]["parent_prompt"],
            prompt_hash=mutation["root_parent_hash"], assigned=assigned,
        )
        metrics = TeamMiniBatchMetrics(
            invalid_delta=mini_eval.competence.invalid_count - mini_incumbent.competence.invalid_count,
            vote_delta=mini_eval.team_outcome.vote_correct_count - mini_incumbent.team_outcome.vote_correct_count,
            target_delta=mini_eval.competence.correct_count - mini_incumbent.competence.correct_count,
            coalition_delta=mini_eval.marginal.net_vote_delta,
            responsibility_delta=mini_eval.marginal.assigned_residual_repair_count,
            broad_delta=mini_eval.member_gain.total_gain_count - mini_incumbent.member_gain.total_gain_count,
        )
        row: dict[str, Any] = {
            **{key: mutation[key] for key in (
                "mutation_id", "target_member", "root_parent_hash", "accepted_candidate_hash",
                "local_delta", "newly_fixed", "newly_broken", "source_proposal_index",
                "source_parent_task_id", "source_phase_b_run_identity",
            )},
            "execution_status": "COMPLETE", "team_minibatch_metrics": asdict(metrics),
            "team_minibatch_status": "PASS" if minibatch_pass(metrics) else "FAIL",
            "full_status": "NOT_REACHED", "full_positive_signal": None,
            "common_safe_status": "NOT_REACHED", "shadow_status": "NOT_REACHED",
        }
        if not minibatch_pass(metrics):
            results.append(row)
            continue
        full_profile = await runtime.profile(
            prompt=prompt, examples=examples, stage="team_full_eval", mutation_id=mutation_id,
        )
        full_eval = _evaluate(
            examples=examples, active_profiles=active_profiles,
            candidate_profile=full_profile, target=target, prompt=prompt,
            prompt_hash=candidate_hash, assigned=assigned,
        )
        active_eval = _evaluate(
            examples=examples, active_profiles=active_profiles,
            candidate_profile=active_profiles[target], target=target,
            prompt=tasks[mutation["source_parent_task_id"]]["parent_prompt"],
            prompt_hash=mutation["root_parent_hash"], assigned=assigned,
        )
        decision = common_safe(full_eval, active_eval)
        row.update({
            "full_status": "PASS", "full_positive_signal": full_positive(full_eval, active_eval),
            **transfer_decomposition(
                examples=examples, active_profiles=active_profiles,
                candidate_profile=full_profile, target_member=target,
                assigned_example_ids=assigned,
                match_answer=lambda a, b: match_answer(a, b, "option_letter"),
                active=active_eval, candidate=full_eval,
            ),
            "common_safe_status": "PASS" if decision.passed else "FAIL",
            "common_safe": asdict(decision),
        })
        if decision.passed:
            if shadow_active_profiles is None:
                baseline_prompt = tasks["seed78_update0_member0"]["parent_prompt"]
                baseline_shadow = await runtime.profile(
                    prompt=baseline_prompt, examples=shadow_examples,
                    stage="team_shadow_eval", mutation_id="shared_control",
                )
                shadow_active_profiles = [baseline_shadow] * 5
            candidate_shadow = await runtime.profile(
                prompt=prompt, examples=shadow_examples,
                stage="team_shadow_eval", mutation_id=mutation_id,
            )
            shadow_active = _evaluate(
                examples=shadow_examples, active_profiles=shadow_active_profiles,
                candidate_profile=shadow_active_profiles[target], target=target,
                prompt=tasks[mutation["source_parent_task_id"]]["parent_prompt"],
                prompt_hash=mutation["root_parent_hash"], assigned=set(),
            )
            shadow_candidate = _evaluate(
                examples=shadow_examples, active_profiles=shadow_active_profiles,
                candidate_profile=candidate_shadow, target=target, prompt=prompt,
                prompt_hash=candidate_hash, assigned=set(),
            )
            shadow = evaluate_shadow_gate(ShadowGateMetrics(
                incumbent_vote_correct=shadow_active.team_outcome.vote_correct_count,
                candidate_vote_correct=shadow_candidate.team_outcome.vote_correct_count,
                incumbent_target_correct=shadow_active.competence.correct_count,
                candidate_target_correct=shadow_candidate.competence.correct_count,
                row_count=len(shadow_examples),
            ))
            row.update({"shadow_status": "PASS" if shadow.passed else "FAIL",
                        "shadow": shadow.sanitized()})
        results.append(row)
    return {
        "execution_status": "EXECUTION_COMPLETE", "experiment_id": IDENTITY,
        "baseline_team_hash": private["baseline_team_hash"], "case_count": len(results),
        "cases": results, "classifier": classify_transfer(results),
        "accounting": {"solver": runtime.evaluator.accounting(),
                       "provider_attempts": runtime.attempts,
                       "provider_successes": runtime.successes,
                       "provider_failures": runtime.failures},
        "isolation": {"GEPA": 0, "Reflection": 0, "write_back": 0,
                      "persistent_realizability_update": 0,
                      "Validation50": 0, "Test50": 0},
    }


async def execute(bundle: Path, output: Path) -> None:
    if os.getenv(AUTH_ENV) != "1":
        raise PermissionError(f"{AUTH_ENV}=1_required")
    manifest, private, freeze = verify_frozen_bundle(bundle)
    require_api_authorization(
        manifest, phase="accepted_mutation_team_replay", role="solver",
        explicit_user_authorized=True,
    )
    if output.exists():
        raise FileExistsError("formal team-transfer run root must be fresh")
    output.mkdir(parents=True)

    async def operation() -> None:
        try:
            from openai import AsyncOpenAI
            _, key = resolve_api_key("DASHSCOPE_API_KEY")
            _, base = resolve_base_url("DASHSCOPE_BASE_URL")
            if not key or not base:
                raise ValueError("provider_credentials_unavailable")
            async with AsyncOpenAI(api_key=key, base_url=base, max_retries=0,
                                   timeout=CONTRACT_SPEC.timeout_seconds) as client:
                runtime = Runtime(client=client, ledger_path=output / "provider_ledger.jsonl")
                result = await replay(private, runtime)
            if runtime.successes > 800 or runtime.attempts > 3200:
                raise RuntimeError("team replay hard budget exceeded")
            verify_frozen_bundle(bundle)
            write_new(output / "execution.json", result)
            write_new(output / "completion.json", {
                "status": "EXECUTION_COMPLETE", "source_hashes_verified_after_execution": True,
            })
        except BaseException as exc:
            write_new(output / "abort.json", {
                "status": "EXECUTION_ABORTED", "error_type": type(exc).__name__,
            })
            raise

    await run_with_lifecycle(
        output / "run_lifecycle.json",
        identity={"experiment_id": IDENTITY, "frozen_commit": freeze["commit"],
                  "read_only_team_replay": True},
        operation=operation,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "execute"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "preflight":
        manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
        print(json.dumps({
            "status": "PASS" if args.bundle.joinpath("freeze.json").exists() else "AUTHORIZATION_FREEZE_REQUIRED",
            "api_authorized": manifest["api_authorization"]["authorized"],
            "formal_run_root_absent": not args.output.exists(),
        }))
    else:
        asyncio.run(execute(args.bundle, args.output))


if __name__ == "__main__":
    main()
