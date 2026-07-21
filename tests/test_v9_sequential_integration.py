import asyncio
import hashlib
from types import SimpleNamespace

from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.optimization.prompt_update_controller import PromptUpdateMixin
from multi_dataset_diverse_rl.persistence.checkpoint import build_training_checkpoint, restore_system_state
from multi_dataset_diverse_rl.policy import AgentState
from multi_dataset_diverse_rl.sequential_state import sequential_team_metrics
from multi_dataset_diverse_rl.utils import plurality_vote_with_diagnostics


def _profile(prompt, correctness, answers):
    return {
        "prompt": prompt,
        "prompt_hash": prompt,
        "answer_vector": answers,
        "correctness_vector": correctness,
        "invalid_vector": [0, 0],
        "trace_embedding_vector_per_question": [[1.0, 0.0], [0.0, 1.0]],
        "question_hashes": ["q0", "q1"],
        "gold_answers": ["A", "A"],
    }


class _SequentialSmokeSystem(PromptUpdateMixin):
    def __init__(self):
        self.cfg = Config(
            method_version="v9_state_conditioned_error",
            state_diversity_constraints_enabled=False,
            state_full_probe_acceptance_candidates=1,
            epochs=1,
            train_size=2,
        )
        self.agents = [AgentState(f"p{index}") for index in range(5)]
        self.fixed_acceptance_probe_data = [
            {"question": "q0", "answer": "A"},
            {"question": "q1", "answer": "A"},
        ]
        self.task_spec = SimpleNamespace(
            parse_gold=lambda answer, question: str(answer),
            match_answer=lambda answer, gold: str(answer) == str(gold),
        )
        profiles = [
            _profile("p0", [0, 0], ["B", "B"]),
            _profile("p1", [1, 0], ["A", "B"]),
            _profile("p2", [1, 0], ["A", "B"]),
            _profile("p3", [0, 1], ["B", "A"]),
            _profile("p4", [0, 1], ["B", "A"]),
        ]
        self.profile_map = {(index, f"p{index}"): profile for index, profile in enumerate(profiles)}
        self.profile_map[(0, "p0_new")] = _profile("p0_new", [1, 1], ["A", "A"])
        self.current_sequential_profiles = profiles
        self.initial_sequential_profiles = [dict(profile) for profile in profiles]
        self.initial_sequential_team_metrics = [
            sequential_team_metrics(
                profiles, ["A", "A"], ["q0", "q1"], agent_id, self.cfg,
                vote_fn=plurality_vote_with_diagnostics,
                match_fn=self.task_spec.match_answer,
            )
            for agent_id in range(5)
        ]
        self.fixed_probe_state_snapshot = {
            "active_prompt_hashes": [f"p{index}" for index in range(5)]
        }
        self.fixed_probe_snapshot_refresh_count = 0
        self.full_probe_missing_pair_evaluation_count = 0
        self.full_probe_cache_hit_count = 0
        self._profile_cache = set()
        self.sequential_update_history = []
        self.sequential_recent_accepted_prompt_hashes = []
        self.sequential_agent_order_index_by_epoch = {"1": 1}
        self.update_logs = []
        self.prompt_history = []
        self.cost_summary = {}
        self.state_parent_selection_source_counts = {}
        self.state_search_diagnostics = {}
        self.total_agent_update_count = 0
        self.recent_window_records = []

    @staticmethod
    def _hash(value):
        return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _normalized_prompt_hash(prompt):
        return str(prompt)

    @staticmethod
    def _make_beam_item(prompt, score, metrics, parent_id, generation, candidate_id=None):
        return {
            "id": candidate_id or str(prompt), "candidate_id": candidate_id or str(prompt),
            "prompt": str(prompt), "prompt_hash": str(prompt), "score": score,
            "metrics": dict(metrics), "parent_id": parent_id, "generation": generation,
        }

    async def _evaluate_prompt_on_stable_probe(self, agent_id, prompt, probe_data):
        key = (agent_id, str(prompt))
        if key not in self._profile_cache:
            self._profile_cache.add(key)
            self.full_probe_missing_pair_evaluation_count += len(probe_data)
        return dict(self.profile_map[key])

    async def refresh_state_conditioned_fixed_probe_snapshot(self, probe_data, *, epoch):
        self.current_sequential_profiles = [
            dict(self.profile_map[(agent_id, agent.current_prompt)])
            for agent_id, agent in enumerate(self.agents)
        ]
        self.fixed_probe_state_snapshot = {
            "snapshot_epoch": epoch,
            "active_prompt_hashes": [agent.current_prompt for agent in self.agents],
        }
        self.fixed_probe_snapshot_refresh_count += 1
        return self.fixed_probe_state_snapshot

    @staticmethod
    def _candidate_pool_source(item):
        return str(item.get("candidate_pool_source", item.get("source", "")))

    @staticmethod
    def _base_log_fields():
        return {}

    @staticmethod
    def _empty_cost_summary():
        return {}

    def _flush_jsonl(self, filename, rows):
        return None

    def _append_prompt_history_event(self, *args):
        return None

    def flush_prompt_history(self):
        return None


def test_deterministic_sequential_stage_b_smoke_and_checkpoint_resume():
    system = _SequentialSmokeSystem()
    context = SimpleNamespace(
        agent_id=0,
        agent=system.agents[0],
        epoch_id=1,
        step_id=10,
        evaluated=[{
            "candidate_id": "candidate",
            "prompt": "p0_new",
            "prompt_hash": "p0_new",
            "generation": 1,
            "source": "optimizer",
            "candidate_pool_source": "optimizer",
            "metrics": {"candidate_target_accuracy": 1.0, "candidate_team_accuracy": 1.0, "candidate_invalid_rate": 0.0},
        }],
        candidate_pool=[{"candidate_pool_source": "optimizer"}],
        parent_jobs=[{}],
        requested=1,
        optimizer_generation_records=[],
        optimizer_generation_summary={
            "student_candidate_count_raw": 1,
            "student_candidate_count_final": 1,
            "student_failure_stage": "none",
        },
    )
    changed, summary = asyncio.run(system._run_v9_sequential_stage_b(context))

    assert changed is True
    assert summary["active_prompt_changed_count"] == 1
    assert [agent.current_prompt for agent in system.agents] == ["p0_new", "p1", "p2", "p3", "p4"]
    assert system.fixed_probe_snapshot_refresh_count == 1
    assert system.fixed_probe_state_snapshot["active_prompt_hashes"][0] == "p0_new"
    assert len(system.agents[0].prompt_memory) <= 5
    rollback = next(
        item for item in system.agents[0].prompt_memory
        if item["prompt_memory_slot"] == "rollback_or_recent_success"
    )
    assert rollback["prompt_hash"] == "p0"
    assert system.sequential_update_history[-1]["previous_active_prompt_hash"] == "p0"
    assert system.sequential_update_history[-1]["rollback_prompt_hash"] == "p0"
    assert system.sequential_update_history[-1]["joint_team_combination_count"] == 0
    assert system.state_search_diagnostics["accepted_accuracy_regression_count"] == 0
    assert system.total_agent_update_count == 1
    assert system.sequential_update_history[-1]["optimizer_generation_diagnostics"][
        "student_candidate_count_final"
    ] == 1
    assert summary["student_failure_stage"] == "none"

    checkpoint = build_training_checkpoint(
        system.cfg, system, epoch_index=0, cursor=1, order=[0, 1],
        train_accumulators={}, best_score=0.0, best_epoch=0,
        epochs_without_improvement=0, stopped_early=False,
        no_effective_evolution_counter=0, no_effective_evolution_stopped=False,
        no_effective_evolution_reason="",
    )
    assert checkpoint["state"]["sequential_agent_order_index_by_epoch"] == {"1": 1}
    restored = _SequentialSmokeSystem()
    restore_system_state(restored, checkpoint["state"])
    assert restored.sequential_agent_order_index_by_epoch == {"1": 1}
    assert restored.agents[0].current_prompt == "p0_new"
    assert restored.agents[0].prompt_memory == system.agents[0].prompt_memory
