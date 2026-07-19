"""Extracted TraceBeamSearchSystem responsibility mixin."""

from ..system_shared import *


class ArtifactMethodsMixin:
    def save_state(self, name: str, extra: Optional[Dict[str, Any]] = None):
        payload = {
            **self._base_log_fields(),
            "agents": [
                {
                    "agent_id": i,
                    "initial_prompt": a.initial_prompt,
                    "current_prompt": a.current_prompt,
                    "prompt_beam": a.prompt_beam,
                    "history": a.history,
                    "accept_count": a.accept_count,
                    "reject_count": a.reject_count,
                    **a.trajectory_state_dict(),
                }
                for i, a in enumerate(self.agents)
            ],
            "extra": extra or {},
        }
        with open(os.path.join(self.cfg.out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _flush_jsonl(self, filename: str, rows: List[Dict[str, Any]]):
        if not hasattr(self, "artifact_writer"):
            from .artifacts import ArtifactWriter
            self.artifact_writer = ArtifactWriter(self.cfg.out_dir)
        self.artifact_writer.append_jsonl(filename, rows)

    def flush_update_logs(self):
        self._flush_jsonl("update_logs.jsonl", self.update_logs)
        self.update_logs = []
        self._flush_jsonl("trajectory_events.jsonl", self.trajectory_events)
        self.trajectory_events = []

    def flush_train_step_logs(self):
        self._flush_jsonl("train_step_logs.jsonl", self.train_step_logs)
        self.train_step_logs = []

    def flush_train_trace_history_logs(self):
        self._flush_jsonl("train_trace_history.jsonl", self.train_trace_history_logs)
        self.train_trace_history_logs = []

    def flush_test_trace_history_logs(self):
        self._flush_jsonl("test_trace_history.jsonl", self.test_trace_history_logs)
        self.test_trace_history_logs = []

    def _write_json_snapshot(self, filename: str, payload: Any):
        if not hasattr(self, "artifact_writer"):
            from .artifacts import ArtifactWriter
            self.artifact_writer = ArtifactWriter(self.cfg.out_dir)
        self.artifact_writer.write_json(filename, payload)

    def flush_prompt_history(self):
        self._write_json_snapshot("prompt_history.json", self.prompt_history)

    def flush_llm_call_logs(self):
        self._flush_jsonl("llm_calls.jsonl", self.llm_call_logs)
        self.llm_call_logs = []

    def write_cost_summary(self):
        self.cost_summary.update({
            "full_probe_cache_hits": int(getattr(self, "full_probe_cache_hit_count", 0)),
            "full_probe_missing_pair_evaluations": int(getattr(self, "full_probe_missing_pair_evaluation_count", 0)),
            "embedding_cache_hits": int(getattr(self, "mechanism_embedding_cache_hit_count", 0)),
            "embedding_cache_misses": int(getattr(self, "mechanism_embedding_cache_miss_count", 0)),
            "legacy_beam_refresh_calls": int(getattr(self, "legacy_beam_refresh_call_count", 0)),
            "joint_refresh_count": int(getattr(self, "joint_refresh_count", 0)),
            "joint_refresh_skipped_count": int(getattr(self, "joint_refresh_skipped_count", 0)),
            "new_full_probe_prompt_count": int(getattr(self, "new_full_probe_prompt_count", 0)),
            "new_full_probe_pair_count": int(getattr(self, "full_probe_missing_pair_evaluation_count", 0)),
            "offline_team_combination_count": int(getattr(self, "offline_team_combination_count", 0)),
            "team_level_solver_calls": int(getattr(self, "joint_team_solver_call_count", 0)),
            "tcs_teacher_calls": int(self.cost_summary.get("tcs_teacher_calls", 0)),
            "tcs_critic_calls": int(self.cost_summary.get("tcs_critic_calls", 0)),
            "tcs_rewrite_calls": int(self.cost_summary.get("tcs_rewrite_calls", 0)),
            "tcs_student_calls": int(self.cost_summary.get("tcs_student_calls", 0)),
            "open_exploration_calls": int(self.cost_summary.get("open_exploration_calls", 0)),
            "calls_saved_by_skipped_joint_refresh": int(getattr(self, "joint_refresh_skipped_count", 0))
            * int(getattr(self.cfg, "joint_representative_beam_size", 3))
            * len(getattr(self, "agents", [])),
            "calls_saved_by_dirty_prompt_cache": int(getattr(self, "full_probe_cache_hit_count", 0)),
            "calls_saved_by_tcs_round_reduction": int(self.cost_summary.get("calls_saved_by_tcs_round_reduction", 0)),
        })
        self._write_json_snapshot("cost_summary.json", self.cost_summary)
