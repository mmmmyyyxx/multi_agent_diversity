"""Integrity failures and counterfactual controls for read-only proposal replay."""
from copy import deepcopy
import json

import pytest

from multi_dataset_diverse_rl.evaluation.proposal_contract_audit import (
    audit_proposals, extract_proposal, paired_identity_audit, replay_boundary, text_hash, verify_annotation,
)


class FakeIdentityBroker:
    def _request_identity(self, *, role, split, messages):
        body = dict(messages=messages, extra_body={"enable_thinking": False})
        return body, text_hash(json.dumps([role, split, body], sort_keys=True))


def fixture():
    broker = FakeIdentityBroker()
    prompt = "Check the final answer against all constraints."
    parent = "Derive a result."
    body, key = broker._request_identity(role="reflection", split="optimize", messages=[{"content": "synthetic"}])
    diagnostics = dict(nonthinking_wire_confirmed=True, reasoning_tokens=None,
                       diagnostics=dict(complete_single_fence=True, repetition_pathology=False))
    raw = "```\n" + prompt + "\n```"
    trace = [dict(role="reflection", request=body, request_sha256=key,
                  response=dict(text=raw, output_tokens=12, finish_reason="stop",
                                optimizer_generation_diagnostics=diagnostics))]
    annotation = dict(proposal_sha256=text_hash(prompt), semantic_class="BENIGN_REASONING",
                      evidence_spans=[dict(start=0, end=len(prompt), span_sha256=text_hash(prompt.casefold()))])
    callback = dict(event_type="proposal_end", proposal_hash=text_hash(prompt), iteration=1,
                    failed_checks=["forbidden_final_answer_marker"])
    trajectory = []
    for i in range(2):
        op = dict(opportunity_id=str(i), parent_prompt=parent, parent_state_id="state",
                  target_member=0, evidence=dict(mutation_evidence=[], search_validation_evidence=[]))
        trajectory.extend([dict(stage="OPPORTUNITY", opportunity=op),
                           dict(stage="EVALUATION", opportunity_id=str(i), evaluated=[],
                                search=dict(candidates=[], search_state=dict(callback_events=[callback])))])
    ledger = [dict(kind="SUCCESS", role="reflection", split="optimize", request_sha256=key,
                   response_sha256=text_hash(raw)),
              dict(kind="CACHE_HIT", role="reflection", split="optimize", request_sha256=key)]
    return trace, trajectory, ledger, {}, [annotation], broker


def test_join_preserves_null_metadata_and_distinguishes_unique_from_logical():
    result = audit_proposals(*fixture())
    assert (result["physical_proposals"], result["logical_proposals"], result["cache_replays"]) == (1, 2, 1)
    assert result["rows"][0]["reasoning_tokens"] is None
    assert result["strict_proposer_semantic_counterfactual_admissible"] == 1
    assert result["candidate_admission_changed"] is False


@pytest.mark.parametrize("attack", ["request", "response", "cache_order", "callback", "annotation", "parent", "heldout"])
def test_evidence_corruption_fails_closed(attack):
    trace, trajectory, ledger, examples, annotations, broker = deepcopy(fixture())
    if attack == "request": trace[0]["request_sha256"] = "0" * 64
    if attack == "response": ledger[0]["response_sha256"] = "0" * 64
    if attack == "cache_order": ledger.reverse()
    if attack == "callback": trajectory[1]["search"]["search_state"]["callback_events"][0]["failed_checks"] = []
    if attack == "annotation": annotations[0]["evidence_spans"][0]["span_sha256"] = "0" * 64
    if attack == "parent": trajectory[2]["opportunity"]["parent_prompt"] = "Other reasoning."
    if attack == "heldout": ledger[0]["split"] = "validation"
    with pytest.raises(ValueError):
        audit_proposals(trace, trajectory, ledger, examples, annotations, broker)


def test_lexical_mask_can_admit_output_controls_and_cannot_fix_append_only():
    benign = replay_boundary("Verify the final answer against constraints.", "Reason.", ())
    control = replay_boundary("Keep commentary out of the final answer block.", "Reason.", ())
    appended = replay_boundary("Reason. Verify the final answer.", "Reason.", ())
    assert benign["phrase_mask_boundary_accepts"] and control["phrase_mask_boundary_accepts"]
    assert appended["phrase_mask_failed_checks"] == ["append_only"]


def test_pinned_extractor_and_annotation_unicode_coordinates():
    assert extract_proposal("```text\nVerify constraints.\n```") == "Verify constraints."
    prompt = "ＦＩＮＡＬ answer"
    annotation = dict(proposal_sha256=text_hash(prompt), semantic_class="BENIGN_REASONING",
                      evidence_spans=[dict(start=0, end=12, span_sha256=text_hash("final answer"))])
    verify_annotation(prompt, annotation)
    annotation["evidence_spans"][0]["end"] = 100
    with pytest.raises(ValueError, match="out of bounds"):
        verify_annotation(prompt, annotation)


def test_persisted_validation_stage_names_and_duplicate_or_changed_scores():
    initial = dict(team="validation_initial", example_id="synthetic-id", member_correct=[True],
                   member_valid=[True], oracle_correct=True, vote_correct=False, request_output_hashes=["hash"])
    final = {**initial, "team": "validation_final"}
    rows = [initial, final]
    assert paired_identity_audit(rows, 1)["oracle_correct_vote_wrong"] == 1
    with pytest.raises(ValueError, match="inventory"):
        paired_identity_audit(rows + [initial], 1)
    with pytest.raises(ValueError, match="differ"):
        paired_identity_audit([initial, {**final, "vote_correct": True}], 1)
    with pytest.raises(ValueError, match="unknown"):
        paired_identity_audit([initial, {**final, "team": "final"}], 1)
