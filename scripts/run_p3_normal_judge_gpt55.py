#!/usr/bin/env python
"""GPT-5.5 rejudge with normal strategy-judge-equivalent information.

Population: target does not include option_contrast, but the original automatic
judge's primary label is option_contrast.

GPT-5.5 is not shown the target strategy, model identity, run name, or original
automatic labels. It is shown the same kind of information as the normal
single-trace family judge: taxonomy labels, major-family tree, family
definitions, agent/task/hash/answer metadata, and the single trace.

For auditability, the generated packet stores the taxonomy and the normal
judge context in each row, and also writes the shared context to a separate
JSON file.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.utils import infer_strategy_family_major, strategy_family_major_categories  # noqa: E402
from run_p3_prompt_following_validation import (  # noqa: E402
    extract_json_object,
    load_candidates,
    sample_candidates,
    truncate,
    write_csv,
    write_jsonl,
)


NORMAL_JUDGE_PROMPT_VERSION = "normal_single_trace_family_judge_equivalent"


SYSTEM_PROMPT = """You judge the reasoning strategy family for exactly one agent trace.
Ignore answer correctness. Judge only the reasoning trajectory.
Do not use other agents, group behavior, vote results, or gold answers.
Choose the most specific existing leaf family that captures the trace.
Do not output major/coarse category labels and never output an 'other' family.
Return strict JSON only."""


REASONING_SUMMARY_REQUIREMENTS = """Reasoning summary requirements:
- Write a detailed natural-language paragraph with at least about 60 words when the trace contains enough information.
- Maximum length: 512 tokens.
- Preserve the semantic structure of the trace as much as possible.
- Describe how the agent understands the problem, what information it prioritizes, how it organizes intermediate reasoning, whether it compares options, reasons backward, constructs constraints, derives equations, estimates, verifies, or handles uncertainty, and how it converges to a final answer.
- Focus on reasoning trajectory and method, not answer correctness.
- Do not include evaluative judgments about quality, such as saying the reasoning is thorough, careful, robust, weak, effective, or well-structured.
- Do not mention gold answers, vote results, other agents, or group behavior.
- Use continuous natural language suitable for embedding-based comparison; do not make this field a bullet list."""


FAMILY_DEFINITIONS = {
    "decomposition": "breaks a task into sub-goals, components, assumptions, or intermediate claims",
    "symbolic_formulation": "translates the problem into variables, symbols, equations, logical forms, or structured representations",
    "algebraic_derivation": "manipulates symbolic relations to derive a needed expression or conclusion",
    "equation_solving": "sets up and solves equations, systems, or inequalities for unknown quantities",
    "direct_computation": "computes, recalls, or applies a short formula directly with minimal intermediate structure",
    "case_analysis": "splits the solution into mutually relevant cases, branches, or conditions",
    "exhaustive_enumeration": "lists possibilities systematically and evaluates them to find the valid answer",
    "constraint_propagation": "uses constraints to narrow possible states, values, or relationships step by step",
    "option_elimination": "removes impossible or less plausible answer options until a remaining option is selected",
    "comparative_reasoning": "compares alternatives, quantities, hypotheses, or criteria to decide among them",
    "backward_reasoning": "starts from the desired result, answer form, or goal condition and reasons backward",
    "consistency_verification": "checks whether an intermediate or final conclusion satisfies the original conditions",
    "counterexample_search": "searches for disconfirming examples or edge cases",
    "proof_by_contradiction": "assumes the negation or an incompatible claim and derives a contradiction",
    "invariant_reasoning": "uses conserved quantities, stable relationships, monotonicity, or unchanged properties",
    "symmetry_reasoning": "uses interchangeable roles, mirrored structures, or symmetry to simplify the task",
    "probabilistic_reasoning": "reasons with likelihoods, conditional probabilities, uncertainty, or stochastic structure",
    "expected_value_reasoning": "uses averages, expectation, weighted outcomes, or long-run value calculations",
    "combinatorial_counting": "counts arrangements, selections, paths, or possibilities using combinatorial structure",
    "pattern_generalization": "detects a recurring pattern and extends or generalizes it",
    "inductive_reasoning": "infers a general rule from examples, smaller cases, or observed regularities",
    "analogy_mapping": "maps the problem to a similar known structure or transfers a parallel solution pattern",
    "causal_reasoning": "tracks cause-effect relationships, mechanisms, interventions, or explanatory chains",
    "temporal_sequential_reasoning": "reasons over order, time, process steps, or before-after relationships",
    "spatial_visualization": "uses mental diagrams, geometry, layout, orientation, or spatial transformations",
    "definition_application": "applies definitions, terminology, or conceptual criteria directly",
    "rule_based_classification": "matches facts to a rule, category, diagnostic criterion, or decision procedure",
    "theorem_property_application": "uses a known theorem, law, identity, property, or domain principle",
    "edge_case_analysis": "tests boundary conditions, special cases, degeneracies, or limiting scenarios",
    "dimensional_unit_analysis": "uses units, dimensions, scales, or quantity types to constrain the answer",
    "optimization_extremal_reasoning": "seeks maxima, minima, worst cases, best cases, or extremal configurations",
    "approximation_bounding": "uses estimates, bounds, orders of magnitude, or inequalities to locate the answer",
    "simulation_tracing": "steps through a process, algorithm, scenario, or state transition explicitly",
    "recursive_reasoning": "uses recurrence, self-similar structure, dynamic programming, or reduction to smaller instances",
    "abductive_inference": "selects the best explanation for observed facts among plausible hypotheses",
    "counterfactual_reasoning": "changes an assumption or condition to test what would follow",
    "concept_definition_match": "matches the stem or options to a concept, term, definition, or canonical description",
    "option_contrast": "compares answer choices against each other to decide which best fits the stem",
    "distractor_elimination": "identifies and removes distractor choices based on specific flaws, mismatches, or impossibilities",
    "option_contradiction_check": "tests options for direct contradiction with the stem, known facts, or stated constraints",
    "answer_to_stem_backward_check": "starts from a candidate answer and reasons backward to see whether it explains or satisfies the stem",
    "stem_evidence_alignment": "anchors the decision in explicit clues, quoted wording, data, or evidence from the question stem",
    "scope_qualifier_analysis": "focuses on qualifiers such as all, most, except, always, never, best, least, or not to control the answer scope",
    "negation_exception_handling": "handles negative or exception-seeking stems such as NOT, EXCEPT, false, incorrect, or least likely",
    "rule_or_principle_application": "applies a named rule, law, theorem, framework, or domain principle to select the answer",
    "causal_mechanism_reasoning": "explains mechanisms, cause-effect chains, interventions, or process drivers before choosing",
    "historical_context_reasoning": "uses historical period, actor, event, institution, or context to disambiguate options",
    "scientific_model_reasoning": "uses a scientific model, theory, mechanism, or experimental setup to reason through options",
    "quantitative_formula_application": "selects or applies a quantitative formula, numerical relationship, or measurement rule",
    "classification_taxonomy_reasoning": "classifies the case into a category, type, school, syndrome, structure, or taxonomy",
    "example_counterexample_reasoning": "uses examples or counterexamples to support or reject candidate options",
    "statistical_method_reasoning": "uses statistical design, inference, validation, variable, or error-metric reasoning",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def prompt_hash(text: str) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()[:12]


def load_taxonomy_payload(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def load_taxonomy_labels(path: Path) -> list[str]:
    payload = load_taxonomy_payload(path)
    labels = payload.get("labels", []) if isinstance(payload, dict) else []
    return [str(x) for x in labels]


def make_normal_judge_context(
    labels: list[str],
    taxonomy_path: str = "",
    taxonomy_updated_at: str = "",
) -> dict[str, Any]:
    return {
        "prompt_version": NORMAL_JUDGE_PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "task_instruction": "Assign reasoning family labels for this single trace only.",
        "secondary_label_instruction": "Return primary_family and secondary_family. If only one clear strategy is present, set secondary_family equal to primary_family.",
        "reasoning_summary_requirements": REASONING_SUMMARY_REQUIREMENTS,
        "confidence_instructions": [
            "confidence is your confidence that the assigned reasoning-family labels are accurate.",
            "It is not a score for answer correctness, reasoning quality, or trace quality.",
            "Use high confidence, usually 0.70-0.95, when the trace clearly matches existing leaf families.",
            "If the trace can be mapped to an existing leaf family, give at least 0.60 confidence even when the trace is incomplete, messy, low quality, or answer-incorrect.",
            "Do not lower confidence because the reasoning is weak, verbose, terse, or mathematically wrong; only lower it when the family mapping itself is uncertain.",
            "Use low confidence below 0.40 only when you are genuinely unsure about the family assignment, or when no existing leaf family appears to fit the trace.",
            "Before assigning confidence below 0.40, first try to choose the closest existing leaf family; use low confidence only if that closest mapping would be unreliable.",
        ],
        "evidence_span_requirements": [
            "evidence_spans must be exact contiguous substrings copied from the trace.",
            "Do not paraphrase, shorten by rewriting, or invent evidence spans.",
        ],
        "existing_leaf_families": list(labels),
        "taxonomy_path": taxonomy_path,
        "taxonomy_updated_at": taxonomy_updated_at,
        "label_policy": "You must output only labels from Existing leaf families, or a genuinely new reusable leaf family label.",
        "disallowed_major_labels": [
            "representation_formalization",
            "algebra_computation",
            "logical_proof",
            "probability_statistics",
            "induction_pattern",
            "process_structure_simulation",
            "optimization_boundary_meta",
        ],
        "major_family_tree": strategy_family_major_categories(),
        "family_definitions_base_set": dict(FAMILY_DEFINITIONS),
        "allowed_input_context_fields": [
            "agent_id",
            "task_type",
            "question_hash",
            "trace_hash",
            "trace_length",
            "extracted_answer",
        ],
        "return_json_schema": {
            "agent_id": 0,
            "primary_family": "...",
            "secondary_family": "...",
            "reasoning_summary": "...",
            "strategy_steps": ["...", "..."],
            "distinctive_features": ["...", "..."],
            "evidence_spans": ["short exact span from trace", "..."],
            "confidence": 0.85,
            "reason": "...",
        },
    }


def make_packet_and_key(
    sampled: list[dict[str, Any]],
    max_trace_chars: int,
    normal_judge_context: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    packets: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for i, cand in enumerate(sampled, start=1):
        blinded_id = f"P3NJ{i:04d}"
        clean_trace = normalize_spaces(cand["trace"])
        allowed_input_context = {
            "agent_id": int(cand["agent_id"]),
            "task_type": "mmlu",
            "question_hash": str(cand["question_hash"]),
            "trace_hash": prompt_hash(clean_trace),
            "trace_length": len(clean_trace),
            "extracted_answer": normalize_spaces(cand.get("answer", ""))[:80],
        }
        packet = {
            "blinded_id": blinded_id,
            "normal_judge_equivalent_context": normal_judge_context,
            "allowed_input_context": allowed_input_context,
            "agent_id": allowed_input_context["agent_id"],
            "task_type": allowed_input_context["task_type"],
            "question_hash": allowed_input_context["question_hash"],
            "trace_hash": allowed_input_context["trace_hash"],
            "trace_length": allowed_input_context["trace_length"],
            "extracted_answer": allowed_input_context["extracted_answer"],
            "trace": truncate(clean_trace, max_trace_chars),
            "annotation_fields": {
                "primary_family": "",
                "secondary_family": "",
                "notes_optional": "",
            },
        }
        packets.append(packet)

        key = {k: v for k, v in cand.items() if k != "trace"}
        key["blinded_id"] = blinded_id
        key["target_families"] = "|".join(cand["target_families"])
        key["target_majors"] = "|".join(cand["target_majors"])
        key_rows.append(key)
    return packets, key_rows


def build_user_prompt(packet: dict[str, Any], labels: list[str], max_trace_chars: int) -> str:
    context = packet.get("normal_judge_equivalent_context")
    if not isinstance(context, dict):
        context = make_normal_judge_context(labels)
    context_labels = [str(x) for x in context.get("existing_leaf_families", labels)]
    definitions = context.get("family_definitions_base_set", FAMILY_DEFINITIONS)
    if not isinstance(definitions, dict):
        definitions = FAMILY_DEFINITIONS
    definition_lines = "\n".join([f"- {k}: {v}." for k, v in definitions.items()])
    major_tree = context.get("major_family_tree", strategy_family_major_categories())
    if not isinstance(major_tree, dict):
        major_tree = strategy_family_major_categories()
    major_lines = "\n".join(
        [f"- {major}: {', '.join(str(x) for x in families)}" for major, families in major_tree.items()]
    )
    confidence_lines = "\n".join([f"- {x}" for x in context.get("confidence_instructions", [])])
    evidence_lines = "\n".join([f"- {x}" for x in context.get("evidence_span_requirements", [])])
    disallowed = ", ".join(str(x) for x in context.get("disallowed_major_labels", []))
    cleaned_trace = truncate(normalize_spaces(packet.get("trace", "")), max_trace_chars)
    return (
        "Assign reasoning family labels for this single trace only.\n"
        "Return primary_family and secondary_family. If only one clear strategy is present, set secondary_family equal to primary_family.\n"
        f"{context.get('reasoning_summary_requirements', REASONING_SUMMARY_REQUIREMENTS)}\n"
        "Also provide strategy_steps, distinctive_features, and evidence_spans copied as short spans from the trace.\n\n"
        "Confidence meaning:\n"
        f"{confidence_lines}\n\n"
        "Evidence span requirements:\n"
        f"{evidence_lines}\n\n"
        f"Existing leaf families: {', '.join(context_labels)}.\n"
        f"{context.get('label_policy', 'You must output only labels from Existing leaf families, or a genuinely new reusable leaf family label.')}\n"
        f"Do NOT output major/coarse category labels such as {disallowed}.\n"
        "Major-family tree:\n"
        f"{major_lines}\n\n"
        "Family definitions (base set):\n"
        f"{definition_lines}\n\n"
        "Allowed input context:\n"
        f"- agent_id: {packet.get('agent_id')}\n"
        f"- task_type: {packet.get('task_type')}\n"
        f"- question_hash: {packet.get('question_hash')}\n"
        f"- trace_hash: {packet.get('trace_hash')}\n"
        f"- trace_length: {packet.get('trace_length')}\n"
        f"- extracted_answer: {packet.get('extracted_answer')}\n\n"
        "Single agent trace:\n"
        f"{cleaned_trace}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "agent_id": 0,\n'
        '  "primary_family": "...",\n'
        '  "secondary_family": "...",\n'
        '  "reasoning_summary": "...",\n'
        '  "strategy_steps": ["...", "..."],\n'
        '  "distinctive_features": ["...", "..."],\n'
        '  "evidence_spans": ["short exact span from trace", "..."],\n'
        '  "confidence": 0.85,\n'
        '  "reason": "..."\n'
        "}"
    )


def normalize_eval(blinded_id: str, model: str, raw_response: str, labels: set[str]) -> dict[str, Any]:
    obj = extract_json_object(raw_response)
    primary = normalize_spaces(obj.get("primary_family", "")).lower().replace("-", "_").replace(" ", "_")
    secondary = normalize_spaces(obj.get("secondary_family", "")).lower().replace("-", "_").replace(" ", "_")
    if primary not in labels:
        primary = "invalid_or_unknown"
    if secondary not in labels:
        secondary = primary
    confidence = float(obj.get("confidence", 0.0) or 0.0)
    confidence = max(0.0, min(1.0, confidence))
    exact_option_match = int(primary == "option_contrast")
    option_pair_match = int(primary == "option_contrast" or secondary == "option_contrast")
    diagnosis = "original_judge_supported" if exact_option_match else "judge_taxonomy_questioned"
    return {
        "blinded_id": blinded_id,
        "evaluator_model": model,
        "gpt_primary_family": primary,
        "gpt_secondary_family": secondary,
        "gpt_confidence": confidence,
        "gpt_primary_is_option_contrast": exact_option_match,
        "gpt_primary_or_secondary_is_option_contrast": option_pair_match,
        "diagnosis": diagnosis,
        "gpt_reasoning_summary": str(obj.get("reasoning_summary", "")),
        "gpt_strategy_steps": json.dumps(obj.get("strategy_steps", []), ensure_ascii=False),
        "gpt_distinctive_features": json.dumps(obj.get("distinctive_features", []), ensure_ascii=False),
        "gpt_evidence_spans": json.dumps(obj.get("evidence_spans", []), ensure_ascii=False),
        "gpt_reason": str(obj.get("reason", "")),
        "raw_response": raw_response,
        "parse_ok": int(primary != "invalid_or_unknown"),
    }


async def call_openai_chat(
    client: Any,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    retry_sleep: float,
) -> str:
    last_err = None
    for attempt in range(max(1, max_retries)):
        started = time.time()
        try:
            req = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            resp = await asyncio.wait_for(req, timeout=timeout) if timeout > 0 else await req
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_err = exc
            elapsed = time.time() - started
            print(f"[P3NJ][WARN] attempt={attempt + 1}/{max_retries} elapsed={elapsed:.2f}s error={exc}", flush=True)
            if attempt + 1 < max_retries:
                await asyncio.sleep(max(0.0, retry_sleep) * (attempt + 1))
    raise RuntimeError(f"GPT normal-judge rejudge failed after {max_retries} attempts: {last_err}")


def read_existing(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("blinded_id", "")): row for row in read_jsonl(path) if row.get("blinded_id")}


async def run_gpt(packet_path: Path, out_jsonl: Path, labels: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))
    packets = read_jsonl(packet_path)
    existing = read_existing(out_jsonl) if int(args.resume) else {}
    rows: list[dict[str, Any] | None] = [None] * len(packets)
    label_set = set(labels)
    sem = asyncio.Semaphore(max(1, int(getattr(args, "eval_parallelism", 12))))

    async def _eval_one(idx: int, packet: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        blinded_id = str(packet.get("blinded_id", ""))
        if blinded_id in existing and int(existing[blinded_id].get("parse_ok", 0) or 0):
            return idx, existing[blinded_id]
        async with sem:
            print(f"[P3NJ] {idx + 1}/{len(packets)} blinded_id={blinded_id} model={args.evaluator_model}", flush=True)
            raw = await call_openai_chat(
                client,
                args.evaluator_model,
                build_user_prompt(packet, labels, args.max_trace_chars),
                args.temperature,
                args.max_tokens,
                args.llm_call_timeout,
                args.max_retries,
                args.retry_sleep,
            )
            return idx, normalize_eval(blinded_id, args.evaluator_model, raw, label_set)

    tasks = [asyncio.create_task(_eval_one(idx, packet)) for idx, packet in enumerate(packets)]
    for fut in asyncio.as_completed(tasks):
        idx, row = await fut
        rows[idx] = row
        write_jsonl(out_jsonl, [r for r in rows if r is not None])
    return [r for r in rows if r is not None]


def join_key_eval(key_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row["blinded_id"]): row for row in key_rows}
    rows = []
    for ev in eval_rows:
        bid = str(ev.get("blinded_id", ""))
        if bid in by_id and int(ev.get("parse_ok", 0) or 0):
            joined = {**by_id[bid], **ev}
            targets = [x for x in str(joined.get("target_families", "")).split("|") if x]
            target_majors = {infer_strategy_family_major(x) for x in targets}
            gpt_primary = str(joined.get("gpt_primary_family", ""))
            gpt_secondary = str(joined.get("gpt_secondary_family", ""))
            exact = int(gpt_primary in targets or gpt_secondary in targets)
            same_major = int(
                exact
                or infer_strategy_family_major(gpt_primary) in target_majors
                or infer_strategy_family_major(gpt_secondary) in target_majors
            )
            joined["gpt_target_exact_hit"] = exact
            joined["gpt_target_same_major_hit"] = same_major
            rows.append(joined)
    return rows


def safe_mean(values: list[Any]) -> float:
    nums = [float(v) for v in values if v not in {None, ""}]
    return float(statistics.mean(nums)) if nums else 0.0


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in keys)].append(row)
    out = []
    for key, vals in sorted(groups.items()):
        rec = {k: v for k, v in zip(keys, key)}
        rec.update(
            {
                "n": len(vals),
                "gpt_primary_option_contrast_rate": safe_mean([v.get("gpt_primary_is_option_contrast") for v in vals]),
                "gpt_pair_option_contrast_rate": safe_mean([v.get("gpt_primary_or_secondary_is_option_contrast") for v in vals]),
                "gpt_target_exact_hit_rate": safe_mean([v.get("gpt_target_exact_hit") for v in vals]),
                "gpt_target_same_major_hit_rate": safe_mean([v.get("gpt_target_same_major_hit") for v in vals]),
                "original_judge_supported_rate": safe_mean([int(v.get("diagnosis") == "original_judge_supported") for v in vals]),
                "judge_taxonomy_questioned_rate": safe_mean([int(v.get("diagnosis") == "judge_taxonomy_questioned") for v in vals]),
                "mean_confidence": safe_mean([v.get("gpt_confidence") for v in vals]),
            }
        )
        out.append(rec)
    return out


def md_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(md_cell(v) for v in row) + " |")
    return lines


def build_summary_md(
    sampled_count: int,
    candidate_count: int,
    joined: list[dict[str, Any]],
    context_path: str = "p3_normal_judge_context.json",
) -> str:
    lines = [
        "# P3 GPT-5.5 Normal-Judge Rejudge",
        "",
        "目标：抽样 `target` 不包含 `option_contrast`、但原自动 judge 的 primary 标签为 `option_contrast` 的 trace，让 GPT-5.5 在接近正常 judge 的输入条件下重判。",
        "",
        "GPT-5.5 输入包含：完整 taxonomy leaf labels、major-family tree、base family definitions、reasoning_summary 要求、confidence/evidence 规则、返回 JSON schema、agent_id、task_type、question_hash、trace_hash、trace_length、extracted_answer、Single agent trace。",
        "",
        "GPT-5.5 输入不包含：目标策略、模型身份、run 名称、原自动 judge 标签、gold answer、group/vote 信息。",
        "",
        f"- normal_judge_context_file: `{context_path}`",
        f"- candidate_count: {candidate_count}",
        f"- sampled_count: {sampled_count}",
        f"- evaluated_count: {len(joined)}",
        "- 每条 `p3_normal_judge_packet.jsonl` 也内嵌 `normal_judge_equivalent_context`，方便逐样本审计。",
        "",
        "判读规则：",
        "",
        "- 如果 GPT-5.5 也把 primary 判为 `option_contrast`：原 judge 的 option_contrast 判定被支持，更像模型/trace 本身确实是 option-style。",
        "- 如果 GPT-5.5 判为非 `option_contrast`：原 judge/taxonomy 的 option_contrast 吸附问题被支持。",
        "- 如果 GPT-5.5 把 `option_contrast` 放在 secondary：说明 trace 是混合策略，原 judge 可能把次要的选项格式提升成 primary。",
        "",
    ]
    if not joined:
        lines.extend(
            [
                "当前只生成了待评审样本包，还没有 GPT-5.5 评审结果。",
                "",
                "运行评审命令：",
                "",
                "```powershell",
                "conda run -n DL python scripts\\run_p3_normal_judge_gpt55.py --run_gpt 1 --sample_size 776",
                "```",
            ]
        )
        return "\n".join(lines) + "\n"

    overall = summarize(joined, [])
    by_target = summarize(joined, ["agent_id", "target_families"])
    by_model = summarize(joined, ["model", "agent_id", "target_families"])
    lines.extend(["## Overall", ""])
    lines.extend(
        md_table(
            ["n", "GPT primary option", "GPT pair option", "original judge supported", "judge/taxonomy questioned", "confidence"],
            [
                [
                    overall[0]["n"],
                    overall[0]["gpt_primary_option_contrast_rate"],
                    overall[0]["gpt_pair_option_contrast_rate"],
                    overall[0]["original_judge_supported_rate"],
                    overall[0]["judge_taxonomy_questioned_rate"],
                    overall[0]["mean_confidence"],
                ]
            ],
        )
    )
    lines.extend(["", "## By Target", ""])
    lines.extend(
        md_table(
            ["agent", "target", "n", "GPT primary option", "GPT pair option", "original judge supported", "judge/taxonomy questioned", "confidence"],
            [
                [
                    r["agent_id"],
                    r["target_families"],
                    r["n"],
                    r["gpt_primary_option_contrast_rate"],
                    r["gpt_pair_option_contrast_rate"],
                    r["original_judge_supported_rate"],
                    r["judge_taxonomy_questioned_rate"],
                    r["mean_confidence"],
                ]
                for r in by_target
            ],
        )
    )
    lines.extend(["", "## By Model And Target", ""])
    lines.extend(
        md_table(
            ["model", "agent", "target", "n", "GPT primary option", "GPT pair option", "judge/taxonomy questioned"],
            [
                [
                    r["model"],
                    r["agent_id"],
                    r["target_families"],
                    r["n"],
                    r["gpt_primary_option_contrast_rate"],
                    r["gpt_pair_option_contrast_rate"],
                    r["judge_taxonomy_questioned_rate"],
                ]
                for r in by_model
            ],
        )
    )
    return "\n".join(lines) + "\n"


def build_summary_md_clean(
    sampled_count: int,
    candidate_count: int,
    joined: list[dict[str, Any]],
    context_path: str = "p3_normal_judge_context.json",
) -> str:
    lines = [
        "# P3 GPT-5.5 Normal Taxonomy Judge 复核",
        "",
        "目标：抽样 `target` 不包含 `option_contrast`、但原自动 judge 的 primary 标签为 `option_contrast` 的 trace，让 GPT-5.5 在接近正常 taxonomy judge 的输入条件下重新判定策略标签。",
        "",
        "GPT-5.5 输入包含：完整 taxonomy leaf labels、major-family tree、family definitions、reasoning_summary 要求、confidence/evidence 规则、返回 JSON schema、agent_id、task_type、question_hash、trace_hash、trace_length、extracted_answer、single agent trace。",
        "",
        "GPT-5.5 输入不包含：目标策略、模型身份、run 名称、原自动 judge 标签、gold answer、group/vote 信息。",
        "",
        f"- normal_judge_context_file: `{context_path}`",
        f"- candidate_count: {candidate_count}",
        f"- sampled_count: {sampled_count}",
        f"- evaluated_count: {len(joined)}",
        "- 每条 `p3_normal_judge_packet.jsonl` 也内嵌 `normal_judge_equivalent_context`，方便逐样本审计。",
        "",
        "判读规则：",
        "",
        "- 如果 GPT-5.5 也把 primary 判为 `option_contrast`：原 judge 的 option_contrast 判定得到支持，更像 trace 本身确实是 option-style。",
        "- 如果 GPT-5.5 判为非 `option_contrast`：支持原 judge/taxonomy 存在 option_contrast 吸附问题。",
        "- 如果 GPT-5.5 把 `option_contrast` 放在 secondary：说明 trace 是混合策略，原 judge 可能把次要选项比较提升成 primary。",
        "",
    ]
    if not joined:
        lines.extend(
            [
                "当前只生成了待评审样本包，还没有 GPT-5.5 评审结果。",
                "",
                "运行评审命令：",
                "",
                "```powershell",
                "python scripts\\run_p3_normal_judge_gpt55.py --runs_root prove_experiments\\p3_analysis_runs --run_gpt 1 --sample_size 776",
                "```",
            ]
        )
        return "\n".join(lines) + "\n"

    overall = summarize(joined, [])
    by_target = summarize(joined, ["agent_id", "target_families"])
    by_model = summarize(joined, ["model", "agent_id", "target_families"])
    lines.extend(["## 总体结果", ""])
    lines.extend(
        md_table(
            [
                "n",
                "GPT primary option",
                "GPT pair option",
                "GPT target exact",
                "GPT target same-major",
                "original judge supported",
                "judge/taxonomy questioned",
                "confidence",
            ],
            [
                [
                    overall[0]["n"],
                    overall[0]["gpt_primary_option_contrast_rate"],
                    overall[0]["gpt_pair_option_contrast_rate"],
                    overall[0]["gpt_target_exact_hit_rate"],
                    overall[0]["gpt_target_same_major_hit_rate"],
                    overall[0]["original_judge_supported_rate"],
                    overall[0]["judge_taxonomy_questioned_rate"],
                    overall[0]["mean_confidence"],
                ]
            ],
        )
    )
    lines.extend(["", "## 按目标策略", ""])
    lines.extend(
        md_table(
            [
                "agent",
                "target",
                "n",
                "GPT primary option",
                "GPT pair option",
                "GPT target exact",
                "GPT target same-major",
                "judge/taxonomy questioned",
            ],
            [
                [
                    r["agent_id"],
                    r["target_families"],
                    r["n"],
                    r["gpt_primary_option_contrast_rate"],
                    r["gpt_pair_option_contrast_rate"],
                    r["gpt_target_exact_hit_rate"],
                    r["gpt_target_same_major_hit_rate"],
                    r["judge_taxonomy_questioned_rate"],
                ]
                for r in by_target
            ],
        )
    )
    lines.extend(["", "## 按模型和目标策略", ""])
    lines.extend(
        md_table(
            ["model", "agent", "target", "n", "GPT primary option", "GPT target exact", "GPT target same-major", "judge/taxonomy questioned"],
            [
                [
                    r["model"],
                    r["agent_id"],
                    r["target_families"],
                    r["n"],
                    r["gpt_primary_option_contrast_rate"],
                    r["gpt_target_exact_hit_rate"],
                    r["gpt_target_same_major_hit_rate"],
                    r["judge_taxonomy_questioned_rate"],
                ]
                for r in by_model
            ],
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default="prove_experiments/runs")
    parser.add_argument("--taxonomy_path", default="taxonomies/mmlu_reasoning_family_taxonomy.json")
    parser.add_argument("--out_dir", default="prove_experiments/p3_normal_judge_gpt55")
    parser.add_argument("--sample_size", type=int, default=776)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_trace_chars", type=int, default=3500)
    parser.add_argument("--run_gpt", type=int, default=0)
    parser.add_argument("--evaluator_model", default="gpt-5.5")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1600)
    parser.add_argument("--llm_call_timeout", type=float, default=180.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--eval_parallelism", type=int, default=4)
    parser.add_argument("--resume", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    taxonomy_payload = load_taxonomy_payload(Path(args.taxonomy_path))
    labels = [str(x) for x in taxonomy_payload.get("labels", [])]
    normal_judge_context = make_normal_judge_context(
        labels,
        taxonomy_path=str(Path(args.taxonomy_path)),
        taxonomy_updated_at=str(taxonomy_payload.get("updated_at", "")),
    )
    candidates = load_candidates(Path(args.runs_root))
    sampled = sample_candidates(candidates, args.sample_size, args.seed)
    packets, key_rows = make_packet_and_key(sampled, args.max_trace_chars, normal_judge_context)

    packet_path = out_dir / "p3_normal_judge_packet.jsonl"
    context_path = out_dir / "p3_normal_judge_context.json"
    key_path = out_dir / "p3_normal_judge_key.csv"
    eval_path = out_dir / "p3_normal_judge_evaluations.jsonl"
    rows_path = out_dir / "p3_normal_judge_analysis_rows.csv"
    summary_path = out_dir / "p3_normal_judge_summary.md"

    write_jsonl(packet_path, packets)
    context_path.write_text(json.dumps(normal_judge_context, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    write_csv(key_path, key_rows)

    eval_rows: list[dict[str, Any]] = []
    if int(args.run_gpt):
        eval_rows = asyncio.run(run_gpt(packet_path, eval_path, labels, args))
    elif eval_path.exists():
        eval_rows = read_jsonl(eval_path)

    joined = join_key_eval(key_rows, eval_rows)
    write_csv(rows_path, joined)
    summary_path.write_text(
        build_summary_md_clean(len(sampled), len(candidates), joined, context_path.name),
        encoding="utf-8-sig",
    )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
