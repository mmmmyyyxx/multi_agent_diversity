"""Bridge COMMON_SOLVER_CONTRACT_V1 into the Diversity optimization system."""

from __future__ import annotations

import hashlib

from multi_dataset_diverse_rl.evaluation.prompt_question import PromptAnswer

from .evaluator import CommonSolverEvaluator


class CommonContractSolverAdapter:
    """Return the domain PromptAnswer while retaining exact common accounting."""

    def __init__(self, evaluator: CommonSolverEvaluator) -> None:
        self.evaluator = evaluator
        self.last_transport_attempts = 0
        self.last_cache_hit = False

    async def solve(self, question: str, agent_id: int, prompt: str) -> PromptAnswer:
        del agent_id  # Agent identity must not enter the common request.
        result = await self.evaluator.evaluate(
            decision_procedure=prompt,
            question=question,
        )
        self.last_transport_attempts = result.transport_attempts
        self.last_cache_hit = result.cache_hit
        parsed = result.response
        raw = self.evaluator.cache[result.request_identity]
        response_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return PromptAnswer(
            answer=parsed.answer,
            trace=raw,
            valid=parsed.valid,
            validity_status=parsed.status,
            raw_final_answer_payload=parsed.answer,
            final_answer_line_count=parsed.final_answer_line_count,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
            response_hash=response_hash,
            request_identity=result.request_identity,
            solver_attempt_count=1,
            first_attempt_valid=parsed.valid,
            recovered_from_invalid=False,
            terminal_invalid=not parsed.valid,
            raw_invalid_attempt_count=int(not parsed.valid),
            attempt_validity_statuses=(parsed.status,),
            attempt_finish_reasons=(result.finish_reason,),
            attempt_response_hashes=(response_hash,),
            attempt_prompt_tokens=(result.prompt_tokens,),
            attempt_completion_tokens=(result.completion_tokens,),
            attempt_total_tokens=(result.prompt_tokens + result.completion_tokens,),
        )

    def accounting(self) -> dict[str, int]:
        return self.evaluator.accounting()
