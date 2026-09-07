from __future__ import annotations

from scripts.run_common_contract_seed77_sensitivity import (
    CONDITIONS,
    _variant_request,
)


QUESTION = "Question?\nOptions:\n(A) A\n(B) B"
PROMPT = "Use evidence carefully."


def test_factorial_contains_exactly_four_unique_conditions() -> None:
    assert len(CONDITIONS) == 4
    assert len({name for name, _, _ in CONDITIONS}) == 4
    assert {(newline, sent) for _, newline, sent in CONDITIONS} == {
        ("LF", False),
        ("LF", True),
        ("CRLF", False),
        ("CRLF", True),
    }


def test_request_variants_change_only_newline_and_provider_seed() -> None:
    lf = _variant_request(prompt=PROMPT, question=QUESTION, newline="LF", send_seed=False)
    crlf = _variant_request(prompt=PROMPT, question=QUESTION, newline="CRLF", send_seed=False)
    seeded = _variant_request(prompt=PROMPT, question=QUESTION, newline="LF", send_seed=True)
    assert "seed" not in lf
    assert seeded["seed"] == 77
    assert {key: value for key, value in seeded.items() if key != "seed"} == lf
    assert lf["messages"][0] == crlf["messages"][0]
    assert "\r\n" not in lf["messages"][1]["content"]
    assert "\r\n" in crlf["messages"][1]["content"]
    assert lf["model"] == crlf["model"] == seeded["model"] == "qwen3-8b"
    assert lf["extra_body"] == {"enable_thinking": False}
