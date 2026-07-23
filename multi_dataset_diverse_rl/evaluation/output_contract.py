from __future__ import annotations


SOLVER_OUTPUT_CONTRACT_VERSION = "task_output_contract_v1"


def solver_output_contract(answer_format: str) -> str:
    fmt = str(answer_format or "").strip().lower()
    if fmt == "option_letter":
        payload = (
            "The final line must be exactly:\n"
            "FINAL_ANSWER: X\n\n"
            "Replace X with one uppercase option letter that appears in the question. "
            "Do not add parentheses, punctuation, explanation, or any other text after the letter."
        )
    elif fmt == "yes_no":
        payload = "The final line must be exactly FINAL_ANSWER: yes or FINAL_ANSWER: no."
    elif fmt == "boolean":
        payload = "The final line must be exactly FINAL_ANSWER: true or FINAL_ANSWER: false."
    elif fmt == "valid_invalid":
        payload = "The final line must be exactly FINAL_ANSWER: valid or FINAL_ANSWER: invalid."
    elif fmt == "numeric":
        payload = (
            "The final line must be exactly FINAL_ANSWER: N, where N is only the numeric answer "
            "with no units, punctuation, or explanation."
        )
    else:
        payload = (
            "The final line must be exactly FINAL_ANSWER: <answer>, with only the answer payload "
            "after the colon and no trailing explanation."
        )
    return (
        f"Solver output contract ({SOLVER_OUTPUT_CONTRACT_VERSION}):\n"
        f"{payload}\n"
        "There must be exactly one FINAL_ANSWER line."
    )
