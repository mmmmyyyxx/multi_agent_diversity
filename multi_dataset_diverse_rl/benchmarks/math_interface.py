"""The sole active immutable MATH inference interface is V6."""
import hashlib
from .. import versions

MATH_SOLVER_INTERFACE_V6 = (
    "Solve the mathematical problem using the supplied decision procedure.\n\n"
    "Provide a clear, logically ordered solution showing the relevant reasoning, "
    "equations, intermediate calculations, and checks.\n\n"
    "End the response with exactly one final-answer line:\n"
    "FINAL_ANSWER: <answer>\n\n"
    "The final-answer payload must contain only the mathematical answer and may "
    "use mathematical or LaTeX notation. Keep the payload on the same line. "
    "The final-answer marker must appear exactly once, on the last nonempty line. "
    "Do not output anything after the final-answer line."
)


def v6_interface_contract():
    return dict(identity=versions.MATH_SOLVER_INTERFACE_V6_VERSION,
        sha256=hashlib.sha256(MATH_SOLVER_INTERFACE_V6.encode()).hexdigest(),
        parser_identity="math_verify_no_fallback_v1",
        solver_max_output_tokens=3600, reflection_max_output_tokens=1800)


solver_interface_contract=v6_interface_contract
