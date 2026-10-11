"""Direct optimizer evidence: reading requires no offsets or decoder tools."""

IDENTITY = 'DIRECT_READABLE_OPTIMIZER_EVIDENCE_V2'
INSTRUCTION = (
    'The solver_trajectory.observed_response string is the complete ordinary '
    'Solver response, including its written steps and final declaration. '
    'Read that text directly. The visible_solution string is only a convenient '
    'bounded preview; when feedback_truncated is true, continue reading the '
    'complete observed_response rather than limiting analysis to the preview. '
    'A response_truncated flag describes an incomplete Solver generation. '
    'These observations are not hidden reasoning. Any separately labeled '
    'reference_solution is the dataset worked solution, not Solver output.\n'
)
