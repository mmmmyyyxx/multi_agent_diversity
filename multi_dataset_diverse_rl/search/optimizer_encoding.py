"""Lossless optimizer-only wire encoding; private observations remain complete."""
from copy import deepcopy
import hashlib

from .schemas import SearchContractError

IDENTITY = 'LOSSLESS_OPTIMIZER_TRAJECTORY_ENCODING_V1'
INSTRUCTION = ('A visible_solution object with encoding '+IDENTITY+' is an exact '
    'Unicode slice [start:end] of the same trajectory observed_response. '
    'Read that slice as the written solution; the complete ordinary response '
    'and all provenance are retained. It is not hidden reasoning.\n')


def encode(value):
    """Reference a duplicated view only when the reference is smaller and exact."""
    if isinstance(value, list):
        return [encode(v) for v in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    result = {k: encode(v) for k, v in value.items()}
    raw, view = value.get('observed_response'), value.get('visible_solution')
    if isinstance(raw, str) and isinstance(view, str) and view:
        start = raw.find(view)
        reference = dict(encoding=IDENTITY, start=start, end=start+len(view),
            sha256=hashlib.sha256(view.encode()).hexdigest())
        # Including hash/offset overhead avoids increasing short packets.
        import json
        if start >= 0 and len(json.dumps(reference, separators=(',', ':'))) < len(json.dumps(view)):
            result['visible_solution'] = reference
    return result


def decode(value):
    """Deterministic verification/analysis view, never a provider response parser."""
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    result = {k: decode(v) for k, v in value.items()}
    view = result.get('visible_solution')
    if isinstance(view, dict) and view.get('encoding') == IDENTITY:
        raw = result.get('observed_response')
        if (set(view) != {'encoding', 'start', 'end', 'sha256'}
                or not isinstance(raw, str)
                or type(view['start']) is not int or type(view['end']) is not int
                or not 0 <= view['start'] <= view['end'] <= len(raw)):
            raise SearchContractError('OPTIMIZER_TRAJECTORY_REFERENCE_INVALID')
        text = raw[view['start']:view['end']]
        if hashlib.sha256(text.encode()).hexdigest() != view['sha256']:
            raise SearchContractError('OPTIMIZER_TRAJECTORY_REFERENCE_INVALID')
        result['visible_solution'] = text
    return result
