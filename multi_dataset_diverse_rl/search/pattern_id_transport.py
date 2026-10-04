"""Lossless opportunity-local Pattern support IDs; strict output membership."""
from copy import deepcopy
from .. import versions
from .schemas import SearchContractError
from .pattern_responsibility import SetLevelPatternProvider


class AliasSetLevelPatternProvider(SetLevelPatternProvider):
    support_id_transport=versions.PATTERN_SUPPORT_ID_ALIAS_VERSION

    def diagnose(self,payload):
        # Preserve frozen example order and every evidence field. IDs alone have
        # a bijective wire representation; no raw/approximate output recovery.
        wire=deepcopy(payload);mapping={}
        ids=[r['example_id'] for r in payload['examples']]
        if len(set(ids))!=len(ids):raise SearchContractError('PATTERN_UNIVERSE_DUPLICATE')
        for index,row in enumerate(wire['examples'],1):
            alias=f'e{index}';mapping[alias]=row['example_id'];row['example_id']=alias
        value=super().diagnose(wire)
        if not isinstance(value,dict) or set(value)!={'patterns','unassigned_ids'} or not isinstance(value['patterns'],list) or not isinstance(value['unassigned_ids'],list):
            raise SearchContractError('PATTERN_DISCOVERY_INVALID')
        decoded=deepcopy(value)
        def decode(ids):
            if not isinstance(ids,list) or any(not isinstance(x,str) or x not in mapping for x in ids):
                raise SearchContractError('PATTERN_DISCOVERY_INVALID_MEMBERSHIP')
            return [mapping[x] for x in ids]
        for p in decoded['patterns']:
            if not isinstance(p,dict) or 'support_ids' not in p:raise SearchContractError('PATTERN_DISCOVERY_INVALID')
            p['support_ids']=decode(p['support_ids'])
        decoded['unassigned_ids']=decode(decoded['unassigned_ids'])
        # Duplicate, overlap and omission remain for the unchanged strict
        # partition scorer. Never invent an assignment or merge an unknown ID.
        return decoded
