"""Historical JSON receipts supply integrity metadata, never runtime behavior."""
import hashlib
import json
from pathlib import Path
from ..search.schemas import SearchContractError


def verify_receipt_dependencies(root, receipt, seen=None):
    root=Path(root).resolve();seen=set() if seen is None else seen
    def visit(value):
        if isinstance(value,list):
            for child in value:visit(child)
        elif isinstance(value,dict):
            for key,relative in value.items():
                if not key.endswith('_path') or not isinstance(relative,str):continue
                digest=value.get('initial_team_artifact_sha256' if key=='initial_team_path' else key[:-5]+'_sha256')
                if digest is None:continue
                path=(root/relative).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise SearchContractError('CURRENT_PROVENANCE_RECEIPT_INVALID')
                actual=hashlib.sha256(path.read_bytes()).hexdigest()
                if key=='verify_settings_path':
                    actual=hashlib.sha256(json.dumps(json.loads(path.read_bytes()),sort_keys=True,separators=(',',':')).encode()).hexdigest()
                if actual!=digest:
                    raise SearchContractError('CURRENT_PROVENANCE_RECEIPT_HASH_MISMATCH: '+relative)
                if path not in seen and path.suffix=='.json':
                    seen.add(path);visit(json.loads(path.read_bytes()))
            for child in value.values():
                if isinstance(child,(dict,list)):visit(child)
    visit(receipt)
    return tuple(sorted(path.relative_to(root).as_posix() for path in seen))
