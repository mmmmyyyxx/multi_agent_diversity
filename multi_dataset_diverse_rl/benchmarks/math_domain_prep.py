"""Isolated source reference projection for evaluator preparation, never search."""
from pathlib import Path
import json
import sys
from .data_freeze import file_hash, reference_answer, SOURCE_PINS

CONTEXT = 'EVALUATOR_CONTRACT_PREP_CONTEXT'


def source_references(canonical_root: Path, *, expected_manifest_sha256: str, context: str):
    if context != CONTEXT:
        raise ValueError('EVALUATOR_PREPARATION_CONTEXT_REQUIRED')
    forbidden=('multi_dataset_diverse_rl.local_optimizers','multi_dataset_diverse_rl.provider_factory',
        'multi_dataset_diverse_rl.search.patterns','multi_dataset_diverse_rl.search.memory')
    if any(name.startswith(forbidden) for name in sys.modules):
        raise ValueError('EVALUATOR_PREPARATION_RUNTIME_IMPORT_FORBIDDEN')
    path=canonical_root/'manifests/math.json'
    if file_hash(path) != expected_manifest_sha256:
        raise ValueError('CANONICAL_MANIFEST_IDENTITY_MISMATCH')
    manifest=json.loads(path.read_bytes())
    if any(manifest['source'].get(k)!=v for k,v in SOURCE_PINS['math'].items()):
        raise ValueError('CANONICAL_SOURCE_IDENTITY_MISMATCH')
    for source,count in [('train',7500),('test',5000)]:
        path=canonical_root/'raw/math'/f'{source}.jsonl'
        expected=next(r for r in manifest['source']['canonical_sources'] if r['name']==source)
        if file_hash(path)!=expected['canonical_sha256']:
            raise ValueError('CANONICAL_SOURCE_HASH_MISMATCH')
        observed=0
        with path.open(encoding='utf-8') as stream:
            for index,line in enumerate(stream):
                row=json.loads(line)
                extracted=reference_answer(row['content']['solution'])
                if extracted!=row['reference_final_answer']:
                    raise ValueError('CANONICAL_REFERENCE_EXTRACTION_MISMATCH')
                # No problem, reasoning solution or prediction is projected.
                yield dict(stable_example_id=row['stable_example_id'],source_split=source,
                    source_index=index,input_sha256=row['input_sha256'],content_sha256=row['content_sha256'],
                    subject=row['content']['type'],level=row['content']['level'],reference=extracted)
                observed+=1
        if observed!=count:
            raise ValueError('MATH_SOURCE_COUNT_MISMATCH')
