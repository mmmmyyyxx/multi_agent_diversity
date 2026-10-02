"""Offline repository metadata validation and deterministic derived views."""
from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
from typing import Mapping

import jsonschema

from .registries import load_yaml, validate_lineage
from .source_identity import build_unified_source_identity

KINDS = {'FORMAL_EXPERIMENT','PILOT','DIAGNOSTIC','ABLATION','FORENSIC_AUDIT',
         'ZERO_API_AUDIT','ARCHITECTURE_REFACTOR','BENCHMARK_MIGRATION',
         'PROTOCOL_FREEZE','PREEXECUTION_FREEZE','INVALID_ATTEMPT',
    'DATASET_FREEZE_AND_PROTOCOL_MIGRATION','REAL_CANARY'}
ERAS = {'V17_V18_MEMBER_AWARE','GEPA_TWO_LAYER','FORMAL_V3_V4',
        'UNIFIED_TEAM_PROMPT_SEARCH','BENCHMARK_GENERALIZATION',
        'HISTORICAL_V15_V16_AND_EARLIER'}
STATUSES = {'COMPLETED','IN_PROGRESS','PREPARED_NOT_EXECUTED','IMPLEMENTED_NOT_EXECUTED',
            'STATUS_UNRESOLVED','SUPERSEDED','ARCHIVED','ABANDONED','INVALID','HOLD',
            'RUNNING','DRAFT','PREREGISTERED','TRAIN_FROZEN','READY','PREEXECUTION_FROZEN'}
REQUIRED = {'experiment_id','kind','era','method_family','parent_ids','status',
            'scientific_status','implementation_commit','execution_commit','result_commit',
            'manifest','report','classifier','active_for_new_work','historical_replay',
            'superseded_by','notes'}
CURRENT_DOCS = ('AGENTS.md','README.md','method.md','docs/CURRENT_ARCHITECTURE.md',
                'docs/design/CURRENT_SPEC.md','docs/research/CURRENT_RESEARCH_STATE.md')


def validate_manifest_v2(root: Path, manifest: Mapping) -> list[str]:
    schema = load_yaml(root/'experiments/schema/experiment_manifest_v2.schema.json')
    errors = [e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(manifest)]
    if manifest.get('lifecycle',{}).get('status') != 'DRAFT':
        for key in ('source_sha','dataset_manifest_identity','split_identity','preregistration_identity'):
            if manifest.get(key) is None:
                errors.append(f'frozen manifest requires {key}')
        for key,value in manifest.get('hash_closure',{}).items():
            if value is None:
                errors.append(f'frozen hash closure requires {key}')
    auth = manifest.get('authorization',{})
    if auth.get('real_api_authorized') and (
        manifest.get('lifecycle',{}).get('status')=='DRAFT' or
        not auth.get('attempt_id') or not auth.get('authorization_identity')):
        errors.append('real API authorization requires frozen attempt and explicit identity')
    return errors


def validate_registry_v2(root: Path, registry: Mapping) -> tuple[list[str],dict]:
    errors=[]; manifests={}; rows=registry.get('experiments',[])
    ids=[row.get('experiment_id') for row in rows]; known=set(ids)
    for ident,count in Counter(ids).items():
        if count>1:errors.append(f'duplicate experiment IDs: {ident}')
    for row in rows:
        ident=row.get('experiment_id')
        if REQUIRED-set(row):errors.append(f'{ident}: missing fields {sorted(REQUIRED-set(row))}')
        for key,allowed in [('kind',KINDS),('era',ERAS),('status',STATUSES)]:
            if row.get(key) not in allowed:errors.append(f'{ident}: invalid {key}')
        for key in ('active_for_new_work','historical_replay'):
            if not isinstance(row.get(key),bool):errors.append(f'{ident}: {key} must be boolean')
        for key in ('parent_ids','superseded_by'):
            values=row.get(key,[])
            if not isinstance(values,list):errors.append(f'{ident}: {key} must be list');continue
            if len(set(values))!=len(values):errors.append(f'{ident}: duplicate {key}')
            for parent in values:
                if parent not in known:errors.append(f'{ident}: dangling {key} {parent}')
        for key in ('manifest','report'):
            path=row.get(key)
            if path:
                target=(root/path).resolve()
                if not target.is_relative_to(root.resolve()) or not target.exists():
                    errors.append(f'{ident}: missing {key} {path}')
            elif not row.get('unavailable',{}).get(key):
                errors.append(f'{ident}: {key} unavailable reason required')
        if row.get('status')=='RUNNING' and not row.get('runtime_evidence'):
            errors.append(f'{ident}: stale RUNNING without runtime evidence')
        path=row.get('manifest')
        if path and (root/path).is_file():
            obj=load_yaml(root/path);manifests[ident]=obj
            if isinstance(obj,dict) and obj.get('schema_version')=='experiment_manifest_v2':
                errors.extend(f'{ident}: {e}' for e in validate_manifest_v2(root,obj))
            elif isinstance(obj,dict) and obj.get('schema_version')=='experiment_manifest_v1':
                from .manifest import validate_manifest
                import hashlib
                legacy_schema=load_yaml(root/'infrastructure/experiment_manifest.schema.json')
                schema_index=load_yaml(root/'experiments/manifest_schema_index.json')
                profile=next((r for r in schema_index['historical_manifests'] if r['path']==path),{})
                if profile.get('validation_profile')=='HISTORICAL_BESPOKE_IMMUTABLE':
                    normalized=(root/path).read_bytes().replace(b'\r\n',b'\n')
                    if hashlib.sha256(normalized).hexdigest()!=profile.get('normalized_git_blob_sha256'):
                        errors.append(f'{ident}: historical bespoke manifest mutated')
                else:
                    errors.extend(f'{ident}: {e}' for e in validate_manifest(obj,legacy_schema))
            # Legacy formats remain immutable; they are indexed, not silently migrated.
    return errors,manifests


def validate_lineage_v2(registry: Mapping,lineage: Mapping) -> list[str]:
    rows=registry.get('experiments',[]);known={r['experiment_id']:r for r in rows}
    errors,_=validate_lineage(lineage,known)
    nodes=lineage.get('nodes',[]);ids=[n.get('experiment_id') for n in nodes]
    if len(ids)!=len(set(ids)):errors.append('duplicate lineage nodes')
    if set(ids)!=set(known):errors.append('registry/lineage node sets differ')
    for n in nodes:
        if n.get('experiment_id') in known and n.get('kind')!=known[n['experiment_id']]['kind']:
            errors.append(f'lineage kind mismatch: {n["experiment_id"]}')
    for row in rows:
        parents={e['from'] for e in lineage.get('edges',[]) if e['to']==row['experiment_id']}
        if parents!=set(row.get('parent_ids',[])):errors.append(f'parent mismatch: {row["experiment_id"]}')
    return errors


def render_lineage_v2(root: Path) -> str:
    registry=load_yaml(root/'experiments/registry.yaml');lineage=load_yaml(root/'experiments/lineage.yaml')
    errors=validate_lineage_v2(registry,lineage)
    if errors:raise ValueError('; '.join(errors))
    rows=registry['experiments'];short={r['experiment_id']:f'n{i}' for i,r in enumerate(rows)}
    lines=['# Experiment Lineage','','DO NOT EDIT DIRECTLY. Generated from experiments/lineage.yaml.',
           'Registry provides node metadata; current_frontier.yaml provides readiness only.','',
           '## Current frontier','', '```yaml',
           (root/'experiments/current_frontier.yaml').read_text(encoding='utf-8').rstrip(),'```','',
           '## Experiment and engineering DAG','','Engineering edges are metadata progression, never efficacy evidence.','',
           '```mermaid','flowchart TD']
    for i,era in enumerate(sorted(ERAS)):
        lines.append(f'  subgraph era{i}["{era}"]')
        for r in rows:
            if r['era']==era:lines.append(f'    {short[r["experiment_id"]]}["{r["experiment_id"]}<br/>{r["kind"]}<br/>{r["status"]}"]')
        lines.append('  end')
    for e in lineage['edges']:lines.append(f'  {short[e["from"]]} -->|{e["relation"]}| {short[e["to"]]}')
    lines.extend(['```','','## Archived branches and unresolved evidence','',
                  '| Node | Era | Lifecycle | Scientific classifier |','|---|---|---|---|'])
    for r in rows:
        if r['historical_replay'] or r['status'] in {'INVALID','SUPERSEDED','STATUS_UNRESOLVED'}:
            lines.append(f'| {r["experiment_id"]} | {r["era"]} | {r["status"]} | {r.get("classifier") or "Not established"} |')
    return '\n'.join(lines)+'\n'


def report_paths(root: Path) -> list[str]:
    tracked=subprocess.check_output(['git','ls-files','--','reports'],cwd=root,text=True).splitlines()
    paths={str(Path(p).parent).replace('\\','/') for p in tracked if Path(p).name=='README.md'}
    paths.update('/'.join(p.split('/')[:2]) for p in tracked if len(p.split('/'))>2)
    # This milestone is the only new report allowed before staging. Untracked
    # retry evidence and arbitrary local directories are never scanned.
    own='reports/repository_hygiene_alignment_v1_20261001'
    if (root/own/'README.md').is_file():paths.add(own)
    return sorted(paths)


def build_report_index(root: Path) -> dict:
    registry=load_yaml(root/'experiments/registry.yaml');rows=registry['experiments'];entries=[]
    mapping={'FORENSIC_AUDIT':'FORENSIC','ZERO_API_AUDIT':'ZERO_API','PREEXECUTION_FREEZE':'PREEXECUTION',
             'PROTOCOL_FREEZE':'ZERO_API','ARCHITECTURE_REFACTOR':'ARCHITECTURE','BENCHMARK_MIGRATION':'MIGRATION',
             'DIAGNOSTIC':'DIAGNOSTIC','INVALID_ATTEMPT':'INVALID_ATTEMPT',
             'DATASET_FREEZE_AND_PROTOCOL_MIGRATION':'MIGRATION','REAL_CANARY':'REAL_CANARY'}
    for path in report_paths(root):
        matches=[r for r in rows if path in r.get('reports',[]) or path==r.get('report') or
                 (r.get('report') and path.startswith(r['report']+'/'))]
        row=matches[0] if matches else None;date=re.findall(r'20\d{6}',path)
        report_kind=mapping.get(row['kind'],'SCIENTIFIC_RESULT') if row else None
        if any(s in path for s in ('preexecution','preflight','_prep','precanary','refreeze')):report_kind='PREEXECUTION'
        if any(s in path for s in ('forensic','postmortem','_abort','incident')):report_kind='FORENSIC'
        if row and row.get('scientific_status','').startswith('INVALID'):report_kind='INVALID_ATTEMPT'
        documents=[p for p in (root/path).glob('*.md') if p.is_file()]
        document=next((p for p in documents if p.name=='README.md'),next(iter(sorted(documents)),None))
        entries.append(dict(path=path,document=document.relative_to(root).as_posix() if document else path,
            date=date[-1] if date else None,era=row['era'] if row else None,kind=report_kind,
            experiment_id=row['experiment_id'] if row else None,status=row['status'] if row else 'STATUS_UNRESOLVED',
            source_commit=row.get('execution_commit') or row.get('implementation_commit') if row else None,
            historical_current='current_governance_milestone' if path.endswith('repository_hygiene_alignment_v1_20261001') else 'historical_evidence',
            registry_referenced=row is not None,registry_ref='experiments/registry.yaml',
            context='Current/active wording refers only to report generation time.'))
    return {'schema_version':'report_index_v1','generated_from':['experiments/registry.yaml','tracked reports filesystem'],
            'reports_are_normative':False,'reports':entries}


def report_index_markdown(index: Mapping) -> str:
    lines=['# Reports Index','','DO NOT EDIT DIRECTLY. Generated from experiments/registry.yaml and tracked report paths.',
           'Reports are immutable evidence, never design authority. Unresolved metadata does not reclassify a result.','',
           '| Report | Date | Era | Kind | Registry node | Lifecycle |','|---|---|---|---|---|---|']
    for r in index['reports']:
        lines.append(f'| [{r["path"].removeprefix("reports/")}]({r["document"].removeprefix("reports/")}) | {r["date"] or "Unknown"} | {r["era"]} | {r["kind"]} | {r["experiment_id"]} | {r["status"]} |')
    return '\n'.join(lines)+'\n'


def import_guard(root: Path) -> list[str]:
    errors=[]
    for area in ('search','benchmarks'):
        for path in (root/'multi_dataset_diverse_rl'/area).rglob('*.py'):
            if path.name=='legacy_bbh_replay.py':continue
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else ([node.module or ''] if isinstance(node,ast.ImportFrom) else [])
                for name in names:
                    if name in {'scripts','reports'} or name.startswith(('scripts.','reports.')) or any(s in name for s in ('mars','team_search.controller','sequential_controller','production_formal_saturation')):
                        errors.append(f'{path.relative_to(root).as_posix()}: forbidden current import {name}')
    return errors


def audit_repository(root: Path, check_generated: bool=True) -> dict:
    registry=load_yaml(root/'experiments/registry.yaml');lineage=load_yaml(root/'experiments/lineage.yaml')
    registry_errors,manifests=validate_registry_v2(root,registry)
    lineage_errors=validate_lineage_v2(registry,lineage)
    docs=[]
    for path in CURRENT_DOCS:
        text=(root/path).read_text(encoding='utf-8')
        if 'sole active research architecture is Unified Team Prompt Search' not in text.replace('\n',' '):docs.append(f'{path}: Unified declaration missing')
        for pattern in [r'active research (?:direction|architecture) is (?:backend-neutral )?Layer',r'canonical runtime is `member_aware_peer_state_v15`']:
            if re.search(pattern,text,re.I):docs.append(f'{path}: stale active claim')
    for path in (*CURRENT_DOCS,'docs/REPOSITORY_MAP.md','docs/research/OPEN_QUESTIONS.md','reports/INDEX.md'):
        if not (root/path).is_file():
            docs.append(f'{path}: missing authority/view');continue
        text=(root/path).read_text(encoding='utf-8')
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
            if '://' in target or target.startswith('#'):continue
            target=target.split('#',1)[0]
            if not (root/path).parent.joinpath(target).exists():docs.append(f'{path}: broken link {target}')
    frontier=load_yaml(root/'experiments/current_frontier.yaml')
    if frontier.get('real_api_authorized') is not False:
        docs.append('frontier cannot authorize real APIs')
    readiness=frontier.get('real_execution_ready')
    if readiness == 'true_for_canary_only':
        path=frontier.get('canary_manifest')
        if not path or not (root/path).is_file():
            docs.append('canary readiness requires a frozen manifest')
        else:
            from .unified_execution import bound_preflight
            if bound_preflight(root,load_yaml(root/path))['blockers']:
                docs.append('canary readiness binding failed closed')
        if frontier.get('validation_access')!='not_authorized' or frontier.get('test_access')!='sealed':
            docs.append('canary readiness cannot unlock held-out access')
    elif readiness is not False:
        docs.append('frontier readiness must be closed or explicitly canary-only')
    registered={r['experiment_id'] for r in registry['experiments']}
    if frontier.get('last_governance_milestone') not in registered:docs.append('frontier node missing')
    current=frontier.get('current_experiment')
    if current != 'NO_AUTHORIZED_REAL_EXPERIMENT' and current not in registered:
        docs.append('current experiment is not registered')
    if readiness == 'true_for_canary_only' and path and (root/path).is_file():
        if load_yaml(root/path).get('experiment_id') != current:
            docs.append('current experiment differs from canary manifest identity')
    imports=import_guard(root)
    identity=build_unified_source_identity(root)
    bad=[f['path'] for f in identity['scopes']['scientific']['files'] if f['path'].startswith(('reports/','docs/archive/')) or f['path'] in {'README.md','method.md','AGENTS.md'}]
    archive_errors=[];archive_map=load_yaml(root/'reports/repository_hygiene_alignment_v1_20261001/historical_archive_map.json')
    import hashlib
    for entry in archive_map['archives']:
        text=(root/entry['archive']).read_bytes()
        if b'archive_status: HISTORICAL_REPLAY_ONLY' not in text:archive_errors.append('unmarked archive '+entry['archive'])
        body=text.split(b'---\n\n',1)[1]
        body=body.removesuffix(b'<!-- END_VERBATIM_ARCHIVE_BODY -->\n')
        if hashlib.sha256(body).hexdigest()!=entry['body_sha256']:archive_errors.append('nonverbatim archive '+entry['archive'])
    historical_spec=(root/'docs/archive/specs/historical_current_spec.md').read_text(encoding='utf-8')
    invariant_index=load_yaml(root/'docs/archive/specs/historical_invariant_index.json')
    if set(re.findall(r'INV-[A-Z0-9-]+',historical_spec))!={r['invariant_id'] for r in invariant_index['invariants']}:
        archive_errors.append('historical invariant index incomplete')
    index=build_report_index(root);index_errors=[r['path'] for r in index['reports'] if not r['registry_referenced']]
    generated=[]
    if check_generated:
        outputs={'docs/experiments/LINEAGE.md':render_lineage_v2(root),'reports/INDEX.md':report_index_markdown(index),'reports/index.json':json.dumps(index,indent=2)+'\n'}
        for path,expected in outputs.items():
            if not (root/path).is_file() or (root/path).read_text(encoding='utf-8')!=expected:generated.append(path+' stale')
    errors=registry_errors+lineage_errors+docs+imports+bad+archive_errors+index_errors+generated
    return {'ok':not errors,'errors':errors,'registry':{'node_count':len(registry['experiments']),'errors':registry_errors},
            'lineage':{'node_count':len(lineage['nodes']),'edge_count':len(lineage['edges']),'errors':lineage_errors},
            'manifest':{'historical_indexed_count':len(manifests),'historical_migrated':False},
            'current_docs':{'errors':docs},'imports':{'errors':imports},'archives':{'count':len(archive_map['archives']),'errors':archive_errors},
            'reports':{'indexed_count':len(index['reports']),'errors':index_errors},'source_identity':identity,
            'generated_errors':generated,'real_api_authorized':False,'real_execution_ready':readiness}
