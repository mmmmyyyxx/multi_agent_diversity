"""Credential-free public source download recipes for Benchmark Data Freeze V1.

Only pinned public dataset objects are fetched. No upstream executable module
is imported. In particular papillon_utils initializes a judge at import time.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlparse

from .data_freeze import GEPA_REVISION, SOURCE_PINS, canonical, file_hash, wrap_rows


class PublicDataDownloader:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.requests = 0
        self.files = []

    def fetch(self, url: str, name: str) -> Path:
        if urlparse(url).scheme != "https" or urlparse(url).hostname not in {"huggingface.co", "raw.githubusercontent.com"}:
            raise ValueError("PUBLIC_DATA_SOURCE_NOT_ALLOWED")
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("DOWNLOAD_FILENAME_INVALID")
        path = self.directory / name
        receipt = path.with_name(path.name + ".receipt.json")
        if path.exists():
            if path.stat().st_size == 0:
                raise ValueError("EMPTY_CACHED_DOWNLOAD")
            if not receipt.exists():
                raise ValueError("CACHED_DOWNLOAD_RECEIPT_MISSING: " + name)
            expected = json.loads(receipt.read_text(encoding="utf-8"))
            if expected["url"] != url or expected["sha256"] != file_hash(path):
                raise ValueError("CACHED_DOWNLOAD_IDENTITY_MISMATCH: " + name)
        else:
            temp = path.with_suffix(path.suffix + ".partial")
            for attempt in range(4):
                try:
                    self.requests += 1
                    with (self.directory / "network_events.jsonl").open("ab") as events:
                        events.write(canonical({"url": url, "attempt": attempt + 1, "provider_attempts": 0}) + b"\n")
                    # Fresh opener; never consumes HF tokens or provider credentials.
                    opener = urllib.request.build_opener() if attempt % 2 == 0 else urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    with opener.open(urllib.request.Request(url, headers={"User-Agent": "benchmark-data-freeze-v1"}), timeout=60) as response, temp.open("wb") as f:
                        total = 0
                        while chunk := response.read(1024 * 1024):
                            f.write(chunk)
                            total += len(chunk)
                        expected = response.headers.get("Content-Length")
                        if expected is not None and total != int(expected):
                            raise ValueError("INCOMPLETE_DOWNLOAD")
                    temp.replace(path)
                    receipt.write_bytes(canonical({"url": url, "sha256": file_hash(path)}) + b"\n")
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(attempt + 1)
        self.files.append({"url": url, "local_name": name, "sha256": file_hash(path), "bytes": path.stat().st_size})
        return path

    def hf(self, benchmark: str, filename: str) -> Path:
        pin = SOURCE_PINS[benchmark]
        return self.fetch(f"https://huggingface.co/datasets/{pin['repository']}/resolve/{pin['revision']}/{filename}", filename.replace("/", "__"))


def fingerprint(rows: list[dict]) -> str:
    from datasets import Dataset
    return Dataset.from_list(rows)._fingerprint


def download_source(root: Path, benchmark: str) -> tuple[dict, dict]:
    import datasets
    if datasets.__version__ != "3.6.0":
        raise ValueError("DATASETS_VERSION_MISMATCH: use datasets==3.6.0")
    dl = PublicDataDownloader(root / "raw" / benchmark)
    primary: dict[str, list[dict]] = {}
    fingerprints = {}
    recipe = "exact pinned source files; no remote executable imports; Dataset.from_list fingerprint"
    if benchmark == "hotpotqa":
        import pyarrow.parquet as pq
        rows = []
        for n in range(2):
            path = dl.hf(benchmark, f"fullwiki/train-{n:05d}-of-00002.parquet")
            rows.extend(pq.read_table(path).to_pylist())
        primary["train"] = wrap_rows(benchmark, "train", rows)
        fingerprints["train"] = fingerprint(rows)
        validation_path = dl.hf(benchmark, "fullwiki/validation-00000-of-00001.parquet")
        validation = pq.read_table(validation_path).to_pylist()
        primary["validation"] = wrap_rows(benchmark, "validation", validation)
        fingerprints["validation"] = fingerprint(validation)
        recipe += "; pinned parquet train shards concatenated in shard/row order and official labeled validation (fullwiki)"
    elif benchmark == "hover":
        pin = SOURCE_PINS[benchmark]
        dl.hf(benchmark, "hover.py")
        path = dl.fetch(f"https://raw.githubusercontent.com/{pin['data_repository']}/{pin['data_revision']}/data/hover/hover_train_release_v1.1.json", "hover_train_release_v1.1.json")
        rows = []
        for i, original in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            rows.append({"id": i, "uid": original["uid"], "claim": original["claim"],
                         "supporting_facts": [{"key": x[0], "value": x[1]} for x in original["supporting_facts"]],
                         "label": ["NOT_SUPPORTED", "SUPPORTED"].index(original["label"]),
                         "num_hops": original["num_hops"], "hpqa_id": original["hpqa_id"]})
        primary["train"] = wrap_rows(benchmark, "train", rows)
        fingerprints["train"] = fingerprint(rows)
        recipe += "; inspected pinned hover.py _generate_examples projection/ClassLabel encoding; separately pinned upstream JSON"
    elif benchmark == "ifbench":
        for split in ("train", "test"):
            filename = f"gepa_artifact/benchmarks/IFBench/data/IFBench_{split}.jsonl"
            path = dl.fetch(f"https://raw.githubusercontent.com/gepa-ai/gepa-artifact/{GEPA_REVISION}/{filename}", f"IFBench_{split}.jsonl")
            raw = path.read_bytes()
            import hashlib
            blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            dl.files[-1].update({"git_blob_sha1": blob, "line_count": len(raw.splitlines())})
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            primary[split] = wrap_rows(benchmark, split, rows)
            fingerprints[split] = fingerprint(rows)
    elif benchmark == "pupa":
        # datasets' CSV loader is used locally so inferred types match the
        # upstream data-files configuration, including nullable columns.
        for config, filename in (("pupa_new", "PUPA_New.csv"), ("pupa_tnb", "PUPA_TNB.csv")):
            path = dl.hf(benchmark, filename)
            data = datasets.load_dataset("csv", data_files={"train": str(path)}, cache_dir=str(root / "raw/_hf_cache"), split="train")
            rows = data.to_list()
            split = "train" if config == "pupa_new" else "auxiliary_tnb"
            primary[split] = wrap_rows(benchmark, split, rows)
            fingerprints[config] = fingerprint(rows)
        recipe = "datasets==3.6.0 local csv loader over pinned HF config files; pupa_tnb is auxiliary only"
    elif benchmark == "math":
        import pyarrow.parquet as pq
        for split in ("train", "test"):
            rows = []
            for subject in SOURCE_PINS[benchmark]["subject_configs"]:
                path = dl.hf(benchmark, f"{subject}/{split}-00000-of-00001.parquet")
                rows.extend(pq.read_table(path).to_pylist())
            if len(rows) != {"train": 7500, "test": 5000}[split]:
                raise ValueError("STOP_MATH_SOURCE_COUNT_MISMATCH")
            primary[split] = wrap_rows(benchmark, split, rows)
            fingerprints[split] = fingerprint(rows)
        recipe += "; canonical MATH mirror original train/test; fixed subject-config then parquet row order; project ordering (not paper identity)"
    else:
        raise ValueError("BENCHMARK_UNKNOWN")
    metadata = {"files": dl.files, "datasets_version": datasets.__version__, "dataset_fingerprints": fingerprints,
                "fingerprint_scope": "local explicit loading recipe, not claimed historical paper fingerprint",
                "loading_recipe": recipe, "upstream_splits": list(primary)}
    # Runtime network counters belong in ignored local audit, not deterministic
    # manifests: cache hits on a repeated run cannot change a frozen identity.
    with (dl.directory / "download_events.jsonl").open("ab") as f:
        f.write(canonical({"http_requests": dl.requests, "provider_attempts": 0, "files": [x["local_name"] for x in dl.files]}) + b"\n")
    return primary, metadata
