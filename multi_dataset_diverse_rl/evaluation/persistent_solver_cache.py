from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from .prompt_question import PromptAnswer


SCHEMA_VERSION = "shared_solver_cache_v1"


class PersistentSolverCache:
    """Cross-process prompt-question cache with short SQLite claim transactions."""

    def __init__(self, path: str | Path, *, stale_after_seconds: float = 1800.0):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.owner_id = f"{os.getpid()}:{uuid.uuid4().hex}"
        self.stale_after_seconds = max(60.0, float(stale_after_seconds))
        self.hits = 0
        self.misses = 0
        self.waits = 0
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            row = connection.execute(
                "SELECT value FROM cache_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is not None and str(row[0]) != SCHEMA_VERSION:
                raise ValueError(
                    f"shared solver cache schema mismatch: expected {SCHEMA_VERSION}, got {row[0]}"
                )
            connection.execute(
                "INSERT OR REPLACE INTO cache_metadata(key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS solver_cache (
                    cache_key TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    state TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    created_at REAL NOT NULL,
                    model_request_identity TEXT NOT NULL,
                    solver_model TEXT NOT NULL,
                    endpoint_identity TEXT NOT NULL,
                    output_contract_version TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    max_tokens INTEGER NOT NULL,
                    evaluation_replica_seed INTEGER NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    question_hash TEXT NOT NULL,
                    answer_json TEXT
                )
                """
            )
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(solver_cache)").fetchall()
            }
            required = {
                "cache_key", "schema_version", "state", "owner_id", "updated_at", "created_at",
                "model_request_identity", "solver_model", "endpoint_identity",
                "output_contract_version", "parser_version", "temperature", "max_tokens",
                "evaluation_replica_seed", "prompt_hash", "question_hash", "answer_json",
            }
            if columns != required:
                raise ValueError("shared solver cache table does not match the frozen schema")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS solver_cache_lookup ON solver_cache(state, updated_at)"
            )

    @staticmethod
    def _answer_from_json(value: str) -> PromptAnswer:
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("cached solver answer must be a JSON object")
        return PromptAnswer(**payload)

    @staticmethod
    def _owner_alive(owner_id: str) -> bool:
        try:
            pid = int(str(owner_id).split(":", 1)[0])
        except (TypeError, ValueError):
            return False
        if pid == os.getpid():
            return True
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _claim_or_read(
        self,
        cache_key: str,
        metadata: Mapping[str, Any],
    ) -> tuple[str, PromptAnswer | None]:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state, owner_id, updated_at, answer_json FROM solver_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is not None and row[0] == "ready":
                return "ready", self._answer_from_json(str(row[3]))
            if row is None:
                connection.execute(
                    """
                    INSERT INTO solver_cache (
                        cache_key, schema_version, state, owner_id, updated_at, created_at,
                        model_request_identity, solver_model, endpoint_identity,
                        output_contract_version, parser_version, temperature, max_tokens,
                        evaluation_replica_seed, prompt_hash, question_hash, answer_json
                    ) VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        cache_key,
                        SCHEMA_VERSION,
                        self.owner_id,
                        now,
                        now,
                        str(metadata["model_request_identity"]),
                        str(metadata["solver_model"]),
                        str(metadata["endpoint_identity"]),
                        str(metadata["output_contract_version"]),
                        str(metadata["parser_version"]),
                        float(metadata["temperature"]),
                        int(metadata["max_tokens"]),
                        int(metadata["evaluation_replica_seed"]),
                        str(metadata["prompt_hash"]),
                        str(metadata["question_hash"]),
                    ),
                )
                return "owner", None
            if (
                not self._owner_alive(str(row[1]))
                or now - float(row[2]) > self.stale_after_seconds
            ):
                connection.execute(
                    """
                    UPDATE solver_cache
                    SET owner_id = ?, updated_at = ?, state = 'pending', answer_json = NULL
                    WHERE cache_key = ?
                    """,
                    (self.owner_id, now, cache_key),
                )
                return "owner", None
            return "wait", None

    def _store(self, cache_key: str, answer: PromptAnswer) -> None:
        payload = json.dumps(asdict(answer), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE solver_cache
                SET state = 'ready', answer_json = ?, updated_at = ?
                WHERE cache_key = ? AND owner_id = ?
                """,
                (payload, time.time(), cache_key, self.owner_id),
            ).rowcount
            if updated != 1:
                raise RuntimeError("persistent solver cache claim was lost before store")

    def _release(self, cache_key: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM solver_cache WHERE cache_key = ? AND owner_id = ? AND state = 'pending'",
                (cache_key, self.owner_id),
            )

    async def resolve(
        self,
        *,
        cache_key: str,
        metadata: Mapping[str, Any],
        producer: Callable[[], Awaitable[PromptAnswer]],
    ) -> PromptAnswer:
        while True:
            state, cached = await asyncio.to_thread(self._claim_or_read, cache_key, metadata)
            if state == "ready":
                self.hits += 1
                if cached is None:
                    raise RuntimeError("ready persistent cache row has no answer")
                return cached
            if state == "owner":
                self.misses += 1
                try:
                    answer = await producer()
                    await asyncio.to_thread(self._store, cache_key, answer)
                    return answer
                except BaseException:
                    await asyncio.to_thread(self._release, cache_key)
                    raise
            self.waits += 1
            await asyncio.sleep(0.1)

    def ready_entry_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM solver_cache WHERE state = 'ready'"
            ).fetchone()
        return int(row[0] if row else 0)

    def ready_content_hash(self) -> str:
        import hashlib

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT cache_key, answer_json FROM solver_cache WHERE state = 'ready' ORDER BY cache_key"
            ).fetchall()
        encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
