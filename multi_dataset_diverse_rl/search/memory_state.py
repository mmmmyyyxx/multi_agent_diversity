"""Generic tuple state and revision transactions, without treatment semantics."""
from .schemas import SearchContractError

class MemoryState:
    def __init__(self, *, top_k_private, top_k_shared, max_context_chars,
                 private_storage_limit, shared_storage_limit):
        limits = (top_k_private, top_k_shared, max_context_chars, private_storage_limit, shared_storage_limit)
        if any(not isinstance(x, int) or x <= 0 for x in limits):
            raise SearchContractError("explicit positive memory limits required")
        self.limits = dict(zip(("top_k_private", "top_k_shared", "max_context_chars", "private_storage_limit", "shared_storage_limit"), limits))
        self.private = (); self.shared = (); self.revision = 0
        self.read_private_count = self.read_shared_count = self.write_count = 0

    def validate_delta(self, delta):
        if delta.revision != self.revision: raise SearchContractError("stale memory outcome")
        if not isinstance(delta.private, tuple) or not isinstance(delta.shared, tuple):
            raise SearchContractError("invalid memory delta")

    def apply_outcome(self, delta):
        # Validated prepared immutable tuples; no provider, persistence or parsing.
        changed = (self.private, self.shared) != (delta.private, delta.shared)
        self.private, self.shared = delta.private, delta.shared
        self.revision += int(changed)
        self.write_count += int(changed)
