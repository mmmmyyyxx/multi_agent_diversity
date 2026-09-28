"""Pre-import outbound-network denial for the Formal V3 fake-provider campaign.

Loaded by Python itself through PYTHONPATH, before pytest or project imports.
Never records destinations or environment values.
"""

from __future__ import annotations

import socket
from threading import local


class NetworkIsolationViolation(RuntimeError):
    pass


_attempt_count = 0
_internal_socketpair_count = 0
_socketpair_context = local()
_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_original_socketpair = socket.socketpair


def _internal_socketpair_address(args) -> bool:
    if not getattr(_socketpair_context, "active", False) or len(args) < 2:
        return False
    address = args[1]
    return isinstance(address, tuple) and address and address[0] in {"127.0.0.1", "::1"}


def _deny(*_args, **_kwargs):
    global _attempt_count
    if _internal_socketpair_address(_args):
        return _original_connect(*_args, **_kwargs)
    _attempt_count += 1
    raise NetworkIsolationViolation("outbound network denied by formal zero-API guard")


def _deny_ex(*args, **kwargs):
    global _attempt_count
    if _internal_socketpair_address(args):
        return _original_connect_ex(*args, **kwargs)
    _attempt_count += 1
    raise NetworkIsolationViolation("outbound network denied by formal zero-API guard")


def _internal_socketpair(*args, **kwargs):
    global _internal_socketpair_count
    _socketpair_context.active = True
    try:
        pair = _original_socketpair(*args, **kwargs)
        _internal_socketpair_count += 1
        return pair
    finally:
        _socketpair_context.active = False


def network_attempt_count() -> int:
    return _attempt_count


socket.socket.connect = _deny
socket.socket.connect_ex = _deny_ex
socket.create_connection = _deny
socket.socketpair = _internal_socketpair
socket.socket.sendto = _deny
socket.getaddrinfo = _deny
