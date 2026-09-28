"""Prove connection primitives fail before transmission in a fresh child."""

import socket

import sitecustomize


def expect_blocked(action) -> None:
    try:
        action()
    except sitecustomize.NetworkIsolationViolation:
        return
    raise AssertionError("network operation escaped zero-API guard")


with socket.socket() as candidate:
    expect_blocked(lambda: candidate.connect(("192.0.2.1", 80)))
    expect_blocked(lambda: candidate.connect_ex(("192.0.2.1", 80)))
    expect_blocked(lambda: candidate.sendto(b"offline", ("192.0.2.1", 80)))
expect_blocked(lambda: socket.create_connection(("192.0.2.1", 80)))
expect_blocked(lambda: socket.getaddrinfo("example.invalid", 80))
assert sitecustomize.network_attempt_count() == 5
print("NETWORK_GUARD_NEGATIVE_CONTROL_PASS")
