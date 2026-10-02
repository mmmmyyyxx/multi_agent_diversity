# MATH installed SDK HTTP error mapping repair V1

Canary V2 stopped after one physical request: the HTTP error branch passed
an unsupported body keyword to the installed SDK. The HTTP status itself
was not persisted and is unknown. The 7,231-token full fallback charge is
permanent. The failed attempt and its authorization remain closed.

The repair uses the installed response-only error mapper and saves private
HTTP status/body evidence for future failures. SDK status exceptions retain
the frozen transport retry policy; every retry reserves and charges separately.
The installed SDK no longer includes bearer authentication in default headers.
The repaired transport uses its ordinary security/header and URL construction,
while sending the same reserved body bytes. Redirects fail closed after one
physical request. The installed SDK version is pinned in the fresh binding;
an offline synthetic negative control reproduces the missing-header defect.
Request serialization, models, thinking, output cap, strict parser, initial
team, splits, GEPA physics, responsibility, aggregation and stopping are unchanged.

A fresh Canary attempt3 has a separate source freeze, prep, cache and single-use
authorization. Zero-API gate and source receipts follow the implementation
commit. Efficacy is unverified. This is operational repair 1 for this root cause.
No push is authorized.
