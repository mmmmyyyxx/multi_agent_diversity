# Stage 2 v3 Failure Report

Stage 2 protocol v3 is invalid for formal comparison and will not be mixed
with later results. On Windows, a transient sharing conflict caused the
single-attempt checkpoint `os.replace` call to fail with `WinError 5` during
geometric_shapes scalar training.

The implementation was minimally fixed in commit
`dcc949238c0366fad1061cd20f97e17924ac7a6b`: checkpoint snapshots now use a
unique temporary filename and retry atomic replacement three times with short
backoff. Persistent failures still raise. Two regression tests were added and
the full suite passed with 155 tests.

All v3 processes were stopped and outputs were retained. Stage 2 restarts from
the beginning in a fresh v4 directory.
