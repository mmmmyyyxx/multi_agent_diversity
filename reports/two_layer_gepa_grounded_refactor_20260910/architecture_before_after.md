
# Architecture before and after

Before, the canonical `system.py` owns responsibility, TCS proposal generation,
candidate rollout, Common-Safe ranking, Shadow, and write-back. Historical
RG-GEPA scripts are non-committing fixed-parent probes with locally reimplemented
selection.

After, the opt-in `two_layer_rg_gepa_v1` path has one backend-neutral boundary.
Layer 1 runs official GEPA or delegates byte-preservingly to legacy TCS and
returns candidates. Layer 2 owns responsibility, quota-based TeamMiniBatch12,
full fixed-peer evaluation, Common-Safe, Shadow, and commit. GEPA local Pareto
and team selection are separately named and cannot substitute for one another.
