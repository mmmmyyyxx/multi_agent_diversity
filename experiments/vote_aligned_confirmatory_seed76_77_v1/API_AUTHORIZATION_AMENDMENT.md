# API authorization amendment

On 2026-09-06, after the prospective protocol and classifier were frozen and
before any Seed76/77 result existed, the user explicitly authorized model API
calls for this confirmatory replication.

The authorization covers only Seed76 and Seed77, with P0 followed by P1 for
each seed and the no-training Static control evaluated from the matched frozen
initial state. It permits the frozen online trajectory and frozen-final-state
Shadow50/Validation50 phases using Solver `qwen3-8b` and
Teacher/Critic/Student/Evaluator `qwen3.7-flash`.

It does not authorize Test50, extra seeds, retries, method changes, scheduler
tuning, added update opportunities, or another experiment. The original
protocol and classifier are unchanged.
