# V2.3 Canary owner supervision

The scientific owner remains responsible for both actual-evidence reviews even
when a scoped Luna task launches the frozen command. A monitor task cannot
approve either receipt. The existing 3600-second deadline, receipt hashes,
startup binding and fail-closed behavior remain unchanged.

Before a paid launch, the handoff identifies the owner, executor and monitoring
command. The owner verifies that it can remain active through initialization and
the first opportunity. An executor must use bounded command yields of at most
60 seconds and report a pending review immediately. A successful launch alone
does not establish continued supervisor availability.

The owner polls the retained process session and pending review files at most
60 seconds apart. At each barrier it independently inspects the actual request
bytes, provenance, trajectories, accounting and applicable evidence flow before
writing the matching decision. A lexical trajectory count alone cannot approve
solution usefulness. No decision may be manufactured after terminal failure.

If the delegated monitor becomes unavailable, the owner takes over read-only
monitoring of the same live process. This does not launch another experiment or
change its source, state, budget, cache or authorization. If owner availability
is also lost, the frozen review deadline continues to apply; it is not extended.

For a future fresh attempt, direct execution by the scientific owner is permitted
only when explicitly included in that attempt's human-approved handoff. This
avoids depending on a second model's account availability during the review.
It retains the sole production command and both independent owner decisions.

After a timeout, preserve the consumed approval and all paid evidence. A new
attempt requires fresh registration, source/startup freeze, cache, accounting,
preflight and exact single-use API approval under AGENTS.md section 8. Previous
profiles and remaining allowance cannot be imported as a scientific restart.
