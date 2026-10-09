# Responsibility fallback and repair evidence V2.5

The current method is `unified_team_prompt_search_v2_5_responsibility_fallback_repair_probe`.
This amendment changes eligibility, diagnostic recovery, evidence allocation,
local evolution and Probe admission. Full transition and Shadow remain unchanged.
Historical V2.4 contracts and reports retain their original meaning.

Five identical structured prompts start with Role `You are a helpful assistant.`,
Strategy `Answer the question.`, and Answer `On the last line, write "Final answer:"
followed by your mathematical answer.` All blocks are editable. System contains
exactly their rendering; User contains exactly the original problem. No extra
reasoning or presentation instruction is added, and written reasoning is optional.

The gold-blind V2 extractor accepts a last nonempty line labeled Final answer
or FINAL_ANSWER, case insensitive with common spaces, including a balanced boxed
payload inside that label. Headings and intermediate boxes are ordinary text.
Standalone expressions and prose guesses are forbidden. Multiple explicit labels
must agree under the pinned mathematical relation, without consulting gold.
Only pinned binary mathematical equivalence rewards a response. First valid wrong
stops recovery; at most four successful semantic draws recover invalid output.
Exhaustion is incorrect in the fixed denominator. All responses and retry reasons
are retained privately. Four repeated format failures are observable Prompt evidence.
Transport and truncation-only 3600/6144 capacity recovery stay unchanged.

Any member with a genuine incorrect Optimize result is eligible; there are no
correct/wrong quotas or minimum accuracy. All correct ends NO_REPAIR_SIGNAL.
Actual D/N/C and failure discount remain. Positive responsibility always wins
above zero; all-zero member choice uses seed, committed state identity and global
opportunity ordinal. Pattern selection follows the same rule among legal actionable
patterns. Fallback uses general Memory, without invented responsibility labels.
Member incorrectness and team incorrectness are separate observations.

Each wrong example permits three Gradient draws, retrying only malformed JSON,
schema or recoverable abstract content. Valid ACTIONABLE or UNCERTAIN stops.
An exhausted example is nonactionable and does not erase other valid gradients.
Pattern whole-response structural failures permit three draws. Consistent known
supports of invalid pattern content move to unassigned, preserving valid patterns.
Missing known aliases also move to unassigned. Duplicate/unknown assignments,
example leakage, provenance, split, persistence, ledger or identity errors fail
closed. No actionable correction ends NO_ACTIONABLE_PATTERN without fabrication.
All draws are charged and share one opportunity; models and sampling stay frozen.

Mutation3 prioritizes selected Pattern wrong supports, then fills from Optimize
by seeded sampling. Independently sampled SearchValidation3 is disjoint. The
Independent TeamProbe6 is sampled from the remaining Optimize membership; for
Canary12 it is exactly the complement. Parent and Child use the same frozen IDs
and target member. No SearchValidation problem, answer or trajectory enters the
mutation request. Only aggregate measured counts can inform the next generation.

Every contract-legal, safe, nonduplicate and actually evaluated local edit becomes
its next temporary parent, including neutral or negative validation effects.
Malformed, duplicate and NO_SAFE_EDIT responses do not advance it. Complete root
and actual parent chains remain reconstructable. Six generations, 42 local metric
calls, four ranked exports and at most two Full promotions stay fixed.
Local scores never independently deploy a team prompt.

Probe re-evaluates selected Pattern wrong IDs present in Mutation against
its official committed parent. This is SEEN_ASSIGNED_REPAIR. At least one actual
binary 0-to-1 repair is required; invalid-to-valid-wrong is not a repair. Exact
request cache realizations are shared within the new attempt, so this logical
remeasurement may hit the cache. It is never independent or generalization evidence.
The separate six-example Probe measures Vote, target, invalidity and collateral
loss with fixed peers and equal plurality. A loss of at least three Vote-correct
or target-correct results net, or at least three old target-correct results, is
catastrophic: it loses half this small panel. This is an explicitly frozen scale
choice, not an estimated population threshold. Smaller regression remains visible
and subject to unchanged Full checks. Mathematical invalidity is incorrect,
observation-only. Seen repair plus passing risk is sufficient; no unrelated
positive submetric is required. Rank Vote gain, target gain, lower collateral,
lower invalid delta, then candidate identity; admit at most two Full candidates.

Full retains its immutable initial competence floor, Vote nonregression and
target-or-team progress over the incumbent. A winner alone enters the unchanged
Shadow gate, then atomic commit. Explicit outcomes distinguish assigned failure,
independent regression, Probe pass / Full rejection, Full pass, Shadow rejection
and commit. Full and all local evidence remain Optimize training observations.

Private Memory retains actual block chains, binary repair/loss, validity
transitions, repeated format failures, seen/independent Probe and Full effects.
Format-valid wrong is not success. Shared OUTPUT_CONTRACT_FAILURE contains only
closed generic risk observations, without raw questions, answers or edits.
Retrieval remains bounded and member-private for edit evidence.

Fresh source, manifest, startup, provider, cache, accounting and single-use scope
are mandatory. The user authorizes exactly one Canary12, seed81, five members,
one opportunity, Solver concurrency up to eight and charged plus reserved <=2M.
No Validation, Test or Pilot is authorized. V2.4's 195 outputs may be replayed
offline for parser diagnosis only; no historical realization enters new scoring.
Owner review barriers audit real evidence and never force an efficacy outcome.
