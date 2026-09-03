# V18 accepted-candidate counterfactual ranking audit

This is a zero-API audit of all feasible candidates in the 15 accepted C updates for Seeds68-70. It does not change ranking, proposal generation, Common-Safe, or historical artifacts.

## Identifiability result

The requested validation counterfactual is **not identifiable from frozen evidence**. All 15 selected winners have complete 50-row validation observations. All 27 unselected feasible candidates have zero validation observations; there are no partial observations. The 12 zero/negative commits contain 23 feasible alternatives, but none has frozen validation evidence.

Therefore this audit cannot honestly answer whether an unselected candidate had better validation target/Vote/Oracle transfer. Doing so would require new validation Solver calls and is outside the zero-API scope.

## What is established without API

- Three nonpositive commits, including one of the three negative commits, had no feasible alternative at all. Ranking cannot explain these cases.
- The three negative commits contained 9 alternatives in total. None train-Pareto-dominated the selected winner on target gain, Vote net, and Vote loss.
- Across all 12 nonpositive commits, no alternative train-Pareto-dominated the winner. Only one alternative had lower train Vote loss while matching or improving Vote net; its validation transfer is unobserved.
- Thus available train-side evidence does not implicate ranking, and there is a direct feasible-set-quality lower bound. However the global ranking-versus-feasible-set distinction remains unresolved because alternative validation outcomes were never evaluated.

Seed69's paired +7 remains decomposed as A -3 and C +4; it is not attributed to three matched extra commits.

No API, new validation, or test call was made.
