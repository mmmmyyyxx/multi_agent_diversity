# Architecture lineage

`main@a6692a04654d5de8bbef5d023abc26581298d051` was the common ancestor for
the backend data-flow experiments.

1. Backend-native feed controls let GEPA and MARS own their example selection.
   They are superseded and frozen at `archive/native-feed-gepa-v1`
   (`d3c7dcd…`) and `archive/native-feed-mars-v1` (`27e47b7…`).
2. Layer2-owned evidence moved target, responsibility, evidence selection and
   team admission into the shared controller.
3. The transition-evidence amendment defined focus as the latest accepted
   parent-to-child correct-to-wrong cases and anchor as wrong-to-correct cases.
   Those branch states are frozen at `archive/layer2-feed-gepa-transition-v2`
   (`d72ea15…`) and `archive/layer2-feed-mars-transition-v2` (`5e6e787…`).
4. The unified replaceable-backend architecture places GEPA_NATIVE,
   GEPA_LAYER2, MARS_NATIVE and MARS_LAYER2 in one codebase with one Layer 2.

Historical code, manifests and reports remain recoverable through the archive
tags. Current development and future experiment freezes use `main`, a commit
SHA, configuration/protocol/data hashes, and immutable experiment tags.
