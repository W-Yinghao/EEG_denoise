# V31 V30 freeze note

V31 is based exactly on V30 terminal `220dcbaaabdef0cb8d1ac91b87b0d1cc8b7109cf`. The checkpoint inventory, all-donor, falsification, privacy, latency, selection, and Git lineage are digest-bound in `v30_binding.json`. V30 remains read-only; `selected_candidate=none`, sealed confirmation remains unauthorized, and this round trains no model.

The externally named v2.3 ledger file was not present on the server. The explicit V31 user instruction was therefore synchronized as the v2.3 start state on top of the complete committed v2.2 ledger, with that provenance recorded rather than inferred silently.
