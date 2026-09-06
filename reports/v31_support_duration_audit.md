# V31 support-duration implementation audit

Verdict: `V30_DURATION_EVIDENCE_SUPERSEDED`. This supersedes only V30 duration rows, not its common-panel, specificity, falsification, natural, latency, privacy, or final-selection findings.

The committed V30 implementation had four defects: `support_starts(5s)` produced overlapping 2 s windows; short-duration EOG center/scale came from the first 120 s; the 120 s condition used 16 windows rather than all 60 non-overlapping windows; and its validator checked uniqueness/bounds but not overlap or future-normalization. V31 leaves V30 untouched and repairs the evidence in a new namespace.
