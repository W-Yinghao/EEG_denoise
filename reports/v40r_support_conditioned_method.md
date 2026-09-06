# Support-conditioned EEGDfus-MC

Stage 1 trains the population EEGDfus-MC on training participants only. Stage 2 freezes the
population backbone and trains a compact support encoder plus two identity-initialized FiLM adapters.
The support encoder applies a temporal CNN to each 2-second EEG+EOG support window, globally pools,
mean-aggregates the set, and emits a 128-d context. Support windows are chronological,
non-overlapping, query-disjoint, and normalized only from their own duration prefix.

POP bypasses both adapters exactly. MATCH, WRONG, SHUFFLED, and population-mean conditions reuse the
same backbone, query, checkpoint, DDIM schedule, and registered noise. Only context changes. The
held-out query receives EEG only; EOG is opened by the evaluator after output creation.

Primary: 30 seconds. Sensitivity: exact POP at 0 seconds, 10 seconds, and 30 seconds. No subject ID,
persistent identity vector, query EOG, clean target, event label, or target-gradient adaptation is an
inference input.
