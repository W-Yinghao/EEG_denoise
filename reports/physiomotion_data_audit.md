# PhysioMotion Artifact data audit

Dataset: `OpenNeuro ds006386 v1.0.1`. Coverage is 30 participants × runs 01–06, with [34] channels and [1000.0] Hz metadata. Ordered layout count is 1.

The metadata-only split freezes 20 development and 10 sealed participants. Sealed IDs are [3, 4, 7, 12, 15, 17, 19, 24, 25, 28]; no sealed signal or annotation was opened.

Only 14/20 development participants have more than one acquisition date. The primary protocol is therefore accurately named `repeated-run`, not cross-day.

Primary artifacts exclude blink, saccade, horizontal/vertical eye movement, and any combination containing them. The retained strata are head motion, chewing, tongue, swallowing, and eyebrow/facial EMG.
