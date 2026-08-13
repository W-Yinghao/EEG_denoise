# V38P support and donor protocol

Each participant contributes 20 chronological Session-1 support trials: the first 10 trials of each
registered MI class. The remaining 80 Session-1 trials form the privacy gallery; all 100 Session-2
trials form the primary query/task set. Support labels are used only to freeze the balanced support
budget. Query donor matching and inference never use true task labels.

For each fold/seed, the frozen V36P EEGNet supplies 128-d representations and two task logits.
Outer-training representations define predicted-class and confidence-tertile means. A participant's
context is the mean support representation after subtracting its frozen predicted-semantics mean,
then normalizing with context statistics from the 36 outer-training participants only. It is called
task-demeaned source-context information, not a pure identity or biometric vector.

The donor bank contains only the 36 outer-training participants. Every dynamic donor differs from
the source participant and matches frozen predicted class plus the same confidence stratum whenever
available, otherwise the nearest stratum. Validation and outer-test representations never enter the
bank. Donor identities and true labels are not model inputs.
