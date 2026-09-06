# SCAD V22 preprocessing contract

`RAW-like` means only channel mapping, physical-unit conversion, and sampling alignment needed for valid tensors. It performs no artifact removal and adds no optional filtering.

`STANDARD` is the V19/V20-compatible 100 Hz, 0.5–15 Hz fourth-order band-pass, common-average reference, microvolt, 8-MAD winsorized representation. The existing immutable V19 prepared assets implement this contract and are reused without modification.

EEGdenoiseNet official-native reproduction retains the pinned upstream preprocessing. Unified-harness results are separately labelled and never substituted for official-native results.
