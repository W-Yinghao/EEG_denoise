# V32P terminal report

## Git lineage

```text
base:                  274b371ed2d3c7c105f2351f4dd88d4464fe3a66
implementation:        b73ff4bedcde4228b816667e34237e542fc078f3
protocol correction:   679e5e2
participant evaluator: 8ac86bf
baseline/method/result: d89d3b9c0871f1b390010992122753ad98d5c677
ledger v2.8:            d89d3b9c0871f1b390010992122753ad98d5c677
engineering tests:     fab624560b53f265df2c4e02e41b5883ea1baf38
terminal:              commit containing this report and terminal manifest
```

Remote/local parity is verified after the terminal commit because a commit
cannot contain its own SHA. No PR is created and master is not merged.

## Execution lineage

```text
accepted:   939641_[0-2]
failed:     939632_[0-2] (P100 sm_60 environment incompatibility)
superseded: 939635_[0-2], 939638_[0-2]
recovery:   939632 -> 939635 -> 939638 -> 939641
```

Recovery did not alter the scientific model, data, seeds, budget, or test
panel. Eighteen checkpoint digests were verified. Targeted tests: 17/17.
Clean git-archive tests: 17/17.

## Governance and decision

```text
waveform sealed reads: 0
query EOG reads:       0
A-track:               unchanged at 0c4f2301c1f873120fe54537cde3c76fff7ea3a2
manuscript:            unchanged; not compiled
current V32P jobs:     none
waveform interaction: deferred_not_comparable
selected candidate:   SANDiff
```

SANDiff is a positive development candidate, not an anonymity guarantee.
