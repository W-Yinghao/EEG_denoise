# V41R transfer-signature protocol

The source acquisition exposes four eye-region electrodes in the registered order
`HEOGL, HEOGR, VEOGU, VEOGL`. V41R constructs exactly two bipolar regressors:

```text
VEOG = VEOGU - VEOGL
HEOG = HEOGL - HEOGR
```

The four electrodes are never passed as four independent EOG sources. For each participant/session/
task context, support EEG and bipolar EOG are temporally centered and a 46×2 ridge transfer is fitted.
The ridge ratio is 0.05 and frozen before results.

Support uses a chronological 30-second prefix, 15 non-overlapping two-second windows, and EOG
center/scale computed only from that prefix. The 10-second sensitivity uses five windows and its own
10-second prefix normalization. Zero support is the outer-training population transfer signature.

The artifact-generating transfer is fitted independently on samples 15000–27000 (Qgen). It is used
to construct paired targets and the explicitly non-deployable ORACLE condition; it never enters a
deployable model condition. Held-out natural inference loads a precomputed support-signature bundle
and query EEG only. Query EOG is opened only after output digest freeze by the evaluator.

Each channel condition contains two transfer values, their log norm, four support-quality values, and
a fixed 46-way sensor identity. Continuous values are normalized with outer-training contexts only.
The condition is an ephemeral artifact-transfer signature, not a subject or biometric embedding.

## Resource provenance

The participant-held-out semi-simulation uses the frozen V19/V24 15-participant session/task panel.
The historical Klados v4 source has 54 records but no recoverable 27-participant record map and is not
silently represented as this participant-held-out panel.
