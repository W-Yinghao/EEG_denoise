# Frozen SGEYESUB natural-EEG diffusion protocol

Frozen on 2026-08-02 before reading any output from this protocol. The machine-
readable specification is
`configs/cgdr/sgeyesub_diffusion_incremental.yaml`. This is a prospective,
exploratory release-internal block-1-to-block-2 comparison. It is not an
official EEGDS reproduction, formal G1/G3 evidence, or a clean-waveform
recovery experiment.

## Scientific question and boundary

The primary question is whether operator-conditioned conditional diffusion has
incremental value over a task-matched multichannel deterministic U-Net when
both receive the same deployment-visible information, weak-supervision pairs,
outer-training participant stems, windows, channel layout, normalization,
operator conditioning, and 6,000 successful optimizer updates. The U-Net uses
one network call. Diffusion uses a 1,000-step linear training schedule and a
100-call deterministic DDIM sampler (`eta=0`). Both use seed `20260802`; model
failures remain in the denominator. The update budget is matched, while the
inference compute is deliberately not described as matched: network calls,
wall time, latency, and peak memory are reported explicitly.

The weak target is observed EEG from a low-artifact block-1 window, defined by
at least 95% artifact class 6. Artifact-source windows contain at least 25%
sample-wise classes 1--5. The same raw window cannot play both roles, and the
participant-specific block-1 P0 transfer recontaminates the weak target with
aligned EOG. This construction is weak supervision, **not paired clean EEG**.
No clean RRMSE, clean correlation, or clean-recovery claim is permitted.

At inference, the held-out block-2 EEG, support-derived projector, outer-
training normalization, and an EEG-only attenuation are visible. Query EOG,
artifact classes, trial labels, trial IDs, and outcomes remain sealed until all
arm outputs for that stem are frozen; they can then be used only for scoring.

## Frozen split and windows

Models, population operators, and normalization are fit separately within each
exact study, ordered EEG layout, as-delivered reference cell, and sampling
rate. There is no cross-cell pooling. Each fold excludes its held-out stems
from learned weights, population geometry, attenuation calibration, and
normalization. The held-out stem's block 1 is support/calibration only; block 2
is query only.

Development has ten folds over all 15 stems in `study01` and `study03`:

- `study01`: five singleton held-out folds, `p01` through `p05`.
- `study03`: `(p03,p19)`, `(p20,p21)`, `(p22,p23)`, `(p24,p25)`, and
  `(p26,p27)`.

Evaluation has fifteen folds over 43 compatible stems:

- `study02`: `(p02,p03,p06)`, `(p07,p08,p09)`, `(p10,p11,p12)`,
  `(p13,p14,p15)`, and `(p16,p17,p18)`.
- `study04`: `(p03,p08,p24)`, `(p28,p29,p30)`, `(p31,p32,p33)`,
  `(p34,p35,p36)`, and `(p37,p38,p39)`.
- `study05` layout 05: `(p06,p10,p24)`, `(p40,p41,p43)`,
  `(p44,p45,p46)`, `(p47,p48)`, and `(p49,p50)`.

`study05_p42` is the sole member of layout 06 and is prospectively
`blocked_no_population`. It contributes no performance value but remains in
the availability denominator, so evaluation coverage is always reported as 43
compatible stems out of 44 available stems.

Windows are trial-local, non-overlapping, complete 2-second segments with no
padding: 400 samples for studies 01--03, 200 for study04, and 512 for study05.
Each observed 8-second trial therefore supplies four complete windows.
Incomplete or cross-trial windows are rejected. Scientific evaluation consumes
every valid block-2 window from every compatible stem; a small integration
input is engineering smoke only.

## Arms and reported evidence

The primary pair is conditional DDIM100 versus the matched deterministic
U-Net. Raw observation, population-projector Qy, matching-projector Qy, and
matching-projector soft proximal are contextual arms. This protocol has no
query-derived oracle projector. It reports participant-stem-level EOG remaining
ratio, EOG coherence reduction, matching-projector artifact attenuation,
non-artifact preservation, PSD and covariance distortion, ERP/task proxies
when label semantics permit, observation change, latency, peak memory, and
failure status. These are natural-EEG proxy and preservation outcomes, not
clean-waveform fidelity.

Metric formulas retain the existing release-internal audit definitions:
artifact intervals are sample-wise classes 1--5 and non-artifact intervals are
class 6; EOG coherence reduction is observed minus output mean absolute EOG
correlation; matching-projector attenuation is
`20 log10(||P y|| / ||P x_out||)` on artifact intervals; preservation is one
minus relative output change on class-6 intervals; PSD and covariance
distortions are output-versus-observation relative distortions on class-6
intervals. The ERP proxy is one minus relative four-condition template change.
Higher is better for coherence reduction, attenuation, preservation, and ERP;
lower is better for remaining ratio and distortion.

All 43 compatible stems remain the scientific denominator. A primary paired
row requires both learned arms to succeed for the same `recording_key`.
Blocked and failed rows remain in coverage. Results are reported both pooled
with equal stem weights and separately for `study02`, `study04`, and
`study05`. A 20,000-replicate participant-stem bootstrap with seed `20260802`
provides descriptive 95% intervals only; it is not a significance test.

## Prospective exploratory decision rule

All deltas below are conditional diffusion minus matched U-Net. A protocol-
scoped positive requires every condition:

- paired success at least 90%: at least 39 of 43 compatible stems;
- pooled mean EOG-coherence-reduction delta greater than zero;
- pooled mean matching-projector artifact-attenuation delta greater than zero;
- at least 60% of paired stems improve on both primary outcomes;
- mean non-artifact-preservation delta at least `-0.02`;
- mean PSD-distortion delta no greater than `+0.05`;
- mean covariance-distortion delta no greater than `+0.05`;
- mean ERP-proxy delta at least `-0.02`;
- conditional-diffusion failure fraction no greater than 0.10 (at most 4 of
  43 compatible stems).

The pooled thresholds govern this exploratory decision. Study-specific
estimates are mandatory heterogeneity diagnostics but are not extra pass/fail
conditions. Descriptive intervals crossing zero or a safety boundary are
reported as uncertainty; thresholds will not be changed after seeing outcomes.

Passing supports only conditional diffusion over the matched U-Net under this
frozen SGEYESUB weak-supervision protocol. Failure supports only “no detectable
incremental value under the tested SGEYESUB protocol.” Insufficient coverage
or mixed safety is `inconclusive`. Neither result licenses “diffusion is
useless,” “diffusion sampler has no value,” “EEG diffusion is disproved,” or
“personalization failed.”
