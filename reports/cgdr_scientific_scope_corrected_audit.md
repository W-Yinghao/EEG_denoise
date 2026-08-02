# CGDR scientific-scope corrected audit

Date: 2026-08-02 (Europe/Paris)

This audit narrows the interpretation of the retained Klados and SGEYESUB
results. It does not replace or modify any historical result, classifier
output, metric table, or path.

## Current statuses

- Klados: `current_M2_no_incremental_value`
- diffusion family: `not_tested`
- SGE operator specificity: `hard_Q_P0_tradeoff_inconclusive`
- engineering priority: `deterministic_first_diffusion_open`
- formal G1: `NOT_RUN_BLOCKED`
- formal G3: `NOT_RUN_BLOCKED`

## Klados boundary

The retained 16-record comparison supports only this statement: under the
current Klados source-record protocol, the existing unconditional clean prior,
100-step deterministic DDIM, and frozen M2 final-hard-Q output did not add
value over deterministic oracle orthogonal subtraction on the audited waveform
metrics. The oracle-projector M2 arm improved over its same-sampler POP arm by
median delta-e_parallel `-0.004969`, but was worse than oracle `Qy` by median
delta-e_parallel about `+1.145`, median delta-RRMSE about `+0.700`, and median
delta-correlation about `-0.178`; `Qy` was better on all 16/16 records.

`Qy` uses a query-derived oracle projector and is non-deployable. It is a
mechanism diagnostic, not evidence that practical deterministic methods in
general beat diffusion. These records are source records with unverified
participant independence and have already been used in model selection and
audit, so any further use is exploratory.

This result does not test conditional diffusion, a dual-prior model, another
training objective, another diffusion sampler, EEGDfus, D4PM, or EEG diffusion
as a method family. A task-matched multichannel deterministic U-Net was absent.
Consequently no diffusion-family or formal G3 conclusion is available.

## SGEYESUB boundary

The SGEYESUB experiment tested operator geometry with deterministic hard-Q and
proximal paths; it did not run diffusion. The old gamma-zero label is retained
only as historical output. Gamma zero means the support-only development
objective selected the population endpoint; it is not an unbiased test of
personalization. The corrected evaluation must separately report performance
on successful compatible stems and feasibility over all 44 stems before any
operator-specificity interpretation.

## Required next comparison

Before discussing diffusion incremental value, use a frozen protocol with the
same data, inputs, windows, split, supervision exposure, and meaningful
training budget for M1 warm start, M2 final hard-Q, M4 stepwise proximal,
deterministic oracle Qy, deterministic soft proximal, and a trained
multichannel deterministic U-Net. At least one non-M2 diffusion configuration
and the matched U-Net must complete. Official EEGDfus is a separate benchmark;
its native protocol and the stricter source-epoch split must remain
separately labeled.
