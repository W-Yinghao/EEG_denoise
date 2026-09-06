# SGE subject-basis × score-LoRA V9

Development exploration only. Historical v8.1 remains `BRIDGE_GATE_FAILED_SCORE_LORA_NOT_RUN`; V9 directly tests Score-LoRA and does not revise that result. Score-LoRA was therefore tested for the first time here, independently of the historical v8.1 bridge gate.

## V8.1 no-training Track A

Raw MATCH120 is the primary geometry control, POP is the baseline, fixed-λ EB is secondary, and full-feature EB is stopped. WRONG donors are scored before averaging. The correction-strength curve is diagnostic only and was not used to select V9.

| method | units | mean RRMSE | mean correlation | mean ΔSNR |
|---|---:|---:|---:|---:|
| DET-EB-FIXED120 | 9 | 0.5247 | 0.8563 | +1.4160 |
| DET-MATCH120 | 9 | 0.5120 | 0.8637 | +1.5920 |
| DET-POP | 9 | 0.5372 | 0.8528 | +1.3831 |
| DET-RAW-SHUFFLED120 | 9 | 0.5283 | 0.8551 | +1.0664 |
| DIFF-EB-FIXED120 | 9 | 0.5164 | 0.8613 | +1.6813 |
| DIFF-MATCH120 | 9 | 0.5017 | 0.8701 | +1.8847 |
| DIFF-POP | 9 | 0.5322 | 0.8549 | +1.6299 |
| DIFF-RAW-SHUFFLED120 | 9 | 0.5238 | 0.8569 | +1.2282 |
| DIFF-RAW-WRONG120-0 | 9 | 0.5522 | 0.8468 | +1.4086 |
| DIFF-RAW-WRONG120-1 | 3 | 0.5054 | 0.8665 | +1.4949 |
| RAW | 9 | 0.7084 | 0.8269 | +0.0000 |

Oracle MATCH−POP ceiling vs actual MATCH−POP correlation: `0.6695`; oracle MATCH−EB vs actual MATCH−EB: `0.4754`. These are development correlations only; the mathematically coupled RAW-shared correlation is not reported.

Track A natural-query raw operating points:

| method | units | EOG reduction | preservation | PSD | covariance |
|---|---:|---:|---:|---:|---:|
| DIFF-POP | 9 | 0.0271 | 0.7479 | 0.1401 | 0.2319 |
| DIFF-MATCH120 | 9 | 0.0233 | 0.7365 | 0.1580 | 0.2602 |
| DIFF-EB-FIXED120 | 9 | 0.0258 | 0.7387 | 0.1514 | 0.2520 |
| DIFF-RAW-SHUFFLED120 | 9 | 0.0230 | 0.7287 | 0.1601 | 0.2784 |

The full γ={0.50, 0.65, 0.75, 0.85, 0.925, 1.00} diagnostic is preserved in `track_a/strength_diagnostic.csv`; it was not used for method selection.

## Six-fold diagnostic routes

Positive U denotes lower paired RRMSE. U_D is diffusion−matched deterministic; U_P is personal−population; U_W and U_S are wrong/shuffled specificity. Safety values use the support-calibrated output, while the primary U values are raw outputs.

| route | U_D | raw U_P | cal U_P | U_W | U_S | EOG red. | preservation | PSD | covariance | coverage | fallback | mean w | promoted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D10 | +0.0087 | -0.0083 | +0.0055 | +0.0014 | +0.0289 | 0.0325 | 0.7788 | 0.1263 | 0.2340 | 78.6% | 3 | 0.311 | False |
| D01 | +0.0118 | -0.0060 | -0.0207 | +0.0178 | +0.0071 | 0.0335 | 0.7625 | 0.1124 | 0.2169 | 78.6% | 3 | 0.671 | False |
| D11 | +0.0128 | +0.0028 | +0.0027 | +0.0225 | +0.0372 | 0.0336 | 0.7614 | 0.1178 | 0.2305 | 78.6% | 3 | 0.507 | False |

Absolute paired performance:

| method | units | RRMSE | correlation | ΔSNR |
|---|---:|---:|---:|---:|
| RAW | 14 | 0.7826 | 0.7915 | +0.0000 |
| DET-D00 | 14 | 0.4725 | 0.8818 | +3.4253 |
| DIFF-D00 | 14 | 0.4629 | 0.8857 | +3.7626 |
| DET-D10 | 14 | 0.4798 | 0.8791 | +3.2742 |
| DIFF-D10 | 14 | 0.4711 | 0.8830 | +3.5447 |
| DIFF-D10-CAL | 14 | 0.4573 | 0.8896 | +4.1026 |
| DET-D01 | 14 | 0.4806 | 0.8750 | +3.3963 |
| DIFF-D01 | 14 | 0.4689 | 0.8830 | +3.6783 |
| DIFF-D01-CAL | 14 | 0.4836 | 0.8764 | +3.6487 |
| DET-D11 | 14 | 0.4729 | 0.8829 | +3.4641 |
| DIFF-D11 | 14 | 0.4601 | 0.8901 | +3.7931 |
| DIFF-D11-CAL | 14 | 0.4602 | 0.8884 | +4.0367 |

Raw factorial effects: geometry G `-0.0083`, LoRA A `-0.0060`, combined C `+0.0028`, interaction I `+0.0171`.
Technical validity: `passed`; three real subject bases; fixed-batch loss reduction `0.999910`; basis response `3.103363`; 28 internal ResBlock convolutions and 31616 rank-4 parameters.

Geometry is D10, score-LoRA is D01, and their joint/interaction route is D11. WRONG donors were scored individually before utility averaging. Study heterogeneity is in `per_study_effects.csv`; the bootstrap is development-descriptive, clustered by outer fold within study, and uses seed 20260811.

No route was promoted. D11 was the strongest combined signal but failed the frozen diagnostic rule because personalization coverage was 11/14 (78.6%, below 80%). Its raw U_P was small and heterogeneous: only 4/14 units were positive and only 2/5 study means were nonnegative. D10 also had negative U_P; D01 had negative U_P and a 21.4% severe-reversal rate.

The study05 units lacked sufficient disjoint support pseudo-pair roles and therefore used the predeclared support-only POP fallback (w=0); all three remain in the denominator. Consequently J5–J7, the 27-fold one-seed extension, and extra seeds were not run.

Recovery record: job 928440 was cancelled before output after the candidate-specific calibration-control fix. Array task 928421_5 then failed on study05 support coverage, making 928422 DependencyNeverSatisfied; recovery jobs 928860 and 928862 applied the predeclared POP fallback, followed by successful aggregation/finalization. Final tests passed in jobs 928867 and 928871. The first clean-checkout job 928868 exposed a node without SLURM_TMPDIR; the fallback path was repaired and clean-checkout import passed in job 928872 without changing science outputs.

Promoted routes: `[]`.

Narrow conclusion: the population artifact-subspace diffusion is a valid and useful estimator in these diagnostic folds and diffusion beats its information-matched deterministic estimator on average for D10/D01/D11. Geometry or LoRA alone did not establish utility over POP. Their combined D11 interaction is a development candidate signal, not an established subject-aware advantage and not eligible for expansion under the frozen rule. No result is confirmation or a family-wide diffusion/personalization conclusion.
