# V32P SANDiff development results

## Method result

SANDiff diffuses only the LEACE-defined subject-linked component in a
128-dimensional EEGNet representation. It conditions on the retained component
and frozen four-class task logits, with no participant ID, persistent subject
embedding, or target support at inference. Its target is a task-matched private
component from another training participant. The matched one-step sanitizer
receives exactly the same non-private information.

The full three-point frontier is:

| method | strength | fixed-head BA | retrained-head BA | adaptive subject BA | verification AUROC | ECE |
|---|---|---:|---:|---:|---:|---:|
| one-step | weak | 0.3187 | 0.3314 | 0.6752 | 0.5879 | 0.3281 |
| one-step | medium | 0.3218 | 0.3281 | 0.6678 | 0.5820 | 0.3326 |
| one-step | strong | 0.3241 | 0.3266 | 0.6611 | 0.5770 | 0.3393 |
| SANDiff | weak | 0.3202 | 0.3328 | 0.6728 | 0.5873 | 0.3327 |
| SANDiff | medium | 0.3225 | 0.3320 | 0.6674 | 0.5801 | 0.3458 |
| SANDiff | strong | 0.3250 | 0.3299 | 0.6541 | 0.5735 | 0.3604 |

## Participant-first selected operating point

Strength was selected independently within each fold and seed using the
participant-disjoint validation group. No outer-test result entered selection.
After nested selection:

| method | fixed-head BA | retrained-head BA | adaptive subject BA | verification BA | verification AUROC |
|---|---:|---:|---:|---:|---:|
| one-step | 0.3227 | 0.3270 | 0.6674 | 0.5874 | 0.5806 |
| SANDiff | 0.3241 | 0.3322 | 0.6647 | 0.5864 | 0.5775 |

Across nine outer-test participants, SANDiff versus RAW changed fixed-head MI
BA by +0.00386 (participant bootstrap 95% interval −0.00367 to +0.01196;
4/9 positive), reduced adaptive subject recall by 0.00386 (−0.02007 to
+0.02932; 5/9), and reduced verification AUROC by 0.01467 (+0.00580 to
+0.02423; 7/9). Compared with LEACE, fixed-head utility was +0.00617
(+0.00097 to +0.01177; 8/9).

SANDiff versus the matched one-step sanitizer was small and heterogeneous:
fixed-head BA +0.00135 (−0.00347 to +0.00714; 4/9), adaptive privacy utility
+0.00270 (−0.01061 to +0.01582; 5/9), and AUROC reduction +0.00272
(−0.00154 to +0.00886; 5/9). Diffusion is therefore a competitive operating
point, not a uniformly superior mechanism.

## Cost and boundary

For batch size 64 on V100, one-step median latency was 0.264 ms; ten-step
SANDiff was 9.345 ms. Trainable parameter counts were 133,248 and 508,384,
respectively. Phase-C waveform interaction is `deferred_not_comparable` because
V27 outputs are SGEYESUB waveforms and do not align directly with BCI-IV-2a
trials. No waveform pipeline was rebuilt.
