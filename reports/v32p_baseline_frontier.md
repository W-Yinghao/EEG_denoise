# V32P baseline privacy–utility frontier

## Protocol

BCI Competition IV-2a was evaluated with one EEGNet representation and a
strict 3/3/3 cyclic participant split. Models train on official session T from
training participants, checkpoints are selected with session E from separate
validation participants, and outer-test participants are never used for model
selection. Adaptive subject classifiers train on sanitized session-T outputs
from the three held-out participants and test on their session-E outputs.
Chance is 0.25 for the MI task and 0.333 for the fold-local subject attack.

Each official trial contributes exactly one 22-channel, 2-second input. Seeds
are repeated computational runs, not biological samples.

## Phase-A results

| method | fixed-head MI BA | retrained-head MI BA | adaptive subject BA | verification AUROC | worst-participant MI BA |
|---|---:|---:|---:|---:|---:|
| RAW | 0.3202 | 0.3318 | 0.6686 | 0.5932 | 0.2604 |
| LEACE | 0.3179 | 0.3275 | 0.6651 | 0.5753 | 0.2662 |
| DANN/GRL | 0.3260 | 0.3279 | 0.6798 | 0.5851 | 0.2627 |
| one-step, strong | 0.3241 | 0.3266 | 0.6611 | 0.5770 | 0.2714 |
| SANDiff, strong | 0.3250 | 0.3299 | 0.6541 | 0.5735 | 0.2755 |

LEACE lowered cross-session verification leakage but did little against the
adaptive classifier. The registered DANN instance did not reduce adaptive
subject classification and had worse calibration (ECE 0.4429 versus RAW
0.3193). Both generative replacement methods trace a smooth strength curve:
stronger replacement reduces leakage while preserving approximately the same
MI utility.

These values show residual subject linkage well above chance. They do not
support formal anonymity.
