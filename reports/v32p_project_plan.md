# V32P SANDiff positive-method pilot

This development-only pilot uses BCI Competition IV-2a, one EEGNet encoder,
the official T/E session boundary, and participant-grouped outer folds. It
does not read waveform sealed data or modify the waveform denoising evidence.

The operational target is a **subject-linked nuisance**: a representation
component that is predictive of participant identity under the registered
attackers but is not necessary for four-class motor-imagery prediction. No
causal anatomical, session, montage, reference, or acquisition attribution is
made.

## Registered protocol

- Outer groups are A01--A03, A04--A06, and A07--A09. Each fold uses a strict
  3/3/3 cyclic train/validation/test assignment.
- EEGNet and privacy methods train on official session T from training
  participants; checkpoint selection uses session E from participant-disjoint
  validation participants.
- Adaptive privacy attackers are retrained on sanitized session-T outputs from
  held-out participants and evaluated on their session-E outputs.
- One non-overlapping 2-second trial window (0.5--2.5 seconds after cue) is used
  per official trial. Trial windows are not treated as repeated biological
  samples.
- RAW, LEACE, DANN/GRL, matched one-step replacement, and SANDiff share the
  split. SANDiff uses K=1 and ten DDIM reverse steps.
- Weak, medium, and strong replacement points are all retained. Two SANDiff
  development seeds are reported; seeds are not biological replicates.

## Boundary

V27 waveform interaction is `deferred_not_comparable`: its frozen outputs are
from SGEYESUB and do not align directly to BCI-IV-2a trials. No waveform
pipeline will be reconstructed for this pilot.
