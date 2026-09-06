# MobileBCI v5 no-training closure

Final label: `PROTOCOL_CORE_VALID / SSVEP_SAFETY_PREVIOUSLY_INVALID / ONE-SEED_RAW_TEMPORAL_ROUTE_NO_GO / DIFFUSION_FAMILY_NOT_TESTED`.

The event onset remains a 100-Hz sample index, while the event duration field is already seconds and is no longer divided by 100. SSVEP safety was recomputed from 1080 frozen method/unit outputs. Training-target clipping cannot be reconstructed because targets were not saved. The historical wrong donors carry a fold-role confound; P-C used old per-window labels rather than bounded-oracle masks; numerical zero EOG inputs were not true token masking. No v5 model was retrained and no sealed participant was opened.
