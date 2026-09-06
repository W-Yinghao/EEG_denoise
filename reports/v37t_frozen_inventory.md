# V37T frozen evidence inventory

V37T binds 115 frozen checkpoint/evidence rows from V22/V25/V26/V27. Of these, 112 checkpoint
files were present and reproduced their registered SHA256; three V27 rows are closed-form energy
configurations with no independent checkpoint. No missing checkpoint was replaced.

The registered primary method is `V27_ENERGY_SDEDIT_L05`: V25 DeepSets raw-support context,
V26 CalibSDEdit (`sigma_start=0.05`, 10 DDIM steps, K=1), followed by V27 final-only
partial-observation energy (`lambda_a=1`, `lambda_y=0.5`). V30 supplies the common paired/natural
panel and V31 supplies the superseding exact support-duration rows. EEGDfus is the frozen V22
baseline. No compatible, registered strong U-Net checkpoint exists in the V30 inventory, so it is
reported as `not_comparable_missing_registered_checkpoint` rather than approximated.

The authoritative machine-readable binding is `results/taas_waveform_v37t/checkpoint_binding.csv`.

