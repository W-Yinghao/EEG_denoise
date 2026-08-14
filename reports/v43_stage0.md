# V43 Stage 0 — panel note and preregistration pointer

Panel provenance (fact note for the manuscript, not an audit): per
`configs/cgdr/counterfactual_operator_headroom_v19.yaml`
(`data_root: /projects/EEG-foundation-model/mobile_bci`), the 15-participant
46-EEG + 4-eye-electrode panel is MobileBCI (OSF R7S9B), 16 development / 8 sealed.

Preregistration: `reports/v43_preregistration.md` — frozen and committed before the
first Slurm submission of V43. No section-4.2 threshold may be tuned afterwards.

Base: branch `codex/cleanroom-calib-saddpm-cond-v42r`, commit
`ec635b2177442cd54620b8075366c81d42d5704a`; work branch `codex/rgcc-v43`.
Frozen inputs: checkpoints under
`/projects/EEG-foundation-model/derived/denoiseNet/calib_saddpm_cond_v42r/job_941770/`
(10 fold/seed cells, all with `best.pt` + `result.json`, verified present), prepared
MobileBCI records under
`/projects/EEG-foundation-model/derived/denoiseNet/counterfactual_operator_headroom_v19/prepared/`,
and the read-only V42R results tree `results/calib_saddpm_cond_v42r/`.
