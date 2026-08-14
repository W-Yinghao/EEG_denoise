# V42R reviewer evidence map

| Revision request | Registered V42R evidence |
|---|---|
| Stronger baselines | Bound official EEGDfus native reproduction, audited official EEGdenoiseNet CNN/DeepSeparator evidence, and frozen joint deterministic reference where protocol-compatible. |
| Subject-agnostic DDPM | POP route of the identical joint model and checkpoint. |
| Subject mechanism | Common-noise POP/MATCH/WRONG/SHUFFLED/ORACLE interventions. |
| Subject embedding | Replaced by a query-disjoint, two-bipolar-EOG transfer estimate; no identity token or raw support waveform enters diffusion. |
| FiLM/residual contribution | `NO_TRANSFER_BRANCH` disables only the transfer residual decoder. |
| Target data amount | Exact POP at 0 s and prefix-correct 10/30 s support. |
| Statistical testing | Participant-first mean, median, paired bootstrap interval, positive count, and complete participant effects. |
| SADDPM versus SADDPM-Cond | One paired conditional (x_0) model with observation (y) at every reverse step. |
| Transductive limitation | No query-gradient adaptation; support and query are disjoint. |
| Privacy | Lightweight linkage audit of the ephemeral transfer state plus session-end deletion recommendation. |
| Exact t/K/sampling | T=1,000 linear diffusion; deterministic DDIM50 primary; no target-selected draw. |

The deterministic reference positions absolute performance but is not an automatic diffusion-elimination gate.

