# V33P SANDiff consolidation plan

V33P does not introduce a new method. It reuses the V32P BCI-IV-2a EEGNet,
LEACE decomposition, cross-subject donor construction, one-step sanitizer,
SANDiff architecture, attackers, and utility metrics.

Each fold has a permanently held-out three-participant outer test group. Stage
A uses the registered three training participants and three disjoint validation
participants to select EEGNet and sanitizer epochs. SANDiff validation uses the
deployed K=1, ten-step reverse sampler at strong replacement strength and an
adaptive attacker trained on validation Session T and tested on Session E.

Stage B discards Stage-A weights and refits from scratch on Session T from all
six non-test participants for exactly the frozen epochs. Outer-test Session T
is then used only to train adaptive attackers; Session E supplies task and
privacy outcomes. Weak and medium replacement remain curve points, while
strong one-step and strong SANDiff are primary.

No waveform sealed data, additional dataset, encoder, diffusion family, or
manuscript asset is opened or modified.
