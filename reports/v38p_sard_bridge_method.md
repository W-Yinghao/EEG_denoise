# V38P SARD-Bridge method

SARD-Bridge models the dynamic, task/confidence-matched cross-subject donor residual distribution. It conditions on the frozen source representation, frozen logits, and a query-disjoint task-demeaned Session-1 support prototype. Training and inference never use a source/donor identity token or true test label. Canonical inference retains all K=8 draws from a 10-step x0-prediction sampler.

The canonical source-adversary weight was 0.1. Because validation showed retained/amplified source leakage, the one registered repair changed only this weight to 0.5. The repair worsened validation leakage and was rejected; its checkpoints and complete results remain preserved.
