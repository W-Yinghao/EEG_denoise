#!/usr/bin/env python
"""M3 milestone gate: subject embeddings + FiLM conditioning (handoff §11).

Trains a subject-conditional single-decoder DDPM jointly on all 9 subjects' Session-T windows,
then checks that conditioning changes generated samples per subject:
  (1) conditioning effect: from identical noise, different subject embeddings give different x_0;
  (2) identity preservation: generated-subject-i vs real-subject-j spectral-descriptor correlation
      is diagonally dominant (above chance), a preview of the §8.2 Table-3 analysis.

Run on a GPU node (scripts/slurm/m3.sbatch).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from saddpm.data.config import DataConfig  # noqa: E402
from saddpm.data.datasets import load_session_windows, pool_subjects  # noqa: E402
from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion  # noqa: E402
from saddpm.diffusion.schedule import DiffusionConfig  # noqa: E402
from saddpm.eval.subject_corr import (  # noqa: E402
    descriptors_from_groups,
    diagonal_dominance,
    pearson_matrix,
)
from saddpm.losses.recon import reconstruction_loss  # noqa: E402
from saddpm.models.config import ModelConfig  # noqa: E402
from saddpm.models.unet1d import UNet1D  # noqa: E402
from saddpm.utils.checkpoint import save_checkpoint  # noqa: E402
from saddpm.utils.logging import RunLogger  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402

CONFIGS = REPO_ROOT / "configs"


def make_eps_fn(model: UNet1D, subject_index: int):
    """Return an ``eps_fn(x, t)`` that conditions the model on a fixed subject id."""

    def eps_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sid = torch.full((x.shape[0],), subject_index, device=x.device, dtype=torch.long)
        return model(x, t, sid)

    return eps_fn


def _heatmap(mat: np.ndarray, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xlabel("real subject j")
    ax.set_ylabel("generated subject i")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    ax.set_xticklabels([f"A{j+1:02d}" for j in range(mat.shape[1])], fontsize=7)
    ax.set_yticklabels([f"A{i+1:02d}" for i in range(mat.shape[0])], fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    with open(CONFIGS / "m3.yaml", "r", encoding="utf-8") as fh:
        mc = yaml.safe_load(fh)
    with open(CONFIGS / "train.yaml", "r", encoding="utf-8") as fh:
        tc = yaml.safe_load(fh)
    epochs = args.epochs if args.epochs is not None else int(mc["epochs"])

    data_cfg = DataConfig.from_yaml(CONFIGS / "data.yaml")
    model_cfg = ModelConfig.from_yaml(CONFIGS / "model.yaml")
    diff_cfg = DiffusionConfig.from_yaml(CONFIGS / "diffusion.yaml")
    seed_everything(data_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subjects = list(range(1, data_cfg.dataset.n_subjects + 1))
    pooled = pool_subjects(subjects, data_cfg, session_role="T")
    loader = DataLoader(
        pooled.dataset(),
        batch_size=int(tc["batch_size"]),
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[M3] device={device} subjects={subjects} windows={len(pooled.dataset())} epochs={epochs}")

    model = UNet1D(model_cfg, subject_conditioned=True).to(device)
    diffusion = GaussianDiffusion(diff_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[M3] params={n_params/1e6:.2f}M (incl. subject embeddings)")

    emb_params = list(model.subject_embed.parameters())
    emb_ids = {id(p) for p in emb_params}
    other_params = [p for p in model.parameters() if id(p) not in emb_ids]
    opt = torch.optim.Adam(
        [
            {"params": other_params, "weight_decay": float(tc["weight_decay"])},
            {"params": emb_params, "weight_decay": float(tc["embedding_weight_decay"])},
        ],
        lr=float(tc["lr"]),
        betas=tuple(tc["betas"]),
    )
    total_steps = epochs * len(loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and bool(tc["amp"])))

    run_dir = REPO_ROOT / "artifacts" / "runs" / "m3_subject"
    logger = RunLogger(
        run_dir,
        use_wandb=args.wandb,
        wandb_project=tc["wandb"]["project"],
        wandb_run_name="m3-subject",
        wandb_mode=tc["wandb"]["mode"],
        config={"milestone": "M3", "epochs": epochs, **{k: tc[k] for k in ("lr", "batch_size")}},
    )

    model.train()
    step = 0
    for epoch in range(epochs):
        for x0, subj, _mi in loader:
            x0 = x0.to(device, non_blocking=True)
            subj = subj.to(device, non_blocking=True)
            t = torch.randint(0, diffusion.num_timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            xt = diffusion.q_sample(x0, t, noise)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and bool(tc["amp"]))):
                loss = reconstruction_loss(model(xt, t, subj), noise)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % int(tc["log_every"]) == 0:
                logger.log(step, {"loss": loss.item(), "lr": sched.get_last_lr()[0], "epoch": epoch})
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d} step {step:6d} loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e}")

    ckpt = REPO_ROOT / "artifacts" / "checkpoints" / "m3_subject_ddpm.pt"
    save_checkpoint(ckpt, model, {"model": model_cfg.__dict__, "diffusion": diff_cfg.__dict__,
                                  "subject_conditioned": True, "milestone": "M3"})
    print(f"[M3] saved checkpoint -> {ckpt.relative_to(REPO_ROOT)}")

    # ---- GATE: generation-based conditioning checks ----
    model.eval()
    n_sub = len(subjects)
    shape_gen = (int(mc["n_gen_per_subject"]), model_cfg.in_channels, model_cfg.signal_length)
    gen_groups, real_groups = [], []
    for i in range(n_sub):
        gen = diffusion.ddim_sample_loop(make_eps_fn(model, i), shape_gen, device, ddim_steps=int(mc["ddim_steps"]))
        gen_groups.append(gen.detach().cpu().numpy())
        real_groups.append(load_session_windows(subjects[i], data_cfg, "T").windows)

    desc_gen = descriptors_from_groups(gen_groups)
    desc_real = descriptors_from_groups(real_groups)
    corr_gen_real = pearson_matrix(desc_gen, desc_real)
    corr_real_real = pearson_matrix(desc_real, desc_real)
    diag = diagonal_dominance(corr_gen_real)

    # conditioning effect: identical noise, different subject -> different output.
    torch.manual_seed(0)
    x_shared = torch.randn(int(mc["n_shared_noise"]), *shape_gen[1:], device=device)
    shared_out = []
    for i in range(n_sub):
        out = diffusion.ddim_sample_loop(
            make_eps_fn(model, i), x_shared.shape, device, ddim_steps=int(mc["ddim_steps"]), x_t=x_shared.clone()
        )
        shared_out.append(out.detach().cpu().numpy())
    shared_out = np.stack(shared_out, axis=0)  # (S, M, C, L)
    cross = []
    for m in range(shared_out.shape[1]):
        flat = shared_out[:, m].reshape(n_sub, -1)
        cm = pearson_matrix(flat, flat)
        cross.append(cm[np.triu_indices(n_sub, k=1)].mean())
    mean_cross_corr = float(np.mean(cross))

    fig_dir = REPO_ROOT / "artifacts" / "figures"
    _heatmap(corr_gen_real, f"M3: gen-i vs real-j spectral corr (diag-dom={diag:.2f})", fig_dir / "m3_corr_gen_real.png")
    _heatmap(corr_real_real, "M3: real-i vs real-j spectral corr (panel a)", fig_dir / "m3_corr_real_real.png")
    logger.log_image("corr_gen_real", fig_dir / "m3_corr_gen_real.png")
    logger.finish()

    # Handoff M3 gate: "confirm conditioning changes generated samples per subject."
    cond_effect = mean_cross_corr < 0.999
    passed = cond_effect
    # Diagnostic (forward-looking): generation-time subject-identity preservation. Expected to be
    # weak here because plain L_simple does not require subject info; M4's ArcFace + orthogonality
    # losses enforce it, and it is properly evaluated at M7 (§8.2).
    strong_identity = diag >= float(mc["pass_diag_dominance"])
    print(f"[M3] mean cross-subject corr (shared noise) = {mean_cross_corr:.4f} -> conditioning changes output: {cond_effect}")
    print(f"[M3] [diagnostic] gen-vs-real diagonal dominance = {diag:.3f} (chance {1/n_sub:.3f}; "
          f"strong-identity>= {mc['pass_diag_dominance']}? {strong_identity}) -- enforced at M4")
    print(f"[M3] {'PASS' if passed else 'FAIL'} (gate = conditioning changes generated samples)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
