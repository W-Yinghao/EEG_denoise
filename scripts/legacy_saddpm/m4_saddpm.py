#!/usr/bin/env python
"""M4 milestone gate: full SADDPM (dual decoder + 3 losses + ArcFace) (handoff §11).

Trains the dual-decoder SADDPM jointly on all 9 subjects' Session-T windows with
``L = λ_r L_r + λ_o L_o + λ_a L_a`` and checks that (a) training is stable and (b) the ArcFace
subject-classification accuracy on held-out Session-E windows exceeds chance. Saves the trained
checkpoint used by M5-M7. Run on a GPU node (scripts/slurm/m4.sbatch).
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
from saddpm.losses.arcface_loss import arcface_loss  # noqa: E402
from saddpm.losses.orthogonality import orthogonality_loss  # noqa: E402
from saddpm.losses.recon import reconstruction_loss  # noqa: E402
from saddpm.models.config import ModelConfig  # noqa: E402
from saddpm.models.dual_decoder import DualDecoderSADDPM  # noqa: E402
from saddpm.utils.checkpoint import save_checkpoint  # noqa: E402
from saddpm.utils.logging import RunLogger  # noqa: E402
from saddpm.utils.seed import seed_everything  # noqa: E402

CONFIGS = REPO_ROOT / "configs"


@torch.no_grad()
def arcface_subject_accuracy(
    model: DualDecoderSADDPM,
    diffusion: GaussianDiffusion,
    data_cfg: DataConfig,
    subjects: list[int],
    eval_t: int,
    eval_batch: int,
    device: torch.device,
) -> float:
    """Subject-classification accuracy from z^s on Session-E windows noised at ``eval_t``."""
    model.eval()
    correct, total = 0, 0
    rng = np.random.default_rng(0)
    for s in subjects:
        sw = load_session_windows(s, data_cfg, "E")
        idx = rng.choice(len(sw.windows), size=min(eval_batch, len(sw.windows)), replace=False)
        x0 = torch.from_numpy(sw.windows[idx]).float().to(device)
        labels = torch.full((x0.shape[0],), s - 1, dtype=torch.long, device=device)
        t = torch.full((x0.shape[0],), eval_t, dtype=torch.long, device=device)
        xt = diffusion.q_sample(x0, t, torch.randn_like(x0))
        out = model(xt, t, labels)  # subject_ids only affect content branch, not z_s
        pred = model.arcface.logits(out["z_s"]).argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += x0.shape[0]
    return correct / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    with open(CONFIGS / "m4.yaml", "r", encoding="utf-8") as fh:
        m4 = yaml.safe_load(fh)
    with open(CONFIGS / "train.yaml", "r", encoding="utf-8") as fh:
        tc = yaml.safe_load(fh)
    lw = tc["loss_weights"]
    epochs = args.epochs if args.epochs is not None else int(m4["epochs"])

    data_cfg = DataConfig.from_yaml(CONFIGS / "data.yaml")
    model_cfg = ModelConfig.from_yaml(CONFIGS / "model.yaml")
    diff_cfg = DiffusionConfig.from_yaml(CONFIGS / "diffusion.yaml")
    seed_everything(data_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subjects = list(range(1, data_cfg.dataset.n_subjects + 1))
    pooled = pool_subjects(subjects, data_cfg, session_role="T")
    loader = DataLoader(
        pooled.dataset(), batch_size=int(tc["batch_size"]), shuffle=True,
        num_workers=4, drop_last=True, pin_memory=(device.type == "cuda"),
    )
    model = DualDecoderSADDPM(model_cfg).to(device)
    diffusion = GaussianDiffusion(diff_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[M4] device={device} windows={len(pooled.dataset())} epochs={epochs} params={n_params/1e6:.2f}M")
    print(f"[M4] loss weights: lr={lw['lambda_r']} lo={lw['lambda_o']} la={lw['lambda_a']}")

    emb_ids = {id(p) for p in model.subject_embed.parameters()}
    other = [p for p in model.parameters() if id(p) not in emb_ids]
    opt = torch.optim.Adam(
        [
            {"params": other, "weight_decay": float(tc["weight_decay"])},
            {"params": list(model.subject_embed.parameters()), "weight_decay": float(tc["embedding_weight_decay"])},
        ],
        lr=float(tc["lr"]), betas=tuple(tc["betas"]),
    )
    total_steps = epochs * len(loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    use_amp = device.type == "cuda" and bool(tc["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    logger = RunLogger(
        REPO_ROOT / "artifacts" / "runs" / "m4_saddpm",
        use_wandb=args.wandb, wandb_project=tc["wandb"]["project"], wandb_run_name="m4-saddpm",
        wandb_mode=tc["wandb"]["mode"], config={"milestone": "M4", "epochs": epochs, **lw},
    )

    hist = {"step": [], "L_r": [], "L_o": [], "L_a": []}
    model.train()
    step = 0
    for epoch in range(epochs):
        last = {}
        for x0, subj, _mi in loader:
            x0 = x0.to(device, non_blocking=True)
            subj = subj.to(device, non_blocking=True)
            t = torch.randint(0, diffusion.num_timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            xt = diffusion.q_sample(x0, t, noise)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(xt, t, subj)
                l_r = reconstruction_loss(out["eps_theta"] + out["eps_phi"], noise)
                l_o = orthogonality_loss(out["z_c"], out["z_s"])
                l_a = arcface_loss(model.arcface, out["z_s"], subj)
                loss = lw["lambda_r"] * l_r + lw["lambda_o"] * l_o + lw["lambda_a"] * l_a
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                print(f"  [warn] non-finite loss at step {step}; skipping")
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            last = {"L_r": l_r.item(), "L_o": l_o.item(), "L_a": l_a.item(), "L": loss.item()}
            if step % int(tc["log_every"]) == 0:
                with torch.no_grad():
                    acc = (model.arcface.logits(out["z_s"]).argmax(1) == subj).float().mean().item()
                logger.log(step, {**last, "lr": sched.get_last_lr()[0], "train_subj_acc": acc})
                for k in ("L_r", "L_o", "L_a"):
                    hist[k].append(last[k])
                hist["step"].append(step)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d} step {step:6d} L={last['L']:.4f} "
                  f"L_r={last['L_r']:.4f} L_o={last['L_o']:.4f} L_a={last['L_a']:.4f}")

    ckpt = REPO_ROOT / "artifacts" / "checkpoints" / "m4_saddpm.pt"
    save_checkpoint(ckpt, model, {"model": model_cfg.__dict__, "diffusion": diff_cfg.__dict__,
                                  "milestone": "M4", "train_mode": tc.get("train_mode", "joint")})
    print(f"[M4] saved checkpoint -> {ckpt.relative_to(REPO_ROOT)}")

    acc = arcface_subject_accuracy(model, diffusion, data_cfg, subjects,
                                   int(m4["eval_t"]), int(m4["eval_batch"]), device)

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in ("L_r", "L_o", "L_a"):
        ax.plot(hist["step"], hist[k], label=k, lw=1.0)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("loss (log)")
    ax.set_title("M4 SADDPM training losses"); ax.legend()
    fig.tight_layout()
    out_fig = REPO_ROOT / "artifacts" / "figures" / "m4_losses.png"
    fig.savefig(out_fig, dpi=130); plt.close(fig)
    logger.log_image("losses", out_fig)
    logger.finish()

    stable = all(np.isfinite(hist[k][-1]) for k in ("L_r", "L_o", "L_a")) and hist["L_r"][-1] < hist["L_r"][0]
    passed = acc > float(m4["pass_arcface_acc"]) and stable
    print(f"[M4] final L_r={hist['L_r'][-1]:.4f} (start {hist['L_r'][0]:.4f}) stable={stable}")
    print(f"[M4] ArcFace subject acc (Session-E, t={m4['eval_t']}) = {acc:.3f} "
          f"(chance {1/len(subjects):.3f}, need > {m4['pass_arcface_acc']})")
    print(f"[M4] {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
