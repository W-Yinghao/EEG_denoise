"""Static Slurm route checks for the CPU-only decision aggregate."""

from pathlib import Path


def test_decision_aggregate_has_one_cpu_non_array_route() -> None:
    cli = Path("src/eeg_cgdr/cli/main.py").read_text(encoding="utf-8")
    job = Path("scripts/slurm/jobs/cgdr.sbatch").read_text(encoding="utf-8")
    submitter = Path("scripts/slurm/submit.sh").read_text(encoding="utf-8")
    assert '"diffusion-incremental-decision"' in cli
    assert 'args.mode == "diffusion-incremental-decision"' in cli
    assert "run_diffusion_incremental_decision" in cli
    assert "diffusion-incremental-decision)" in job
    assert "diffusion-incremental-decision aggregate requires cpu" in job
    assert "diffusion-incremental-decision rejects array tasks" in job
    assert "diffusion-incremental-decision aggregate requires cpu" in submitter
    assert "diffusion-incremental-decision rejects arrays" in submitter


def test_decision_route_uses_frozen_canonical_config() -> None:
    config = Path("configs/cgdr/diffusion_incremental_decision.yaml").read_text(
        encoding="utf-8"
    )
    assert "frozen_before_evaluation_outputs: true" in config
    assert "eegdfus_full_aggregate_summary:" in config
    assert "klados_conditional_v2_summary:" in config
    assert "klados_deterministic_v4_summary:" in config
    assert "output_root: results/cgdr/diffusion_incremental_decision" in config

