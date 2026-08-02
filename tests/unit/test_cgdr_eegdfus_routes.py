"""Static contracts for the Slurm-facing EEGDfus benchmark route.

The submitter cannot be executed from a unit test because a successful call
would submit work.  These assertions keep the frozen CLI/stage/profile
contract reviewable while the scheduled CPU validation exercises the adapter.
"""

from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_PATH = Path("configs/baselines/eegdfus_native_strict.yaml")
CLI_PATH = Path("src/eeg_cgdr/cli/main.py")
JOB_PATH = Path("scripts/slurm/jobs/cgdr.sbatch")
SUBMIT_PATH = Path("scripts/slurm/submit.sh")


def test_eegdfus_cli_routes_cpu_and_gpu_stages_and_preserves_exit_75() -> None:
    cli = CLI_PATH.read_text(encoding="utf-8")

    assert '"eegdfus-benchmark",' in cli
    assert 'if args.stage == "cpu-tests":' in cli
    assert "run_eegdfus_cpu_validation(config, run_dir=args.run_dir)" in cli
    assert 'elif args.stage in {"smoke", "full"}:' in cli
    assert "run_eegdfus_stage(" in cli
    assert '_array_task_index(f"eegdfus-{args.stage}")' in cli
    assert (
        'return_code = 75 if result["status"] == "checkpointed_for_resume" else 0'
        in cli
    )


def test_eegdfus_job_records_run_identity_and_forwards_usr1() -> None:
    job = JOB_PATH.read_text(encoding="utf-8")

    assert "eegdfus-benchmark)" in job
    assert "eegdfus-benchmark cpu-tests requires cpu" in job
    assert "eegdfus-benchmark smoke requires V100-32GB, L40S or gpu-any" in job
    assert "eegdfus-benchmark full requires A100, H100 or gpu-any" in job
    assert "tests/unit/test_cgdr_eegdfus_benchmark.py" in job
    assert "tests/unit/test_cgdr_eegdfus_routes.py" in job
    assert (
        'run_dir="$CODE_ROOT/runs/'
        'cgdr_${mode}${stage_suffix}_${job_id}${task_suffix}"' in job
    )
    assert 'printf \'%s\\n\' "$job_id" > "$run_dir/slurm_job_id.txt"' in job
    assert '"$SLURM_ARRAY_TASK_ID" > "$run_dir/slurm_array_task_id.txt"' in job
    assert "trap forward_usr1 USR1" in job
    assert "kill -USR1 \"$child_pid\"" in job
    assert 'exit "$child_status"' in job


def test_eegdfus_submit_route_freezes_profiles_and_array_shape() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    execution = config["execution"]
    assert execution == {
        "mode": "eegdfus-benchmark",
        "stages": ["cpu-tests", "smoke", "full"],
        "cpu_test_profile": "cpu",
        "smoke_profiles": ["V100-32GB", "L40S", "gpu-any"],
        "full_profiles": ["gpu-any", "A100", "H100"],
        "array": "0-7%8",
        "cpu_test_command": (
            "scripts/slurm/submit.sh cpu cgdr eegdfus-benchmark "
            "configs/baselines/eegdfus_native_strict.yaml cpu-tests"
        ),
        "smoke_command": (
            "scripts/slurm/submit.sh gpu-any cgdr --array 0-7%8 "
            "eegdfus-benchmark configs/baselines/eegdfus_native_strict.yaml smoke"
        ),
        "full_command": (
            "scripts/slurm/submit.sh gpu-any cgdr --array 0-7%8 "
            "eegdfus-benchmark configs/baselines/eegdfus_native_strict.yaml full"
        ),
        "resume_command": (
            "scripts/slurm/submit.sh gpu-any cgdr --array TASK_INDEX "
            "eegdfus-benchmark configs/baselines/eegdfus_native_strict.yaml full"
        ),
    }

    submitter = SUBMIT_PATH.read_text(encoding="utf-8")
    assert "cpu-tests:cpu)" in submitter
    assert "smoke:V100-32GB|smoke:L40S|smoke:gpu-any)" in submitter
    assert "full:A100|full:H100|full:gpu-any)" in submitter
    assert submitter.count("[[ \"$array_spec\" == '0-7%8'") >= 2
    assert submitter.count('"$array_spec" =~ ^[0-7]$') >= 2
    assert "eegdfus-benchmark config must be inside the code root" in submitter
    assert 'light_sbatch_args+=(--signal="$checkpoint_signal")' in submitter
