#!/usr/bin/env python3
"""Validate the administrative Slurm/configuration layer without executing science."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROFILES = {"cpu", "cpu-high", "A100", "H100", "L40S"}
ENVIRONMENTS = {
    "eeg2025": Path("/home/infres/yinwang/anaconda3/envs/eeg2025"),
    "icml": Path("/home/infres/yinwang/anaconda3/envs/icml"),
}
DEFERRED_BACKUP_FILES = {
    "b1_robust_m.yaml",
    "b2_lag_fir.yaml",
    "b3_fb_cov.yaml",
    "b4_graph_ridge.yaml",
    "b5_cca_ged_cov.yaml",
    "b6_pop_shrink.yaml",
}
GATE_FILES = {
    "g1_oracle.yaml",
    "g2_specificity.yaml",
    "g3_diffusion.yaml",
    "g4_attenuation.yaml",
    "g5_drift.yaml",
}


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def construct_unique_mapping(loader: UniqueKeyLoader, node: yaml.Node, deep: bool = False) -> Any:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_unique_mapping
)


def load_yaml(path: Path) -> Any:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)


def load_module(path: Path, module_name: str) -> Any:
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def validate(code_root: Path) -> list[str]:
    failures: list[str] = []
    cluster_path = code_root / "configs/cluster/slurm.yaml"
    environment_path = code_root / "configs/environments.yaml"
    submitter_path = code_root / "scripts/slurm/submit.sh"
    runtime_job_path = code_root / "scripts/slurm/jobs/audit_runtime.sbatch"
    renderer_job_path = code_root / "scripts/slurm/jobs/extract_pdf.sbatch"
    renderer_probe_path = code_root / "scripts/contract/probe_renderer_import.py"
    renderer_verifier_path = code_root / "scripts/contract/verify_renderer_matrix.py"

    cluster = load_yaml(cluster_path)
    if not isinstance(cluster, dict) or cluster.get("schema_version") != 1:
        failures.append("cluster configuration schema_version is not 1")
    if cluster.get("cluster_name") != "gpucluster":
        failures.append("cluster_name is not the audited gpucluster")
    profiles = cluster.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != PROFILES:
        failures.append("cluster profile set differs from the registered five profiles")
    else:
        for profile_name, profile in profiles.items():
            if not isinstance(profile, dict):
                failures.append(f"profile {profile_name} is not a mapping")
                continue
            for field in ("partition", "cpus_per_task", "memory", "walltime", "gres"):
                if field not in profile:
                    failures.append(f"profile {profile_name} lacks {field}")
        for gpu_profile in ("A100", "H100", "L40S"):
            if profiles[gpu_profile].get("gres") != "gpu:1":
                failures.append(f"GPU profile {gpu_profile} does not request exactly one GPU")
        for cpu_profile in ("cpu", "cpu-high"):
            if profiles[cpu_profile].get("gres") is not None:
                failures.append(f"CPU profile {cpu_profile} unexpectedly requests a GRES")
        if profiles["cpu-high"].get("walltime") != "5-00:00:00":
            failures.append("cpu-high full-root inventory walltime is not the audited five-day limit")
        if profiles["cpu-high"].get("checkpoint_signal") != "B:USR1@600":
            failures.append("cpu-high does not request the preregistered pre-termination signal")

    environments = load_yaml(environment_path)
    registered = environments.get("environments") if isinstance(environments, dict) else None
    if not isinstance(registered, dict) or set(registered) != set(ENVIRONMENTS):
        failures.append("environment registry differs from eeg2025/icml")
    else:
        for name, expected_path in ENVIRONMENTS.items():
            observed_path = Path(str(registered[name].get("path", "")))
            if observed_path != expected_path:
                failures.append(f"environment {name} path differs from the registered absolute path")
    policy = environments.get("policy", {}) if isinstance(environments, dict) else {}
    if policy.get("third_environment_allowed") is not False:
        failures.append("third environments are not explicitly forbidden")
    if policy.get("in_place_upgrade_allowed_without_approval") is not False:
        failures.append("unapproved in-place upgrades are not explicitly forbidden")

    backup_root = code_root / "configs/deferred_backups"
    observed_backup_files = {path.name for path in backup_root.glob("*.yaml")}
    if observed_backup_files != DEFERRED_BACKUP_FILES:
        failures.append("deferred backup config set differs from B1-B6")
    for backup_path in sorted(backup_root.glob("*.yaml")):
        backup = load_yaml(backup_path)
        if not isinstance(backup, dict) or backup.get("enabled") is not False:
            failures.append(f"deferred backup {backup_path.name} is not explicitly disabled")
        if backup.get("query_target_routing") != "forbidden":
            failures.append(f"deferred backup {backup_path.name} does not forbid query routing")

    gate_root = code_root / "configs/gates"
    observed_gate_files = {path.name for path in gate_root.glob("*.yaml")}
    if observed_gate_files != GATE_FILES:
        failures.append("gate config set differs from G1-G5")
    for gate_path in sorted(gate_root.glob("*.yaml")):
        gate = load_yaml(gate_path)
        if not isinstance(gate, dict) or gate.get("enabled") is not False:
            failures.append(f"gate {gate_path.name} is not frozen disabled")
        if gate.get("thresholds") != "TBD-PREREG":
            failures.append(f"gate {gate_path.name} has an unapproved threshold value")

    schema_root = code_root / "datasets/schemas"
    for schema_path in sorted(schema_root.glob("*.json")):
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"invalid JSON schema {schema_path.name}: {exc}")
            continue
        if not isinstance(schema, dict) or schema.get("$schema") != (
            "https://json-schema.org/draft/2020-12/schema"
        ):
            failures.append(f"JSON schema {schema_path.name} lacks the frozen draft declaration")

    submitter_text = submitter_path.read_text(encoding="utf-8")
    if "--export=ALL" in submitter_text or "--export=ALL," in submitter_text:
        failures.append("submitter inherits the login environment with --export=ALL")
    required_export_names = {
        "DENOISENET_SUBMIT_CONFIG_SHA256",
        "DENOISENET_ENV_CONFIG_SHA256",
        "DENOISENET_JOB_SCRIPT_SHA256",
        "DENOISENET_SUBMITTER_SHA256",
        "DENOISENET_CONTRACT_BUNDLE_SHA256",
        "DENOISENET_SLURM_JOBS_BUNDLE_SHA256",
        "DENOISENET_PAYLOAD_ARGS_SHA256",
        "DENOISENET_REQUEST_SHA256",
    }
    for name in sorted(required_export_names):
        if name not in submitter_text:
            failures.append(f"submitter does not export {name}")

    runtime_job_text = runtime_job_path.read_text(encoding="utf-8")
    if "latest_job_id" in runtime_job_text:
        failures.append("runtime audit still uses a shared latest-job pointer")
    if "capture_sanitized_command.py" not in runtime_job_text:
        failures.append("runtime audit does not use separated atomic sanitized stream capture")
    if "2>&1 |" in runtime_job_text:
        failures.append("runtime audit still merges lock stdout and stderr")
    if "validate_submission_request.py" not in runtime_job_text:
        failures.append("runtime audit does not strictly bind its pre-sbatch request")
    if (
        "probe_renderer_import.py" not in runtime_job_text
        or "renderer-import-comparison.json" not in runtime_job_text
        or "renderer_launchers=(direct conda_run)" not in runtime_job_text
        or "renderer_preload_plans=(none runtime_full_cuda)" not in runtime_job_text
        or "renderer_warnings_policies=(default error)" not in runtime_job_text
        or "-u CONDA_PREFIX -u CONDA_DEFAULT_ENV" not in runtime_job_text
        or "preimport.json" not in runtime_job_text
        or "renderer-evidence.sha256" not in runtime_job_text
        or "renderer-positive-control-validation.json" not in runtime_job_text
        or "renderer-positive-control-validation.sha256" not in runtime_job_text
        or 'ulimit -f 1024 || exit 91' not in runtime_job_text
        or '[[ "$(ulimit -f)" == 1024 ]] || exit 91' not in runtime_job_text
        or "ulimit -c 0" not in runtime_job_text
    ):
        failures.append("runtime audit lacks the bounded orthogonal renderer comparison")
    if not renderer_probe_path.is_file() or renderer_probe_path.is_symlink():
        failures.append("cold-start renderer probe is absent or symbolic")
    else:
        renderer_probe_text = renderer_probe_path.read_text(encoding="utf-8")
        renderer_probe_requirements = {
            'RUNTIME_IMPORTS["icml"]': "renderer probe does not reuse the runtime import order",
            '"runtime_full_cuda": RUNTIME_FULL_PRELOAD': (
                "renderer probe lacks the exact runtime-full preload mapping"
            ),
            "preload_versions[module_name] = module_version(": (
                "renderer probe lacks runtime-equivalent version reads"
            ),
            "runtime_torch_audit = torch_details(torch)": (
                "renderer probe lacks runtime-equivalent CUDA inspection"
            ),
            "publish_json(\n            preimport_output": (
                "renderer probe lacks its atomic pre-import marker"
            ),
            'EXPECTED_PYMUPDF_VERSION = "1.26.5"': (
                "renderer probe lacks the registered PyMuPDF version"
            ),
        }
        for required_text, failure in renderer_probe_requirements.items():
            if required_text not in renderer_probe_text:
                failures.append(failure)
    if not renderer_verifier_path.is_file() or renderer_verifier_path.is_symlink():
        failures.append("renderer matrix verifier is absent or symbolic")
    else:
        renderer_verifier_text = renderer_verifier_path.read_text(encoding="utf-8")
        renderer_verifier_requirements = {
            "def validate_positive_record(": (
                "renderer verifier lacks positive-record validation"
            ),
            '"preimport_ready"': "renderer verifier lacks pre-import status validation",
            '"import_ok"': "renderer verifier lacks import-success validation",
            "EXPECTED_PYMUPDF_VERSION": (
                "renderer verifier lacks PyMuPDF version validation"
            ),
            "expected_cells = set(": "renderer verifier lacks complete cell validation",
            "validate_evidence_manifest(": (
                "renderer verifier lacks renderer evidence hash validation"
            ),
        }
        for required_text, failure in renderer_verifier_requirements.items():
            if required_text not in renderer_verifier_text:
                failures.append(failure)

    runtime_matrix_requirements = {
        'expected_cell_executions": 16': "runtime audit does not declare all 16 cells",
        'probe_rc=$?\n                    probe_key=': (
            "runtime audit does not capture probe status immediately"
        ),
        'renderer_probe_rcs["$probe_key"]=$probe_rc': (
            "runtime audit does not persist the captured probe status"
        ),
        "renderer_matrix_setup_failed=true": (
            "runtime audit does not fail closed on resource-limit setup"
        ),
        'renderer_verifier_rc=$?': (
            "runtime audit does not capture renderer verifier status"
        ),
        "conda_run:runtime_full_cuda:default:1": (
            "runtime audit lacks positive-control replicate 1"
        ),
        "conda_run:runtime_full_cuda:default:2": (
            "runtime audit lacks positive-control replicate 2"
        ),
        "PYTHONWARNINGS=error python": (
            "runtime audit does not inject strict warnings inside the Conda child"
        ),
    }
    for required_text, failure in runtime_matrix_requirements.items():
        if required_text not in runtime_job_text:
            failures.append(failure)

    renderer_job_text = renderer_job_path.read_text(encoding="utf-8")
    if (
        'readonly CONDA_BIN=/home/infres/yinwang/anaconda3/bin/conda'
        not in renderer_job_text
        or '"$CONDA_BIN" run --no-capture-output -p "$ICML_ENV"' not in renderer_job_text
        or "ulimit -c 0" not in renderer_job_text
    ):
        failures.append("PDF renderer is not cold-started through registered conda run")
    if (
        "readonly RENDERER_STARTUP_AUTHORIZATION=diagnostic_pending"
        not in renderer_job_text
        or '[[ "$RENDERER_STARTUP_AUTHORIZATION" == verified ]]' not in renderer_job_text
    ):
        failures.append("formal PDF renderer lacks its explicit diagnostic-pending gate")

    for job_path in sorted((code_root / "scripts/slurm/jobs").glob("*.sbatch")):
        job_text = job_path.read_text(encoding="utf-8")
        if "grep '^SLURM_'" in job_text or 'grep "^SLURM_"' in job_text:
            failures.append(
                f"job {job_path.name} records the unbounded SLURM environment instead of a whitelist"
            )

    request_validator = load_module(
        code_root / "scripts/contract/validate_submission_request.py",
        "denoisenet_validate_submission_request",
    )
    profile_items = profiles.items() if isinstance(profiles, dict) else []
    for profile_name, profile in profile_items:
        expected_resources = request_validator.PROFILE_RESOURCES.get(profile_name)
        if expected_resources is None:
            failures.append(f"submission request validator lacks profile {profile_name}")
            continue
        for field in ("cpus_per_task", "memory", "walltime"):
            if expected_resources.get(field) != profile.get(field):
                failures.append(
                    f"submission request validator disagrees with {profile_name}.{field}"
                )
        normalized_gres = "null" if profile.get("gres") is None else profile.get("gres")
        normalized_constraint = (
            "null" if profile.get("constraint") is None else profile.get("constraint")
        )
        normalized_checkpoint_signal = (
            "null"
            if profile.get("checkpoint_signal") is None
            else profile.get("checkpoint_signal")
        )
        if expected_resources.get("gres") != normalized_gres:
            failures.append(f"submission request validator disagrees with {profile_name}.gres")
        if expected_resources.get("constraint") != normalized_constraint:
            failures.append(
                f"submission request validator disagrees with {profile_name}.constraint"
            )
        if expected_resources.get("checkpoint_signal") != normalized_checkpoint_signal:
            failures.append(
                f"submission request validator disagrees with {profile_name}.checkpoint_signal"
            )

    allocation_helper = load_module(
        code_root / "scripts/contract/capture_slurm_allocation.py",
        "denoisenet_capture_slurm_allocation",
    )
    for profile_name, profile in profile_items:
        allocation_expectation = allocation_helper.PROFILE_EXPECTATIONS.get(profile_name)
        if allocation_expectation is None:
            failures.append(f"allocation helper lacks profile {profile_name}")
            continue
        if allocation_expectation.get("partition") != profile.get("partition"):
            failures.append(f"allocation helper disagrees with {profile_name}.partition")
        if allocation_expectation.get("cpus") != profile.get("cpus_per_task"):
            failures.append(f"allocation helper disagrees with {profile_name}.cpus_per_task")
        if allocation_expectation.get("time_limit") != profile.get("walltime"):
            failures.append(f"allocation helper disagrees with {profile_name}.walltime")

    sanitizer = load_module(
        code_root / "scripts/contract/sanitize_lock.py", "denoisenet_sanitize_lock"
    )
    redacted, changed = sanitizer.sanitize_url(
        "https://user:secret@example.org/pkg.whl?token=abc#not-a-digest"
    )
    if not changed or redacted != "https://example.org/pkg.whl":
        failures.append("URL sanitizer did not remove userinfo/query/non-digest fragment")
    digest_url = "https://repo.example.org/pkg.conda#" + "a" * 64
    retained, _ = sanitizer.sanitize_url(digest_url)
    if retained != digest_url:
        failures.append("URL sanitizer did not retain a cryptographic digest fragment")
    private_url = "https://repo.example.org/t/private-token/channel/pkg.conda"
    private_redacted, private_changed = sanitizer.sanitize_url(private_url)
    if not private_changed or "private-token" in private_redacted:
        failures.append("URL sanitizer did not redact a private-channel path token")
    assignment_redacted, _ = sanitizer.sanitize_text("token=secret-value Bearer abcdef")
    if "secret-value" in assignment_redacted or "abcdef" in assignment_redacted:
        failures.append("text sanitizer did not redact assignment or bearer credentials")

    submissions_root = code_root / "reports/slurm/submissions"
    if submissions_root.exists():
        for path in submissions_root.rglob("*.json"):
            if path.name.endswith(".partial"):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                failures.append(f"invalid submission JSON {path.relative_to(code_root)}: {exc}")
                continue
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                failures.append(f"unexpected submission schema {path.relative_to(code_root)}")

    if not re.fullmatch(r"[0-9a-f]{64}", __import__("hashlib").sha256(b"").hexdigest()):
        failures.append("host hashlib SHA-256 implementation is unavailable")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    code_root = args.code_root.resolve(strict=True)
    expected_root = Path("/home/infres/yinwang/denoiseNet")
    if code_root != expected_root:
        raise SystemExit(f"unexpected code root: {code_root}")
    failures = validate(code_root)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failures else "failed",
        "failure_count": len(failures),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, args.output, follow_symlinks=False)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink()
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
