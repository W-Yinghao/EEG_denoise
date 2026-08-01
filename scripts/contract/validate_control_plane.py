#!/usr/bin/env python3
"""Validate the administrative Slurm/configuration layer without executing science."""

from __future__ import annotations

import argparse
import hashlib
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


def validate(
    code_root: Path, expected_registration_state: str
) -> tuple[list[str], dict[str, object]]:
    failures: list[str] = []
    registration_evidence: dict[str, object] = {
        "expected_registration_state": expected_registration_state,
        "observed_registration_state": None,
        "immutable_environment_policy_sha256": None,
        "environment_audit_job_ids": {},
        "renderer_validation_sha256": None,
    }
    cluster_path = code_root / "configs/cluster/slurm.yaml"
    environment_path = code_root / "configs/environments.yaml"
    submitter_path = code_root / "scripts/slurm/submit.sh"
    runtime_job_path = code_root / "scripts/slurm/jobs/audit_runtime.sbatch"
    attachment_job_path = code_root / "scripts/slurm/jobs/review_attachments.sbatch"
    renderer_job_path = code_root / "scripts/slurm/jobs/extract_pdf.sbatch"
    renderer_helper_path = code_root / "scripts/contract/extract_pdf_pymupdf.py"
    attachment_helper_path = code_root / "scripts/contract/review_attachments.py"
    renderer_probe_path = code_root / "scripts/contract/probe_renderer_import.py"
    renderer_verifier_path = code_root / "scripts/contract/verify_renderer_matrix.py"
    renderer_startup_path = code_root / "scripts/contract/renderer_startup.py"
    environment_policy_path = code_root / "scripts/contract/environment_policy.py"

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
        renderer_authority = registered["icml"].get("renderer_startup_authority")
        expected_contract_sha256 = (
            "dfa6ace23bcb146e9bf23a50c078c5e3a391b3353e1fff83d337beaae7cb15ae"
        )
        if not isinstance(renderer_authority, dict) or set(renderer_authority) != {
            "schema_version",
            "status",
            "contract_sha256",
            "validation_sha256",
        }:
            failures.append("icml renderer startup authority registry is malformed")
        elif (
            renderer_authority.get("schema_version") != 1
            or renderer_authority.get("contract_sha256")
            != expected_contract_sha256
            or renderer_authority.get("status") != expected_registration_state
        ):
            failures.append("icml renderer startup authority policy differs")
        elif expected_registration_state == "pending":
            if renderer_authority.get("validation_sha256") is not None:
                failures.append("pending renderer authority unexpectedly has validation evidence")
        else:
            validation_sha256 = renderer_authority.get("validation_sha256")
            strict_job_id = registered["icml"].get("strict_reaudit_job_id")
            if (
                not isinstance(validation_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", validation_sha256) is None
                or not isinstance(strict_job_id, str)
                or not strict_job_id.isdigit()
            ):
                failures.append("verified renderer authority lacks pinned audit evidence")
    immutable_environment_policy_sha256: str | None = None
    try:
        environment_policy_module = load_module(
            environment_policy_path, "denoisenet_environment_policy"
        )
        environment_policy_module.validate_registration_state(
            environments, expected_registration_state
        )
        immutable_environment_policy_sha256 = (
            environment_policy_module.immutable_environment_policy_sha256(
                environment_path
            )
        )
        registration_evidence["observed_registration_state"] = (
            expected_registration_state
        )
        registration_evidence["immutable_environment_policy_sha256"] = (
            immutable_environment_policy_sha256
        )
        if isinstance(registered, dict):
            registration_evidence["environment_audit_job_ids"] = {
                name: registered[name].get("strict_reaudit_job_id")
                for name in ("eeg2025", "icml")
            }
            if isinstance(renderer_authority, dict):
                registration_evidence["renderer_validation_sha256"] = (
                    renderer_authority.get("validation_sha256")
                )
    except Exception as exc:
        failures.append(
            f"environment registration state/policy validation failed: {type(exc).__name__}"
        )
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
        or "renderer_candidate_id=conda_standard_default_none" not in runtime_job_text
        or '"design": "fixed_single_candidate"' not in runtime_job_text
        or '"expected_cell_executions": 2' not in runtime_job_text
        or "/usr/bin/env -u PYTHONHOME -u PYTHONPATH -u PYTHONWARNINGS"
        not in runtime_job_text
        or "preimport.json" not in runtime_job_text
        or "renderer-evidence.sha256" not in runtime_job_text
        or "renderer-positive-control-validation.json" not in runtime_job_text
        or "renderer-positive-control-validation.sha256" not in runtime_job_text
        or 'ulimit -f 1024 || exit 91' not in runtime_job_text
        or '[[ "$(ulimit -f)" == 1024 ]] || exit 91' not in runtime_job_text
        or "ulimit -c 0" not in runtime_job_text
        or "-u CONDA_PREFIX" in runtime_job_text
        or "PYTHONWARNINGS=error" in runtime_job_text
        or "renderer_launchers=" in runtime_job_text
    ):
        failures.append("runtime audit lacks the fixed renderer startup candidate")
    if not renderer_startup_path.is_file() or renderer_startup_path.is_symlink():
        failures.append("shared renderer startup module is absent or symbolic")
    else:
        renderer_startup_text = renderer_startup_path.read_text(encoding="utf-8")
        renderer_startup_requirements = {
            'STARTUP_CONTRACT_ID = "pymupdf_conda_standard_default_none_v1"': (
                "shared renderer startup contract ID differs"
            ),
            '"dfa6ace23bcb146e9bf23a50c078c5e3a391b3353e1fff83d337beaae7cb15ae"': (
                "shared renderer startup contract hash differs"
            ),
            "def load_registered_pymupdf(": (
                "shared renderer startup loader is absent"
            ),
            'fitz_module = importlib.import_module("fitz")': (
                "shared renderer startup lacks its unique native import"
            ),
            '"warnings_policy": "default"': (
                "shared renderer startup warnings policy differs"
            ),
            '"preload_plan": "none"': (
                "shared renderer startup preload policy differs"
            ),
            '"stderr_policy": "empty"': (
                "shared renderer startup stderr policy differs"
            ),
            '"stdout_policy": "empty"': (
                "shared renderer startup stdout policy differs"
            ),
            '"pymupdf._extra"': (
                "shared renderer startup does not forbid native extra preloading"
            ),
            'maps_path = Path("/proc/self/maps")': (
                "shared renderer startup lacks native loader-path evidence"
            ),
            'raise RendererStartupError("renderer native component was mapped before import")': (
                "shared renderer startup lacks its native pre-import absence assertion"
            ),
            'preimport_path, marker': (
                "shared renderer startup lacks its pre-import marker"
            ),
            'result_path, result': (
                "shared renderer startup lacks its import-success record"
            ),
        }
        for required_text, failure in renderer_startup_requirements.items():
            if required_text not in renderer_startup_text:
                failures.append(failure)
        try:
            renderer_startup_module = load_module(
                renderer_startup_path, "denoisenet_renderer_startup"
            )
            if (
                renderer_startup_module.startup_contract_sha256()
                != "dfa6ace23bcb146e9bf23a50c078c5e3a391b3353e1fff83d337beaae7cb15ae"
            ):
                failures.append("shared renderer startup canonical contract hash differs")
        except Exception as exc:
            failures.append(
                f"shared renderer startup contract cannot be validated: {type(exc).__name__}"
            )
    if not renderer_probe_path.is_file() or renderer_probe_path.is_symlink():
        failures.append("cold-start renderer probe is absent or symbolic")
    else:
        renderer_probe_text = renderer_probe_path.read_text(encoding="utf-8")
        renderer_probe_requirements = {
            "from renderer_startup import load_registered_pymupdf": (
                "renderer probe does not use the shared startup loader"
            ),
            'role=f"probe_r{args.replicate}"': (
                "renderer probe lacks its fixed replicate role"
            ),
            'authority={"kind": "self_audit_candidate", "audit_job_id": job_id}': (
                "renderer probe lacks its self-audit authority binding"
            ),
        }
        for required_text, failure in renderer_probe_requirements.items():
            if required_text not in renderer_probe_text:
                failures.append(failure)
        for forbidden_text in ("import fitz", "importlib.import_module", "--mode", "--preload", "--warnings-policy"):
            if forbidden_text in renderer_probe_text:
                failures.append(f"renderer probe exposes forbidden startup path: {forbidden_text}")
    if not renderer_verifier_path.is_file() or renderer_verifier_path.is_symlink():
        failures.append("renderer candidate verifier is absent or symbolic")
    else:
        renderer_verifier_text = renderer_verifier_path.read_text(encoding="utf-8")
        renderer_verifier_requirements = {
            "def validate_record(": (
                "renderer verifier lacks exact startup-record validation"
            ),
            '"preimport_ready"': "renderer verifier lacks pre-import status validation",
            '"import_ok"': "renderer verifier lacks import-success validation",
            'EXPECTED_CONTRACT_SHA256 = (': (
                "renderer verifier lacks its independently frozen contract hash"
            ),
            "expected_cells = {": "renderer verifier lacks complete fixed-cell validation",
            "validate_evidence_manifest(": (
                "renderer verifier lacks renderer evidence hash validation"
            ),
            "len(process_ids) == 2": (
                "renderer verifier does not require two independent processes"
            ),
            "component_hashes == observed_components": (
                "renderer verifier does not compare loaded components across replicates"
            ),
        }
        for required_text, failure in renderer_verifier_requirements.items():
            if required_text not in renderer_verifier_text:
                failures.append(failure)

    runtime_matrix_requirements = {
        'expected_cell_executions": 2': "runtime audit does not declare both fixed probes",
        'probe_rc=$?\n        renderer_probe_rcs["$replicate"]=$probe_rc': (
            "runtime audit does not capture fixed-probe status immediately"
        ),
        "renderer_candidate_setup_failed=true": (
            "runtime audit does not fail closed on resource-limit setup"
        ),
        'renderer_verifier_rc=$?': (
            "runtime audit does not capture renderer verifier status"
        ),
        "renderer-import-conda_run-none-warnings_default-r1.json": (
            "runtime audit lacks fixed candidate replicate 1"
        ),
        "renderer-import-conda_run-none-warnings_default-r2.json": (
            "runtime audit lacks fixed candidate replicate 2"
        ),
        '"$CONDA_BIN" run --no-capture-output -p "$env_path"': (
            "runtime audit does not use standard Conda execution"
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
        or "/usr/bin/env -u PYTHONHOME -u PYTHONPATH -u PYTHONWARNINGS"
        not in renderer_job_text
        or "ulimit -c 0" not in renderer_job_text
        or "renderer_stream_evidence_stable" not in renderer_job_text
        or "MAX_RENDERER_STREAM_BYTES" not in renderer_job_text
        or "formal-renderer-stream-validation.json" not in renderer_job_text
        or "! -s \"$formal_renderer_stdout\"" not in renderer_job_text
        or "! -s \"$formal_renderer_stderr\"" not in renderer_job_text
        or "PYTHONWARNINGS=error" in renderer_job_text
    ):
        failures.append("PDF renderer is not cold-started through registered conda run")
    if (
        "RENDERER_STARTUP_AUTHORIZATION" in renderer_job_text
        or "diagnostic_pending" in renderer_job_text
    ):
        failures.append("formal PDF renderer still uses a source-edited startup gate")
    renderer_helper_text = renderer_helper_path.read_text(encoding="utf-8")
    if (
        "from renderer_startup import (" not in renderer_helper_text
        or "load_registered_pymupdf(" not in renderer_helper_text
        or "validate_formal_startup_evidence(" not in renderer_helper_text
        or '"renderer-startup.preimport.json"' not in renderer_helper_text
        or '"renderer-startup.json"' not in renderer_helper_text
        or '"renderer_startup_result_sha256"' not in renderer_helper_text
        or "def validate_child_parent_dependency(" not in renderer_helper_text
        or renderer_helper_text.count("validate_child_parent_dependency(") != 3
        or 'contract.get("request_dependency") != f"afterok:{parent_job_id}"'
        not in renderer_helper_text
        or "import fitz" in renderer_helper_text
        or "load_fitz_renderer" in renderer_helper_text
    ):
        failures.append("formal PDF helper does not exclusively use the shared startup loader")
    attachment_helper_text = attachment_helper_path.read_text(encoding="utf-8")
    if (
        "def validate_renderer_startup_authority(" not in attachment_helper_text
        or '"renderer_startup_authority": renderer_startup_authority' not in attachment_helper_text
        or "validate_candidate(audit_dir, audit_job_id)" not in attachment_helper_text
        or 'validate_registration_state(config, "verified")' not in attachment_helper_text
        or "def load_pinned_contract_validation(" not in attachment_helper_text
    ):
        failures.append("attachment contract lacks data-driven renderer authority validation")
    attachment_job_text = attachment_job_path.read_text(encoding="utf-8")
    if (
        'contract_validation_start_sha256=""' not in attachment_job_text
        or "--expected-contract-validation-sha256" not in attachment_job_text
        or 'sha256_of "$run_dir/contract-validation-start.json"' not in attachment_job_text
    ):
        failures.append("attachment parent does not pin its start contract validation")

    if expected_registration_state == "verified" and isinstance(registered, dict):
        try:
            attachment_contract_module = load_module(
                attachment_helper_path, "denoisenet_attachment_contract"
            )
            strict_ids = {
                name: str(registered[name]["strict_reaudit_job_id"])
                for name in ("eeg2025", "icml")
            }
            if strict_ids["eeg2025"] == strict_ids["icml"]:
                raise ValueError("registered environment audits are not distinct")
            current_provenance = {
                "cluster_config_sha256": attachment_contract_module.sha256_file(
                    cluster_path
                ),
                "immutable_environment_policy_sha256": (
                    immutable_environment_policy_sha256
                ),
                "job_script_sha256": attachment_contract_module.sha256_file(
                    runtime_job_path
                ),
                "submitter_sha256": attachment_contract_module.sha256_file(
                    submitter_path
                ),
                "contract_bundle_sha256": (
                    attachment_contract_module.directory_bundle_sha256(
                        code_root / "scripts/contract", ".py"
                    )
                ),
                "slurm_jobs_bundle_sha256": (
                    attachment_contract_module.directory_bundle_sha256(
                        code_root / "scripts/slurm/jobs", ".sbatch"
                    )
                ),
            }
            for name in ("eeg2025", "icml"):
                audit_id = strict_ids[name]
                audit_dir = code_root / "reports/environments" / name / "jobs" / audit_id
                status_path = audit_dir / "status.json"
                status = attachment_contract_module.load_json_beneath(
                    code_root, status_path
                )
                expected_profile = "cpu" if name == "eeg2025" else "L40S"
                expected_environment_path = str(ENVIRONMENTS[name])
                if (
                    not isinstance(status, dict)
                    or status.get("schema_version") != 1
                    or status.get("job") != "audit_runtime"
                    or str(status.get("job_id")) != audit_id
                    or status.get("environment_name") != name
                    or status.get("environment_path") != expected_environment_path
                    or status.get("profile") != expected_profile
                    or status.get("state") != "completed"
                    or status.get("provenance_complete") is not True
                    or status.get("exit_code") != 0
                ):
                    raise ValueError(f"registered {name} audit status differs")
                for field, expected in current_provenance.items():
                    if not isinstance(expected, str) or status.get(field) != expected:
                        raise ValueError(f"registered {name} audit is stale for {field}")
                policy_record = attachment_contract_module.read_regular_beneath(
                    code_root, audit_dir / "environment-policy.sha256", 4096
                ).decode("utf-8").strip()
                if policy_record != immutable_environment_policy_sha256:
                    raise ValueError(f"registered {name} policy record differs")
                explicit_path = audit_dir / "conda-explicit.txt"
                pip_path = audit_dir / "pip-freeze.txt"
                explicit_hash = hashlib.sha256(
                    attachment_contract_module.read_regular_beneath(
                        code_root, explicit_path, 256 * 1024**2
                    )
                ).hexdigest()
                pip_hash = hashlib.sha256(
                    attachment_contract_module.read_regular_beneath(
                        code_root, pip_path, 256 * 1024**2
                    )
                ).hexdigest()
                if (
                    explicit_hash != registered[name]["explicit_manifest_sha256"]
                    or pip_hash != registered[name]["pip_manifest_sha256"]
                    or attachment_contract_module.read_regular_beneath(
                        code_root, audit_dir / "conda-explicit.sha256", 4096
                    ).decode("utf-8").strip()
                    != f"{explicit_hash}  {explicit_path}"
                    or attachment_contract_module.read_regular_beneath(
                        code_root, audit_dir / "pip-freeze.sha256", 4096
                    ).decode("utf-8").strip()
                    != f"{pip_hash}  {pip_path}"
                ):
                    raise ValueError(f"registered {name} environment locks differ")
                submission = attachment_contract_module.load_json_beneath(
                    code_root,
                    code_root / "reports/slurm/submissions" / f"{audit_id}.json",
                )
                expected_dependency = (
                    "" if name == "eeg2025" else f"afterok:{strict_ids['eeg2025']}"
                )
                if (
                    not isinstance(submission, dict)
                    or str(submission.get("job_id")) != audit_id
                    or submission.get("job") != "audit_runtime"
                    or submission.get("profile") != expected_profile
                    or submission.get("dependency") != expected_dependency
                    or submission.get("request_sha256") != status.get("request_sha256")
                ):
                    raise ValueError(f"registered {name} audit submission differs")
            renderer_evidence = (
                attachment_contract_module.validate_renderer_startup_authority(
                    code_root, registered["icml"], strict_ids["icml"]
                )
            )
            registration_evidence["renderer_validation_sha256"] = (
                renderer_evidence["validation_sha256"]
            )
            registration_evidence["renderer_evidence_manifest_sha256"] = (
                renderer_evidence["evidence_manifest_sha256"]
            )
        except Exception as exc:
            failures.append(
                f"verified environment/renderer evidence validation failed: {type(exc).__name__}"
            )

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
    return failures, registration_evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-registration-state",
        choices=("pending", "verified"),
        required=True,
    )
    args = parser.parse_args()

    code_root = args.code_root.resolve(strict=True)
    expected_root = Path("/home/infres/yinwang/denoiseNet")
    if code_root != expected_root:
        raise SystemExit(f"unexpected code root: {code_root}")
    failures, registration_evidence = validate(
        code_root, args.expected_registration_state
    )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failures else "failed",
        "failure_count": len(failures),
        "failures": failures,
        "registration_evidence": registration_evidence,
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
