#!/usr/bin/env python3
"""Hash the immutable portion of the two-environment registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


REGISTERED_CONFIG = Path("/home/infres/yinwang/denoiseNet/configs/environments.yaml")
MUTABLE_ENVIRONMENT_FIELDS = {
    "audit_job_id",
    "responsibility_status",
    "explicit_manifest_sha256",
    "pip_manifest_sha256",
    "compatibility_status",
    "strict_reaudit_job_id",
}
MUTABLE_RENDERER_AUTHORITY_FIELDS = {"status", "validation_sha256"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def construct_unique_mapping(
    loader: UniqueKeyLoader, node: yaml.Node, deep: bool = False
) -> Any:
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


def immutable_environment_policy(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != 1:
        raise ValueError("environment registry schema differs")
    environments = config.get("environments")
    if not isinstance(environments, dict) or set(environments) != {"eeg2025", "icml"}:
        raise ValueError("environment registry set differs")
    normalized = json.loads(json.dumps(config))
    normalized_environments = normalized["environments"]
    for environment_name in ("eeg2025", "icml"):
        entry = normalized_environments.get(environment_name)
        if not isinstance(entry, dict):
            raise ValueError("environment registry entry differs")
        for field in MUTABLE_ENVIRONMENT_FIELDS:
            if field not in entry:
                raise ValueError(f"environment mutable registration field is absent: {field}")
            del entry[field]
    renderer_authority = normalized_environments["icml"].get(
        "renderer_startup_authority"
    )
    if not isinstance(renderer_authority, dict):
        raise ValueError("renderer authority policy is absent")
    for field in MUTABLE_RENDERER_AUTHORITY_FIELDS:
        if field not in renderer_authority:
            raise ValueError(f"renderer mutable registration field is absent: {field}")
        del renderer_authority[field]
    return normalized


def validate_registration_state(
    config: dict[str, Any], expected_state: str
) -> dict[str, str | None]:
    if expected_state not in {"pending", "verified"}:
        raise ValueError("environment registration state differs")
    environments = config.get("environments")
    if not isinstance(environments, dict) or set(environments) != {"eeg2025", "icml"}:
        raise ValueError("environment registration set differs")
    observed_ids: dict[str, str | None] = {}
    for name in ("eeg2025", "icml"):
        entry = environments.get(name)
        if not isinstance(entry, dict):
            raise ValueError("environment registration entry differs")
        strict_job_id = entry.get("strict_reaudit_job_id")
        observed_ids[name] = str(strict_job_id) if strict_job_id is not None else None
        if not isinstance(entry.get("audit_job_id"), str) or not str(
            entry["audit_job_id"]
        ).isdigit():
            raise ValueError("base environment audit ID differs")
        if not isinstance(entry.get("explicit_manifest_sha256"), str) or HEX64.fullmatch(
            str(entry["explicit_manifest_sha256"])
        ) is None:
            raise ValueError("environment explicit lock hash differs")
        if not isinstance(entry.get("pip_manifest_sha256"), str) or HEX64.fullmatch(
            str(entry["pip_manifest_sha256"])
        ) is None:
            raise ValueError("environment pip lock hash differs")
        if expected_state == "pending":
            if (
                strict_job_id is not None
                or entry.get("compatibility_status") != "pending_strict_reaudit"
                or entry.get("responsibility_status")
                != "audit_completed_current_bundle_stale"
            ):
                raise ValueError("pending environment registration fields differ")
        elif (
            not isinstance(strict_job_id, str)
            or not strict_job_id.isdigit()
            or entry.get("audit_job_id") != strict_job_id
            or entry.get("compatibility_status") != "compatible"
            or entry.get("responsibility_status") != "verified_strict_reaudit"
        ):
            raise ValueError("verified environment registration fields differ")
    renderer_authority = environments["icml"].get("renderer_startup_authority")
    if not isinstance(renderer_authority, dict):
        raise ValueError("renderer authority registration is absent")
    if expected_state == "pending":
        if (
            renderer_authority.get("status") != "pending"
            or renderer_authority.get("validation_sha256") is not None
        ):
            raise ValueError("pending renderer authority fields differ")
    elif (
        renderer_authority.get("status") != "verified"
        or not isinstance(renderer_authority.get("validation_sha256"), str)
        or HEX64.fullmatch(str(renderer_authority["validation_sha256"])) is None
    ):
        raise ValueError("verified renderer authority fields differ")
    return observed_ids


def immutable_environment_policy_sha256(path: Path = REGISTERED_CONFIG) -> str:
    if Path(path).resolve(strict=True) != REGISTERED_CONFIG:
        raise ValueError("environment policy path differs")
    metadata = path.lstat()
    if path.is_symlink() or not path.is_file() or metadata.st_size > 1024 * 1024:
        raise ValueError("environment policy file is unsafe")
    config = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    if not isinstance(config, dict):
        raise ValueError("environment registry is not a mapping")
    normalized = immutable_environment_policy(config)
    canonical = json.dumps(
        normalized, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--expected-registration-state", choices=("pending", "verified"), required=True
    )
    args = parser.parse_args()
    config = yaml.load(args.config.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    if not isinstance(config, dict):
        raise ValueError("environment registry is not a mapping")
    validate_registration_state(config, args.expected_registration_state)
    print(immutable_environment_policy_sha256(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
