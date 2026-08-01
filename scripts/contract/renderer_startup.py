#!/usr/bin/env python3
"""Single fail-closed startup path for the registered PyMuPDF renderer."""

from __future__ import annotations

import faulthandler
import hashlib
import importlib
import importlib.util
import json
import os
import resource
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ICML_ENV = Path("/home/infres/yinwang/anaconda3/envs/icml")
ICML_PYTHON = ICML_ENV / "bin" / "python"
CONDA_BIN = Path("/home/infres/yinwang/anaconda3/bin/conda")
EXPECTED_PYMUPDF_VERSION = "1.26.5"
STARTUP_CONTRACT_ID = "pymupdf_conda_standard_default_none_v1"
STARTUP_CONTRACT: dict[str, object] = {
    "candidate_id": "conda_standard_default_none",
    "conda_bin": str(CONDA_BIN),
    "environment_name": "icml",
    "environment_path": str(ICML_ENV),
    "launcher": "conda_run",
    "preload_plan": "none",
    "profile": "L40S",
    "pymupdf_version": EXPECTED_PYMUPDF_VERSION,
    "python_command": "python",
    "pythonhome": "unset",
    "pythonpath": "unset",
    "pythonwarnings": "unset",
    "schema_version": 1,
    "stderr_policy": "empty",
    "stdout_policy": "empty",
    "warnings_policy": "default",
}
STARTUP_CONTRACT_CANONICAL = json.dumps(
    STARTUP_CONTRACT, ensure_ascii=True, separators=(",", ":"), sort_keys=True
).encode("ascii")
# This value is frozen independently into configs/environments.yaml before audit.
EXPECTED_STARTUP_CONTRACT_SHA256 = (
    "dfa6ace23bcb146e9bf23a50c078c5e3a391b3353e1fff83d337beaae7cb15ae"
)
FORBIDDEN_PRELOADED_MODULES = (
    "fitz",
    "pymupdf",
    "pymupdf.mupdf",
    "pymupdf._mupdf",
    "pymupdf._extra",
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "mne",
    "yaml",
    "torch",
    "einops",
)
COMPONENT_RELATIVE_PATHS = (
    "fitz/__init__.py",
    "pymupdf/__init__.py",
    "pymupdf/mupdf.py",
    "pymupdf/_mupdf.so",
    "pymupdf/_extra.so",
    "pymupdf/libmupdf.so.26.10",
    "pymupdf/libmupdfcpp.so.26.10",
)
NATIVE_COMPONENT_RELATIVE_PATHS = (
    "pymupdf/_mupdf.so",
    "pymupdf/_extra.so",
    "pymupdf/libmupdf.so.26.10",
    "pymupdf/libmupdfcpp.so.26.10",
)
ALLOWED_ROLES = {"probe_r1", "probe_r2", "formal_pdf_extraction"}


class RendererStartupError(RuntimeError):
    """A fixed renderer-startup contract failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def startup_contract_sha256() -> str:
    observed = hashlib.sha256(STARTUP_CONTRACT_CANONICAL).hexdigest()
    if observed != EXPECTED_STARTUP_CONTRACT_SHA256:
        raise RendererStartupError("renderer startup contract hash is not frozen")
    return observed


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def startup_module_sha256() -> str:
    return sha256_path(Path(__file__).resolve(strict=True))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    parent = path.parent
    parent_metadata = parent.lstat()
    if not stat.S_ISDIR(parent_metadata.st_mode) or stat.S_ISLNK(parent_metadata.st_mode):
        raise RendererStartupError("renderer startup output parent is unsafe")
    if path.exists() or path.is_symlink():
        raise RendererStartupError("renderer startup output already exists")
    temporary = path.with_name(path.name + ".partial")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("renderer startup JSON write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink()
    _fsync_directory(parent)


def _require_regular_component(path: Path) -> Path:
    resolved_environment = ICML_ENV.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(resolved_environment)
    except ValueError as exc:
        raise RendererStartupError("renderer component leaves the registered environment") from exc
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RendererStartupError("renderer component is not a regular non-symbolic file")
    return resolved


def _site_packages() -> Path:
    candidates = sorted((ICML_ENV / "lib").glob("python*/site-packages"))
    if len(candidates) != 1:
        raise RendererStartupError("registered renderer site-packages path is ambiguous")
    candidate = candidates[0]
    metadata = candidate.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RendererStartupError("registered renderer site-packages path is unsafe")
    return candidate


def _validate_import_candidates(site_packages: Path) -> None:
    expected = {
        "fitz": (site_packages / "fitz" / "__init__.py").resolve(strict=True),
        "pymupdf": (site_packages / "pymupdf" / "__init__.py").resolve(strict=True),
    }
    for module_name, expected_origin in expected.items():
        specification = importlib.util.find_spec(module_name)
        if specification is None or specification.origin is None:
            raise RendererStartupError("registered renderer import candidate is absent")
        observed_origin = _require_regular_component(Path(specification.origin))
        if observed_origin != expected_origin:
            raise RendererStartupError("renderer import candidate origin differs")


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (os.major(metadata.st_dev), os.minor(metadata.st_dev), metadata.st_ino)


def _mapped_regular_file_identities() -> dict[Path, set[tuple[int, int, int]]]:
    maps_path = Path("/proc/self/maps")
    maps_metadata = maps_path.lstat()
    if not stat.S_ISREG(maps_metadata.st_mode) or maps_metadata.st_size > 16 * 1024 * 1024:
        raise RendererStartupError("native loader map evidence is unsafe")
    loaded: dict[Path, set[tuple[int, int, int]]] = {}
    native_basenames = {Path(relative).name for relative in NATIVE_COMPONENT_RELATIVE_PATHS}
    with maps_path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            fields = line.rstrip("\n").split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith("/"):
                mapped_name = fields[5]
                if mapped_name.endswith(" (deleted)"):
                    if Path(mapped_name[: -len(" (deleted)")]).name in native_basenames:
                        raise RendererStartupError("mapped renderer component was deleted")
                    continue
                candidate = Path(mapped_name)
                if not candidate.exists() or candidate.is_symlink():
                    continue
                try:
                    device_major, device_minor = (
                        int(value, 16) for value in fields[3].split(":", 1)
                    )
                    inode = int(fields[4], 10)
                except (TypeError, ValueError) as exc:
                    raise RendererStartupError("native loader identity is malformed") from exc
                if inode < 1:
                    continue
                resolved = candidate.resolve(strict=True)
                loaded.setdefault(resolved, set()).add(
                    (device_major, device_minor, inode)
                )
    return loaded


def _native_component_paths(site_packages: Path) -> set[Path]:
    return {
        _require_regular_component(site_packages / relative)
        for relative in NATIVE_COMPONENT_RELATIVE_PATHS
    }


def _mapped_renderer_native_identities(
    site_packages: Path,
) -> dict[Path, set[tuple[int, int, int]]]:
    expected = _native_component_paths(site_packages)
    native_basenames = {path.name for path in expected}
    mapped = _mapped_regular_file_identities()
    relevant = {
        path: identities for path, identities in mapped.items() if path.name in native_basenames
    }
    unexpected = set(relevant) - expected
    if unexpected:
        raise RendererStartupError("renderer native basename is mapped outside the registered environment")
    return relevant


def _hash_component_from_descriptor(path: Path) -> tuple[Path, tuple[int, int, int], str]:
    resolved = _require_regular_component(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        path_metadata = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(path_metadata.st_mode)
            or _file_identity(before) != _file_identity(path_metadata)
        ):
            raise RendererStartupError("renderer component descriptor identity differs")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            _file_identity(after) != _file_identity(before)
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise RendererStartupError("renderer component changed while hashing")
        return resolved, _file_identity(after), digest.hexdigest()
    finally:
        os.close(descriptor)


def _component_hashes(site_packages: Path, fitz_module: Any) -> dict[str, str]:
    module_origins = {
        "fitz/__init__.py": getattr(fitz_module, "__file__", None),
        "pymupdf/__init__.py": getattr(sys.modules.get("pymupdf"), "__file__", None),
        "pymupdf/mupdf.py": getattr(sys.modules.get("pymupdf.mupdf"), "__file__", None),
        "pymupdf/_mupdf.so": getattr(sys.modules.get("pymupdf._mupdf"), "__file__", None),
        "pymupdf/_extra.so": getattr(sys.modules.get("pymupdf._extra"), "__file__", None),
    }
    hashes: dict[str, str] = {}
    identities: dict[Path, tuple[int, int, int]] = {}
    for relative in COMPONENT_RELATIVE_PATHS:
        path = site_packages / relative
        resolved, identity, digest = _hash_component_from_descriptor(path)
        observed_module_origin = module_origins.get(relative)
        if observed_module_origin is not None and Path(observed_module_origin).resolve(strict=True) != resolved:
            raise RendererStartupError("loaded renderer module origin differs")
        hashes[relative] = digest
        identities[resolved] = identity
    if any(module_origins.get(name) is None for name in module_origins):
        raise RendererStartupError("registered renderer did not load its expected modules")
    mapped = _mapped_renderer_native_identities(site_packages)
    for path in _native_component_paths(site_packages):
        if path not in mapped or identities.get(path) not in mapped[path]:
            raise RendererStartupError("registered native renderer mapping identity differs")
    return hashes


def _validate_authority(
    role: str, job_id: str, authority: Mapping[str, object]
) -> dict[str, object]:
    normalized = dict(authority)
    if role in {"probe_r1", "probe_r2"}:
        expected = {"kind": "self_audit_candidate", "audit_job_id": job_id}
        if normalized != expected:
            raise RendererStartupError("renderer probe authority differs")
    else:
        expected_keys = {
            "schema_version",
            "authorized",
            "audit_job_id",
            "contract_sha256",
            "validation_sha256",
            "evidence_manifest_sha256",
            "component_sha256",
            "startup_module_sha256",
        }
        if (
            set(normalized) != expected_keys
            or normalized.get("schema_version") != 1
            or normalized.get("authorized") is not True
            or not str(normalized.get("audit_job_id", "")).isdigit()
            or normalized.get("contract_sha256") != startup_contract_sha256()
            or not all(
                isinstance(normalized.get(name), str)
                and len(str(normalized[name])) == 64
                and all(character in "0123456789abcdef" for character in str(normalized[name]))
                for name in (
                    "validation_sha256",
                    "evidence_manifest_sha256",
                    "startup_module_sha256",
                )
            )
            or not isinstance(normalized.get("component_sha256"), dict)
        ):
            raise RendererStartupError("formal renderer authority differs")
    return normalized


def load_registered_pymupdf(
    *,
    role: str,
    job_id: str,
    preimport_path: Path,
    result_path: Path,
    authority: Mapping[str, object],
) -> Any:
    """Load PyMuPDF exactly once under the one registered startup contract."""
    if role not in ALLOWED_ROLES or not job_id.isdigit():
        raise RendererStartupError("renderer startup role or job ID differs")
    if preimport_path.parent != result_path.parent or preimport_path == result_path:
        raise RendererStartupError("renderer startup evidence paths differ")
    if not preimport_path.is_absolute() or not result_path.is_absolute():
        raise RendererStartupError("renderer startup evidence paths are not absolute")
    contract_sha256 = startup_contract_sha256()
    normalized_authority = _validate_authority(role, job_id, authority)
    if Path(sys.prefix).resolve(strict=True) != ICML_ENV.resolve(strict=True):
        raise RendererStartupError("renderer process prefix differs")
    if Path(os.path.abspath(sys.executable)) != ICML_PYTHON:
        raise RendererStartupError("renderer Python executable differs")
    if os.environ.get("CONDA_PREFIX") != str(ICML_ENV):
        raise RendererStartupError("renderer Conda prefix differs")
    if os.environ.get("CONDA_DEFAULT_ENV") != "icml":
        raise RendererStartupError("renderer Conda environment name differs")
    conda_shlvl = os.environ.get("CONDA_SHLVL", "")
    if not conda_shlvl.isdigit() or int(conda_shlvl) < 1:
        raise RendererStartupError("renderer Conda activation depth is absent")
    if any(os.environ.get(name) is not None for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONWARNINGS", "LD_PRELOAD")):
        raise RendererStartupError("renderer inherited a forbidden startup override")
    if sys.warnoptions:
        raise RendererStartupError("renderer warnings policy is not the default")
    preloaded = [name for name in FORBIDDEN_PRELOADED_MODULES if name in sys.modules]
    if preloaded:
        raise RendererStartupError("renderer process contains a forbidden preload")
    core_soft, _ = resource.getrlimit(resource.RLIMIT_CORE)
    if core_soft != 0:
        raise RendererStartupError("renderer core-dump limit is not zero")
    faulthandler.enable(all_threads=True)
    if not faulthandler.is_enabled():
        raise RendererStartupError("renderer faulthandler is not enabled")
    site_packages = _site_packages()
    _validate_import_candidates(site_packages)
    if any(name in sys.modules for name in FORBIDDEN_PRELOADED_MODULES):
        raise RendererStartupError("renderer discovery imported a forbidden module")
    if _mapped_renderer_native_identities(site_packages):
        raise RendererStartupError("renderer native component was mapped before import")
    replicate = 1 if role == "probe_r1" else 2 if role == "probe_r2" else None
    module_sha256 = startup_module_sha256()
    if (
        role == "formal_pdf_extraction"
        and normalized_authority.get("startup_module_sha256") != module_sha256
    ):
        raise RendererStartupError("formal renderer startup module differs from authority")
    invocation_id = hashlib.sha256(
        f"{contract_sha256}\0{job_id}\0{role}".encode("ascii")
    ).hexdigest()
    common: dict[str, object] = {
        "schema_version": 1,
        "startup_contract_id": STARTUP_CONTRACT_ID,
        "startup_contract_sha256": contract_sha256,
        "startup_module_sha256": module_sha256,
        "job_id": job_id,
        "role": role,
        "replicate": replicate,
        "invocation_id": invocation_id,
        "process_id": os.getpid(),
        "launcher": "conda_run",
        "warnings_policy": "default",
        "preload_plan": "none",
        "stderr_policy": "empty",
        "stdout_policy": "empty",
        "deliberate_preloaded_modules": [],
        "forbidden_preloaded_modules_observed": [],
        "native_components_mapped_preimport": [],
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python_version": sys.version,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_shlvl": conda_shlvl,
        "pythonwarnings_environment": None,
        "pythonhome_environment": None,
        "pythonpath_environment": None,
        "ld_preload_environment": None,
        "python_warnoptions": [],
        "core_dump_soft_limit": 0,
        "faulthandler_enabled": True,
        "authority": normalized_authority,
    }
    marker = dict(common)
    marker.update({"status": "preimport_ready", "generated_at_utc": utc_now()})
    publish_json(preimport_path, marker)
    try:
        fitz_module = importlib.import_module("fitz")
        observed_version = getattr(fitz_module, "VersionBind", None)
        if observed_version != EXPECTED_PYMUPDF_VERSION:
            raise RendererStartupError("registered PyMuPDF version differs")
        components = _component_hashes(site_packages, fitz_module)
        if role == "formal_pdf_extraction" and components != normalized_authority.get(
            "component_sha256"
        ):
            raise RendererStartupError("formal renderer components differ from audit authority")
        result = dict(common)
        result.update(
            {
                "status": "import_ok",
                "pymupdf_version": observed_version,
                "component_sha256": components,
                "generated_at_utc": utc_now(),
            }
        )
        publish_json(result_path, result)
        return fitz_module
    except Exception as exc:
        if not result_path.exists() and not result_path.is_symlink():
            failed = dict(common)
            failed.update(
                {
                    "status": "failed",
                    "failure_stage": "renderer_import_or_component_validation",
                    "error_type": type(exc).__name__,
                    "generated_at_utc": utc_now(),
                }
            )
            publish_json(result_path, failed)
        if isinstance(exc, RendererStartupError):
            raise
        raise RendererStartupError("registered PyMuPDF import failed") from exc
