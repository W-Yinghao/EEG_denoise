#!/usr/bin/env python3
"""Run one argv vector and atomically retain separately sanitized stdout/stderr."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import re
from pathlib import Path

from sanitize_lock import sanitize_text


CONDA_URL_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://\S+$")
PIP_PIN_LINE = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?(?:==|===)\S+$")
PIP_DIRECT_LINE = re.compile(
    r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])? @ [A-Za-z][A-Za-z0-9+.-]*://\S+$"
)


def validate_stdout_format(
    text: str, format_name: str
) -> tuple[list[int], int, list[str]]:
    invalid_lines: list[int] = []
    requirement_count = 0
    explicit_lines: list[int] = []
    conda_url_lines: list[int] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        if format_name == "conda-explicit":
            valid = line == "@EXPLICIT" or bool(CONDA_URL_LINE.fullmatch(line))
            if line == "@EXPLICIT":
                explicit_lines.append(line_number)
            elif CONDA_URL_LINE.fullmatch(line):
                conda_url_lines.append(line_number)
                requirement_count += 1
        elif format_name == "pip-freeze":
            valid = bool(
                PIP_PIN_LINE.fullmatch(line)
                or PIP_DIRECT_LINE.fullmatch(line)
                or (line.startswith("-e ") and len(line.split()) == 2)
            )
            if valid:
                requirement_count += 1
        else:
            valid = True
        if not valid:
            invalid_lines.append(line_number)
    structural_failures: list[str] = []
    if format_name == "conda-explicit":
        if len(explicit_lines) != 1:
            structural_failures.append("conda output must contain exactly one @EXPLICIT marker")
        if not conda_url_lines:
            structural_failures.append("conda output must contain at least one package URL")
        if explicit_lines and any(line <= explicit_lines[0] for line in conda_url_lines):
            structural_failures.append("conda package URLs must follow @EXPLICIT")
    elif format_name == "pip-freeze" and requirement_count < 1:
        structural_failures.append("pip freeze output must contain at least one requirement")
    return invalid_lines, requirement_count, structural_failures


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path, follow_symlinks=False)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout-out", type=Path, required=True)
    parser.add_argument("--stderr-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument(
        "--stdout-format", choices=("unstructured", "conda-explicit", "pip-freeze"), required=True
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command argv vector is required after --")

    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_text = completed.stdout.decode("utf-8", errors="replace")
    stderr_text = completed.stderr.decode("utf-8", errors="replace")
    sanitized_stdout, stdout_counts = sanitize_text(
        stdout_text, redact_non_url_secrets=False
    )
    sanitized_stderr, stderr_counts = sanitize_text(stderr_text)
    invalid_stdout_lines, requirement_count, structural_failures = validate_stdout_format(
        sanitized_stdout, args.stdout_format
    )

    atomic_write(args.stdout_out, sanitized_stdout)
    atomic_write(
        args.stderr_out,
        "stderr text suppressed; raw bytes and SHA-256 are recorded in the sanitization audit\n",
    )
    audit = {
        "schema_version": 1,
        "command_executable": command[0],
        "command_argument_count": len(command) - 1,
        "return_code": completed.returncode,
        "stdout_format": args.stdout_format,
        "stdout_format_valid": not invalid_stdout_lines and not structural_failures,
        "invalid_stdout_line_numbers": invalid_stdout_lines[:100],
        "stdout_requirement_count": requirement_count,
        "stdout_structure_failures": structural_failures,
        "replayability": (
            "redacted_non_replayable"
            if stdout_counts["sanitized_url_count"]
            else "verbatim_replayable"
        ),
        "raw_stdout_bytes": len(completed.stdout),
        "raw_stderr_bytes": len(completed.stderr),
        "raw_stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "raw_stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "stdout_sanitization": stdout_counts,
        "stderr_sanitization": stderr_counts,
        "policy": (
            "raw streams were hashed in memory and were never written to disk; sanitized stdout "
            "was retained, while stderr text was suppressed"
        ),
    }
    atomic_write(args.audit_out, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if completed.returncode == 0 and (invalid_stdout_lines or structural_failures):
        return 124
    if completed.returncode < 0 or completed.returncode > 125:
        return 125
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
