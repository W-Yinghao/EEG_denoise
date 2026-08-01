#!/usr/bin/env python3
"""Fail-closed, job-scoped PyMuPDF extraction and artifact promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from review_attachments import (
    CONTROL_MARKERS,
    HEX64,
    MAX_PDF_ANNOTATIONS,
    MAX_PDF_IMAGES,
    MAX_PDF_LINKS,
    MAX_PDF_METADATA_BYTES,
    MAX_PDF_OBJECT_RECORD_BYTES,
    MAX_PDF_TEXT_BYTES,
    MAX_PDF_XREF_OBJECTS,
    REGISTERED_PDF_RENDERER_PROFILE,
    artifact_records,
    atomic_json,
    directory_bundle_sha256,
    directory_bytes,
    load_json_beneath,
    relative_parts,
    require_disk_capacity,
    safe_existing_directory,
    safe_exclusive_write,
    safe_member_path,
    safe_open_source,
    scan_bytes_for_credentials,
    scan_tree_for_credentials,
    secure_create_directory,
    sha256_file,
    snapshot_attachment,
    source_identity,
    validate_artifact_record_set,
    validate_code_root,
    validate_contract_evidence,
    verify_original_source,
    verify_review_artifacts,
)


class ExtractionError(RuntimeError):
    """A non-secret fail-closed extraction error."""


fitz: Any = None


def load_fitz_renderer() -> None:
    """Load the native renderer only in the extraction process, after pure-Python validation."""
    global fitz
    if fitz is not None:
        return
    try:
        import fitz as fitz_module
    except Exception as exc:
        raise ExtractionError(f"PyMuPDF import failed: {type(exc).__name__}") from exc
    fitz = fitz_module


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def serializable(value: Any, depth: int = 0) -> Any:
    if depth > 12:
        raise ExtractionError("PDF metadata nesting exceeds budget")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple)):
        if len(value) > 100_000:
            raise ExtractionError("PDF metadata sequence exceeds budget")
        return [serializable(item, depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > 100_000:
            raise ExtractionError("PDF metadata mapping exceeds budget")
        return {str(key): serializable(item, depth + 1) for key, item in value.items()}
    return str(value)


def bounded_json(value: Any, maximum_bytes: int, label: str) -> tuple[Any, bytes]:
    normalized = serializable(value)
    encoded = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    if len(encoded) > maximum_bytes:
        raise ExtractionError(f"{label} bytes exceed budget")
    detected = scan_bytes_for_credentials(encoded)
    if detected:
        raise ExtractionError(f"high-confidence credential pattern {detected} in {label}")
    return normalized, encoded


def warning_text() -> str:
    getter = getattr(fitz.TOOLS, "mupdf_warnings", None)
    if not callable(getter):
        raise ExtractionError("PyMuPDF warning audit API is unavailable")
    return str(getter() or "").strip()


def reset_warnings() -> None:
    resetter = getattr(fitz.TOOLS, "reset_mupdf_warnings", None)
    if callable(resetter):
        resetter()
    else:
        warning_text()


def collect_annotations(page: fitz.Page) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    annotation = page.first_annot
    while annotation is not None:
        if len(records) >= MAX_PDF_ANNOTATIONS:
            raise ExtractionError("PDF annotation count exceeds budget")
        annotation_type = serializable(annotation.type)
        if "FileAttachment" in str(annotation_type).replace(" ", ""):
            raise ExtractionError("PDF file-attachment annotation is refused")
        records.append(
            {
                "type": annotation_type,
                "rect": serializable(annotation.rect),
                "info": serializable(annotation.info),
                "flags": annotation.flags,
                "colors": serializable(annotation.colors),
                "opacity": serializable(annotation.opacity),
            }
        )
        annotation = annotation.next
    bounded_json(records, MAX_PDF_OBJECT_RECORD_BYTES, "PDF annotations")
    return records


def collect_links(page: fitz.Page) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    link = page.first_link
    while link is not None:
        if len(records) >= MAX_PDF_LINKS:
            raise ExtractionError("PDF link count exceeds budget")
        linked_file = serializable(getattr(link, "file", None))
        if linked_file:
            raise ExtractionError("PDF file link is refused")
        record = {
            "rect": serializable(getattr(link, "rect", None)),
            "uri": serializable(getattr(link, "uri", None)),
            "xref": serializable(getattr(link, "xref", None)),
            "is_external": serializable(getattr(link, "is_external", None)),
            "file": linked_file,
        }
        bounded_json(record, 64 * 1024, "PDF link")
        records.append(record)
        link = link.next
    bounded_json(records, MAX_PDF_OBJECT_RECORD_BYTES, "PDF links")
    return records


def validate_parent_attachment(
    *,
    code_root: Path,
    parent_job_id: str,
    expected_manifest_sha256: str,
    input_path: Path,
    expected_source_sha256: str,
    parent_member_path: str | None,
    renderer_max_input_bytes: int,
) -> dict[str, Any]:
    if not parent_job_id.isdigit() or not HEX64.fullmatch(expected_manifest_sha256):
        raise ExtractionError("parent attachment binding is malformed")
    if renderer_max_input_bytes < 1:
        raise ExtractionError("renderer input budget must be positive")
    parent_dir = safe_existing_directory(
        code_root, code_root / "reports" / "attachment_jobs" / parent_job_id
    )
    status = load_json_beneath(code_root, parent_dir / "status.json")
    current_parent_helper_sha256 = sha256_file(
        code_root / "scripts" / "contract" / "review_attachments.py"
    )
    current_parent_job_sha256 = sha256_file(
        code_root / "scripts" / "slurm" / "jobs" / "review_attachments.sbatch"
    )
    if (
        status.get("schema_version") != 2
        or status.get("job") != "review_attachments"
        or str(status.get("job_id")) != parent_job_id
        or status.get("state")
        != "parent_attachment_phase_complete_pending_registered_pdf_renderers_and_full_read"
        or status.get("exit_code") != 0
        or status.get("review_complete") is not False
        or status.get("pdf_defer_to_registered_renderer") is not True
        or status.get("helper_sha256") != current_parent_helper_sha256
        or status.get("submitted_job_script_sha256") != current_parent_job_sha256
    ):
        raise ExtractionError("parent attachment job is not a successful extraction-only review")
    parent_helper_sha256 = status.get("helper_sha256")
    if not isinstance(parent_helper_sha256, str) or not HEX64.fullmatch(
        parent_helper_sha256
    ):
        raise ExtractionError("parent attachment helper hash is absent or malformed")
    output_root = safe_existing_directory(code_root, parent_dir / "outputs")
    extraction_root = safe_existing_directory(
        code_root,
        code_root
        / "reports"
        / "attachment_review_extract"
        / "jobs"
        / parent_job_id
        / "review",
    )
    snapshot_root = safe_existing_directory(code_root, parent_dir / "snapshots")
    parent_verification = verify_review_artifacts(
        code_root=code_root,
        output_root=output_root,
        extract_root=extraction_root,
        snapshot_root=snapshot_root,
        expected_helper_sha256=parent_helper_sha256,
        expected_job_id=parent_job_id,
        require_complete=True,
    )
    manifest_path = output_root / "attachment_manifest.json"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ExtractionError("parent attachment manifest hash mismatch")
    complete = load_json_beneath(code_root, output_root / "EXTRACTION_COMPLETE.json")
    manifest = load_json_beneath(code_root, manifest_path)
    parent_contract_path = parent_dir / "contract-validation-start.json"
    parent_contract = load_json_beneath(code_root, parent_contract_path)
    if (
        complete.get("state") != "PARENT_PHASE_COMPLETE"
        or complete.get("phase") != "parent_attachment_phase"
        or complete.get("extraction_only") is not True
        or complete.get("review_complete") is not False
        or complete.get("outstanding_renderer") is not True
        or int(complete.get("deferred_pdf_count", 0)) < 1
        or str(complete.get("slurm_job_id")) != parent_job_id
        or complete.get("attachment_manifest_sha256") != expected_manifest_sha256
        or complete.get("contract_validation_sha256")
        != sha256_file(parent_contract_path)
        or manifest.get("pdf_defer_to_registered_renderer") is not True
        or manifest.get("phase") != "parent_attachment_phase"
        or manifest.get("outstanding_renderer") is not True
        or int(manifest.get("deferred_pdf_count", 0)) < 1
    ):
        raise ExtractionError("parent attachment COMPLETE marker is invalid")
    parent_request_id = parent_contract.get("request_id")
    if (
        not isinstance(parent_request_id, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", parent_request_id)
    ):
        raise ExtractionError("parent contract lacks a safe submission request ID")
    parent_request_path = (
        code_root
        / "reports"
        / "slurm"
        / "submissions"
        / "requests"
        / f"{parent_request_id}.json"
    )
    parent_request = load_json_beneath(code_root, parent_request_path)
    current_parent_provenance = {
        "cluster_config_sha256": sha256_file(code_root / "configs/cluster/slurm.yaml"),
        "environment_config_sha256": sha256_file(code_root / "configs/environments.yaml"),
        "job_script_sha256": current_parent_job_sha256,
        "submitter_sha256": sha256_file(code_root / "scripts/slurm/submit.sh"),
        "contract_bundle_sha256": directory_bundle_sha256(
            code_root / "scripts/contract", ".py"
        ),
        "slurm_jobs_bundle_sha256": directory_bundle_sha256(
            code_root / "scripts/slurm/jobs", ".sbatch"
        ),
    }
    if (
        parent_request.get("schema_version") != 1
        or parent_request.get("request_id") != parent_request_id
        or parent_request.get("job") != "review_attachments"
        or parent_contract.get("request_sha256") != sha256_file(parent_request_path)
    ):
        raise ExtractionError("parent request/contract binding is invalid")
    for field, expected_value in current_parent_provenance.items():
        if parent_request.get(field) != expected_value:
            raise ExtractionError(f"parent attachment provenance is stale for {field}")
    input_relative = input_path.relative_to(code_root).as_posix()
    matching: list[dict[str, Any]] = []
    if parent_member_path is None:
        expected_defer_record_id = hashlib.sha256(
            b"registered-pdf-defer-v1\0" + expected_source_sha256.encode("ascii")
        ).hexdigest()
        for record in manifest.get("attachments", []):
            if not isinstance(record, dict):
                continue
            source = record.get("source_snapshot")
            if (
                isinstance(source, dict)
                and source.get("source_relative_path") == input_relative
                and source.get("source_sha256") == expected_source_sha256
                and source.get("snapshot_sha256") == expected_source_sha256
                and record.get("sha256") == expected_source_sha256
                and record.get("media_type") == "application/pdf"
                and record.get("pdf_header_validated") is True
                and record.get("pdf_file_probe_verified") is True
                and record.get("read_status")
                == "deferred_to_registered_pdf_renderer"
                and record.get("defer_record_id") == expected_defer_record_id
                and isinstance(record.get("extraction"), dict)
                and record["extraction"].get("kind") == "pdf"
                and record["extraction"].get("status")
                == "deferred_to_registered_pdf_renderer"
                and record["extraction"].get("renderer_environment") == "icml"
                and record["extraction"].get("renderer_profile")
                == REGISTERED_PDF_RENDERER_PROFILE
                and record["extraction"].get("renderer_job") == "extract_pdf"
                and record["extraction"].get("renderer_max_input_bytes")
                == renderer_max_input_bytes
                and record["extraction"].get("rendered") is False
                and record["extraction"].get("defer_record_id")
                == expected_defer_record_id
            ):
                matching.append(record)
        source_kind = "top_level_attachment"
    else:
        safe_member, unsafe_reason = safe_member_path(parent_member_path)
        if (
            unsafe_reason is not None
            or safe_member is None
            or safe_member.as_posix() != parent_member_path
        ):
            raise ExtractionError("parent ZIP member path is unsafe or non-canonical")
        expected_defer_record_id = ""
        for record in manifest.get("attachments", []):
            if not isinstance(record, dict):
                continue
            archive_sha256 = record.get("sha256")
            extraction = record.get("extraction")
            archive_snapshot = record.get("source_snapshot")
            if (
                not isinstance(archive_sha256, str)
                or not HEX64.fullmatch(archive_sha256)
                or not isinstance(extraction, dict)
                or not isinstance(archive_snapshot, dict)
                or extraction.get("kind") != "zip"
                or extraction.get("status") != "safely_extracted"
                or record.get("read_status") != "safely_extracted"
            ):
                continue
            destination = extraction.get("destination")
            embedded = extraction.get("pdf_files")
            if not isinstance(destination, str) or not isinstance(embedded, list):
                continue
            destination_path = code_root / destination
            relative_parts(code_root, destination_path)
            snapshot_relative = archive_snapshot.get("snapshot_relative_path")
            if not isinstance(snapshot_relative, str):
                continue
            safe_stem = re.sub(
                r"[^A-Za-z0-9._-]+", "_", Path(snapshot_relative).stem
            )[:80]
            expected_destination = extraction_root / f"{safe_stem}-{archive_sha256}"
            if destination_path != expected_destination:
                continue
            safe_existing_directory(code_root, destination_path)
            for member in embedded:
                if not isinstance(member, dict):
                    continue
                member_sha256 = member.get("sha256")
                if not isinstance(member_sha256, str) or not HEX64.fullmatch(
                    member_sha256
                ):
                    continue
                candidate_defer_id = hashlib.sha256(
                    b"archive-pdf-defer-v1\0"
                    + archive_sha256.encode("ascii")
                    + b"\0"
                    + parent_member_path.encode("utf-8")
                    + b"\0"
                    + member_sha256.encode("ascii")
                ).hexdigest()
                candidate_path = destination_path / "payload" / safe_member
                if (
                    member.get("path") == parent_member_path
                    and member.get("extracted_relative_path")
                    == f"payload/{parent_member_path}"
                    and member_sha256 == expected_source_sha256
                    and candidate_path.relative_to(code_root).as_posix() == input_relative
                    and member.get("read_status")
                    == "deferred_to_registered_pdf_renderer"
                    and member.get("defer_record_id") == candidate_defer_id
                    and member.get("media_type") == "application/pdf"
                    and member.get("pdf_header_validated") is True
                    and member.get("pdf_file_probe_verified") is True
                    and member.get("renderer_environment") == "icml"
                    and member.get("renderer_profile")
                    == REGISTERED_PDF_RENDERER_PROFILE
                    and member.get("renderer_job") == "extract_pdf"
                    and member.get("renderer_max_input_bytes")
                    == renderer_max_input_bytes
                    and isinstance(member.get("bytes"), int)
                    and 0 < int(member["bytes"]) <= renderer_max_input_bytes
                    and member.get("rendered") is False
                    and sha256_file(candidate_path) == expected_source_sha256
                ):
                    expected_defer_record_id = candidate_defer_id
                    matching.append(member)
        source_kind = "zip_member"
    if len(matching) != 1:
        raise ExtractionError("PDF input is not uniquely bound to the parent attachment manifest")
    return {
        "parent_job_id": parent_job_id,
        "parent_manifest_sha256": expected_manifest_sha256,
        "parent_complete_sha256": sha256_file(output_root / "EXTRACTION_COMPLETE.json"),
        "parent_status_sha256": sha256_file(parent_dir / "status.json"),
        "parent_artifact_manifest_sha256": parent_verification[
            "artifact_manifest_sha256"
        ],
        "defer_record_id": expected_defer_record_id,
        "source_kind": source_kind,
        "parent_member_path": parent_member_path,
    }


def extract_pdf(args: argparse.Namespace) -> None:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise ExtractionError("numeric SLURM_JOB_ID is required")
    code_root = validate_code_root(args.code_root)
    helper = Path(__file__).resolve(strict=True)
    observed_helper_sha256 = sha256_file(helper)
    if observed_helper_sha256 != args.expected_helper_sha256:
        raise ExtractionError("PDF helper hash differs from the job-registered hash")
    if not all(
        value > 0
        for value in (
            args.dpi,
            args.max_input_bytes,
            args.max_pages,
            args.max_page_pixels,
            args.max_total_pixels,
            args.max_output_bytes,
        )
    ):
        raise ExtractionError("all PDF budgets must be positive")
    contract = load_json_beneath(code_root, args.contract_validation)
    if (
        contract.get("provenance_complete") is not True
        or str(contract.get("job_id")) != job_id
        or contract.get("job") != "extract_pdf"
        or contract.get("profile") != REGISTERED_PDF_RENDERER_PROFILE
        or contract.get("environment_name") != "icml"
    ):
        raise ExtractionError("PDF contract validation is absent or incomplete")
    contract_sha256 = sha256_file(args.contract_validation)
    expected_run_dir = code_root / "reports" / "attachment_jobs" / job_id
    if Path(os.path.abspath(args.contract_validation.parent)) != expected_run_dir:
        raise ExtractionError("PDF contract validation is outside the registered job directory")

    input_descriptor_path = args.input if args.input.is_absolute() else code_root / args.input
    lexical_input = Path(os.path.abspath(input_descriptor_path))
    relative_parts(code_root, lexical_input)
    try:
        parent_binding = validate_parent_attachment(
            code_root=code_root,
            parent_job_id=args.parent_attachment_job,
            expected_manifest_sha256=args.parent_manifest_sha256,
            input_path=lexical_input,
            expected_source_sha256=args.expected_sha256,
            parent_member_path=args.parent_member_path,
            renderer_max_input_bytes=args.max_input_bytes,
        )
    except (ExtractionError, OSError, ValueError, RuntimeError) as exc:
        detail = str(exc)
        encoded_detail = detail.encode("utf-8", errors="replace")
        if (
            len(encoded_detail) > 4096
            or scan_bytes_for_credentials(encoded_detail) is not None
            or any(character < " " and character not in "\t\n\r" for character in detail)
        ):
            detail = "suppressed_by_diagnostic_safety_policy"
        trace_frames: list[dict[str, Any]] = []
        for frame in traceback.extract_tb(exc.__traceback__):
            frame_path = Path(frame.filename)
            try:
                frame_relative = frame_path.relative_to(code_root).as_posix()
            except ValueError:
                continue
            trace_frames.append(
                {
                    "file": frame_relative,
                    "function": frame.name,
                    "line": frame.lineno,
                }
            )
        try:
            atomic_json(
                expected_run_dir / "parent-validation-failure.json",
                {
                    "schema_version": 1,
                    "job_id": job_id,
                    "stage": "parent_attachment_validation",
                    "error_type": type(exc).__name__,
                    "error_detail": detail,
                    "traceback": trace_frames[-16:],
                    "generated_at_utc": utc_now(),
                },
            )
        except OSError:
            pass
        raise
    expected_snapshot_root = (
        code_root / "reports" / "attachment_jobs" / job_id / "pdf-snapshot"
    )
    if Path(os.path.abspath(args.snapshot_root)) != expected_snapshot_root:
        raise ExtractionError("PDF snapshot root is not the registered job-private path")
    snapshot_root = secure_create_directory(code_root, expected_snapshot_root, exclusive_last=True)
    snapshot = snapshot_attachment(
        lexical_input, code_root, snapshot_root, 1, args.max_input_bytes
    )
    if snapshot["source_sha256"] != args.expected_sha256:
        raise ExtractionError("PDF source hash differs from the submitted hash")
    source = code_root / str(snapshot["snapshot_relative_path"])
    source_descriptor, _ = safe_open_source(code_root, source)
    try:
        if b"%PDF-" not in os.read(source_descriptor, 1024):
            raise ExtractionError("input lacks a PDF header in its first 1024 bytes")
    finally:
        os.close(source_descriptor)

    output_root = code_root / "reports" / "attachment_review_extract" / "jobs" / job_id
    if Path(os.path.abspath(args.output_dir)) != output_root:
        raise ExtractionError("PDF output root differs from the job-scoped path")
    output_root = secure_create_directory(code_root, output_root, exclusive_last=True)
    work_root = secure_create_directory(code_root, output_root / "pymupdf", exclusive_last=True)
    pages_dir = secure_create_directory(code_root, work_root / "pages", exclusive_last=True)
    text_dir = secure_create_directory(code_root, work_root / "text", exclusive_last=True)
    require_disk_capacity(code_root, args.max_output_bytes + int(snapshot["snapshot_bytes"]))
    load_fitz_renderer()

    document: fitz.Document | None = None
    try:
        reset_warnings()
        document = fitz.open(source)
        if document.needs_pass or document.is_encrypted:
            raise ExtractionError("encrypted PDF is refused")
        if bool(getattr(document, "is_repaired", False)):
            raise ExtractionError("repaired or structurally recovered PDF is refused")
        if document.page_count < 1 or document.page_count > args.max_pages:
            raise ExtractionError("PDF page count is outside the registered budget")
        xref_length = int(document.xref_length())
        if xref_length < 1 or xref_length > MAX_PDF_XREF_OBJECTS:
            raise ExtractionError("PDF xref object count is outside the registered budget")
        if list(document.embfile_names()):
            raise ExtractionError("PDF embedded files are refused")
        catalog_xref = int(document.pdf_catalog())
        associated_type, associated_value = document.xref_get_key(catalog_xref, "AF")
        if associated_type not in {"null", "none"} or associated_value not in {
            "null",
            "",
        }:
            raise ExtractionError("PDF associated files are refused")

        metadata, _ = bounded_json(
            document.metadata, MAX_PDF_METADATA_BYTES, "PDF metadata"
        )
        toc_raw = document.get_toc(simple=False)
        if len(toc_raw) > 20_000:
            raise ExtractionError("PDF table-of-contents entry count exceeds budget")
        toc, _ = bounded_json(toc_raw, MAX_PDF_OBJECT_RECORD_BYTES, "PDF table of contents")
        scale = args.dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        estimated_total_pixels = 0
        dimensions: list[tuple[int, int, int]] = []
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            width = max(1, math.ceil(float(page.rect.width) * scale))
            height = max(1, math.ceil(float(page.rect.height) * scale))
            pixels = width * height
            if pixels > args.max_page_pixels:
                raise ExtractionError("estimated PDF page pixels exceed budget")
            estimated_total_pixels += pixels
            if estimated_total_pixels > args.max_total_pixels:
                raise ExtractionError("estimated total PDF pixels exceed budget")
            dimensions.append((width, height, pixels))

        document_text_parts: list[str] = []
        document_text_bytes = 0
        actual_total_pixels = 0
        total_annotation_count = 0
        total_link_count = 0
        total_image_count = 0
        page_records: list[dict[str, Any]] = []
        for page_index in range(document.page_count):
            page_number = page_index + 1
            page = document.load_page(page_index)
            text = page.get_text("text", sort=True)
            encoded_text = text.encode("utf-8")
            detected = scan_bytes_for_credentials(encoded_text)
            if detected:
                raise ExtractionError(
                    f"high-confidence credential pattern {detected} in PDF text"
                )
            section = (
                f"\n===== BEGIN PAGE {page_number} =====\n".encode("utf-8")
                + encoded_text
                + f"\n===== END PAGE {page_number} =====\n".encode("utf-8")
            )
            document_text_bytes += len(section)
            if document_text_bytes > MAX_PDF_TEXT_BYTES:
                raise ExtractionError("PDF extracted text bytes exceed budget")
            text_name = f"page-{page_number:04d}.txt"
            safe_exclusive_write(text_dir / text_name, encoded_text)
            document_text_parts.append(section.decode("utf-8"))

            annotations = collect_annotations(page)
            links = collect_links(page)
            images = page.get_images(full=True)
            if len(images) > MAX_PDF_IMAGES:
                raise ExtractionError("PDF per-page image count exceeds budget")
            total_annotation_count += len(annotations)
            total_link_count += len(links)
            total_image_count += len(images)
            if total_annotation_count > MAX_PDF_ANNOTATIONS:
                raise ExtractionError("PDF total annotation count exceeds budget")
            if total_link_count > MAX_PDF_LINKS:
                raise ExtractionError("PDF total link count exceeds budget")
            if total_image_count > MAX_PDF_IMAGES:
                raise ExtractionError("PDF total image count exceeds budget")

            pixmap = page.get_pixmap(matrix=matrix, alpha=False, annots=True)
            actual_pixels = int(pixmap.width) * int(pixmap.height)
            if actual_pixels > args.max_page_pixels:
                raise ExtractionError("actual PDF page pixels exceed budget")
            actual_total_pixels += actual_pixels
            if actual_total_pixels > args.max_total_pixels:
                raise ExtractionError("actual total PDF pixels exceed budget")
            png = pixmap.tobytes("png")
            if len(png) > args.max_output_bytes:
                raise ExtractionError("single rendered page exceeds output byte budget")
            image_name = f"page-{page_number:04d}.png"
            safe_exclusive_write(pages_dir / image_name, png)
            page_record = {
                "page": page_number,
                "rect": serializable(page.rect),
                "rotation": page.rotation,
                "estimated_width": dimensions[page_index][0],
                "estimated_height": dimensions[page_index][1],
                "estimated_pixels": dimensions[page_index][2],
                "actual_width": int(pixmap.width),
                "actual_height": int(pixmap.height),
                "actual_pixels": actual_pixels,
                "text_bytes": len(encoded_text),
                "text_file": f"text/{text_name}",
                "render_file": f"pages/{image_name}",
                "annotations": annotations,
                "links": links,
                "image_count": len(images),
            }
            bounded_json(page_record, MAX_PDF_OBJECT_RECORD_BYTES, "PDF page record")
            page_records.append(page_record)
            emitted_warning = warning_text()
            if emitted_warning:
                raise ExtractionError("PyMuPDF emitted a warning during page extraction")

        safe_exclusive_write(
            work_root / "document.txt", "".join(document_text_parts).encode("utf-8")
        )
        report = {
            "schema_version": 2,
            "source_sha256": snapshot["source_sha256"],
            "source_snapshot": snapshot,
            "helper_sha256": observed_helper_sha256,
            "contract_validation_sha256": contract_sha256,
            "parent_attachment": parent_binding,
            "generated_at_utc": utc_now(),
            "slurm_job_id": job_id,
            "renderer_profile": REGISTERED_PDF_RENDERER_PROFILE,
            "pymupdf_version": fitz.VersionBind,
            "encrypted": False,
            "repaired": False,
            "embedded_files": False,
            "page_count": document.page_count,
            "xref_object_count": xref_length,
            "estimated_total_pixels": estimated_total_pixels,
            "actual_total_pixels": actual_total_pixels,
            "text_bytes": document_text_bytes,
            "annotation_count": total_annotation_count,
            "link_count": total_link_count,
            "image_count": total_image_count,
            "metadata": metadata,
            "toc": toc,
            "pages": page_records,
            "budgets": {
                "max_input_bytes": args.max_input_bytes,
                "max_pages": args.max_pages,
                "max_page_pixels": args.max_page_pixels,
                "max_total_pixels": args.max_total_pixels,
                "max_output_bytes": args.max_output_bytes,
                "max_text_bytes": MAX_PDF_TEXT_BYTES,
                "max_annotations": MAX_PDF_ANNOTATIONS,
                "max_links": MAX_PDF_LINKS,
                "max_images": MAX_PDF_IMAGES,
                "max_xref_objects": MAX_PDF_XREF_OBJECTS,
            },
            "credential_findings": 0,
            "review_complete": False,
            "review_blocker": "semantic and visual inspection of every page is pending",
        }
        atomic_json(work_root / "pymupdf_report.json", report)
        verify_original_source(code_root, snapshot)
        scan_tree_for_credentials(work_root)
        scan_tree_for_credentials(snapshot_root)
        records = artifact_records(
            work_root,
            {"artifacts_manifest.json", "READY.json", "EXTRACTION_COMPLETE.json"},
        )
        artifact_manifest = {
            "schema_version": 2,
            "source_sha256": snapshot["source_sha256"],
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "slurm_job_id": job_id,
            "renderer_profile": REGISTERED_PDF_RENDERER_PROFILE,
            "helper_sha256": observed_helper_sha256,
            "contract_validation_sha256": contract_sha256,
            "parent_manifest_sha256": args.parent_manifest_sha256,
            "parent_defer_record_id": parent_binding["defer_record_id"],
            "source_kind": parent_binding["source_kind"],
            "parent_member_path": parent_binding["parent_member_path"],
            "generated_at_utc": utc_now(),
            "artifact_count": len(records),
            "artifact_bytes": sum(int(record["bytes"]) for record in records),
            "artifacts": records,
            "credential_findings": 0,
        }
        manifest_path = work_root / "artifacts_manifest.json"
        atomic_json(manifest_path, artifact_manifest)
        ready = {
            "schema_version": 1,
            "state": "READY",
            "extraction_only": True,
            "review_complete": False,
            "source_sha256": snapshot["source_sha256"],
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "helper_sha256": observed_helper_sha256,
            "renderer_profile": REGISTERED_PDF_RENDERER_PROFILE,
            "contract_validation_sha256": contract_sha256,
            "parent_manifest_sha256": args.parent_manifest_sha256,
            "parent_defer_record_id": parent_binding["defer_record_id"],
            "source_kind": parent_binding["source_kind"],
            "parent_member_path": parent_binding["parent_member_path"],
            "artifact_manifest_sha256": sha256_file(manifest_path),
            "slurm_job_id": job_id,
            "generated_at_utc": utc_now(),
            "credential_findings": 0,
        }
        atomic_json(work_root / "READY.json", ready)
        if directory_bytes(output_root) > args.max_output_bytes:
            raise ExtractionError("PDF output bytes exceed budget")
    except Exception as exc:
        try:
            if not (work_root / "EXTRACTION_COMPLETE.json").exists():
                atomic_json(
                    work_root / "EXTRACTION_FAILED.json",
                    {
                        "schema_version": 1,
                        "state": "FAILED",
                        "slurm_job_id": job_id,
                        "error_type": type(exc).__name__,
                        "generated_at_utc": utc_now(),
                        "review_complete": False,
                    },
                )
        except OSError:
            pass
        if isinstance(exc, ExtractionError):
            raise
        raise ExtractionError(f"PDF extraction failed closed: {type(exc).__name__}") from exc
    finally:
        if document is not None:
            document.close()


def verify_pdf_artifacts(args: argparse.Namespace, require_complete: bool) -> dict[str, Any]:
    code_root = validate_code_root(args.code_root)
    work_root = safe_existing_directory(code_root, args.output_root / "pymupdf")
    manifest_path = work_root / "artifacts_manifest.json"
    ready_path = work_root / "READY.json"
    contract_validation_sha256 = sha256_file(args.contract_validation)
    manifest = load_json_beneath(code_root, manifest_path)
    ready = load_json_beneath(code_root, ready_path)
    controls = {"artifacts_manifest.json", "READY.json"}
    if require_complete:
        controls.add("EXTRACTION_COMPLETE.json")
    validate_artifact_record_set(
        work_root,
        manifest.get("artifacts"),
        controls,
        require_root_controls=True,
    )
    artifact_entries = manifest.get("artifacts")
    if not isinstance(artifact_entries, list):
        raise ExtractionError("PDF artifact manifest entries are malformed")
    observed_artifact_bytes = sum(
        int(record.get("bytes", -1))
        for record in artifact_entries
        if isinstance(record, dict)
    )
    if (
        manifest.get("schema_version") != 2
        or str(manifest.get("slurm_job_id")) != args.job_id
        or manifest.get("renderer_profile") != REGISTERED_PDF_RENDERER_PROFILE
        or manifest.get("helper_sha256") != args.expected_helper_sha256
        or manifest.get("artifact_count") != len(artifact_entries)
        or manifest.get("artifact_bytes") != observed_artifact_bytes
        or manifest.get("credential_findings") != 0
        or ready.get("schema_version") != 1
        or ready.get("state") != "READY"
        or ready.get("extraction_only") is not True
        or ready.get("review_complete") is not False
        or str(ready.get("slurm_job_id")) != args.job_id
        or ready.get("renderer_profile") != REGISTERED_PDF_RENDERER_PROFILE
        or ready.get("helper_sha256") != args.expected_helper_sha256
        or ready.get("artifact_manifest_sha256") != sha256_file(manifest_path)
        or ready.get("credential_findings") != 0
    ):
        raise ExtractionError("PDF artifact provenance or READY marker is mismatched")
    report = load_json_beneath(code_root, work_root / "pymupdf_report.json")
    snapshot = report.get("source_snapshot")
    if not isinstance(snapshot, dict):
        raise ExtractionError("PDF report lacks source snapshot provenance")
    verify_original_source(code_root, snapshot)
    parent = report.get("parent_attachment")
    if not isinstance(parent, dict):
        raise ExtractionError("PDF report lacks parent attachment provenance")
    source_relative_path = snapshot.get("source_relative_path")
    if not isinstance(source_relative_path, str):
        raise ExtractionError("PDF source record lacks its relative path")
    report_budgets = report.get("budgets")
    if not isinstance(report_budgets, dict) or not isinstance(
        report_budgets.get("max_input_bytes"), int
    ):
        raise ExtractionError("PDF report lacks its renderer input budget")
    observed_parent = validate_parent_attachment(
        code_root=code_root,
        parent_job_id=str(parent.get("parent_job_id", "")),
        expected_manifest_sha256=str(parent.get("parent_manifest_sha256", "")),
        input_path=code_root / source_relative_path,
        expected_source_sha256=str(snapshot.get("source_sha256", "")),
        parent_member_path=(
            str(parent["parent_member_path"])
            if parent.get("parent_member_path") is not None
            else None
        ),
        renderer_max_input_bytes=int(report_budgets["max_input_bytes"]),
    )
    if observed_parent != parent:
        raise ExtractionError("parent attachment evidence changed")
    if (
        report.get("schema_version") != 2
        or report.get("source_sha256") != snapshot.get("source_sha256")
        or report.get("renderer_profile") != REGISTERED_PDF_RENDERER_PROFILE
        or report.get("helper_sha256") != args.expected_helper_sha256
        or report.get("contract_validation_sha256")
        != contract_validation_sha256
        or report.get("parent_attachment") != observed_parent
        or str(report.get("slurm_job_id")) != args.job_id
        or manifest.get("source_sha256") != snapshot.get("source_sha256")
        or manifest.get("snapshot_sha256") != snapshot.get("snapshot_sha256")
        or manifest.get("contract_validation_sha256")
        != contract_validation_sha256
        or manifest.get("parent_manifest_sha256")
        != observed_parent.get("parent_manifest_sha256")
        or manifest.get("parent_defer_record_id")
        != observed_parent.get("defer_record_id")
        or manifest.get("source_kind") != observed_parent.get("source_kind")
        or manifest.get("parent_member_path")
        != observed_parent.get("parent_member_path")
        or ready.get("source_sha256") != snapshot.get("source_sha256")
        or ready.get("snapshot_sha256") != snapshot.get("snapshot_sha256")
        or ready.get("contract_validation_sha256")
        != contract_validation_sha256
        or ready.get("parent_manifest_sha256")
        != observed_parent.get("parent_manifest_sha256")
        or ready.get("parent_defer_record_id")
        != observed_parent.get("defer_record_id")
        or ready.get("source_kind") != observed_parent.get("source_kind")
        or ready.get("parent_member_path")
        != observed_parent.get("parent_member_path")
    ):
        raise ExtractionError("PDF report/manifest/READY hash chain is mismatched")
    snapshot_path = code_root / str(snapshot.get("snapshot_relative_path", ""))
    if (
        snapshot_path.is_symlink()
        or not snapshot_path.is_file()
        or sha256_file(snapshot_path) != snapshot.get("snapshot_sha256")
    ):
        raise ExtractionError("PDF job-private snapshot changed")
    scan_tree_for_credentials(work_root)
    complete_sha256 = None
    if require_complete:
        complete_path = work_root / "EXTRACTION_COMPLETE.json"
        complete = load_json_beneath(code_root, complete_path)
        if (
            complete.get("schema_version") != 1
            or complete.get("state") != "COMPLETE"
            or complete.get("extraction_only") is not True
            or complete.get("review_complete") is not False
            or str(complete.get("slurm_job_id")) != args.job_id
            or complete.get("renderer_profile")
            != REGISTERED_PDF_RENDERER_PROFILE
            or complete.get("helper_sha256") != args.expected_helper_sha256
            or complete.get("artifact_manifest_sha256") != sha256_file(manifest_path)
            or complete.get("ready_sha256") != sha256_file(ready_path)
            or complete.get("contract_validation_sha256")
            != contract_validation_sha256
            or complete.get("source_sha256") != snapshot.get("source_sha256")
            or complete.get("snapshot_sha256") != snapshot.get("snapshot_sha256")
            or complete.get("parent_manifest_sha256")
            != observed_parent.get("parent_manifest_sha256")
            or complete.get("parent_defer_record_id")
            != observed_parent.get("defer_record_id")
            or complete.get("source_kind") != observed_parent.get("source_kind")
            or complete.get("parent_member_path")
            != observed_parent.get("parent_member_path")
            or complete.get("credential_findings") != 0
        ):
            raise ExtractionError("PDF COMPLETE marker is mismatched")
        complete_sha256 = sha256_file(complete_path)
    return {
        "schema_version": 1,
        "verified_at_utc": utc_now(),
        "slurm_job_id": args.job_id,
        "renderer_profile": REGISTERED_PDF_RENDERER_PROFILE,
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "ready_sha256": sha256_file(ready_path),
        "complete_sha256": complete_sha256,
        "source_sha256": snapshot.get("source_sha256"),
        "snapshot_sha256": snapshot.get("snapshot_sha256"),
        "parent_manifest_sha256": observed_parent.get("parent_manifest_sha256"),
        "parent_defer_record_id": observed_parent.get("defer_record_id"),
        "source_kind": observed_parent.get("source_kind"),
        "parent_member_path": observed_parent.get("parent_member_path"),
        "credential_findings": 0,
        "review_complete": False,
    }


def finalize_pdf(args: argparse.Namespace) -> dict[str, Any]:
    verification = verify_pdf_artifacts(args, False)
    work_root = args.output_root / "pymupdf"
    atomic_json(
        work_root / "EXTRACTION_COMPLETE.json",
        {
            "schema_version": 1,
            "state": "COMPLETE",
            "extraction_only": True,
            "review_complete": False,
            "review_blocker": "primary-agent full semantic and visual read is pending",
            "slurm_job_id": args.job_id,
            "renderer_profile": REGISTERED_PDF_RENDERER_PROFILE,
            "helper_sha256": args.expected_helper_sha256,
            "artifact_manifest_sha256": verification["artifact_manifest_sha256"],
            "ready_sha256": verification["ready_sha256"],
            "contract_validation_sha256": sha256_file(args.contract_validation),
            "source_sha256": verification["source_sha256"],
            "snapshot_sha256": verification["snapshot_sha256"],
            "parent_manifest_sha256": verification["parent_manifest_sha256"],
            "parent_defer_record_id": verification["parent_defer_record_id"],
            "source_kind": verification["source_kind"],
            "parent_member_path": verification["parent_member_path"],
            "credential_findings": 0,
            "generated_at_utc": utc_now(),
        },
    )
    return verify_pdf_artifacts(args, True)


def add_verification_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-helper-sha256", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--contract-validation", type=Path, required=True)
    parser.add_argument("--output-record", type=Path)


def dispatch() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("PDF helper action is required")
    action = sys.argv[1]
    argv = sys.argv[2:]
    if action == "bootstrap":
        parser = argparse.ArgumentParser()
        parser.add_argument("--code-root", type=Path, required=True)
        parser.add_argument("--directory", type=Path, required=True)
        args = parser.parse_args(argv)
        secure_create_directory(
            validate_code_root(args.code_root), args.directory, exclusive_last=True
        )
        return 0
    if action == "contract":
        parser = argparse.ArgumentParser()
        parser.add_argument("--code-root", type=Path, required=True)
        parser.add_argument("--job", choices=("extract_pdf",), required=True)
        parser.add_argument(
            "--profile", choices=(REGISTERED_PDF_RENDERER_PROFILE,), required=True
        )
        parser.add_argument("--payload-sha256", required=True)
        parser.add_argument("--payload-count", type=int, required=True)
        parser.add_argument("--current-job-id", required=True)
        parser.add_argument("--runtime-audit-job", required=True)
        parser.add_argument("--dependency-job", required=True)
        parser.add_argument("--environment-name", choices=("icml",), required=True)
        parser.add_argument("--environment-path", type=Path, required=True)
        parser.add_argument("--allocation-json", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(argv)
        atomic_json(args.output, validate_contract_evidence(args))
        return 0
    if action == "extract":
        parser = argparse.ArgumentParser()
        parser.add_argument("--code-root", type=Path, required=True)
        parser.add_argument("--input", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--snapshot-root", type=Path, required=True)
        parser.add_argument("--contract-validation", type=Path, required=True)
        parser.add_argument("--expected-sha256", required=True)
        parser.add_argument("--expected-helper-sha256", required=True)
        parser.add_argument("--parent-attachment-job", required=True)
        parser.add_argument("--parent-manifest-sha256", required=True)
        parser.add_argument("--parent-member-path")
        parser.add_argument("--dpi", type=int, required=True)
        parser.add_argument("--max-input-bytes", type=int, required=True)
        parser.add_argument("--max-pages", type=int, required=True)
        parser.add_argument("--max-page-pixels", type=int, required=True)
        parser.add_argument("--max-total-pixels", type=int, required=True)
        parser.add_argument("--max-output-bytes", type=int, required=True)
        args = parser.parse_args(argv)
        for value in (
            args.expected_sha256,
            args.expected_helper_sha256,
            args.parent_manifest_sha256,
        ):
            if not HEX64.fullmatch(value):
                raise ExtractionError("submitted PDF hashes must be lowercase SHA-256")
        extract_pdf(args)
        return 0
    if action in {"verify", "finalize"}:
        parser = argparse.ArgumentParser()
        add_verification_args(parser)
        parser.add_argument("--require-complete", action="store_true")
        args = parser.parse_args(argv)
        payload = finalize_pdf(args) if action == "finalize" else verify_pdf_artifacts(
            args, args.require_complete
        )
        if args.output_record is not None:
            atomic_json(args.output_record, payload)
        return 0
    raise SystemExit(f"unsupported PDF helper action: {action}")


if __name__ == "__main__":
    try:
        raise SystemExit(dispatch())
    except (ExtractionError, OSError, ValueError, RuntimeError) as exc:
        print(f"PDF helper failed closed: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(3) from exc
