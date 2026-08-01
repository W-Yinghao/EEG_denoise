#!/usr/bin/env python3
"""Remove URL credentials/query strings while hashing the exact input stream."""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit


URL_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z0-9+.-]*://[^\s]+",
    flags=re.IGNORECASE,
)
SENSITIVE_PATH_PATTERN = re.compile(
    r"(?i)(/(?:t|token|auth|signed|credential)/)[^/?#]+"
)
SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b([A-Za-z0-9_]*(?:TOKEN|PASSWORD|PASSWD|SECRET|API[_-]?KEY|"
    r"ACCESS[_-]?KEY|PRIVATE[_-]?KEY|COOKIE))([ \t]*[=:][ \t]*)([^\s,;]+)"
)
AUTHORIZATION_LINE_PATTERN = re.compile(
    r"(?im)\b(authorization|proxy-authorization|cookie|set-cookie)([ \t]*:[ \t]*)[^\r\n]+"
)
BEARER_PATTERN = re.compile(r"(?i)\bbearer[ \t]+[^\s,;]+")
BASIC_AUTH_PATTERN = re.compile(r"(?i)\bbasic[ \t]+[A-Za-z0-9+/=_-]+")


def sanitize_url(raw_url: str) -> tuple[str, bool]:
    trailing = ""
    while raw_url and raw_url[-1] in ",;)]}":
        trailing = raw_url[-1] + trailing
        raw_url = raw_url[:-1]
    parsed = urlsplit(raw_url)
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is not None:
        host = f"{host}:{port}"
    safe_fragment = ""
    if re.fullmatch(r"[0-9a-fA-F]{32,128}", parsed.fragment):
        safe_fragment = parsed.fragment
    elif re.fullmatch(r"(?:md5|sha256|sha512)=[0-9a-fA-F]{32,128}", parsed.fragment):
        safe_fragment = parsed.fragment
    safe_path = SENSITIVE_PATH_PATTERN.sub(r"\1<redacted>", parsed.path)
    sanitized = urlunsplit((parsed.scheme, host, safe_path, "", safe_fragment))
    changed = sanitized != raw_url
    return sanitized + trailing, changed


def sanitize_text(
    text: str, *, redact_non_url_secrets: bool = True
) -> tuple[str, dict[str, int]]:
    url_count = 0
    changed_url_count = 0

    def replace_url(match: re.Match[str]) -> str:
        nonlocal url_count, changed_url_count
        url_count += 1
        value, changed = sanitize_url(match.group(0))
        changed_url_count += int(changed)
        return value

    sanitized = URL_PATTERN.sub(replace_url, text)
    authorization_count = 0
    assignment_count = 0
    bearer_count = 0
    basic_auth_count = 0
    if redact_non_url_secrets:
        sanitized, authorization_count = AUTHORIZATION_LINE_PATTERN.subn(
            lambda match: f"{match.group(1)}{match.group(2)}<redacted>", sanitized
        )
        sanitized, assignment_count = SENSITIVE_ASSIGNMENT_PATTERN.subn(
            lambda match: f"{match.group(1)}{match.group(2)}<redacted>", sanitized
        )
        sanitized, bearer_count = BEARER_PATTERN.subn("Bearer <redacted>", sanitized)
        sanitized, basic_auth_count = BASIC_AUTH_PATTERN.subn("Basic <redacted>", sanitized)
    return sanitized, {
        "url_count": url_count,
        "sanitized_url_count": changed_url_count,
        "sensitive_assignment_count": assignment_count,
        "authorization_line_count": authorization_count,
        "bearer_count": bearer_count,
        "basic_auth_count": basic_auth_count,
    }
