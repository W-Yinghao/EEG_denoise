"""Consolidated-branch test policy.

The experimental lineage (69 branches, see BRANCHES.md) shipped provenance tests
that are bound to the worktree they were written in: absolute paths such as
/home/infres/yinwang/denoiseNet_<branch>, git-ancestry assertions against that
branch's HEAD, or ledger files at their pre-consolidation locations. Those tests
are records, not regressions; they are kept verbatim and SKIPPED here unless
DENOISENET_LINEAGE_TESTS=1 is set (run them from the original branch worktree).
The module list lives in tests/LINEAGE_PROVENANCE_TESTS.txt.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

_LIST = Path(__file__).with_name("LINEAGE_PROVENANCE_TESTS.txt")
_SKIP = {line.strip() for line in _LIST.read_text().splitlines()
         if line.strip() and not line.startswith("#")} if _LIST.is_file() else set()


def pytest_collection_modifyitems(config, items):
    if os.environ.get("DENOISENET_LINEAGE_TESTS") == "1" or not _SKIP:
        return
    root = Path(str(config.rootpath))
    marker = pytest.mark.skip(reason="lineage provenance test bound to its original "
                                     "branch worktree; set DENOISENET_LINEAGE_TESTS=1 to run")
    for item in items:
        rel = str(Path(str(item.fspath)).resolve().relative_to(root))
        if rel in _SKIP:
            item.add_marker(marker)
