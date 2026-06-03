#!/usr/bin/env python
"""Wire the locally pre-staged BCI-IV-2a into MOABB's cache so nothing is re-downloaded.

The full BCI-IV-2a is pre-staged on this server at ``/projects/EEG-foundation-model/BCI-IV``
as ``A0{1..9}{T,E}.mat`` (byte-identical to MOABB's ``BNCI2014_001`` source files). MOABB
expects them under ``<MNE_DATA>/MNE-bnci-data/~bci/database/001-2014/``. This script symlinks
them there (idempotent), so ``moabb`` loads them directly without hitting the network.

Usage:
    python scripts/link_local_dataset.py
    python scripts/link_local_dataset.py --source /path/to/BCI-IV --check
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_SOURCE = Path("/projects/EEG-foundation-model/BCI-IV")
SUBJECTS = range(1, 10)
SESSIONS = ("T", "E")


def moabb_cache_dir() -> Path:
    """Resolve ``<MNE_DATA>/MNE-bnci-data/~bci/database/001-2014`` (MOABB's BNCI2014_001 path)."""
    try:
        import mne

        root = mne.get_config("MNE_DATA") or str(Path.home() / "mne_data")
    except ImportError:
        root = str(Path.home() / "mne_data")
    return Path(root) / "MNE-bnci-data" / "~bci" / "database" / "001-2014"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="dir holding A0X{T,E}.mat")
    parser.add_argument("--check", action="store_true", help="only report status, do not modify")
    args = parser.parse_args()

    cache = moabb_cache_dir()
    print(f"source: {args.source}")
    print(f"cache : {cache}")
    if not args.check:
        cache.mkdir(parents=True, exist_ok=True)

    missing_src, linked, ok = [], 0, 0
    for s in SUBJECTS:
        for sess in SESSIONS:
            name = f"A{s:02d}{sess}.mat"
            src = args.source / name
            dst = cache / name
            if not src.exists():
                missing_src.append(name)
                continue
            if dst.exists() and not dst.is_symlink():
                ok += 1  # already a real cached file (e.g. previously downloaded)
                continue
            if args.check:
                status = "linked" if dst.is_symlink() else "ABSENT"
                print(f"  {name}: {status}")
                continue
            dst.symlink_to(src)
            linked += 1
    if missing_src:
        print(f"WARNING: missing in source: {missing_src}")
    if not args.check:
        print(f"linked {linked} file(s); {ok} already present as real files.")
    present = sorted(p.name for p in cache.glob("A0*.mat"))
    print(f"cache now has {len(present)}/18 BCI-IV-2a .mat files.")
    return 0 if len(present) == 18 and not missing_src else 1


if __name__ == "__main__":
    raise SystemExit(main())
