"""Metadata-only datalake locator for the SHU multi-session dataset."""
from __future__ import annotations

import io
import json
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np


def describe(value: bytes) -> dict[str, object]:
    result: dict[str, object] = {"bytes": len(value), "prefix_hex": value[:16].hex()}
    for name, loader in (
        ("pickle", lambda: pickle.loads(value)),
        ("numpy", lambda: np.load(io.BytesIO(value), allow_pickle=False)),
    ):
        try:
            obj = loader()
            result["codec"] = name
            result["type"] = type(obj).__name__
            if isinstance(obj, np.ndarray):
                result["shape"] = list(obj.shape); result["dtype"] = str(obj.dtype)
            elif isinstance(obj, dict):
                result["keys"] = sorted(map(str, obj))[:50]
                result["value_summaries"] = {str(k): {"type": type(v).__name__, "shape": list(v.shape) if hasattr(v, "shape") else None} for k, v in list(obj.items())[:20]}
            else:
                result["repr"] = repr(obj)[:500]
            return result
        except Exception:
            pass
    return result


def main(path: Path) -> None:
    env = lmdb.open(str(path), subdir=path.is_dir(), readonly=True, lock=False, readahead=False, max_dbs=32)
    output: dict[str, object] = {"path": str(path), "info": env.info(), "stat": env.stat(), "records": []}
    with env.begin() as txn:
        cursor = txn.cursor()
        for index, (key, value) in enumerate(cursor):
            if index < 200:
                record = {"index": index, "key": key.decode("utf-8", errors="replace"), "value_bytes": len(value)}
                if index < 5: record["value"] = describe(value)
                output["records"].append(record)
            else:
                break
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
