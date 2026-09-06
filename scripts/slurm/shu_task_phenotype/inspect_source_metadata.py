"""Read development-only SHU/Physio metadata required to freeze J0."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import lmdb
import numpy as np


def first_rech() -> dict[str, object]:
    env = lmdb.open(
        "/projects/EEG-foundation-model/RECH202/data_preprocessed/SHUMI/shumi.lmdb",
        subdir=False, readonly=True, lock=False, readahead=False,
    )
    with env.begin() as txn:
        _, raw = next(iter(txn.cursor()))
        record = pickle.loads(raw)
    return {
        "subject_id": str(record["subject_id"]),
        "sfreq": float(record["sfreq"]),
        "shape": list(np.asarray(record["x"]).shape),
        "ch_names": list(map(str, record["ch_names"])),
        "label": int(record["label"]),
    }


def physiomotion_channels() -> dict[str, object]:
    path = Path(
        "/home/infres/yinwang/denoiseNet_physiomotion_subject_restoration/"
        "results/cgdr/physiomotion_subject_restoration/prepared/participant_01.npz"
    )
    with np.load(path, allow_pickle=False) as data:
        keys = sorted(data.files)
        output: dict[str, object] = {"path": str(path), "keys": keys}
        for key in keys:
            lower = key.lower()
            if "channel" in lower or "ch_name" in lower or lower in {"ch_names", "channels"}:
                output[key] = np.asarray(data[key]).astype(str).tolist()
    return output


def main() -> None:
    output = {"rech_development_example": first_rech(), "physiomotion": physiomotion_channels()}
    destination = Path("results/cgdr/shu_task_phenotype_diffusion/j0/source_metadata.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
