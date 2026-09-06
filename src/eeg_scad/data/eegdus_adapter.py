"""Clean-room EEGdenoiseNet materialization for the EEGDfus audit baseline."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def _standardize(value: np.ndarray) -> np.ndarray:
    value=np.asarray(value,np.float64);return (value-np.mean(value,axis=-1,keepdims=True))/np.maximum(np.std(value,axis=-1,keepdims=True),1e-8)


def materialize_eegdenoisenet(source: Path, target: Path, seed: int=20260826) -> dict[str, Any]:
    source=source/"github-8d290661146c7189c98cc04812d37371d4b9426c"
    eeg=np.load(source/"EEG_all_epochs.npy",mmap_mode="r");eog=np.load(source/"EOG_all_epochs.npy",mmap_mode="r");n=min(len(eeg),len(eog));rng=np.random.Generator(np.random.PCG64DXSM(seed));order=rng.permutation(n);n_train=int(.8*n);n_validation=int(.1*n);groups={"train":order[:n_train],"validation":order[n_train:n_train+n_validation],"test":order[n_train+n_validation:]};target.mkdir(parents=True,exist_ok=True);manifest=[]
    for split,indices in groups.items():
        clean=_standardize(np.asarray(eeg[indices]));noise=_standardize(np.asarray(eog[np.roll(indices,1)]));snr_values=np.linspace(-5,5,11) if split=="test" else rng.uniform(-5,5,len(indices));repeats=11 if split=="test" else 1
        clean=np.repeat(clean,repeats,axis=0);noise=np.repeat(noise,repeats,axis=0);snr=np.tile(snr_values,len(indices)) if split=="test" else snr_values;ratio=np.sqrt(np.mean(clean*clean,axis=1)/np.maximum(np.mean(noise*noise,axis=1)*10**(.1*snr),1e-12));artifact=noise*ratio[:,None];noisy=clean+artifact
        np.savez_compressed(target/f"{split}.npz",clean=clean[:,None].astype(np.float32),noisy=noisy[:,None].astype(np.float32),artifact=artifact[:,None].astype(np.float32),snr_db=snr.astype(np.float32),source_record=np.repeat(indices,repeats).astype(np.int32))
        for index in indices:manifest.append({"split":split,"source_record":int(index)})
    with (target/"source_split_manifest.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=("split","source_record"),lineterminator="\n");writer.writeheader();writer.writerows(manifest)
    return {"source_records":n,"train":len(groups["train"]),"validation":len(groups["validation"]),"test":len(groups["test"]),"source_record_disjoint":True,"official_native_scaling":True,"seed":seed}
