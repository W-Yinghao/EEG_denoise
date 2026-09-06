from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


def load_folds(path:Path)->list[dict[str,Any]]:return list(yaml.safe_load(path.read_text(encoding="utf-8"))["folds"])


def validate_folds(folds:list[dict[str,Any]],participants:list[str])->None:
    expected=set(participants)
    assert len(folds)==5
    test=[]
    for row in folds:
        train=set(row["train"]);val=set(row["validation"]);held=set(row["test"])
        assert not train&val and not train&held and not val&held and train|val|held==expected
        assert len(train)==9 and len(val)==len(held)==3;test+=row["test"]
    assert sorted(test)==sorted(participants)

