from dataclasses import dataclass

@dataclass(frozen=True)
class EEGDfusSource:
    commit:str
    classification:str
    official_native_available:bool

