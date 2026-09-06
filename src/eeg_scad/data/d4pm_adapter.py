from dataclasses import dataclass

@dataclass(frozen=True)
class D4PMSource:
    commit:str
    classification:str
    official_native_available:bool
