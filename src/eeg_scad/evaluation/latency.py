from dataclasses import dataclass

@dataclass(frozen=True)
class LatencyRecord:
    method:str
    milliseconds_per_window:float
    nfe:int
    peak_memory_mb:float

