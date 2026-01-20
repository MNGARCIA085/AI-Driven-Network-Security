from dataclasses import dataclass

"""
Base = orchestration
Child = search space
"""


#----------Base class-----------------------
@dataclass(frozen=True)
class BaseTuningConfig:
    num_samples: int
    average: str # common for metric computations
    