from dataclasses import dataclass


@dataclass
class ScorerConfig:
    pass


@dataclass
class RMScorerConfig(ScorerConfig):
    """Generation configuration parameters"""
    max_length: int = 4096
    batch_size: int = 8

