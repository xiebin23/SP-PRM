from dataclasses import dataclass
from typing import Optional
from transformers import GenerationConfig


@dataclass
class BaseGeneratorConfig(GenerationConfig):
    max_new_tokens: Optional[int] = None
    do_sample: Optional[bool] = False
    temperature: Optional[float] = 1.
    top_k: Optional[int] = 50
    top_p: Optional[float] = 1.


@dataclass
class BestOfNGeneratorConfig(BaseGeneratorConfig):
    num_return_sequences: int = 16
    do_sample: bool


@dataclass
class ARGSGeneratorConfig(BaseGeneratorConfig):
    mode: int = 1 # 1: argmax, 2: categorical
    w: float = 1.0


@dataclass
class CBSGeneratorConfig(BaseGeneratorConfig):
    beam_width: int = 2 # 对应的是top-k
    successors_per_state: int = 2 # 每次生成几个序列
    chunk_length: int = 30 # 每个序列的长度


@dataclass
class CARDSGeneratorConfig(BaseGeneratorConfig):
    alpha: float = 0.5
    beta: float = 0.7
    entropy_threshold: float = 3.0
    reward_threshold: float = 8.5
    option: str = "soft"
