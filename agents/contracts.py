from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Symbol:
    symbol_id: str
    text: str
    image_path: Optional[str] = None

@dataclass
class Candidate:
    symbol: Symbol
    score: float

@dataclass
class PredictionRequest:
    sequence: List[Symbol]
    prev_turn: List[str]
    user_id: str
    k: int = 5

@dataclass
class SentenceResult:
    text: str
    attempt: int

@dataclass
class VerifyResult:
    ok: bool
    missing: List[str] = field(default_factory=list)
    added: List[str] = field(default_factory=list)
    feedback: str = ""
