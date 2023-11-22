from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class WikiContext:
    id: int
    title: str
    text: str
    score: float
    has_answer: bool

@dataclass
class NQExample:
    id: int
    question: str
    answers: List[str]
    contexts: List[WikiContext]
    masked_embedding: np.ndarray 

@dataclass
class NQ:
    dataset: List[NQExample]

@dataclass
class Example:
    question: str
    answers: List[str]
    random_top1: str
    random_exact: str
    random_unanswerable: str
    cbr_top1: str
    cbr_exact: str
    cbr_unanswerable: str