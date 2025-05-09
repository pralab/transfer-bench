r"""Ensemble-based attacks that levarege the victim feedback."""

from .bases import BASES
from .dswea import DSWEA
from .gaa import GAA
from .naive_avg import NaiveAvg, NaiveAvg1k, NaiveAvg10

__all__ = [
    "BASES",
    "DSWEA",
    "GAA",
    "NaiveAvg",
    "NaiveAvg1k",
    "NaiveAvg10",
]
