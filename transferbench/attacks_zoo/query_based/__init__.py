r"""Ensemble-based attacks that levarege the victim feedback."""

from .bases import BASES
from .dswea import DSWEA
from .gaa import GAA
from .gfcs import GFCS
from .naive_avg import NaiveAvg, NaiveAvg1k, NaiveAvg10

__all__ = [
    "BASES",
    "DSWEA",
    "GAA",
    "GFCS",
    "NaiveAvg",
    "NaiveAvg1k",
    "NaiveAvg10",
]
