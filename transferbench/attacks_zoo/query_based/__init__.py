r"""Ensemble-based attacks that levarege the victim feedback."""

from .bases import BASES
from .dswea import DSWEA
from .gaa import GAA
from .gfcs import GFCS
from .hybrid import Hybrid
from .naive_avg import NaiveAvg, NaiveAvg1k, NaiveAvg10
from .simba_ods import SimbaODS
from .subspace import SubSpace

__all__ = [
    "BASES",
    "DSWEA",
    "GAA",
    "GFCS",
    "Hybrid",
    "NaiveAvg",
    "NaiveAvg1k",
    "NaiveAvg10",
    "SimbaODS",
    "SubSpace",
]
