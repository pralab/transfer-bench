r"""Ensemble-based attacks that levarege the victim feedback."""

from .bases import BASES
from .dswea import DSWEA
from .gaa import GAA
from .gfcs import GFCS
from .hybrid import HybridAttack
from .naive_avg import NaiveAvg, NaiveAvg1k, NaiveAvg10
from .nes import NES
from .simba_ods import SimbaODS
from .subspace import SubSpace

__all__ = [
    "BASES",
    "DSWEA",
    "GAA",
    "GFCS",
    "HybridAttack",
    "NaiveAvg",
    "NaiveAvg1k",
    "NaiveAvg10",
    "NES",
    "SimbaODS",
    "SubSpace",
]
