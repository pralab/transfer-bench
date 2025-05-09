r"""Attacks zoo."""

from .query_based.bases import BASES
from .query_based.dswea import DSWEA
from .query_based.gaa import GAA
from .query_based.naive_avg import NaiveAvg, NaiveAvg1k, NaiveAvg10

__OPTIONAL__ = {
    "AdaEA": "transferbench.attacks_zoo.zero_query.AdaEA",
    "CWA": "transferbench.attacks_zoo.zero_query.CWA",
    "ENS": "transferbench.attacks_zoo.zero_query.ENS",
    "LGV": "transferbench.attacks_zoo.zero_query.LGV",
    "MBA": "transferbench.attacks_zoo.zero_query.MBA",
    "SASD_WS": "transferbench.attacks_zoo.zero_query.SASD_WS",
    "SMER": "transferbench.attacks_zoo.zero_query.SMER",
    "SVRE": "transferbench.attacks_zoo.zero_query.SVRE",
}


def __getattr__(name: str):  # noqa: ANN202
    if name in __OPTIONAL__:
        module_path, class_name = __OPTIONAL__[name].rsplit(".", 1)
        try:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except ImportError as error:
            msg = (
                f"It seems you are trying to use '{name}' attack which is wrapped "
                f"with TransferBench from an external library. If you started the"
                f"evaluation by cloning the repository, please run"
                f"'git submodule update --init --recursive' to update the submodules."
                f"Otherwise, please install the TransferBench library by running"
                f"'pip install transferbench' "
            )
            raise ImportError(msg) from error
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "BASES",
    "CWA",
    "DSWEA",
    "ENS",
    "GAA",
    "LGV",
    "MBA",
    "SASD_WS",
    "SMER",
    "SVRE",
    "AdaEA",
    "NaiveAvg",
    "NaiveAvg1k",
    "NaiveAvg10",
]
