r"""Attacks zoo."""

from .bases import BASES
from .dswea import DSWEA
from .gaa import GAA
from .naive_avg import NaiveAvg, NaiveAvg1k, NaiveAvg10

__OPTIONAL__ = {
    "AdaEA": "transferbench.attacks_zoo.oneshot_attacks.AdaEA",
    "CWA": "transferbench.attacks_zoo.oneshot_attacks.CWA",
    "ENS": "transferbench.attacks_zoo.oneshot_attacks.ENS",
    "LGV": "transferbench.attacks_zoo.oneshot_attacks.LGV",
    "MBA": "transferbench.attacks_zoo.oneshot_attacks.MBA",
    "SASD_WS": "transferbench.attacks_zoo.oneshot_attacks.SASD_WS",
    "SMER": "transferbench.attacks_zoo.oneshot_attacks.SMER",
    "SVRE": "transferbench.attacks_zoo.oneshot_attacks.SVRE",
}


def __getattr__(name: str):  # noqa: ANN202
    if name in __OPTIONAL__:
        module_path, class_name = __OPTIONAL__[name].rsplit(".", 1)
        try:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except ImportError as error:
            msg = (
                f"Attack '{name}' requires 'full' option. "
                f"Install with: pip install 'transferbench[full]'"
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
