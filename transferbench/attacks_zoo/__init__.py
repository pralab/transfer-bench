r"""Attacks zoo."""

from .naive_avg import NaiveAvg

__OPTIONAL__ = {
    "BASES": "transferbench.attacks_zoo.bases.BASES",
    "GAA": "transferbench.attacks_zoo.gaa.GAA",
    "DSWEA": "transferbench.attacks_zoo.dswea.DSWEA",
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


__all__ = ["BASES", "DSWEA", "GAA", "NaiveAvg"]
