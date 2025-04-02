r"""Attacks zoo."""

from transferbench.transfer_attacks import AttackStep


def my_amazing_attack() -> None:  # noqa: D104
    """Hello, world."""


MyAmazingAttack: AttackStep = my_amazing_attack

__all__ = [
    "MyAmazingAttack",
]
