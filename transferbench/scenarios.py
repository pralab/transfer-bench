r"""Define scenario for the evaluations."""

from dataclasses import dataclass

from torch.nn import Module
from torch.utils.data import Dataset

from .types import TransferAttack
from .wrappers import HyperParameters


@dataclass
class TransferScenario:
    r"""Define the sceneario for evaluating the transferability metric."""

    hp: HyperParameters
    transfer_attack: str | TransferAttack
    dataset: str | Dataset


@dataclass
class AttackScenario:
    r"""Define the scenario for evaluaring the transferability metric."""

    hp: HyperParameters
    victim_model: str | Module
    surrogate_models: list[str | Module]
    dataset: str | Dataset
