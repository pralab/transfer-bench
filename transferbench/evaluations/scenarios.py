r"""Define scenario for the evaluations."""

from dataclasses import dataclass

from torch.nn import Module
from torch.utils.data import Dataset

from transferbench.attacks import AttackStep, BaseHyperParameters


@dataclass
class TransferScenario:
    r"""Define the sceneario for evaluating the transferability metric."""

    hp: BaseHyperParameters
    attack_step: str | AttackStep
    dataset: str | Dataset


@dataclass
class AttackScenario:
    r"""Define the scenario for evaluaring the transferability metric."""

    hp: BaseHyperParameters
    victim_model: str | Module
    surrogate_models: list[str | Module]
    dataset: str | Dataset
