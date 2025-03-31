from dataclasses import dataclass
from transferbench.attacks import BaseHyperParameters, AttackStep
from torch.utils.data import Dataset


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
    victim_model: str
    surrogate_models: list
