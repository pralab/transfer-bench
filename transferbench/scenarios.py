r"""Define scenario for the evaluations."""

from dataclasses import dataclass

from torch.nn import Module
from torch.utils.data import Dataset

from .types import TransferAttack
from .wrappers import HyperParameters
from transferbench.config import CONFIG_DIR


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


def load_scenario(scenario_name: str) -> dict[str, AttackScenario]:
    """Load and parse a YAML scenario file.

    Args:
        file_name: Name of the YAML file to load (e.g., 'scenarios.yaml')

    Returns:
        Parsed dictionary containing scenario configuration

    Raises:
        FileNotFoundError: If specified file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    yaml_path = CONFIG_DIR / file_name
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
