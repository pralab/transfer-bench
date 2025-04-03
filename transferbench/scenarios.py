r"""Define scenario for the evaluations."""

from dataclasses import dataclass
from pathlib import Path

import yaml
from torch.nn import Module
from torch.utils.data import Dataset

from transferbench.types import TransferAttack
from transferbench.wrappers import HyperParameters


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


__SCENARIO_DIR__ = Path(__file__).parent / "config" / "scenarios"


def get_scenarios_paths() -> dict[str, str]:
    """List all available scenario keys from YAML files in the scenarios directory.

    Returns
    -------
    dict[str,dict] of scenarios names
    """
    scenarios_paths = {}
    for yaml_file in __SCENARIO_DIR__.glob("*.yaml"):
        content = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        if content:  # Only process if file is not empty
            curr_scenario_path = dict(
                zip(content.keys(), (yaml_file.stem,) * len(content), strict=True)
            )
            scenarios_paths = {**scenarios_paths, **curr_scenario_path}
    return dict(sorted(scenarios_paths.items()))


def list_scenarios() -> list[str]:
    """List all available scenario keys from YAML files in the scenarios directory.

    Returns
    -------
    list[str] of scenarios names
    """
    return list(get_scenarios_paths().keys())


def load_attack_scenario(scenario_name: str) -> AttackScenario:
    r"""Load the attack scenario from the YAML file.

    Parameters
    ----------
    scenario_name : str
        The name of the scenario to load.

    Returns
    -------
    AttackScenario
        The loaded attack scenario.
    """
    scenarios_path = get_scenarios_paths()
    if scenario_name not in scenarios_path:
        msg = f"Scenario '{scenario_name}' not found."
        raise ValueError(msg)

    scenario_path = __SCENARIO_DIR__ / f"{scenarios_path[scenario_name]}.yaml"
    scenarios_dict = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    scenarios_list = scenarios_dict[scenario_name]
    return [
        AttackScenario(hp=HyperParameters(**scn.pop("hp")), **scn)
        for scn in scenarios_list
    ]
