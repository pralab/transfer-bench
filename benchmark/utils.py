from dataclasses import dataclass
from hashlib import sha1
from itertools import product

from transferbench.attacks_zoo import __all__ as attacks_list
from transferbench.scenarios import AttackScenario, list_scenarios, load_attack_scenario
from transferbench.types import TransferAttack

from .config import ALLOWED_SCENARIOS


@dataclass
class Run:
    """Class to represent a campaign."""

    attack: str | TransferAttack
    scenario: AttackScenario
    campaign: str

    def __post_init__(self) -> None:
        """Post init method to set the id."""
        self.id = sha1(str(self).encode("utf-8")).hexdigest()[-5:]  # noqa: S324


def get_path_from_run(run: Run) -> str:
    """Get the path to the specific attack on a dataset, campaign and scenario."""
    scenario_infos = (
        f"{run.scenario.victim_model}_"
        f"q-{run.scenario.hp.maximum_queries}_"
        f"p-{run.scenario.hp.p}_"
        f"eps-{run.scenario.hp.eps}"
    )
    return (
        f"results/{run.attack}/{run.scenario.dataset}/{run.campaign}/{scenario_infos}"
    )


def get_config_from_run(run: Run) -> dict:
    """Get the config from the run."""
    wandb_config = {
        "attack": run.attack,
        "dataset": run.scenario.dataset,
        "campaign": run.campaign,
        "victim_model": run.scenario.victim_model,
        "maximum_queries": run.scenario.hp.maximum_queries,
        "p": run.scenario.hp.p,
        "eps": run.scenario.hp.eps,
        "id": run.id,
    }
    for i, surrog in enumerate(sorted(run.scenario.surrogate_models)):
        wandb_config[f"surrogate_model_{i}"] = surrog

    return wandb_config


def make_run_list() -> None:
    """Make a list of all the runs."""
    scenario_names = [scn for scn in list_scenarios() if scn in ALLOWED_SCENARIOS]
    run_list = []
    for scn_name, attack in product(scenario_names, attacks_list):
        scns = load_attack_scenario(scn_name)
        for scn in scns:
            run = Run(attack=attack, scenario=scn, campaign=scn_name.split("-")[0])
            run_list.append(run)
    return run_list
