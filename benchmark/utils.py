from hashlib import sha1

from transferbench.scenarios import AttackScenario, list_scenarios, load_attack_scenario

from .config import CAMPAIGNS


def get_path_from(attack: str, dataset: str, campaign: str, scenario: str):
    """Get the path to the specific attack on a dataset, campaign and scenario."""
    return f"results/{attack}/{dataset}/{campaign}/{scenario}"


def get_scenario_info(scn: AttackScenario) -> dict[str, str]:
    """Get the scenario info from a list of scenarios."""
    return {
        "dataset": scn.dataset,
        "victim_model": scn.victim_model,
        "surrogate_models": scn.surrogate_models,
        "max_queries": str(scn.hp.maximum_queries),
        "p": str(scn.hp.p),
        "eps": str(scn.hp.eps),
    }


def make_run_list() -> None:
    """Make a list of all the runs."""
    scenario_names = list_scenarios()
    run_list = []
    for scn_name in scenario_names:
        if scn_name in CAMPAIGNS:
            scns = load_attack_scenario(scn_name)
            for scn in scns:
                scn_dict = get_scenario_info(scn)
                scn_dict["id"] = sha1(str(scn_dict).encode("utf-8")).hexdigest()[-5:]  # noqa: S324
                run_list.append(scn_dict)
    return run_list
