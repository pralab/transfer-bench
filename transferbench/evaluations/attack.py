r"""Define the class for the attack evaluation."""

from dataclasses import asdict

import torch
from tqdm.autonotebook import tqdm

from transferbench import attacks
from transferbench.datasets import DataLoader, get_loader
from transferbench.models import get_model

from .scenarios import AttackScenario

SCENARIOS = {
    "bases": (
        AttackScenario(
            hp=attacks.BaseHyperParameters(maximum_queries=50, p="inf", eps=16 / 255),
            victim_model="ResNeXt50",
            surrogate_models=["ResNet50", "DenseNet121"],
            dataset="ImageNetT",
        ),
    )
}


class AttackEval:
    r"""Class for the evaluation of a black-box attack."""

    def __init__(self, attack: str | attacks.AttackStep) -> None:
        r"""Initialize the class for the evaluation of a black-box attack.

        Parameters
        ----------
        - attack (str | AttackStep): The attack step.
        """
        self.attack = attack
        self.set_scenarios("bases")

    def set_scenarios(self, *scenarios: str | AttackScenario) -> None:
        r"""Set the scenarios to be evaluated."""
        self.scenarios = []
        for scn in scenarios:
            if isinstance(scn, AttackScenario):
                self.scenarios.append(scn)
            elif isinstance(scn, str):
                self.scenarios.extend(SCENARIOS[scn])

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"TransferEval(victim_model={self.victim_model}",
            f"surrogate_models={self.surrogate_models})",
            f"scenarios={self.scenarios})",
        )

    def run(self, batch_size: int = 128, device: torch.device = "cuda") -> None:
        r"""Run the evaluation."""
        results = []
        for scenario in self.scenarios:
            print(f"Evaluating scenario: {scenario}")
            result = self.evaluate_scenario(scenario, batch_size, device)
            results.append(
                {
                    **asdict(scenario.hp),
                    "attack_step": scenario.attack_step,
                    "dataset": scenario.dataset,
                    **result,
                }
            )

    def evaluate_scenario(
        self,
        scenario: AttackScenario,
        batch_size: int = 128,
        device: torch.device = "cuda",
    ) -> None:
        r"""Evaluate the transferability metric"."""
        # Get the hyperparameters
        hp = scenario.hp
        # Get the attack step
        attack_step = (
            getattr(attacks, self.attack_step)
            if isinstance(self.attack_step, str)
            else self.attack_step
        )
        # Get the dataloader
        data_loader = (
            get_loader(scenario.dataset, batch_size=batch_size, device=device)
            if isinstance(scenario.dataset, str)
            else DataLoader(scenario.dataset, batch_size=batch_size, device=device)
        )
        # Import model if string is given
        if isinstance(scenario.victim_model, str):
            victim_model = get_model(
                scenario.victim_model, data_loader.dataset.mean, data_loader.dataset.std
            )
        else:
            victim_model = scenario.victim_model

        # Import surrogate models if list of string is given
        surrogate_models = [
            get_model(model, data_loader.dataset.mean, data_loader.dataset.std)
            if isinstance(model, str)
            else model
            for model in scenario.surrogate_models
        ]

        # Set the device
        victim_model.to(device)
        for surrogate_model in surrogate_models:
            surrogate_model.to(device)

        # Get the attack
        attack = attacks.TransferAttack(
            victim_model=victim_model,
            surrogate_models=surrogate_models,
            attack_step=attack_step,
            hyperparameters=hp,
        )
        pbar = tqdm(
            data_loader,
            total=len(data_loader),
            desc="Evaluating Attack",
        )
        success = 0
        queries = 0
        for inputs, *labels in pbar:
            inputs = inputs.to(device)
            labels = [label.to(device) for label in labels]
            result = attack.run(inputs, *labels)
            success += result["success"].sum().item()
            failures = (~result["success"]).sum().item()
            bqueries = result["batched_queries"]
            queries += result["samples_queries"] - failures * bqueries

            pbar.set_postfix(
                {
                    "ASR": success / len(data_loader.dataset),
                    "AvgQ": queries / len(data_loader.dataset),
                    "ASPQ": success / queries if queries > 0 else 0,
                }
            )
        return {
            "success": success,
            "queries": queries,
        }
