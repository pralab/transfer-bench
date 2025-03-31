from dataclasses import dataclass

import torch
from torch import nn
from tqdm import tqdm

from transferbench import attacks
from transferbench.datasets import get_loader
from transferbench.models.utils import add_normalization


@dataclass
class Scenario:
    hp: dict
    attack_step: str
    dataset: str


SCENARIOS = {
    "oneshot": (
        Scenario(
            hp={"maximum_queries": 1, "p": "inf", "eps": 16 / 255},
            attack_step="NaiveAvg",
            dataset="ImageNetT",
        )
    ),
    "fast": (
        Scenario(
            hp={"maximum_queries": 10, "p": "inf", "eps": 16 / 255},
            attack_step="NaiveAvg",
            dataset="ImageNetT",
        )
    ),
}


class TransferabilityEvaluator:
    r"""Class to evaluate the transferability metric."""

    def __init__(self, vicim_model: nn.Module, *surrogate_models: nn.Module) -> None:
        r"""Initialize the class for the evaluation of the transferability.

        Parameters
        ----------
        - victim_model (nn.Module): The victim model.
        - surrogate_models (list[nn.Module]): The surrogate models.
        """
        self.victim_model = vicim_model
        self.surrogate_models = surrogate_models
        self.set_scenarios("oneshot", "fast")

    def set_scenarios(self, *scenarios: str) -> None:
        r"""Set the scenarios to be evaluated."""
        self.scenarios = [scn for scn in scenarios if scn in SCENARIOS]

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"TrasnferabilityEvaluator(victim_model={self.victim_model}",
            f"surrogate_models={self.surrogate_models})",
            f"scenarios={SCENARIOS})",
        )

    def run(self, batch_size: int = 128, device: torch.device = "cuda") -> None:
        r"""Run the evaluation."""
        for scenario in self.scenarios:
            print(f"Evaluating scenario : {scenario}")
            result = self.evaluate_scenario(scenario, batch_size, device)
            print(f"Finished evaluating scenario {scenario}")
            print(result)

    def evaluate_scenario(
        self,
        scenario: str = "oneshot",
        batch_size: int = 128,
        device: torch.device = "cuda",
    ) -> None:
        r"""Evaluate the transferability metric"."""
        # Get the scenario
        scenario = SCENARIOS[scenario]
        # Get the hyperparameters
        hp = attacks.BaseHyperParameters(**scenario.hp)
        # Get the attack step
        attack_step = getattr(attacks, scenario.attack_step)
        # Get the dataloader
        data_loader = get_loader(scenario.dataset, batch_size=batch_size, device=device)
        # Add normalization to the model
        mean, std = data_loader.dataset.mean, data_loader.dataset.std
        victim_model = add_normalization(self.victim_model, mean, std)
        surrogate_models = [
            add_normalization(model, mean, std) for model in self.surrogate_models
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
            desc="Evaluating transferability",
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
                    "successes_per_queries": success / queries,
                }
            )
        return {
            "success": success,
            "queries": queries,
        }
