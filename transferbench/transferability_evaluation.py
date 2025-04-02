r"""Class and script for evaluating the transferability metric."""

from dataclasses import asdict

import torch
from torch import nn
from tqdm.autonotebook import tqdm

from . import wrappers
from .datasets import DataLoader, get_loader
from .models import get_model
from .scenarios import TransferScenario
from .wrappers import AttackWrapper, HyperParameters

SCENARIOS = {
    "oneshot": (
        TransferScenario(
            hp=HyperParameters(maximum_queries=1, p="inf", eps=16 / 255),
            transfer_attack="NaiveAvg",
            dataset="ImageNetT",
        ),
    ),
    "fast": (
        TransferScenario(
            hp=HyperParameters(maximum_queries=10, p="inf", eps=16 / 255),
            transfer_attack="NaiveAvg",
            dataset="ImageNetT",
        ),
    ),
    "full": (
        TransferScenario(
            hp=HyperParameters(maximum_queries=50, p="inf", eps=16 / 255),
            transfer_attack="NaiveAvg",
            dataset="ImageNetT",
        ),
        TransferScenario(
            hp=HyperParameters(maximum_queries=50, p="inf", eps=16 / 255),
            transfer_attack="NaiveAvg",
            dataset="CIFAR10T",
        ),
    ),
}


class TransferEval:
    r"""Class to evaluate the transferability."""

    def __init__(
        self, vicim_model: str | nn.Module, surrogate_models: str | nn.Module
    ) -> None:
        r"""Evaluate the transferability given a victim model and a surrogate model.

        The victim model is the model to be attacked, and the surrogate model is
        the model used to generate the adversarial examples. The transferability is
        evaluated on default or customized scenarios (defaults are defined in config).

        Parameters
        ----------
        - victim_model (nn.Module): The victim model.
        - surrogate_models (list[nn.Module]): The surrogate models.
        """
        self.victim_model = vicim_model
        self.surrogate_models = surrogate_models
        self.set_scenarios("oneshot")

    def set_scenarios(self, *scenarios: str | TransferScenario) -> None:
        r"""Set the scenarios to be evaluated."""
        self.scenarios = []
        for scn in scenarios:
            if isinstance(scn, TransferScenario):
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
                    "transfer_attack": scenario.transfer_attack,
                    "dataset": scenario.dataset,
                    **result,
                }
            )

    def evaluate_scenario(
        self,
        scenario: TransferScenario,
        batch_size: int = 128,
        device: torch.device = "cuda",
    ) -> None:
        r"""Evaluate the transferability metric"."""
        # Get the hyperparameters
        hp = scenario.hp
        # Get the attack step
        transfer_attack = (
            getattr(wrappers, scenario.transfer_attack)
            if isinstance(scenario.transfer_attack, str)
            else scenario.transfer_attack
        )
        # Get the dataloader
        data_loader = (
            get_loader(scenario.dataset, batch_size=batch_size, device=device)
            if isinstance(scenario.dataset, str)
            else DataLoader(scenario.dataset, batch_size=batch_size, device=device)
        )
        # Import model if string is given
        if isinstance(self.victim_model, str):
            self.victim_model = get_model(
                self.victim_model, data_loader.dataset.mean, data_loader.dataset.std
            )
        # Import surrogate models if list of string is given
        self.surrogate_models = [
            get_model(model, data_loader.dataset.mean, data_loader.dataset.std)
            if isinstance(model, str)
            else model
            for model in self.surrogate_models
        ]

        # Set the device
        self.victim_model.to(device)
        for surrogate_model in self.surrogate_models:
            surrogate_model.to(device)

        # Get the attack
        attack = AttackWrapper(
            victim_model=self.victim_model,
            surrogate_models=self.surrogate_models,
            transfer_attack=transfer_attack,
            hyperparameters=hp,
        )
        pbar = tqdm(
            data_loader,
            total=len(data_loader),
            desc="Evaluating Transferability",
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
