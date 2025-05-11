r"""Define the class for the attack evaluation."""

import logging
from collections.abc import Generator

import torch
from tqdm.autonotebook import tqdm

from transferbench.types import TransferAttack

from . import attacks_zoo
from .datasets import DataLoader, get_loader
from .models import get_model
from .scenarios import load_attack_scenario
from .types import AttackResult, AttackScenario, EvaluationResult
from .wrappers import AttackWrapper


class AttackEval:
    r"""Class for the evaluation of a black-box attack."""

    def __init__(self, transfer_attack: str | TransferAttack) -> None:
        r"""Evaluate the performance of a black-box attack.

        The transfer attack is evaluated on default or customized scenarios (defaults
        defined in config). Each scenario encapsulate the victim and surrogates models,
        along with the dataset and the hyperparameters of the attack.


        Parameters
        ----------
        - attack (str | TransferAttack): The attack step.
        """
        self.transfer_attack = transfer_attack
        self.set_scenarios("etero-imagenet-inf")

    def set_scenarios(self, *scenarios: str | AttackScenario) -> None:
        r"""Set the scenarios to be evaluated."""
        self.scenarios = []
        for scn in scenarios:
            if isinstance(scn, AttackScenario):
                self.scenarios.append(scn)
            elif isinstance(scn, str):
                self.scenarios.extend(load_attack_scenario(scn))

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"TransferEval(transfer_attack={self.transfer_attack}, "
            f"scenarios={self.scenarios})"
        )

    def run(
        self, batch_size: int = 128, device: torch.device = "cuda"
    ) -> list[EvaluationResult]:
        r"""Run the evaluation."""
        results = []
        for scenario in self.scenarios:
            logging.info("Evaluating scenario", extra={"scenario": scenario})  # noqa: LOG015
            result = list(self.evaluate_scenario_(scenario, batch_size, device))
            results.append(
                EvaluationResult(
                    attack=self.transfer_attack,
                    results=result,
                )
            )
        return results

    def evaluate_scenario_(
        self,
        scenario: AttackScenario,
        batch_size: int = 128,
        device: torch.device = "cuda",
    ) -> Generator[AttackResult]:
        r"""Evaluate a single scenario. For internal use only.

        Parameters
        ----------
        scenario : AttackScenario
            The scenario to evaluate.
        batch_size : int
            The batch size to use for the evaluation.
        device : torch.device
            The device to use for the evaluation.

        Yields
        ------
        AttackResult
            The result of the attack evaluation.
        """
        # Get the hyperparameters
        hp = scenario.hp
        # Get the attack step
        transfer_attack = (
            getattr(attacks_zoo, self.transfer_attack)
            if isinstance(self.transfer_attack, str)
            else self.transfer_attack
        )
        # Get the dataloader
        data_loader = (
            get_loader(scenario.dataset, batch_size=batch_size, device=device)
            if isinstance(scenario.dataset, str)
            else DataLoader(scenario.dataset, batch_size=batch_size, device=device)
        )
        # Import model if string is given
        if isinstance(scenario.victim_model, str):
            victim_model = get_model(scenario.victim_model)
        else:
            victim_model = scenario.victim_model

        # Import surrogate models if list of string is given
        surrogate_models = [
            get_model(model) if isinstance(model, str) else model
            for model in scenario.surrogate_models
        ]

        # Set the device
        victim_model.to(device)
        for surrogate_model in surrogate_models:
            surrogate_model.to(device)

        # Get the attack
        attack = AttackWrapper(
            victim_model=victim_model,
            surrogate_models=surrogate_models,
            transfer_attack=transfer_attack,
            hyperparameters=hp,
        )
        pbar = tqdm(
            data_loader,
            total=len(data_loader),
            desc="Evaluating Attack",
        )
        success = 0
        queries = 0
        processed_samples = 0
        for inputs, *labels in pbar:
            inputs = inputs.to(device)
            labels = [label.to(device) for label in labels]
            result = attack.run(inputs, *labels)
            success += result["success"].sum().item()
            queries += (result["queries"] * result["success"]).sum().item()
            processed_samples += result["success"].shape[0]
            pbar.set_postfix(
                {
                    "ASR": success / processed_samples,
                    "AvgQ": queries / success if success > 0 else None,
                    "ASPQ": success / queries if queries > 0 else 0,
                }
            )
            # move to cpu
            result = {
                key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in result.items()
            }
            yield result
        pbar.close()
