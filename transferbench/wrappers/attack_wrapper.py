r"""Module for transfer attacks."""

from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from transferbench.types import TransferAttack

from .model_wrapper import ModelWrapper
from .utils import lp_constraint


# Hyperparameters class, to be inherited by the user for his own attack
@dataclass(frozen=True)
class HyperParameters:
    r"""Hyperparameters for the attack."""

    maximum_queries: int  # Maximum number of queries
    p: float | str  # Norm of the constraint
    eps: float  # Epsilon of the constraint


class AttackWrapper:
    r"""Wrapper class for the transfer attack type."""

    def __init__(
        self,
        victim_model: nn.Module,
        surrogate_models: list[nn.Module],
        transfer_attack: TransferAttack,
        hyperparameters: HyperParameters,
    ) -> None:
        r"""Wrap models and the transfer attack to handle queries and constraints.

        Parameters
        ----------
        - victim_model (nn.Module): The victim model.
        - surrogate_models list(nn.Module): The surrogate models.
        - transfer_attack (TransferAttack): The attack step to be performed.
        - hyperparameters (BaseHyperParameters): The hyperparameters of the attack.


        The victim model is the model that is being attacked, taking as input
        the input samples and returning the logits. The surrogate models are
        a list of models that are used to craft the adversarial examples.
        The attack step is the function that is called to craft the adversarial
        examples.
        """
        self.transfer_attack = transfer_attack
        self.hp = hyperparameters
        self.wrap_models(victim_model, *surrogate_models)
        self.sanity_check()

    def sanity_check(self) -> None:
        r"""Sanity check for the attack."""
        assert isinstance(self.hp, HyperParameters), (
            "Hyperparameters should be a dataclass."
        )
        assert isinstance(self.transfer_attack, TransferAttack), (
            "The attack signature must satisfy the `types.TransferAttack` protocol."
        )

    def wrap_models(
        self, victim_model: nn.Module, *surrogate_models: nn.Module
    ) -> None:
        r"""Wrap the models for the attack."""
        self.victim_model = ModelWrapper(victim_model)
        self.surrogate_models = [
            ModelWrapper(surrogate_model) for surrogate_model in surrogate_models
        ]

    def reset(self, inputs: Tensor) -> None:
        r"""Set models in eval and reset the counters."""
        self.victim_model.eval()
        self.victim_model.counter.reset(inputs)
        for model in self.surrogate_models:
            model.eval()
            model.counter.reset(inputs)

    def check_constraints(self, inputs: Tensor, adv: Tensor) -> bool:
        r"""Check if an example satisfies the constraints."""
        assert lp_constraint(inputs, adv, self.hp.eps, self.hp.p), (
            "Constraint violated. The adversarial example is not a legitimate image"
        )

    def check_queries(self) -> None:
        r"""Check that the maximum number of queries has not been exceeded."""
        forwards = self.victim_model.counter.get_forwards()
        assert forwards <= self.hp.maximum_queries, "Query budget exceeded."

    def check_black_box(self) -> None:
        r"""Check if the gradient of the victim has been used."""
        assert self.victim_model.counter.get_backwards() == 0, (
            "The gradient of the victim model has been used. "
            "The attack is not black-box."
        )

    @torch.no_grad()
    def evaluate_success(
        self, inputs: Tensor, labels: Tensor, targets: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        r"""Check if the attack was successful."""
        logits = self.victim_model(inputs)
        predictions = logits.argmax(dim=-1)
        if targets is not None:
            return predictions, predictions == targets
        return predictions, logits, predictions != labels

    def run(
        self, inputs: Tensor, labels: Tensor, targets: Optional[Tensor] = None
    ) -> dict:
        r"""Run the attack on the given input."""
        self.reset(inputs)
        adv = self.transfer_attack(
            self.victim_model.__call__,
            [model.__call__ for model in self.surrogate_models],
            inputs,
            labels,
            targets,
            **asdict(self.hp),
        )
        self.check_constraints(adv, inputs)
        self.check_queries()
        preds, logits, success = self.evaluate_success(adv, labels, targets)
        queries = self.victim_model.counter.get_queries()
        return {
            "adv": adv,
            "logits": logits,
            "predictions": preds,
            "success": success,
            "queries": queries,
        }

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"{self.__class__.__name__}("
            f"Hyperparameters: {self.hyperparameters}, "
            f"Attack: {self.transfer_attack})"
            f"Victim: {self.victim_model.__class__.__name__},"
            f"Surrogates: {[m.__class__.__name__ for m in self.surrogate_models]}"
        )
