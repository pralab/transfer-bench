r"""Module for transfer attacks."""

from dataclasses import asdict, dataclass
from typing import Optional, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from transferbench.models.utils import ModelWrapper

from .utils import lp_constraint


# Type aliases
class Model(Protocol):
    r"""A model is a callable that takes a tensor and returns a tensor."""

    @staticmethod
    def __call__(inputs: Tensor) -> Tensor: ...  # noqa: D102


# Hyperparameters class, to be inherited by the user for his own attack
@runtime_checkable
class AttackStep(Protocol):
    r"""Attack step protocol."""

    def __call__(
        self,
        victim_model: Model,
        surrogate_models: list[Model],
        inputs: Tensor,
        labels: Tensor,
        targets: Optional[Tensor] = None,
        eps: Optional[float] = None,
        p: Optional[float | str] = None,
        maximum_queries: Optional[int] = None,
    ) -> Tensor:
        r"""Perform the attack on a batch of data.

        Parameters
        ----------
        - victim_model (VictimModel): The victim model.
        - surrogate_models (SurrogateModels): The surrogate models.
        - inputs (Tensor): The input samples.
        - labels (Tensor): The labels.
        - targets (Tensor): The target labels for targeted-attack.
        - eps (float): The epsilon of the constraint.
        - p (float): The norm of the constraint.
        - maximum_queries (int): The maximum number of queries.

        Returns
        -------
        - Tensor: The adversarial examples.

        The attack step function should have the following signature:
        ```
        def attack_step(
            victim_model: Model,
            *surrogate_models: Model,
            inputs: Tensor,
            labels: Tensor,
            targets: Optional[Tensor] = None,
            eps: Optional[float] = None,
            p: Optional[float | str] = None,
            maximum_queries: Optional[int] = None,
        ) -> Tensor:
            ...
        ```
        N.B the attack can work either in batch or single sample mode, nevertheless the
        queries of the victim are counted batch-wise and not sample-wise, hence avoid
        for loops, and prefer masks.
        """


# Hyperparameters class, to be inherited by the user for his own attack
@dataclass(frozen=True)
class BaseHyperParameters:
    r"""Hyperparameters for the attack."""

    maximum_queries: int  # Maximum number of queries
    p: float | str  # Norm of the constraint
    eps: float  # Epsilon of the constraint


class TransferAttack:
    r"""Base class for transfer attacks."""

    def __init__(
        self,
        victim_model: nn.Module,
        surrogate_models: list[nn.Module],
        attack_step: AttackStep,
        hyperparameters: BaseHyperParameters,
    ) -> None:
        r"""Initialize the attack.

        Parameters
        ----------
        - victim_model (nn.Module): The victim model.
        - surrogate_models list(nn.Module): The surrogate models.
        - attack_step (callable): The attack step to be performed.
        - hyperparameters (HyperParameters): The hyperparameters of the attack.


        The victim model is the model that is being attacked, taking as input
        the input samples and returning the logits. The surrogate models are
        a list of models that are used to craft the adversarial examples.
        The attack step is the function that is called to craft the adversarial
        examples.
        """
        self.attack_step = attack_step
        self.hp = hyperparameters
        self.wrap_models(victim_model, *surrogate_models)
        self.sanity_check()

    def sanity_check(self) -> None:
        r"""Sanity check for the attack."""
        assert isinstance(self.hp, BaseHyperParameters), (
            "Hyperparameters should be a dataclass."
        )
        assert isinstance(self.attack_step, AttackStep), (
            "Attack step should be the AttackStep Protocol."
        )

    def wrap_models(
        self, victim_model: nn.Module, *surrogate_models: nn.Module
    ) -> None:
        r"""Wrap the models for the attack."""
        self.victim_model = ModelWrapper(victim_model)
        self.surrogate_models = [
            ModelWrapper(surrogate_model) for surrogate_model in surrogate_models
        ]

    def reset(self) -> None:
        r"""Set models in eval and reset the counters."""
        self.victim_model.eval()
        self.victim_model.counter.reset()
        for model in self.surrogate_models:
            model.eval()
            model.counter.reset()

    def check_constraints(self, inputs: Tensor, adv: Tensor) -> bool:
        r"""Check if an example satisfies the constraints."""
        assert lp_constraint(inputs, adv, self.hp.eps, self.hp.p), (
            "Constraint violated. The adversarial example is not a legitimate image"
        )

    def check_queries(self) -> None:
        r"""Check that the maximum number of queries has not been exceeded."""
        bqueries = self.victim_model.counter.forwarded_batches
        assert bqueries <= self.hp.maximum_queries, "Query budget exceeded."
        return bqueries, self.victim_model.counter.get_forwards()

    def check_black_box(self) -> None:
        r"""Check if the gradient of the victim has been used."""
        assert len(self.victim_model.counter.get_backwards()) == 0, (
            "The gradient of the victim model has been used. "
            "The attack is not black-box."
        )

    @torch.no_grad()
    def evaluate_success(
        self, inputs: Tensor, labels: Tensor, targets: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        r"""Check if the attack was successful."""
        predictions = self.victim_model(inputs).argmax(dim=-1)
        if targets is not None:
            return predictions, predictions == targets
        return predictions, predictions != labels

    def run(
        self, inputs: Tensor, labels: Tensor, targets: Optional[Tensor] = None
    ) -> dict:
        r"""Run the attack on the given input."""
        self.reset()
        adv = self.attack_step(
            self.victim_model.__call__,
            [model.__call__ for model in self.surrogate_models],
            inputs,
            labels,
            targets,
            **asdict(self.hp),
        )
        self.check_constraints(adv, inputs)
        bqueries, queries = self.check_queries()
        pred, success = self.evaluate_success(adv, labels, targets)
        return {
            "adv": adv,
            "predictions": pred,
            "success": success,
            "batched_queries": bqueries,
            "samples_queries": queries,
        }

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"{self.__class__.__name__}("
            f"Hyperparameters: {self.hyperparameters}, "
            f"Attack: {self.attack_step})"
            f"Victim: {self.victim_model.__class__.__name__},"
            f"Surrogates: {[m.__class__.__name__ for m in self.surrogate_models]}"
        )
