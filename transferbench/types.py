r"""Define the types for the transferbench package."""

from typing import Optional, Protocol, runtime_checkable

from torch import Tensor


class CallableModel(Protocol):
    r"""A model is a callable that takes a tensor and returns a tensor."""

    def __call__(self, inputs: Tensor, forward_mask: Optional[Tensor] = None) -> Tensor:
        r"""Callable that take a tensor as input and optionally a binary mask.

        The forward_mask is used to counting the actual forward passes.
        """


# Hyperparameters class, to be inherited by the user for his own attack
@runtime_checkable
class TransferAttack(Protocol):
    r"""Attack step protocol."""

    def __call__(
        self,
        victim_model: CallableModel,
        surrogate_models: list[CallableModel],
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
            victim_model: CallableModel,
            *surrogate_models: CallableModel,
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
