r"""Utility functions for the attacks."""

from collections.abc import Callable

import torch
from torch import Tensor, nn


def lp_constraint(inputs: Tensor, adv: Tensor, eps: float, p: float | str) -> bool:
    r"""Check if the adversarial example satisfies the constraints.

    The constraints are:
    - Box constraints: 0 <= adv <= 1
    - Lp norm constraints: ||inputs - adv||_p <= eps

    Parameters
    ----------
    - inputs (torch.Tensor): The original inputs.
    - adv (torch.Tensor): The adversarial examples.
    - eps (float): The maximum perturbation allowed.
    - p (float | str): The norm to be used.

    """
    box_constraints = torch.all(inputs >= 0) and torch.all(inputs <= 1)
    lp_norms = torch.all(
        torch.linalg.vector_norm(inputs - adv, float(p), dim=(1, 2, 3)) <= eps + 1e-7
    )
    return box_constraints and lp_norms


def hinge_loss(logits: Tensor, labels: Tensor, kappa: int = 200) -> Tensor:
    r"""Compute the hinge loss.

    Args:
    -----
        logits: (batch_size x num_classes) tensor of logits
        labels: (batch_size) tensor of target labels
        kappa: (float) The margin of the hinge loss
    Return:
    -------
        (batch_size) tensor of losses
    """
    margins = logits - logits.gather(1, labels.view(-1, 1))
    worst_margin = margins.scatter(1, labels.view(-1, 1), -float("inf")).amax(1)
    return torch.max(worst_margin, -torch.tensor(kappa))


class BaseEnsembleLoss(nn.Module):
    r"""Base class for ensemble loss functions."""

    def __init__(self, surrogates: list[Callable]) -> None:
        r"""Initialize the ensemble loss.

        Args:
        -----
            surrogates: (list of nn.Module) The surrogate models
        """
        super().__init__()
        self.surrogates = surrogates
        self.register_buffer(
            "weights",
            torch.ones(len(surrogates), dtype=torch.float32) / len(surrogates),
        )

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        r"""Return the ensemble loss not aggregated.

        Args:
        -----
            inputs: (batch_size x dim1 x dim2)  input tensor
            labels: (batch_size) target labels

        Return:
        -------
            (batch_size) tensor of losses
        """
        raise NotImplementedError


class LogitEnsemble(BaseEnsembleLoss):
    r"""Ensemble loss based on the logits."""

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        r"""Return the ensemble loss not aggregated.

        Args:
        -----
            inputs: (batch_size x dim1 x dim2)  input tensor
            labels: (batch_size) target labels

        Return:
        -------
            (batch_size) tensor of losses
        """
        logits = torch.stack([model(inputs) for model in self.surrogates])
        ens_logits = (logits * self.weights.view(-1, 1, 1)).sum(0)
        return hinge_loss(ens_logits, labels)


class AggregatedEnsemble(BaseEnsembleLoss):
    r"""Ensemble loss based on the aggregated hinge loss."""

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        r"""Return the ensemble loss aggregated.

        Args:
        -----
            inputs: (batch_size x dim1 x dim2)  input tensor
            labels: (batch_size) target labels

        Return:
        -------
            (1) tensor of loss
        """
        ens_loss = [hinge_loss(model(inputs), labels) for model in self.surrogates]
        ens_loss = torch.stack(ens_loss)
        weights = self.weights.view(-1, 1).to(ens_loss.device)
        return (ens_loss * weights).sum(0)
