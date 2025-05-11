r"""Utility functions for the attacks."""

from collections.abc import Callable
from random import choice
from typing import Optional

import torch
from torch import Tensor, autograd, nn


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


def grad_projection(g: Tensor, p: float | str) -> Tensor:
    r"""Return the unitary version of psi() function."""
    if float(p) == float("inf"):
        return g.sign()
    q = 1 / (1 - 1 / p)
    if float(p) == 1:
        q = 100  # trick to handling all the norms
    g = g.abs().pow(q - 1) * g.sign()
    return g / torch.norm(g, p=p, dim=(1, 2, 3), keepdim=True)


def lp_projection(x: Tensor, adv: Tensor, eps: float, p: float | str) -> Tensor:
    r"""Return the projection of x on the lp ball."""
    if float(p) == float("inf"):
        return torch.clamp(adv - x, -eps, eps) + x
    if float(p) == 1:
        raise NotImplementedError
    return (adv - x).renorm(p=p, dim=0, maxnorm=eps) + x


def projected_gradient_descent(
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    inputs: Tensor,
    x_init: Tensor,
    labels: Tensor,
    targets: Optional[Tensor],
    ball_projection: Callable[[Tensor], Tensor],
    dot_projection: Callable[[Tensor], Tensor],
    alpha: float,
    inner_iterations: int,
) -> Tensor:
    r"""Implement the projected gradient descent."""
    x = x_init.clone()
    for _ in range(inner_iterations):
        x.requires_grad_()
        loss = -loss_fn(x, labels) if targets is None else loss_fn(x, targets)
        loss.sum().backward()
        with torch.no_grad():
            x = ball_projection(inputs, x - alpha * dot_projection(x.grad))
    return torch.clamp(x, 0, 1)  # , loss


class ODSDirection:
    """Class to generate random directions for the ODS attack."""

    nclasses_: Optional[int] = None

    def update_nclasses(self, nclasses: int) -> None:
        """Update the number of classes."""
        self.nclasses_ = nclasses

    @property
    def nclasses(self) -> int:
        """Get the number of classes."""
        return self.nclasses_

    def __call__(
        self,
        surrogate_models: list[nn.Module],
        inputs: Tensor,
        nclasses: Optional[int] = None,
    ) -> Tensor:
        """Generate a random direction for the ODS attack."""
        surrogate = choice(surrogate_models)
        if nclasses is not None and self.nclasses is None:
            self.update_nclasses(nclasses)
        elif nclasses is None and self.nclasses is None:
            logits = surrogate(inputs)
            self.update_nclasses(logits.shape[-1])
            weights = 2 * torch.rand_like(logits, dtype=torch.float) - 1
            grad_w = autograd.grad((logits * weights).sum(), inputs)[0]
            return grad_w / torch.norm(grad_w, p=2, dim=(1, 2, 3), keepdim=True)

        batch_size = inputs.shape[0]
        weights = 2 * torch.rand(batch_size, self.nclasses, device=inputs.device) - 1
        grad_w = autograd.grad((surrogate(inputs) * weights).sum(), inputs)[0]
        return grad_w / torch.norm(grad_w, p=2, dim=(1, 2, 3), keepdim=True)
