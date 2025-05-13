r"""Implementation of the subspace attack.

Subspace Attack: Exploiting Promising Subspaces for Query-Efficient Black-box Attacks.
`https://papers.nips.cc/paper_files/paper/2019/file/2cad8fa47bbef282badbb8de5374b894-Paper.pdf`

"""

from dataclasses import dataclass
from functools import partial
from random import choice

import torch
from torch import Tensor, nn

from transferbench.types import CallableModel, TransferAttack

from .utils import grad_projection, hinge_loss, lp_projection


def add_dropout(module: nn.Module, dropout_prob: float) -> nn.Module:
    r"""Add Dropout2d after each Conv2d layer in a given module."""
    if isinstance(module, nn.Conv2d):
        drop_layer = nn.Dropout2d(dropout_prob).train()
        return nn.Sequential(module, drop_layer)
    if isinstance(module, nn.Module):
        # Recursively process child modules
        for name, child in module.named_children():
            setattr(module, name, add_dropout(child, dropout_prob))
        return module
    return module


def del_dropout(module: nn.Module) -> nn.Module:
    r"""Remove Dropout2d from a given module."""
    if (
        isinstance(module, nn.Sequential)
        and isinstance(module[0], nn.Conv2d)
        and isinstance(module[1], nn.Dropout2d)
    ):
        # Remove the Dropout2d layer
        return module[0]
    if isinstance(module, nn.Module):
        # Recursively process child modules
        for name, child in module.named_children():
            setattr(module, name, del_dropout(child))
        return module
    if isinstance(module, nn.Dropout2d):
        return module[0]
    return module


def update_dropout(module: nn.Module, dropout_prob: float) -> nn.Module:
    r"""Update Dropout2d probability in a given module."""
    if isinstance(module, nn.Dropout2d):
        module.p = dropout_prob
        module.train()
    else:
        for child in module.children():
            update_dropout(child, dropout_prob)
    return module


def subspace(
    victim_model: CallableModel,
    surrogate_models: list[nn.Module],
    inputs: Tensor,
    labels: Tensor,
    targets: Tensor | None = None,
    eps: float | None = None,
    p: float | str | None = None,
    maximum_queries: int | None = None,
    dropout_ratio: float = 0.05,
    tau: float = 0.1,
    delta: float = 0.1,
    eta_grad: float = 0.1,
    eta: float = 1 / 255,
) -> Tensor:
    r"""Subspace attack."""
    # Add dropout to surrogate models
    [add_dropout(model, dropout_ratio) for model in surrogate_models]
    # Set surrogate models to training mode
    for model in surrogate_models:
        model.train()

    # loss and projection functions
    grad_proj_p = partial(grad_projection, p=p)
    ball_projection = partial(lp_projection, eps=eps, p=p)
    loss_fn = partial(hinge_loss, kappa=float("inf"))

    grad = torch.zeros_like(inputs)
    success = torch.zeros_like(labels, dtype=torch.bool)
    x_adv = inputs.clone().detach()
    x_adv_left = x_adv_right = x_adv.clone()
    for _ in range(maximum_queries // 3):
        curr_grad = grad[~success]
        curr_orig = inputs[~success].clone()
        curr_labels = targets[~success] if targets is not None else labels[~success]
        surrogate = choice(surrogate_models)
        curr_adv = x_adv[~success].clone()
        curr_adv.requires_grad = True
        # Get surrogate model prediction
        loss = loss_fn(surrogate(curr_adv), curr_labels).sum()
        prior_grad = torch.autograd.grad(loss, curr_adv)[0]
        with torch.no_grad():
            # left and right local gradients
            grad_left = curr_grad - tau * prior_grad
            grad_left = grad_left / grad_left.norm(p=2, dim=(1, 2, 3), keepdim=True)
            grad_right = curr_grad + tau * prior_grad
            grad_right = grad_right / grad_right.norm(p=2, dim=(1, 2, 3), keepdim=True)
            # left and right attempts
            x_adv_left[~success] = curr_adv - delta * grad_left[~success]
            x_adv_right[~success] = curr_adv + delta * grad_right[~success]
            l_left = loss_fn(victim_model(x_adv_left, ~success), curr_labels)
            l_right = loss_fn(victim_model(x_adv_right, ~success), curr_labels)
            delta_t = (l_right - l_left)[:, None, None, None] * prior_grad
            delta_t = delta_t / (2 * delta * tau)
            # Update the gradient estimation
            grad[~success] = grad_proj_p(grad[~success] + delta_t[~success] * eta_grad)
            # Perturb using the gradient and project
            curr_adv = ball_projection(curr_orig, curr_adv + eta * grad[~success])
            x_adv[~success] = curr_adv.clamp(0, 1)

            # Evaluate the success
            preds = victim_model(x_adv, ~success).argmax(1)
            success = preds != labels if targets is None else preds == targets
            if success.all():
                break
            # Update dropout
            dropout_ratio = min(0.5, dropout_ratio + 0.01)
            [update_dropout(model, dropout_ratio) for model in surrogate_models]
    # Remove dropout from surrogate models
    [del_dropout(model) for model in surrogate_models]
    return x_adv.detach()


@dataclass
class SubSpaceHyperParams:
    """Hyperparameters for the subspace attack."""

    dropout_ratio: float = 0.05
    tau: float = 1.0
    delta: float = 0.1
    eta: float = 1 / 255
    eta_grad: float = 0.1


SubSpace: TransferAttack = partial(
    subspace,
    **vars(SubSpaceHyperParams()),
)
