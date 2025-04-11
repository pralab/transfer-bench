from collections.abc import Callable
from functools import partial
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .utils import hinge_loss
from transferbench.types import CallableModel, TransferAttack

from .naive_avg import grad_projection, lp_projection


def compute_weights(
        gradients: list[Tensor] | Tensor,
        sigma: float,
) -> Tensor:
    grad_norms = torch.Tensor([grad.norm() for grad in gradients])
    weights_unnorm = torch.exp(-grad_norms / (sigma**2))
    return weights_unnorm / torch.sum(weights_unnorm)


def ensemble_gradient(
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    targets: Tensor,
    surrogate_models: list[Callable[[Tensor], Tensor]],
    weights: Tensor,
) -> tuple[list[Tensor], Tensor, list[Tensor]]:
    """
    Compute the ensemble gradient for a batch of samples.

    Parameters
    ----------
        loss_fn: A loss function that takes (prediction, target) and returns a scalar loss.
        x: An input tensor of shape  b x 3 x h x w.
        targets: The labels of shape b .
        surrogate_models: List of models, each callable as model(x).
        weights: Tensor of weights (same length as surrogate_models) used to weight each model's gradient.

    Returns
    -------
        A tuple (grads_ens, g_ens, loss_ens):
            grads_ens: A list of gradients for each model (each matching the shape of x).
            g_ens: The weighted sum of gradients (same shape as x).
            loss_ens: A list containing each model's scalar loss.
    """
    xs_ens = [x.clone().requires_grad_() for _ in surrogate_models]
    loss_ens = []
    for model, x_loc in zip(surrogate_models, xs_ens):
        pred = model(x_loc)
        loss = loss_fn(pred, targets).mean(dim=-1)
        loss_ens.append(loss)
    grads_ens = torch.autograd.grad(loss_ens, xs_ens)
    g_ens = sum(w * grad for w, grad in zip(weights, grads_ens))
    return grads_ens, g_ens, loss_ens


def dswea(
    victim_model: CallableModel,    
    surrogate_models: list[CallableModel],
    inputs: Tensor,
    labels: Tensor,
    targets: Optional[Tensor] = None,
    maximum_queries: int = 50,
    p: float | str = "inf",
    eps: float = 16 / 255,
    alpha: float = 0.01,
    T: int = 10,
    M: int = 8,
    sigma: float = 2.5,
) -> Tensor:
    r"""
    Parameters
    ----------
    - victim_model (CallableModel): The victim model.
    - surrogate_models (list[CallableModel]): The surrogate models.
    - inputs (Tensor): The original input samples.
    - labels (Tensor): The labels.
    - targets (Tensor, optional): The target labels for targeted attack.
    - eps (float): The epsilon of the constraint.
    - p (float or str): The norm for the constraint.
    - maximum_queries (int): The maximum number of queries.
    - alpha (float): The step size.
    - T (int): Number of external iterations.
    - M (int): Number of internal iterations.
    - sigma (float): Parameter used in computing weights.

    Returns
    -------
    - Tensor: The adversarial examples (same shape as `inputs`).
    """
    # Save original inputs/labels (and targets) for constraint checking and indexing.
    x_orig = inputs.clone()
    labels_orig = labels.clone()
    if targets is not None:
        targets_orig = targets.clone()
    else:
        targets_orig = None

    num_surrogates = len(surrogate_models)
    loss_fn = hinge_loss
    grads_ens = [None] * num_surrogates
    weights = torch.ones(num_surrogates, device=inputs.device) / num_surrogates
    ball_projection = partial(lp_projection, eps=eps, p=p)
    dot_projection = partial(grad_projection, p=p)
    # success is per original sample.
    success = torch.zeros_like(labels_orig, dtype=torch.bool)

    # x_star keeps the full batch of adversarial examples.
    x_star = x_orig.clone()

    for q in range(maximum_queries):
        if q > 0:
            weights = compute_weights(grads_ens, sigma)

        # Instead of shrinking the batch, we identify the unresolved indices.
        unresolved_mask = ~success
        if not unresolved_mask.any():
            break

        # External iteration
        for _ in range(T):
            grads_ens, g_ens, loss_ens = ensemble_gradient(
                loss_fn=loss_fn,
                x=x_star,
                targets=targets,
                surrogate_models=surrogate_models,
                weights=weights,
            )

            G_bar = torch.zeros_like(x_star)
            x_bar = x_star.clone()

            sorted_idx = sorted(
                range(num_surrogates), key=loss_ens.__getitem__, reverse=True
            )

            # Internal iteration
            for m in range(M):
                k = sorted_idx[m % num_surrogates]

                x_bar.requires_grad = True
                x_bar.grad = None

                pred_x_bar = surrogate_models[k](x_bar)
                loss_x_bar = loss_fn(pred_x_bar, targets).mean(dim=-1)
                grad_x_bar = torch.autograd.grad(loss_x_bar, x_bar)[0]
                g_bar = weights[k] * (grad_x_bar - grads_ens[k]) + g_ens
                with torch.no_grad():
                    G_bar = G_bar + g_bar
                    x_bar - alpha * torch.sign(G_bar)
                    x_bar = (x_bar - x_orig).clamp(-eps, eps) + x_orig
                    x_bar = x_bar.clamp(0, 1)
            x_star = x_bar

        with torch.no_grad():
            victim_logits = victim_model(x_star)
            victim_pred = victim_logits.argmax(dim=1)


        if targets_orig is None:
            success = victim_pred != labels_orig
        else:
            success = victim_pred == targets_orig

        if success.all():
            return x_star

    # If query limit is exceeded, return the full batch of (possibly partially attacked) examples.
    return x_star


@dataclass
class DSWEAHyperParams:
    r"""Hyperparameters for DSWEA attack."""
    alpha: float = 0.01
    T: int = 10
    M: int = 8 
    sigma: float = 2.5

# Wrap the attack to be used in the evaluators.
DSWEA: TransferAttack = partial(dswea, **asdict(DSWEAHyperParams()))
