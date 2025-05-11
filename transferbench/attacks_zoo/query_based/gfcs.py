r"""Batched Implementation of the GFCS attack.

Adapted from the paper:
`Attacking deep networks with surrogate-based adversarial black-box methods is easy`
`https://openreview.net/forum?id=Zf4ZdI4OQPV`
"""

from dataclasses import dataclass
from functools import partial
from random import choice
from typing import Optional

import torch
from torch import Tensor, autograd, nn

from transferbench.types import CallableModel, TransferAttack

from .utils import ODSDirection, grad_projection, hinge_loss, lp_projection

margin_loss = partial(hinge_loss, kappa=float("inf"))
cross_entropy_loss = partial(torch.nn.functional.cross_entropy, reduction="none")


def gfcs(
    victim_model: CallableModel,
    surrogate_models: list[nn.Module],
    inputs: Tensor,
    labels: Tensor,
    targets: Optional[Tensor] = None,
    eps: float = 0.01,
    p: str | float = 2,
    maximum_queries: int = 500,
    step_length: float = 2.0,
) -> Tensor:
    r"""GFCS attack.

    Batched version of the attack as describeed in the original paper.
    """
    best_adv = inputs.clone()
    test_adv = inputs.clone()
    # partial functions for projection and direction
    grad_projection_p = partial(grad_projection, p=p)
    proj_p_eps = partial(lp_projection, eps=eps, p=p)
    loss_fn = margin_loss if targets is None else cross_entropy_loss
    get_ods_direction = ODSDirection()
    # Initialize the losses and adv
    best_loss = torch.zeros_like(labels, dtype=torch.float)
    best_loss.fill_(float("inf") if targets is not None else -float("inf"))
    success = torch.zeros_like(labels, dtype=torch.bool)
    left_surr = torch.zeros_like(labels, dtype=torch.int).fill_(len(surrogate_models))
    all_curr_labels = labels if targets is None else targets
    for _ in range(0, maximum_queries, 2):
        curr_adv = best_adv[~success]
        curr_adv.requires_grad_()
        curr_labels = all_curr_labels[~success]
        # Get the surrogate model direction
        directions = torch.zeros_like(curr_adv)
        if (left_surr > 0).any():
            surrogate = choice(surrogate_models)
            loss = loss_fn(surrogate(curr_adv), curr_labels)
            grad = autograd.grad(loss.sum(), curr_adv, create_graph=True)[0]
            directions = grad_projection_p(grad)

        if (left_surr <= 0).any():  # random ods
            ods_direction = get_ods_direction(surrogate_models, curr_adv)
            ods_sampling = left_surr[~success] <= 0
            directions[ods_sampling] = ods_direction[ods_sampling]
        prev_best = torch.zeros_like(all_curr_labels, dtype=torch.bool)
        for alpha in {-step_length, step_length}:
            with torch.no_grad():
                mask = ~success & ~prev_best
                loc_test = curr_adv + grad_projection_p(directions) * alpha
                test_adv[mask] = loc_test[~prev_best[~success]]
                test_adv[mask] = proj_p_eps(inputs[mask], test_adv[mask])
                # Query the victim model
                logits = victim_model(test_adv, mask)
                loss_test = loss_fn(logits, all_curr_labels)
                criterion = loss_test[mask] > best_loss[mask]
                criterion = criterion if targets is None else ~criterion
                # Update the best adv
                best_loss[mask] = torch.where(
                    criterion,
                    loss_test[mask],
                    best_loss[mask],
                )
                best_adv[mask] = torch.where(
                    criterion[:, None, None, None],
                    test_adv[mask],
                    best_adv[mask],
                ).clamp(0, 1)
                prev_best[mask] = criterion

        left_surr = torch.where(prev_best, len(surrogate_models), left_surr - 1)
        # Update the success
        preds = logits.argmax(dim=-1)
        success = preds != labels if targets is None else preds == targets
        if success.all():
            break
    return best_adv.detach()


@dataclass
class GFCSHyperParams:
    r"""Hyperparameters for the SimBA ODS attack."""

    step_length: float = 2.0


GFCS: TransferAttack = partial(gfcs, **vars(GFCSHyperParams()))
