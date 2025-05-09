r"""Implementation of the GFCS attack.

Adapted from the paper:
`Attacking deep networks with surrogate-based adversarial black-box methods is easy`
`https://openreview.net/forum?id=Zf4ZdI4OQPV`
"""

from functools import partial
from random import choice
from typing import Optional

import torch
from torch import Tensor, autograd, nn

from transferbench.types import CallableModel, TransferAttack

from .utils import grad_projection, hinge_loss, lp_projection


def gfcs(
    victim_model: CallableModel,
    surrogate_models: list[nn.Module],
    inputs: Tensor,
    labels: Tensor,
    targets: Optional[Tensor] = None,
    eps: float = 0.01,
    p: str | float = 2,
    maximum_queries: int = 500,
) -> Tensor:
    r"""GFCS attack.

    Batched version of the attack as describeed in the original paper.
    """
    best_adv = inputs.clone()
    best_adv_test = inputs.clone()
    grad_projection_p = partial(grad_projection, p=p)
    step_length: float = Tensor(inputs.shape[2:]).prod() * (1e-3) ** 0.5
    proj_p_nu = partial(lp_projection, eps=eps, p=p)
    margin_loss = partial(hinge_loss, kappa=float("inf"))
    best_loss = torch.zeros_like(labels, dtype=torch.float)
    success = torch.zeros_like(labels, dtype=torch.bool)
    left_surr = torch.zeros_like(labels, dtype=torch.int).fill_(len(surrogate_models))
    for _ in range(maximum_queries):
        curr_adv = best_adv[~success]
        curr_adv.requires_grad_()
        curr_labels = labels[~success] if targets is None else targets[~success]
        # Get the surrogate model direction
        if (left_surr > 0).any():
            surrogate = choice(surrogate_models)
            best_adv.requires_grad_()
            loss = margin_loss(surrogate(curr_adv), curr_labels)
            grad = autograd.grad(loss.sum(), curr_adv, create_graph=True)[0]
            direction = grad_projection_p(grad)

        if (left_surr <= 0).any():  # random ods
            surrogate = choice(surrogate_models)
            logits = surrogate(curr_adv)
            weights = 2 * torch.rand_like(logits, dtype=torch.float) - 1
            grad_w = autograd.grad((logits * weights).sum(), curr_adv)
            direction_ods = grad_projection_p(grad_w)
            ods_sampling = (left_surr <= 0)[~success]
            direction[ods_sampling] = direction_ods[ods_sampling]
        for alpha in {-step_length, step_length}:
            with torch.no_grad():
                curr_adv_test = curr_adv - direction * alpha
                best_adv_test[~success] = proj_p_nu(curr_adv_test, inputs[~success])
                logits = victim_model(best_adv_test, ~success)
                loss_test = margin_loss(logits, curr_labels)
                criterion = loss_test[~success] > best_loss[~success]
                criterion = criterion if targets is None else ~criterion
                # Update the best adv
                best_loss[~success] = torch.where(
                    criterion, loss_test, best_loss[~success]
                )
                best_adv[~success] = torch.where(
                    criterion[:, None, None, None],
                    best_adv_test,
                    best_adv[~success],
                )
                # Update the success
                success = best_loss < 0 if targets is None else best_loss > 0
                left_surr[success] = len(surrogate_models)
        left_surr[~success] -= 1
        if success.all():
            break
    return best_adv.detach()


GFCS: TransferAttack = gfcs
