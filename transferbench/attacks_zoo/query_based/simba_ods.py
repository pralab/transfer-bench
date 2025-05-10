r"""SimBA attack implementation with ODS.

The code is adapted from the repo `https://github.com/ermongroup/ODS.git`
"""

from functools import partial
from random import choice
from typing import Optional

import torch
from torch import Tensor, autograd, nn

from transferbench.types import CallableModel, TransferAttack

from .utils import grad_projection, hinge_loss, lp_projection

margin_loss = partial(hinge_loss, kappa=float("inf"))
cross_entropy_loss = partial(torch.nn.functional.cross_entropy, reduction="none")


def simba_ods(
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
    r"""SimbaODS attack.

    Batched version of the attack as describeed in the original paper.
    """
    best_adv = inputs.clone()
    best_adv_test = inputs.clone()
    grad_projection_p = partial(grad_projection, p=p)
    proj_p_nu = partial(lp_projection, eps=eps, p=p)
    margin_loss = partial(hinge_loss, kappa=float("inf"))
    loss_fn = margin_loss if targets is None else cross_entropy_loss
    best_loss = torch.zeros_like(labels, dtype=torch.float)
    best_loss.fill_(float("inf") if targets is not None else -float("inf"))
    success = torch.zeros_like(labels, dtype=torch.bool)
    for _ in range(0, maximum_queries, 2):
        curr_adv = best_adv[~success]
        curr_adv.requires_grad_()
        curr_labels = labels[~success] if targets is None else targets[~success]

        # Get a random surrogate model direction
        surrogate = choice(surrogate_models)
        logits = surrogate(curr_adv)
        weights = 2 * torch.rand_like(logits, dtype=torch.float) - 1
        grad_w = autograd.grad((logits * weights).sum(), curr_adv)[0]
        direction = grad_projection_p(grad_w)
        for alpha in {-step_length, step_length}:
            with torch.no_grad():
                curr_adv_test = curr_adv + direction * alpha
                best_adv_test[~success] = proj_p_nu(inputs[~success], curr_adv_test)
                loss_test = loss_fn(victim_model(best_adv_test, ~success), curr_labels)
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
                ).clamp(0, 1)
                # Update the success
                success = best_loss > 0 if targets is None else best_loss < 0
        if success.all():
            break
    return best_adv.detach()


SimbaODS: TransferAttack = partial(simba_ods, step_length=2.0)
