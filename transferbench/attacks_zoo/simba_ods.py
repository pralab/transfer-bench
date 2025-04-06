r"""SimBA attack implementation with ODS.

The code is adapted from the repo `https://github.com/ermongroup/ODS.git`
"""

from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional

import torch
from torch import Tensor, nn

from transferbench.types import CallableModel, TransferAttack

from .utils import hinge_loss


def simba_ods(
    victim_model: CallableModel,
    surrogate_models: CallableModel,
    inputs: Tensor,
    labels: Tensor,
    targets: Optional[Tensor] = None,
    eps: float = 0.01,
    p: str | float = "inf",
    maximum_queries: int = 500,
    step_size: float = 0.01,
) -> Tensor:
    r"""SimBA attack with ODS."""
    # Define the loss function
    loss_fn = (
        nn.CrossEntropyLoss(reduction="none")
        if targets is not None
        else partial(hinge_loss, kappa=0)
    )

    adv = inputs.clone()
    logits = victim_model(adv)  # query all the samples once
    loss_best = loss_fn(logits, labels if targets is None else targets)
    loss_best = loss_best * (-1 if targets is not None else 1)
    preds = logits.argmax(1)
    success = (preds != labels) if targets is None else (preds == targets)

    tot_forwards = 1
    for _ in range(maximum_queries):
        if success.all():
            break
        random_directions = torch.rand_like(logits) * 2 - 1
        indices = torch.randint(
            len(surrogate_models),
            (inputs.size(0),),
            device=inputs.device,
            dtype=torch.long,
        )
        adv.requires_grad_()
        with torch.enable_grad():
            loss = torch.stack(
                [model(adv[~success]) for model in surrogate_models], dim=0
            )
            tot_forwards += 1
            loss = (
                loss[indices, torch.arange(inputs.shape[0]), :][~success]
                * random_directions[~success]
            ).sum()
            loss.backward()
            delta = adv.grad / adv.grad.norm(p=2)  # not for all p
        with torch.no_grad():
            for sign in [1, -1]:
                curr_adv = adv + step_size * sign * delta  # Unbouded
                curr_adv = torch.clamp(curr_adv, 0, 1)
                logits_new = victim_model(curr_adv, ~success)
                tot_forwards += 1
                loss_new = loss_fn(logits, labels if targets is None else targets)
                loss_new = loss_new * (-1 if targets is not None else 1)
                adv[loss_best < loss_new] = curr_adv[loss_best < loss_new]
                logits[loss_best < loss_new] = logits_new[loss_best < loss_new]
                if (loss_best < loss_new).all():
                    break
                loss_best[loss_best < loss_new] = loss_new[loss_best < loss_new]
        preds = logits.argmax(1)
        success = (preds != labels) if targets is None else (preds == targets)
        if success.all() or tot_forwards >= maximum_queries:
            break
    return adv.detach()


@dataclass
class SimbaODSHyperparameters:
    """SimBA ODS hyperparameters."""

    step_size: float = 0.2


SimbaODS: TransferAttack = partial(simba_ods, **asdict(SimbaODSHyperparameters()))
