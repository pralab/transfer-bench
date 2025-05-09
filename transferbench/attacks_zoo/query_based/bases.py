r"""Batched immplementation of the bases attack from the paper.

`https://proceedings.neurips.cc/paper_files/paper/2022/file/23b9d4e18b151ba2108fb3f1efaf8de4-Paper-Conference.pdf`.
"""

from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional

import torch
from torch import Tensor

from transferbench.types import CallableModel, TransferAttack

from .utils import (
    grad_projection,
    hinge_loss,
    lp_projection,
    projected_gradient_descent,
)

LR_BOOTSTRAP = 5  # Learning rate is divided by 2 only after 5 queries


def batched_aggregated_loss(
    inputs: Tensor,
    labels: Tensor,
    surrogate_models: list[CallableModel],
    weights: Tensor,
) -> Tensor:
    r"""Compute the aggregated loss for the surrogate models.

    Parameters
    ----------
    - inputs (Tensor): The input samples of shape b x 3 x h x w
    - labels (Tensor): The labels of shape b .
    - surrogate_models (list[CallableModel]): The surrogate m surrogate models.
    - weights (Tensor): The weights of the surrogate models of shpe b x m.
    """
    loss = 0
    for id_model, model in enumerate(surrogate_models):
        loss += hinge_loss(model(inputs), labels) * weights[:, id_model].unsqueeze(1)
    return loss.sum(dim=-1)


def normalize_weights(weights: Tensor) -> Tensor:
    r"""Normalize the weights."""
    return weights.softmax(dim=-1)


def bases(
    victim_model: CallableModel,
    surrogate_models: list[CallableModel],
    inputs: Tensor,
    labels: Tensor,
    targets: Optional[Tensor] = None,
    eps: float = 16 / 255,
    p: float | str = "inf",
    maximum_queries: int = 50,
    lr: float = 5e-3,
    inner_iterations: int = 10,
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

    """
    alpha = 3 * eps / 10  # Default value from the paper

    # Initialize the weights for the surrogate models
    weights = [1.0 / len(surrogate_models)] * len(surrogate_models)
    weights = Tensor(weights).unsqueeze(0).repeat(inputs.shape[0], 1)
    weights = normalize_weights(weights)
    weights = weights.to(inputs.device)

    # Initialize the projections
    ball_projection = partial(lp_projection, eps=eps, p=p)
    dot_projection = partial(grad_projection, p=p)
    loss_fn = partial(batched_aggregated_loss, surrogate_models=surrogate_models)
    # First attempt
    adv = projected_gradient_descent(
        loss_fn=partial(loss_fn, weights=weights),
        inputs=inputs,
        x_init=inputs,
        labels=labels,
        targets=targets,
        ball_projection=ball_projection,
        dot_projection=dot_projection,
        alpha=alpha,
        inner_iterations=inner_iterations,
    )
    preds = victim_model(adv).argmax(1)  # first query is always done for all samples
    tot_forward = 1
    success = preds != labels if targets is None else preds == targets
    if success.all():
        return adv
    curr_adv = adv.clone()
    # Loop over the queries
    for query in range(maximum_queries):
        weight_id = query % len(surrogate_models)
        # Attack with
        weights_left = weights.clone()
        weights_left[:, weight_id] -= lr
        weights_left = normalize_weights(weights_left)
        # Attack with the left weights
        pre_left_curr_adv = curr_adv.clone()
        adv = projected_gradient_descent(
            loss_fn=partial(loss_fn, weights=weights_left),
            inputs=inputs[~success],
            x_init=curr_adv[~success],
            labels=labels[~success],
            targets=targets[~success] if targets is not None else None,
            ball_projection=ball_projection,
            dot_projection=dot_projection,
            alpha=alpha,
            inner_iterations=inner_iterations,
        )
        curr_adv[~success] = adv
        logits = victim_model(curr_adv, ~success)
        tot_forward += 1
        preds = logits.argmax(1)
        victim_loss_left = hinge_loss(logits, labels if targets is None else targets)
        success = preds != labels if targets is None else preds == targets

        if success.all() or tot_forward >= maximum_queries:
            break

        # Attack with the right weight
        weights_right = weights.clone()
        weights_right[:, weight_id] += lr
        weights_right = normalize_weights(weights_right)
        adv = projected_gradient_descent(
            loss_fn=partial(loss_fn, weights=weights_right),
            inputs=inputs[~success],
            x_init=pre_left_curr_adv[~success],  # using old adv if not successful
            labels=labels[~success],
            targets=targets[~success] if targets is not None else None,
            ball_projection=ball_projection,
            dot_projection=dot_projection,
            alpha=alpha,
            inner_iterations=inner_iterations,
        )
        curr_adv[~success] = adv
        # Check the success of the attack
        logits = victim_model(curr_adv, ~success)
        tot_forward += 1
        preds = logits.argmax(1)
        victim_loss_right = hinge_loss(logits, labels if targets is None else targets)
        success = preds != labels if targets is None else preds == targets
        if success.all() or tot_forward >= maximum_queries:
            break

        # Update the weights
        weights[:, weight_id] = torch.where(
            victim_loss_left < victim_loss_right,
            weights_left[:, weight_id],
            weights_right[:, weight_id],
        )
        weights = normalize_weights(weights)

        if query > LR_BOOTSTRAP and query % len(surrogate_models) == 0:
            # Normalize the weights
            lr /= 2
    return curr_adv


@dataclass
class BASESHyperparameters:
    """Hyperparameters for the BASES attack."""

    lr: float = 5e-3
    inner_iterations: int = 10


BASES: TransferAttack = partial(bases, **asdict(BASESHyperparameters()))
