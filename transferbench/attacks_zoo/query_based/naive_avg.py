r"""Naive Average Attack."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional

import torch
from torch import Tensor

from transferbench.types import CallableModel, TransferAttack

from .utils import AggregatedEnsemble


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


def naive_avg(
    victim_model: CallableModel,
    surrogate_models: list[CallableModel],
    inputs: Tensor,
    labels: Tensor,
    targets: Optional[Tensor] = None,
    maximum_queries: int = 50,
    p: float | str = "inf",
    eps: float = 16 / 255,
    alpha: float = 0.01,
    inner_iterations: int = 100,
) -> Tensor:
    r"""Implement the naive average attack."""
    ball_projection = partial(lp_projection, eps=eps, p=p)
    dot_projection = partial(grad_projection, p=p)
    loss_fn = AggregatedEnsemble(surrogate_models)
    success = torch.zeros_like(labels).bool()
    best_adv = inputs.clone()
    for q in range(maximum_queries + 1):
        if q > 0:  # first query is avoided
            # victim model use the mask to properly count the forward passes sample-wise
            preds = victim_model(best_adv, ~success).argmax(1)
            success = preds != labels if targets is None else preds == targets
        if success.all():
            break

        new_adv = projected_gradient_descent(
            loss_fn=loss_fn,
            inputs=inputs[~success],
            x_init=best_adv[~success],
            labels=labels[~success],
            targets=targets[~success] if targets is not None else None,
            ball_projection=ball_projection,
            dot_projection=dot_projection,
            alpha=alpha,
            inner_iterations=inner_iterations,
        )

        best_adv[~success] = new_adv
    return best_adv


@dataclass
class NaiveAvgHyperParams:
    r"""Hyperparameters for the naive average attack."""

    inner_iterations: int = 100
    alpha: float = 3 * 16 / 255 / 10


## Wrap the attack to be used in the evaluators
NaiveAvg: TransferAttack = partial(naive_avg, **asdict(NaiveAvgHyperParams()))

NaiveAvg10: TransferAttack = partial(
    naive_avg, **asdict(NaiveAvgHyperParams(inner_iterations=10))
)

NaiveAvg1k: TransferAttack = partial(
    naive_avg, **asdict(NaiveAvgHyperParams(inner_iterations=1000))
)  ##Very slow dont
