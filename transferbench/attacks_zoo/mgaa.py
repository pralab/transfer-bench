r"""Meta Gradient Attack implementation.

This implements the Meta Gradient Attack (MGAA) as described in the paper
`https://openaccess.thecvf.com/content/ICCV2021/papers/Yuan_Meta_Gradient_Adversarial_Attack_ICCV_2021_paper.pdf`
"""

from dataclasses import dataclass
from functools import partial
from random import sample
from typing import Optional

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy

from transferbench.types import CallableModel, TransferAttack


def ensemble_loss(
    models: list[CallableModel],
    inputs: Tensor,
    labels: Tensor,
) -> Tensor:
    r"""Compute the ensemble loss."""
    logits = torch.stack([model(inputs) for model in models])
    ens_logits = logits.mean(0)  # weights are all 1/N in the paper
    return cross_entropy(ens_logits, labels)


def mgaa(
    victim_model: CallableModel,
    surrogate_models: list[CallableModel],
    inputs: Tensor,
    labels: Tensor,
    targets: Optional[Tensor] = None,
    eps: Optional[float] = None,
    p: Optional[float | str] = None,
    maximum_queries: Optional[int] = None,
    alpha: float = 1,
    beta: float = 0.4,
    n: int = 5,
    T: int = 40,  # noqa: N803
    K: int = 5,  # noqa: N803
) -> Tensor:
    r"""Meta Gradient Attack."""
    x_i = inputs.clone()
    beta = eps / T
    n = min(n, len(surrogate_models) - 1)
    assert p == "inf", "MGAA only supports p=inf"
    for _ in range(T):
        # Step 2: Randomly sample n+1 models from the pool
        task_models = sample(surrogate_models, n + 1)
        ensemble_models = task_models[:-1]  # first n models
        final_model = task_models[-1]  # the (n+1)-th model
        xi_j = x_i.clone()  # Step 3: initialize inner loop input
        xi_j.requires_grad_()
        for _ in range(K):
            # Step 5: compute ensemble loss
            loss = (
                ensemble_loss(ensemble_models, xi_j, labels)
                if targets is None
                else -ensemble_loss(ensemble_models, xi_j, targets)
            )
            # Step 6: gradient ascent step
            grad = torch.autograd.grad(loss, xi_j)[0]
            # Step 7: update xi_j
            with torch.no_grad():
                xi_j = xi_j + alpha * torch.sign(grad)
            xi_j.requires_grad_()
        # Step 8: compute loss under final model
        final_loss = (
            cross_entropy(final_model(xi_j), labels)
            if targets is None
            else -cross_entropy(final_model(xi_j), targets)
        )

        # Step 9: model transfer step
        final_grad = torch.autograd.grad(final_loss, xi_j)[0]
        with torch.no_grad():
            xi_mt = xi_j + beta * torch.sign(final_grad)

        # Step 10: update x for the next iteration
        x_i = x_i + (xi_mt - xi_j)
        x_i = torch.clamp(x_i, 0, 1)  # Not in the paper, but needed
    return x_i


@dataclass
class MGGAHyperParams:
    r"""Hyperparameters for the Meta Gradient Attack."""

    alpha: float = 1
    n: int = 5
    T: int = 4
    K: int = 5


MGAA: TransferAttack = partial(mgaa, **vars(MGGAHyperParams()))
