r"""GAA attack implementation adapted from `https://github.com/HaloMoto/GradientAlignedAttack`."""

from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional

import torch
from torch import Tensor, nn

from transferbench.types import CallableModel, TransferAttack
import numpy as np

L2 = float("2")
Linf = float("inf")


class GradientAlignedLoss(nn.Module):  # noqa: D101
    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(
        self, outputs: Tensor, labels: Tensor, outputs_victim: Tensor
    ) -> Tensor:
        r"""Calculate the gradient aligned loss."""
        prob_victim_ = outputs_victim.softmax(-1)
        target_onehot_bool = (
            torch.nn.functional.one_hot(labels, outputs.shape[-1]).float()
        ).bool()
        prob_victim_[target_onehot_bool] = prob_victim_[target_onehot_bool] - 1.0
        w = prob_victim_ / torch.log(torch.tensor([2.0], device=outputs.device))
        return (w * outputs).sum(1)


def gaa(
    victim_model: CallableModel,
    surrogate_models: list[CallableModel],
    inputs: Tensor,
    labels: Tensor,
    targets: Optional[Tensor] = None,
    eps: Optional[float] = None,
    p: Optional[float | str] = None,
    maximum_queries: Optional[int] = None,
    steps: int = 10,
    decay: float = 1.0,
    alpha: float = 2 / 255,
) -> Tensor:
    r"""Implement the GAA attack."""
    # Initialize momentum
    momentum = torch.zeros_like(inputs)
    loss_fn = GradientAlignedLoss()

    adv_inputs = inputs.clone().detach().requires_grad_()
    outputs_victim = None
    success = torch.zeros_like(labels, dtype=torch.bool)
    maximum_queries = min(maximum_queries, steps)
    query_position = [
        int(np.floor(steps * i / maximum_queries)) for i in range(maximum_queries)
    ]
    for _ in range(steps):
        if _ in query_position:
            with torch.no_grad():
                outputs_victim = victim_model(adv_inputs, ~success)

        # evaluate the success
        pred_victim = outputs_victim.argmax(-1)
        success = pred_victim != labels if targets is None else pred_victim == targets

        if success.all():
            break

        # compute the loss

        verse = -1 if targets is not None else 1
        loc_inputs = inputs[~success]
        loc_adv_inputs = adv_inputs[~success].requires_grad_()
        loc_targets = targets[~success] if targets is not None else labels[~success]
        loc_outputs_victim = outputs_victim[~success]
        cost = sum(
            loss_fn(model(loc_adv_inputs), loc_targets, loc_outputs_victim)
            for model in surrogate_models
        ) / len(surrogate_models)

        # compute the gradient
        grad = torch.autograd.grad(cost.sum(), loc_adv_inputs)[0]
        loc_momentum = momentum[~success]
        with torch.no_grad():
            if float(p) == Linf:
                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                grad = grad + loc_momentum * decay
                momentum[~success] = grad
                loc_adv_inputs = loc_adv_inputs + verse * alpha * grad.sign()
                loc_adv_inputs = (
                    torch.clamp(loc_adv_inputs - loc_inputs, -eps, eps) + loc_inputs
                )

            elif float(p) == L2:
                grad = grad / torch.sqrt(
                    torch.sum(torch.square(grad), dim=(1, 2, 3), keepdim=True)
                )
                grad = grad + loc_momentum * decay
                momentum[~success] = grad

                loc_adv_inputs = loc_adv_inputs + verse * alpha * grad
                _norm = torch.sqrt(
                    torch.sum(
                        torch.square(loc_adv_inputs - loc_inputs),
                        dim=(1, 2, 3),
                        keepdim=True,
                    )
                )
                factor = torch.minimum(torch.tensor(1), eps / _norm)
                loc_adv_inputs = loc_inputs + (loc_adv_inputs - loc_inputs) * factor

            loc_adv_inputs = torch.clamp(loc_adv_inputs, min=0, max=1)
            adv_inputs[~success] = loc_adv_inputs

    return adv_inputs


@dataclass
class GAAHyperParams:
    r"""Hyperparameters for the GAA attack."""

    decay: int = 1
    alpha: float = 0.01


GAA: TransferAttack = partial(gaa, **asdict(GAAHyperParams()))
