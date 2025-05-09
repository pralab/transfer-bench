r"""TransferAttack wrapper for the TransferBench framework."""

from typing import Optional

import torch
from torch import Tensor, nn

from transferbench.attacks_zoo.externals.TransferAttack import transferattack
from transferbench.types import CallableModel

from transferbench.attacks_zoo.externals.TransferAttack.transferattack.utils import (
    EnsembleModel,
)


class LoadModelWrapper:
    r"""Partial wrapper for the TransferAttack class to load models."""

    def load_model(self, surrogate_models: list[nn.Module]) -> EnsembleModel:
        r"""Load the surrogate models into the ensemble model."""
        return EnsembleModel(surrogate_models)

    def DI(self, inputs: Tensor, **kwargs) -> Tensor:  # SASD_WS workaround
        r"""Override the DI function to disable it."""
        # This is a placeholder for the DI function.
        return inputs


def transfer_attack(
    victim_model: CallableModel,
    surrogate_models: list[nn.Module],
    inputs: Tensor,
    labels: Tensor,
    targets: Optional[Tensor] = None,
    maximum_queries: int = 0,
    epoch: int = 100,
    p: float | str = "inf",
    eps: float = 16 / 255,
    attack_name: str = "ens",
    attack_kwargs: dict | None = None,
) -> Tensor:
    r"""Implement a wrapper for TransferAttack attacks."""
    attack_cls = transferattack.load_attack_class(attack_name)

    # replace load model function with custom fn to bypass model names param
    class TransferAttackWrapper(LoadModelWrapper, attack_cls): ...

    if attack_kwargs is None:
        attack_kwargs = {}
    # Common arguments
    norm = "linfty" if float(p) == float("inf") else f"l{p}"
    targeted = targets is not None

    # Initialize the attack
    attacker = TransferAttackWrapper(
        model_name=surrogate_models,
        targeted=targeted,
        epsilon=eps,
        epoch=epoch,
        norm=norm,
        **attack_kwargs,
    )
    if attack_name in ["ens", "lgv", "mba", "sasd_ws", "smer"]:
        labels = torch.stack([labels, targets]) if targeted else labels
    elif attack_name in ["adaea", "cwa", "svre"]:
        labels = labels if not targeted else targets
    perturbations = attacker(inputs, labels).detach()
    return inputs + perturbations
