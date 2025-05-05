r"""TransferAttack wrapper for the TransferBench framework."""

from functools import partial
from typing import Optional

from torch import Tensor, nn

from transferbench.attacks_zoo.TransferAttack import transferattack
from transferbench.types import CallableModel, TransferAttack

from .TransferAttack.transferattack.utils import EnsembleModel


class LoadModelWrapper:
    r"""Partial wrapper for the TransferAttack class to load models."""

    def load_model(self, surrogate_models: list[nn.Module]) -> EnsembleModel:
        r"""Load the surrogate models into the ensemble model."""
        return EnsembleModel(surrogate_models)


def transfer_attack(
    victim_model: CallableModel,
    surrogate_models: list[nn.Module],
    inputs: Tensor,
    labels: Tensor,
    targets: Optional[Tensor] = None,
    maximum_queries: int = 1,
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
    norm = "linfity" if float(p) == float("inf") else f"l{p}"
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
    labels = [labels, targets]
    perturbations = attacker(inputs, labels).detach()
    return inputs + perturbations


ATTACK_CLASSES = [
    "ghost",
    "svre",
    "lgv",
    "mba",
    "adaea",
    "cwa",
    "smer",
    "sasd_ws",  # targeted
]

EPOCH = 100  # For a fair evaluation.

AdaEA: TransferAttack = partial(transfer_attack, attack_name="adaea", epoch=EPOCH)
CWA: TransferAttack = partial(transfer_attack, attack_name="cwa", epoch=EPOCH)
ENS: TransferAttack = partial(transfer_attack, attack_name="ens", epoch=EPOCH)
LGV: TransferAttack = partial(transfer_attack, attack_name="lgv", epoch=EPOCH)
MBA: TransferAttack = partial(transfer_attack, attack_name="mba", epoch=EPOCH)
SASD_WS: TransferAttack = partial(transfer_attack, attack_name="sasd_ws", epoch=EPOCH)
SMER: TransferAttack = partial(transfer_attack, attack_name="smer", epoch=EPOCH)
SVRE: TransferAttack = partial(transfer_attack, attack_name="svre", epoch=EPOCH)

# ensemble
