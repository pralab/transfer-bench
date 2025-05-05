from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional

from torch import Tensor

from transferbench.attacks_zoo.TransferAttack import transferattack
from transferbench.attacks_zoo.TransferAttack.transferattack.utils import EnsembleModel
from transferbench.types import CallableModel, TransferAttack


class LoadModelWrapper(object):
    def load_model(self, surrogate_models) -> EnsembleModel:
        return EnsembleModel(surrogate_models)


def transfer_attack(
    victim_model: CallableModel,
    surrogate_models: list[CallableModel],
    inputs: Tensor,
    labels: Tensor,
    targets: Optional[Tensor] = None,
    maximum_queries: int = 1,
    p: float | str = "inf",
    eps: float = 16 / 255,
    alpha: float = 0.01,
    inner_iterations: int = 100,
    attack_name: str = "ens",
    attack_kwargs: dict | None = None,
) -> Tensor:
    r"""Implement a wrapper for TransferAttack attacks."""
    attack_cls = transferattack.load_attack_class(attack_name)

    # replace load model function with custom fn to bypass model names param
    class TransferAttackWrapper(LoadModelWrapper, attack_cls): ...

    if attack_kwargs is None:
        attack_kwargs = {}
    attacker = TransferAttackWrapper(
        model_name=surrogate_models,
        targeted=True,
        epsilon=eps,
        alpha=alpha,
        epoch=inner_iterations,
        device=None,  # FIXME add device from external APIs?
        **attack_kwargs,
    )
    labels = [labels, targets]
    perturbations = attacker(inputs, labels).detach()
    return inputs + perturbations


@dataclass
class TransferAttackEnsHyperParams:
    r"""Hyperparameters for the ENS attack from TransferAttack."""

    attack_name = "ens"
    inner_iterations: int = 100
    alpha: float = 3 * 16 / 255 / 10

    # todo add these to interface
    decay = 1.0
    targeted = False
    random_start = False
    norm = "linfty"
    loss = "crossentropy"


## Wrap the attack to be used in the evaluators
ens: TransferAttack = partial(transfer_attack, **asdict(TransferAttackEnsHyperParams()))
# ensemble
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
