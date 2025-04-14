from typing import Literal
import torch
from torch.utils.data import Subset
import wandb
from benchmark.wandb_logger import WandbLogger
from transferbench.attack_evaluation import AttackEval
from transferbench.scenarios import AttackScenario
from transferbench.wrappers.attack_wrapper import HyperParameters
from transferbench.datasets.datasets import ImageNetT
import logging


def attack(scenario, batch_size=128):
    DEFAUlT_DEVICE: Literal["cuda"] | Literal["cpu"] = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    """Evaluate Transferability from surrogate to victim on a specific scenario."""
    evaluator = AttackEval(transfer_attack="NaiveAvg")

    if scenario is not None:
        evaluator.set_scenarios(scenario)
    return evaluator.run(batch_size=batch_size, device=DEFAUlT_DEVICE)


def main():
    scenario = AttackScenario(
        hp=HyperParameters(maximum_queries=3, p="inf", eps=16 / 255),
        victim_model="resnet18",
        surrogate_models=["resnet18", "resnet18"],
        dataset=Subset(ImageNetT(), indices=list(range(4))),
    )

    logging.basicConfig(
        filename="outputs.log",
        encoding="utf-8",
        level=logging.INFO,
        filemode="w",
    )

    results = attack(scenario, batch_size=3)
    with WandbLogger("transfer-bench") as w:
        w.upload(results)


if __name__ == "__main__":
    main()
