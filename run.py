import torch
from torch.utils.data import Subset
from transferbench.attack_evaluation import AttackEval
from transferbench.scenarios import AttackScenario
from transferbench.wrappers.attack_wrapper import HyperParameters
from transferbench.datasets.datasets import ImageNetT
import logging
import wandb

from wandb_logger import WandbHandler


def attack(scenario, batch_size=128):
    DEFAUlT_DEVICE: Literal["cuda"] | Literal["cpu"] = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    """Evaluate Transferability from surrogate to victim on a specific scenario."""
    evaluator = AttackEval(transfer_attack="NaiveAvg")

    if scenario is not None:
        evaluator.set_scenarios(scenario)
    result = evaluator.run(batch_size=batch_size, device=DEFAUlT_DEVICE)
    print("Evaluation completed.")
    return result


def main():
    scenario = AttackScenario(
        hp=HyperParameters(maximum_queries=4, p="inf", eps=16 / 255),
        victim_model="vgg19",
        surrogate_models=["resnet18", "resnet18"],
        dataset=Subset(ImageNetT(), indices=list(range(3))),
    )

    logging.basicConfig(
        filename="outputs.log",
        encoding="utf-8",
        level=logging.INFO,
        filemode="w",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    wandb_handler = WandbHandler(project_name="my_project", config=scenario.__dict__)
    logger.addHandler(wandb_handler)

    attack(scenario)


if __name__ == "__main__":
    main()
