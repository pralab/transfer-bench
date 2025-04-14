r"""Logger for Weights & Biases."""

from typing import Optional

import wandb


class WandbLogger:
    r"""Custom logger for Weights & Biases."""

    def __init__(
        self,
        project_name: str,
        entity: Optional[str] = None,
    ) -> None:
        r"""Wandb logger for Weights & Biases.

        Parameters
        ----------
        - project_name (str): Name of the project.
        - entity (str, optional): Name of the entity. Defaults to None.
        - config (dict, optional): Configuration dictionary. Defaults to None.
        """
        self.entity = entity
        self.project_name = project_name

    def __enter__(self) -> "WandbLogger":
        """Open the wandb connection."""
        wandb.login()
        return self

    def upload(self, results: list) -> None:
        r"""Add rows to table."""
        for scenario in results:
            sample_idx = 0
            scenario_results = scenario.pop("results")
            scenario["dataset"] = (
                scenario["dataset"]
                if isinstance(scenario["dataset"], str)
                else "Custom"
            )
            wandb.init(project=self.project_name, entity=self.entity, config=scenario)
            columns = ["id", "predictions", "label", "target", "success", "queries"]
            table = wandb.Table(columns=columns)
            for batch in scenario_results:
                advs = batch.pop("adv")
                for i in range(advs.shape[0]):
                    table.add_data(
                        sample_idx,
                        batch["predictions"][i].item(),
                        batch["labels"][i].item(),
                        batch["targets"][i].item(),
                        batch["success"][i].item(),
                        batch["queries"][i].item(),
                    )
                    sample_idx += 1
            wandb.log({"results": table})

    def __exit__(self, *args) -> None:
        """Close the wandb connection."""
        wandb.finish()
