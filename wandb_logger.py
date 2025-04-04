r"""Logger for Weights & Biases."""

import logging
from typing import Optional

import wandb


class WandbHandler(logging.Handler):
    r"""Custom logging handler for Weights & Biases."""

    def __init__(
        self,
        project_name: str,
        entity: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        r"""WandbHandler handler for Weights & Biases.

        Parameters
        ----------
        - project_name (str): Name of the project.
        - entity (str, optional): Name of the entity. Defaults to None.
        - config (dict, optional): Configuration dictionary. Defaults to None.
        """  # to do Provide a bit of infos on the enitity and config
        super().__init__()
        wandb.init(project=project_name, entity=entity, config=config)
        columns = ["scores", "predictions", "asr", "queries"]
        self.table = wandb.Table(columns=columns)

    def emit(self, record: str) -> None:
        r"""Emit a log record."""
        new_table = wandb.Table(columns=self.table.columns, data=self.table.data)
        new_table.add_data(*record.msg)
        wandb.log({"data": new_table}, commit=False)
        self.table = new_table

    @staticmethod
    def __del__() -> None:  # noqa: D105
        wandb.finish()
