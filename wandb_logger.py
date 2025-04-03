"""Logger for Weights & Biases"""

import logging
import wandb


class WandbHandler(logging.Handler):
    def __init__(self, project_name: str, entity: str = None, config: dict = None):
        super().__init__()
        wandb.init(project=project_name, entity=entity, config=config)
        columns = ["scores", "predictions", "asr", "queries"]
        self.table = wandb.Table(columns=columns)

    def emit(self, record):
        new_table = wandb.Table(columns=self.table.columns, data=self.table.data)
        new_table.add_data(*record.msg)
        wandb.log({"data": new_table}, commit=False)
        self.table = new_table

    @staticmethod
    def __del__():
        wandb.finish()
