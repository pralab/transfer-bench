r"""Handle results stored in Weights & Biases."""

from pathlib import PosixPath
from types import TracebackType
from typing import Optional

import torch
import wandb
from wandb.apis.public import Run

from .config import cfg
from pandas import DataFrame


class WandbReader:
    """Reader for Weigths & Biases data."""

    def __init__(
        self, entity: str = cfg.project_entity, project_name: str = cfg.project_name
    ) -> None:
        """Create WnadbReader."""
        self.entity = entity
        self.project_name = project_name
        self.connection_url = f"{entity}/{project_name}"
        self.api = wandb.Api()

    def get_runs(self) -> list[Run]:
        return self.api.runs(self.connection_url)

    def get_configs(self):
        return [r.config for r in self.get_runs()]

    def get_runs_states(self):
        return [r.state for r in self.get_runs()]

    def download_results(self, root: PosixPath) -> None:
        """Download the results from Weights & Biases."""
        for r in self.get_runs():
            table_name = self.connection_url + f"/run-{r.id}-numerical-results:latest"
            try:
                table = self.api.artifact(table_name)
                table.download(root / "tables" / r.id)
            except Exception as e:
                _ = f"Error while downloading {table_name}.\n {e}"
            finally:
                pass


class WandbRun:
    r"""Custom run for Weights & Biases."""

    def __init__(self, run_id: str, config: str, path: str) -> None:
        r"""Wandb run for Weights & Biases.

        Parameters
        ----------
        - run_id (str): ID of the run.
        """
        self.run_id = run_id
        self.config = config
        self.dir = path
        self.project_name = cfg.project_name
        self.entity = cfg.project_entity

    def __enter__(self) -> "WandbRun":
        """Open the wandb connection."""
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            id=self.run_id,
            resume="allow",
            config=self.config,
            dir=self.dir,
        )
        return self

    def upload_data(self, file_path: PosixPath) -> None:
        """Save data to wandb as an artifact."""
        artifact = wandb.Artifact(
            name=self.run_id + "-" + file_path.stem,
            type="full-results",
            description="Results of the attack",
            metadata=self.config,
        )
        artifact.add_file(file_path)
        artifact.save()
        wandb.log_artifact(artifact)

    def get_data(self, artifact_name: str, art_type: str) -> None:
        """Get data from wandb as an artifact."""
        artifact = self.run.use_artifact(artifact_name, type=art_type)
        artifact_dir = PosixPath("./tmp")
        data_path = artifact.file(artifact_dir)
        return torch.load(data_path, weights_only=False)

    def upload_table(self, dataframe: DataFrame) -> None:
        """Update the wandb table with new data."""
        table = wandb.Table(dataframe=dataframe)
        table.metadata = self.config
        wandb.log({"numerical-results": table})

    def log(self, **data) -> None:
        """Log data to wandb."""
        wandb.log(data)

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Ensure proper run termination with error logging."""
        if exc_type is not None:
            wandb.log(
                {
                    "error/type": exc_type.__name__,
                    "error/value": str(exc_value),
                    "error/traceback": str(traceback),
                }
            )

        wandb.finish(exit_code=1 if exc_type else 0)
