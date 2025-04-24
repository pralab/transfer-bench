r"""Handle results stored in Weights & Biases."""

from pathlib import PosixPath
from types import TracebackType
from typing import Optional

import wandb
from wandb.apis.public import Run

from .config import cfg


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

    def download_results(self):
        msg = "This function is not implemented yet."
        raise NotImplementedError(msg)
        for r in self.get_runs():
            results = r.summary.get("results")
            table_path = results.get("path")
            r.file(table_path).download(f"results/{r.id}")


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
        self.table = None

    def __enter__(self) -> "WandbRun":
        """Open the wandb connection."""
        wandb.init(
            project=cfg.project_name,
            entity=cfg.project_entity,
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

    def __init_table__(self, columns: list[str]) -> None:
        """Initialize the wandb table."""
        self.table = wandb.Table(
            columns=columns,
        )
        self.table.metadata = self.config

    def update_table(self, **data) -> None:
        """Update the wandb table with new data."""
        if self.table is None:
            self.__init_table__(columns=list(data.keys()))
        lenght = len(next(iter(data.values())))
        for idx in range(lenght):
            row = [data[key][idx].item() for key in data]
            self.table.add_data(*row)
        wandb.log({"numerical-results": self.table})

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
                    "error/traceback": traceback.format_exc() if traceback else None,
                }
            )

        wandb.finish(exit_code=1 if exc_type else 0)
