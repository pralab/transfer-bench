import wandb


class WandbReader:
    """Reader for Weigths & Biases data."""

    def __init__(self, entity: str, project_name: str) -> None:
        """Create WnadbReader."""
        self.entity = entity
        self.project_name = project_name
        self.connection_url = f"{entity}/{project_name}"
        self.api = wandb.Api()

    def get_runs(self):
        return self.api.runs(self.connection_url)

    def get_configs(self):
        return [r.config for r in self.get_runs()]

    def get_runs_states(self):
        return [r.state for r in self.get_runs()]

    def download_results(self):
        for r in self.get_runs():
            results = r.summary.get("results")
            table_path = results.get("path")
            r.file(table_path).download(f"results/{r.id}")
