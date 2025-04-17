r"""Module for providing functionality to collect and format runs."""

import pandas as pd
import torch
from transferbench.attack_evaluation import AttackEval
from transferbench.types import AttackResult

from .config import COLUMNS, LOCAL_RESULT_ROOT
from .utils import get_config_from_run, get_path_from_run, get_run_list
from .wandb_helpers import WandbReader, WandbRun


def collect_runs() -> pd.DataFrame:
    r"""Format the runs for display."""
    # Initialize the WandbReader
    wandb_connections = WandbReader()
    remote_runs = wandb_connections.get_runs()
    local_runs = get_run_list()
    df_remote_runs = pd.DataFrame(
        [{**run.config, "status": run.state} for run in remote_runs]
    )
    ## get local pandas dataframe
    df_local_runs = pd.DataFrame([get_config_from_run(run) for run in local_runs])
    # Merge the two dataframes
    if df_remote_runs.empty:
        df_remote_runs = pd.DataFrame(columns=COLUMNS)
        # df_remote_runs = df_local_runs.copy().iloc[:10, :]  # noqa: ERA001
        # df_remote_runs["status"] = "Finished"  # noqa: ERA001
    cols_to_merge = [col for col in COLUMNS if col != "status"]
    df_runs = df_local_runs.merge(
        df_remote_runs,
        how="outer",
        on=cols_to_merge,
        indicator=True,
    )
    df_runs.status = df_runs.status.fillna("Missing")
    return df_runs.loc[:, COLUMNS].sort_values(by="status").reset_index(drop=True)


def run_single_scenario(run_id: str, batch_size: int, device: torch.device) -> None:
    r"""Run a single scenario."""
    # Get the run list
    run_list = get_run_list()
    # Get the run
    run = next((run for run in run_list if run.id == run_id), None)
    if run is None:
        msg = f"Run with id {run_id} not found."
        raise ValueError(msg)
    # Get the path to the run
    evaluator = AttackEval(transfer_attack=run.attack)
    # Run the evaluation
    results = evaluator.evaluate_scenario_(
        scenario=run.scenario, batch_size=batch_size, device=device
    )
    config = get_config_from_run(run)
    path = get_path_from_run(run)
    numerical_res_names = list(AttackResult.__required_keys__)
    _ = numerical_res_names.pop("adv")
    _ = numerical_res_names.pop("logits")
    df_results = pd.DataFrame([], columns=numerical_res_names)
    with WandbRun(run_id=run_id, config=config, path=path) as w:
        for part, res in enumerate(results):
            # Save the results to a pth local file
            data_file_name = LOCAL_RESULT_ROOT / path / f"part{part}.pth"
            torch.save(res, data_file_name)
            # Save the results to wandb
            w.upload_data(data_file_name)
            numerical_data = {
                key: value
                for key, value in results.items()
                if key in numerical_res_names
            }
            # Update wandb table
            w.update_table(**numerical_data)
            # Updata dataframe and save it
            df_loc = pd.DataFrame(numerical_data)
            df_results = pd.concat([df_results, df_loc], ignore_index=True)
            df_results.to_csv(
                LOCAL_RESULT_ROOT / path / "results.csv",
                index=False,
            )
    return df_results
