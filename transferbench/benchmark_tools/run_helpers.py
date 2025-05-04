r"""Module for providing functionality to collect and format runs."""

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from pandas import DataFrame
from wandb.errors import CommError

from transferbench.attack_evaluation import AttackEval
from transferbench.types import AttackResult, AttackScenario

from .config import cfg
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
        df_remote_runs = pd.DataFrame(columns=cfg.columns)
        # df_remote_runs = df_local_runs.copy().iloc[:10, :]  # noqa: ERA001
        # df_remote_runs["status"] = "Finished"  # noqa: ERA001
    cols_to_merge = [col for col in cfg.columns if col not in {"status", "available"}]
    df_runs = df_local_runs.merge(
        df_remote_runs,
        how="outer",
        on=cols_to_merge,
        indicator=True,
    )
    df_runs = df_runs.rename(columns={"_merge": "available"})
    df_runs.available = df_runs.available.replace(
        {"left_only": True, "right_only": False, "both": True}
    )
    df_runs.status = df_runs.status.fillna("missing")
    return df_runs.loc[:, cfg.columns].sort_values(by="status").reset_index(drop=True)


def init_numerical_results(run: WandbRun, resume: bool) -> tuple[DataFrame, int]:
    r"""Load partial results from a run.

    Args:
        run (WandbRun): The run to load the results from.
        resume (bool): Whether to resume prevous data or not.

    Returns:
        tuple: A tuple containing the results and the last part id.
    """
    # Load the results
    part_id = 0
    numerical_res_names = list(AttackResult.__required_keys__)
    numerical_res_names.remove("adv")
    numerical_res_names.remove("logits")
    df_results = pd.DataFrame([], columns=numerical_res_names)
    wandb_path = f"{run.entity}/{run.project_name}/"
    art_type = "full-results"

    while resume:
        artifact_name = wandb_path + f"{run.run_id}-full-batch{part_id}:latest"
        try:
            data = run.get_data(artifact_name, art_type=art_type)
        except CommError:
            break
        numerical_data = {
            key: value for key, value in data.items() if key in numerical_res_names
        }
        # Updata dataframe and save it
        df_loc = pd.DataFrame(numerical_data)
        df_results = pd.concat([df_results, df_loc], ignore_index=True)
        # continue cycle if not error
        part_id += 1
    return df_results, part_id


def init_dataset(scenario: AttackScenario, start_from: int = 0) -> None:
    r"""Initialize the dataset for the scenario.

    Args:
        scenario (AttackScenario): The scenario to initialize.
        start_from (int): The starting index for the dataset. Defaults to 0.
    """
    # Initialize the dataset
    from torch.utils.data import Subset

    from transferbench.datasets import datasets

    dataset = scenario.dataset
    dataset = getattr(datasets, dataset)() if isinstance(dataset, str) else dataset()
    if start_from > 0:
        dataset = Subset(dataset, range(start_from, len(dataset)))
    scenario.dataset = dataset


def run_single_scenario(
    run_id: str, batch_size: int, device: torch.device, resume: bool = True
) -> None:
    r"""Run a single scenario.

    Args:
        run_id (str): The id of the run to evaluate.
        batch_size (int): The batch size to use for the evaluation.
        device (torch.device): The device to use for the evaluation.
        resume (bool): Whether to resume the run or not. Defaults to True.
    """
    # Get the run list
    run_list = get_run_list()
    # Get the run
    run = next((run for run in run_list if run.id == run_id), None)
    if run is None:
        msg = f"Run with id {run_id} not found."
        raise ValueError(msg)
    # Get the path to the run
    evaluator = AttackEval(transfer_attack=run.attack)
    config = get_config_from_run(run)
    path = Path(cfg.results_root) / get_path_from_run(run)

    with WandbRun(run_id=run_id, config=config, path=path) as w:
        df_results, start_part_id = init_numerical_results(w, resume)
        # reduce dataset to the remaining parts
        init_dataset(run.scenario, start_from=len(df_results))
        # Initialize the evaluation
        results = evaluator.evaluate_scenario_(
            scenario=run.scenario,
            batch_size=batch_size,
            device=device,
        )
        for part_id, res in enumerate(results, start=start_part_id):
            # Save the results to a pth local file
            data_file_name = path / f"full-batch{part_id}.pth"
            torch.save(res, data_file_name)
            # Save the results to wandb
            w.upload_data(data_file_name)
            numerical_data = {
                key: value for key, value in res.items() if key in df_results.columns
            }
            # Updata dataframe and save it
            df_loc = pd.DataFrame(numerical_data)
            df_results = pd.concat([df_results, df_loc], ignore_index=True)
            # Update wandb table
            w.upload_table(df_results)
            csv_path = path / "results.csv"
            df_results.to_csv(
                csv_path,
                index=False,
            )
            w.log(
                asr=df_results["success"].mean(),
                avg_q=df_results[df_results.success == 1]["queries"].mean(),
            )
        # Save final results to wandb
        w.upload_table(df_results)
    return df_results


def get_filtered_runs(status: str = "all", query: Optional[str] = None) -> DataFrame:
    r"""Retrieve and filter run information based on the specified criteria."""
    runs = collect_runs()
    if status != "all":
        runs = runs[runs["status"] == status]
    if query is not None:
        runs = runs.query(query)
    return runs
