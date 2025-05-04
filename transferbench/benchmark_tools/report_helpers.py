r"""Report helpers for transferbench."""

import json
from pathlib import Path

import pandas as pd

from .config import cfg
from .run_helpers import get_filtered_runs
from .wandb_helpers import WandbReader


MODEL_NAMES = {
    "resnext101_32x8d": "\\resnext{101}",
    "vgg19": "\\vgg{19}",
    "vit_b_16": "\\vit{16}",
    "Amini2024MeanSparse_Swin-L": "\\amini",
    "Xu2024MIMIR_Swin-L": "\\mimir",
    "imagenet_resnet50_pubdef": "\\pubdef",
}
COLUMN_NAMES = {"avg_success": "ASR", "avg_queries": "$\\bar q$"}
SCENARIO_NAMES = {"etero": "\\etero", "omeo": "\\omeo", "robust": "\\robust"}


def collect_results(download: bool) -> list[dict]:
    r"""Collect the results from the runs.

    Args:
        run_ids (list[str]): List of run ids to collect results from.
        download (bool): Whether to download the results or not. Defaults to False.
    """
    df_runs = get_filtered_runs("finished", 'campaign != "debug"')
    # Get the run ids
    run_ids = df_runs["id"].tolist()
    # Initialize the WandbReader
    reader = WandbReader()
    # check if directory exists
    report_dir = Path(cfg.report_root)
    if not report_dir.exists() or download:
        # create the directory
        report_dir.mkdir(parents=True, exist_ok=True)
        reader.download_results(report_dir)
    ## Aggregate all the runs
    results = []
    for run_id in run_ids:
        # open the json file in id directory
        run_dir = report_dir / "tables" / run_id
        # check if the directory exists
        if run_dir.exists():
            # open the json file
            table = run_dir / "numerical-results.table.json"
            # load the json file
            with open(table, "r") as f:
                # read the json file
                table = f.read()
                json_data = json.loads(table)
            columns = json_data["columns"]
            data = json_data["data"]
            df_run = pd.DataFrame(
                data,
                columns=columns,
            )
            df_run["id"] = run_id
            # print(f"Loaded {len(df_run)} rows from {run_id}")
            results.append(df_run)
    df_results = pd.concat(results, ignore_index=True)
    # merge with configurations
    return df_results.merge(
        df_runs,
        how="left",
        on="id",
    )


def make_tabulars(df_results: pd.DataFrame) -> list[pd.DataFrame]:
    r"""Make a latex and markdown tabular from the results.

    Args:
        df_results (pd.DataFrame): DataFrame with the results.

    Return
        str: list of pandas dataframes.
    """
    datasets = df_results["dataset"].unique()
    tabulars = []

    for dataset in datasets:
        df_loc = df_results[df_results["dataset"] == dataset]
        df_loc = df_loc[df_loc["campaign"].isin(SCENARIO_NAMES.keys())]

        agg_df = (
            df_loc.groupby(["attack", "campaign", "victim_model"])
            .agg(
                avg_success=("success", "mean"),
                avg_queries=("queries", "mean"),
                count=("success", "count"),
            )
            .reset_index()
        )
        agg_df.avg_success *= 100
        agg_df = agg_df.rename(
            columns={"campaign": "scenario", "victim_model": "victim"}
        )

        pivot_df = agg_df.pivot_table(
            index="attack",
            columns=["scenario", "victim"],
            values=["avg_success", "avg_queries"],
        )

        # Rename MultiIndex columns using the mapping dictionaries
        new_columns = []
        for col in pivot_df.columns:
            metric, scenario, model = col
            new_metric = COLUMN_NAMES.get(metric, metric)
            new_scenario = SCENARIO_NAMES.get(scenario, scenario)
            new_model = MODEL_NAMES.get(model, model)
            new_columns.append((new_scenario, new_model, new_metric))

        pivot_df.columns = pd.MultiIndex.from_tuples(new_columns)

        # Rearrange the columns to match the order: scenario, model, {ASR, \bar q}
        def metric_key(row):
            return 0 if row[2] == "ASR" else 1

        pivot_df = pivot_df[
            sorted(
                sorted(
                    sorted(pivot_df.columns, key=metric_key), key=lambda row: row[1]
                ),
                key=lambda row: row[0],
            )
        ]

        # Write to LaTeX
        pivot_df.to_latex(
            Path(cfg.report_root) / f"tabular_{dataset}.tex",
            caption=f"Results for {dataset}",
            label=f"tab:{dataset}",
            float_format="%.2f",
            column_format="l" + "|cc" * (len(pivot_df.columns) // 2) + "|",
            na_rep="-",
            escape=False,  # Important: allows LaTeX commands to be rendered properly
        )

        tabulars.append(pivot_df)
    return tabulars
