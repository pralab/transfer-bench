r"""Report helpers for transferbench."""

import json
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import cfg
from .run_helpers import get_filtered_runs
from .wandb_helpers import WandbReader

MODEL_NAMES = {
    "resnext101_32x8d": "\\resnext{101}",
    "imagenet_resnet50_pubdef": "\\pubdef",
    "vgg19": "\\vgg{19}",
    "Amini2024MeanSparse_Swin-L": "\\amini",
    "vit_b_16": "\\vit{16}",
    "Xu2024MIMIR_Swin-L": "\\mimir",
    "cifar10_vgg19_bn": "\\vgg{19-bn}",
    "cifar10_resnet56": "\\resnet{56}",
    "cifar10_vit_b16": "\\vit{16}",
    "cifar10_beit_b16": "\\beit{16}",
    "Peng2023Robust": "\\peng",
    "Bartoldson2024Adversarial_WRN-94-16": "\\bartold",
}
PLOT_MODEL_NAMES = {
    "resnext101_32x8d": "resnext101",
    "imagenet_resnet50_pubdef": "pubdef-resnet-50",
    "vgg19": "vgg-19",
    "Amini2024MeanSparse_Swin-L": "Amini-Swin-L",
    "vit_b_16": "vit-16/b",
    "Xu2024MIMIR_Swin-L": "Mimir-Swin-L",
    "cifar10_vgg19_bn": "vgg-19-bn",
    "cifar10_resnet56": "resnet-56",
    "cifar10_vit_b16": "Vit-16/t",
    "cifar10_beit_b16": "BeIT-16/b",
    "Peng2023Robust": "Peng2023Robust",
    "Bartoldson2024Adversarial_WRN-94-16": "Barto-WRN-94-16",
}
METRIC_NAME = {"avg_success": "ASR", "avg_queries": "$\\bar q$"}
SCENARIO_NAMES = {"omeo": "\\omeo", "etero": "\\etero", "robust": "\\robust"}
ATTACK_NAMES = {
    "BASES": "BASES",
    "DSWEA": "DSWEA",
    "GAA": "GAA",
    "NaiveAvg": "NaiveAvg",
    "NaiveAvg10": "NaiveAvg10",
    "ENS": "ENS",
    "CWA": "CWA",
    "LGV": "LGV",
    "MBA": "MBA",
    "SASD_WS": "SASD\\_WS",
    "SVRE": "SVRE",
}


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
                with Path.open(table, "r") as f:
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
                results.append(df_run)
        df_results = pd.concat(results, ignore_index=True)
        # merge with configurations
        df_results = df_results.merge(
            df_runs,
            how="left",
            on="id",
        )
        # Save the file for speedup
        df_results.to_csv(report_dir / "results.csv", index=False)
        return df_results
    return pd.read_csv(report_dir / "results.csv")


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
        # filter out unwanted campaigns
        df_loc = df_loc[df_loc["campaign"].isin(SCENARIO_NAMES.keys())]
        # Filter unwnanted attacks
        df_loc = df_loc[df_loc["attack"].isin(ATTACK_NAMES.keys())]
        # Replace queries with nan when success is 0
        df_loc.loc[df_loc["success"] == 0, "queries"] = float("nan")

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
        agg_df: pd.DataFrame = agg_df.rename(
            columns={"campaign": "scenario", "victim_model": "victim"}
        )

        pivot_df = agg_df.pivot_table(
            index="attack",
            columns=["scenario", "victim"],
            values=["avg_success", "avg_queries"],
            dropna=False,
        )

        # Rearrange the columns

        def metric_key(row: tuple) -> int:
            return list(METRIC_NAME.keys()).index(row[0])

        def scenario_key(row: tuple) -> int:
            return list(SCENARIO_NAMES.keys()).index(row[1])

        def model_key(row: tuple) -> int:
            return list(MODEL_NAMES.keys()).index(row[2])

        pivot_df = pivot_df[
            sorted(
                sorted(sorted(pivot_df.columns, key=metric_key), key=scenario_key),
                key=model_key,
            )
        ]
        # Rename MultiIndex columns using the mapping dictionaries
        new_columns = []
        for col in pivot_df.columns:
            metric, scenario, model = col
            new_metric = METRIC_NAME.get(metric, metric)
            new_scenario = SCENARIO_NAMES.get(scenario, scenario)
            new_model = MODEL_NAMES.get(model, model)
            new_columns.append((new_model, new_scenario, new_metric))

        pivot_df.columns = pd.MultiIndex.from_tuples(new_columns)

        actual_attacks = [attack for attack in ATTACK_NAMES if attack in pivot_df.index]
        pivot_df = pivot_df.loc[actual_attacks, :].rename(index=ATTACK_NAMES)

        # Write to LaTeX
        pivot_df.to_latex(
            Path(cfg.report_root) / f"tabular_{dataset}.tex",
            caption=f"Results for {dataset}",
            label=f"tab:{dataset}",
            float_format="%.1f",
            column_format="l" + "|cc" * (len(pivot_df.columns) // 2) + "|",
            multicolumn_format="c",
            na_rep="-",
            escape=False,  # Important: allows LaTeX commands to be rendered properly
        )

        tabulars.append(pivot_df)
    return tabulars


def make_barplots(df_results: pd.DataFrame) -> None:
    for dataset in df_results["dataset"].unique():
        df_loc = df_results[df_results["dataset"] == dataset]
        df_loc = df_loc[df_loc["campaign"].isin(SCENARIO_NAMES.keys())]
        # Replace queries with nan when success is 0
        df_loc.loc[df_loc["success"] == 0, "queries"] = float("nan")
        agg_df = (
            df_loc.groupby(["attack", "campaign", "victim_model"])
            .agg(
                avg_success=("success", "mean"),
                avg_queries=("queries", "mean"),
                count=("queries", "count"),
            )
            .reset_index()
        )

        agg_df.avg_success *= 100
        agg_df = agg_df.rename(
            columns={"campaign": "scenario", "victim_model": "victim"}
        )
        order = (
            agg_df.groupby("attack")["avg_success"]
            .mean()
            .sort_values(ascending=False)
            .index
        )
        print(order)
        plt.figure(figsize=(10, 3.5))
        sns.barplot(
            data=agg_df,
            x="attack",
            y="avg_success",
            order=order,
            hue="scenario",
            hue_order=SCENARIO_NAMES.keys(),
            palette="Set2",
            errorbar="se",
        )

        plt.title(f"Success Rate for {dataset}")
        plt.xlabel("Attack")
        plt.ylabel("Success Rate (%)")
        plt.xticks(rotation=45)
        plt.legend(title="Scenario")
        # add grid
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plot_path = Path(cfg.report_root) / f"barplot_{dataset}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.clf()


def make_line_plots(df_results: pd.DataFrame) -> None:
    r"""Make side-by-side plots (one per scenario) from the results.

    Args:
        df_results (pd.DataFrame): DataFrame with the results.
    """
    for dataset in df_results["dataset"].unique():
        df_loc = df_results[df_results["dataset"] == dataset]
        df_sel = df_loc[["campaign", "victim_model"]].drop_duplicates()
        df_sel = sorted(
            df_sel.to_numpy(), key=lambda x: list(SCENARIO_NAMES.keys()).index(x[0])
        )
        df_sel = sorted(
            df_sel,
            key=lambda x: list(PLOT_MODEL_NAMES.keys()).index(x[1]),
        )
        buckets = defaultdict(list)
        for key, value in df_sel:
            buckets[key].append(value)
        for key in SCENARIO_NAMES:
            buckets[key]

        rows = list(
            zip_longest(*(buckets[scn] for scn in SCENARIO_NAMES), fillvalue=None)
        )

        for row in rows:
            # Set up subplots
            fig, axes = plt.subplots(
                1,
                len(row),
                figsize=(4 * len(row), 3),
                sharey=True,
            )
            for ax, scenario, victim in zip(axes, SCENARIO_NAMES, row, strict=False):
                df_scen_vict = df_loc[df_loc["campaign"] == scenario]
                df_scen_vict = df_scen_vict[df_scen_vict["victim_model"] == victim]
                df_scen_vict.loc[df_scen_vict["success"] == 0, "queries"] = float("nan")
                df_scen_vict = df_scen_vict.sort_values("queries")
                df_scen_vict["success"] = df_scen_vict.groupby(
                    ["attack", "victim_model"]
                )["success"].transform(lambda x: x.cumsum() / x.count())
                df_scen_vict["success"] *= 100
                df_scen_vict = df_scen_vict.sort_values("attack")
                df_oneshot = df_scen_vict[df_scen_vict["queries"] == 0]
                df_agg_oneshot = df_oneshot.groupby("attack").agg(
                    success=("success", "max"), queries=("queries", "mean")
                )
                df_nquery = df_scen_vict[df_scen_vict["queries"] > 0]
                legend = scenario == "robust"
                if not df_nquery.empty:
                    # Plot the success rate for each attack
                    sns.lineplot(
                        data=df_nquery,
                        x="queries",
                        y="success",
                        hue="attack",
                        markers=True,
                        dashes=False,
                        legend=legend,
                        ax=ax,
                        estimator=None,
                    )

                if not df_agg_oneshot.empty:
                    # Plot the success rate for each attack
                    sns.scatterplot(
                        data=df_agg_oneshot.reset_index(),
                        x="queries",
                        y="success",
                        style="attack",
                        legend=legend,
                        ax=ax,
                    )
                if victim is not None:
                    plt_name = (
                        PLOT_MODEL_NAMES[victim]
                        + " on "
                        + scenario.capitalize()
                        + "Pool"
                    )
                    ax.set_title(plt_name)
                    ax.set_xlabel("Queries")
                    ax.set_ylabel("Attack Success Rate [%]")
                    ax.set_xscale("symlog", base=2)
                if legend:
                    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
                ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.suptitle(f"ASR and queries-per-success on {dataset}", fontsize=16)
            plt.tight_layout()
            plot_path = Path(cfg.report_root) / f"plot_{dataset}_{row[0]}.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, bbox_inches="tight")
            plt.clf()


def make_plots(df_results: pd.DataFrame) -> None:
    r"""Make plots from the results.

    Args:
        df_results (pd.DataFrame): DataFrame with the results.
    """
    make_line_plots(df_results)
    make_barplots(df_results)
