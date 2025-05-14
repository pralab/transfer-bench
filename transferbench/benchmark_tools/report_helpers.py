r"""Report helpers for transferbench."""

import json
from collections import OrderedDict, defaultdict
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
    "vgg19": "\\vgg{19}",
    "vit_b_16": "\\vit{16}",
    "imagenet_resnet50_pubdef": "\\pubdef",
    "Amini2024MeanSparse_Swin-L": "\\amini",
    "Xu2024MIMIR_Swin-L": "\\mimir",
    "cifar10_resnet56": "\\resnet{56}",
    "cifar10_vgg19_bn": "\\vgg{19-bn}",
    "cifar10_vit_b16": "\\vit{16}",
    "cifar10_beit_b16": "\\beit{16}",
    "Peng2023Robust": "\\peng",
    "Bartoldson2024Adversarial_WRN-94-16": "\\bartold",
}
PLOT_MODEL_NAMES = {
    "resnext101_32x8d": "ResNeXt-101",
    "imagenet_resnet50_pubdef": "Pub-RN-50",
    "vgg19": "VGG-19",
    "Amini2024MeanSparse_Swin-L": "Amini-Sw-L",
    "vit_b_16": "ViT-B/16",
    "Xu2024MIMIR_Swin-L": "Mim-Sw-L",
    "cifar10_vgg19_bn": "VGG-19-bn",
    "cifar10_resnet56": "ResNet-56",
    "cifar10_vit_b16": "ViT-B/16",
    "cifar10_beit_b16": "BeIT-B/16",
    "Peng2023Robust": "Peng-RW-RN-70",
    "Bartoldson2024Adversarial_WRN-94-16": "Barto-WRN-94",
}
METRIC_NAME = {"avg_success": "ASR", "avg_queries": "$\\bar q$"}
SCENARIO_NAMES = {"omeo": "\\omeo", "etero": "\\etero", "robust": "\\robust"}
PLOT_SCENARIO_NAMES = {
    "omeo": "HoS",
    "etero": "HeS",
    "robust": "HoS+R",
}
ATTACK_NAMES = OrderedDict(
    {  # TODO(@fabio): move to config # https://github.com/your-repo/issues/future-issue
        "BASES": "BASES",
        "DSWEA": "DSWEA",
        "GAA": "GAA",
        "GFCS": "GFCS",
        "SimbaODS": "SimbaODS",
        "NaiveAvg10": "NaiveAvg10",
        "NaiveAvg": "NaiveAvg100",
        "ENS": "ENS",
        "CWA": "CWA",
        "LGV": "LGV",
        "MBA": "MBA",
        "SASD_WS": "SASD\\_WS",
        "SVRE": "SVRE",
    }
)
PLOT_ATTACK_NAMES = OrderedDict(
    {  # TODO(@fabio): move to config # https://github.com/your-repo/issues/future-issue
        "BASES": "BASES",
        "DSWEA": "DSWEA",
        "GAA": "GAA",
        "GFCS": "GFCS",
        "SimbaODS": "SimbaODS",
        "NaiveAvg10": "NaiveAvg10",
        "NaiveAvg": "NaiveAvg100",
        "ENS": "ENS",
        "CWA": "CWA",
        "LGV": "LGV",
        "MBA": "MBA",
        "SASD_WS": "SASD_WS",
        "SVRE": "SVRE",
    }
)


def collect_results(download: bool) -> list[dict]:
    r"""Collect the results from the runs.

    Args:
        run_ids (list[str]): List of run ids to collect results from.
        download (bool): Whether to download the results or not. Defaults to False.
    """
    df_runs = get_filtered_runs(
        "finished", f'campaign != "debug" and attack in {list(ATTACK_NAMES.keys())}'
    )
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
        # Replace queries with nan when success is 0
        df_loc.loc[df_loc["success"] == 0, "queries"] = float("nan")

        agg_df = df_loc.groupby(["attack", "campaign", "victim_model"]).agg(
            avg_success=("success", "mean"),
            avg_queries=("queries", "mean"),
            count=("success", "count"),
        )
        # Set queries to -1 when success is 0
        agg_df.loc[agg_df["avg_success"] == 0, "avg_queries"] = -1
        agg_df = agg_df.reset_index()
        agg_df.avg_success *= 100
        agg_df: pd.DataFrame = agg_df.rename(
            columns={"campaign": "scenario", "victim_model": "victim"}
        )

        pivot_df = agg_df.pivot_table(
            index="attack",
            columns=["scenario", "victim"],
            values=["avg_success", "avg_queries"],
        )

        # replace -1 with nan
        pivot_df = pivot_df.replace(-1, float("nan"))

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
            column_format="l" + "|cc" * (len(pivot_df.columns) // 2),
            multicolumn_format="c",
            na_rep="-",
            escape=False,  # Important: allows LaTeX commands to be rendered properly
        )

        tabulars.append(pivot_df)
    return tabulars


def make_json_summary(df_results: pd.DataFrame) -> dict:
    r"""Generate a nested JSON summary from the evaluation results.

    Args:
        df_results (pd.DataFrame): DataFrame with the results.

    Returns:
        dict: Nested dictionary in the desired JSON structure.
    """
    datasets = df_results["dataset"].unique()
    json_output = {}

    for dataset in datasets:
        df_loc = df_results[df_results["dataset"] == dataset]
        df_loc = df_loc[df_loc["campaign"].isin(SCENARIO_NAMES.keys())]
        df_loc.loc[df_loc["success"] == 0, "queries"] = float("nan")

        agg_df = (
            df_loc.groupby(["attack", "campaign", "victim_model"])
            .agg(
                avg_success=("success", "mean"),
                avg_queries=("queries", "mean"),
            )
            .reset_index()
        )
        agg_df.avg_success = agg_df.avg_success.round(4)
        agg_df.avg_queries = agg_df.avg_queries.round(0).astype("Int64")  # handle NaNs

        agg_df = agg_df.rename(
            columns={"campaign": "scenario", "victim_model": "victim"}
        )

        # Create nested structure: dataset -> victim -> scenario -> [entries]
        nested_dict = {}

        for _, row in agg_df.iterrows():
            attack_name = row["attack"]
            scenario = PLOT_SCENARIO_NAMES.get(row["scenario"], row["scenario"])
            victim = PLOT_MODEL_NAMES.get(row["victim"], row["victim"])
            asr = float(row["avg_success"])
            queries = None if pd.isna(row["avg_queries"]) else int(row["avg_queries"])

            nested_dict.setdefault(victim, {}).setdefault(scenario, []).append({
                "attack_name": attack_name,
                "ASR": asr,
                "queries": queries
            })

        json_output[dataset] = nested_dict

    output_path = Path(cfg.report_root) / "data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    with open(output_path, "w") as f:
        json.dump(json_output, f, indent=2)
    return json_output


def make_barplots(df_results: pd.DataFrame) -> None:
    for dataset in df_results["dataset"].unique():
        df_loc = df_results[df_results["dataset"] == dataset]
        # Replace queries with nan when success is 0
        df_loc.loc[df_loc["success"] == 0, "queries"] = float("nan")
        agg_df = (
            df_loc.groupby(["attack", "campaign", "victim_model"])
            .agg(
                avg_success=("success", "mean"),
                avg_queries=("queries", "mean"),
            )
            .reset_index()
        )

        agg_df.avg_success *= 100
        agg_df = agg_df.rename(
            columns={"campaign": "scenario", "victim_model": "victim"}
        )
        # rename scenarios
        agg_df["scenario"] = agg_df["scenario"].map(PLOT_SCENARIO_NAMES)
        # rename attacks
        agg_df["attack"] = agg_df["attack"].map(PLOT_ATTACK_NAMES)
        order = (
            agg_df.groupby("attack")["avg_success"]
            .mean()
            .sort_values(ascending=False)
            .index
        )
        plt.figure(figsize=(10, 3.5))
        sns.barplot(
            data=agg_df,
            x="attack",
            y="avg_success",
            order=order,
            hue="scenario",
            hue_order=PLOT_SCENARIO_NAMES.values(),
            palette="Set2",
            errorbar=("pi", 50),
        )

        plt.title(f"Attack Success Rate for {dataset}")
        plt.ylabel("Attack Success Rate [%]")
        plt.xlabel("")
        plt.xticks(rotation=15)
        plt.legend(title="Scenario")
        # add grid
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plot_path = Path(cfg.report_root) / f"barplot_{dataset}.pdf"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.clf()


def make_line_plots(df_results: pd.DataFrame, add_oneshot: bool = False) -> None:
    r"""Make side-by-side plots (one per scenario) from the results.

    Args:
        df_results (pd.DataFrame): DataFrame with the results.
    """
    for dataset in df_results["dataset"].unique():
        df_loc = df_results[df_results["dataset"] == dataset]
        if not add_oneshot:
            df_loc = df_loc[df_loc["queries"] > 0]
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
        # Rename the attacks
        df_loc["attack"] = df_loc["attack"].map(PLOT_ATTACK_NAMES)
        hue_order = [
            attack
            for attack in PLOT_ATTACK_NAMES.values()
            if attack in df_loc["attack"].unique()
        ][::-1]
        palette = sns.color_palette("tab10", len(hue_order))

        for row in rows:
            # Set up subplots
            fig, axes = plt.subplots(
                1,
                len(row),
                figsize=(4 * len(row), 3),
                sharey=True,
            )
            all_handles = []
            all_labels = []
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
                if not df_nquery.empty:
                    # Plot the success rate for each attack
                    sns.lineplot(
                        data=df_nquery,
                        x="queries",
                        y="success",
                        hue="attack",
                        hue_order=hue_order,
                        palette=palette,
                        markers=True,
                        dashes=False,
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
                        ax=ax,
                    )
                if victim is not None:
                    plt_name = (
                        f"{PLOT_MODEL_NAMES[victim]} ({PLOT_SCENARIO_NAMES[scenario]})"
                    )
                    ax.set_title(plt_name)
                    ax.set_xlabel("Queries")
                    ax.set_ylabel("Attack Success Rate [%]")
                    ax.set_xscale("log", base=2)

                    ax.grid(axis="y", linestyle="--", alpha=0.7)

                handles, labels = ax.get_legend_handles_labels()
                all_handles.extend(handles)
                all_labels.extend(labels)
            # Remove duplicate handles and labels
            handles_by_labels = OrderedDict(zip(all_labels, all_handles, strict=False))
            [ax.get_legend().remove() for ax in axes if ax.get_legend() is not None]
            # Add a legend to the last subplot

            axes[-1].legend(
                handles_by_labels.values(),
                handles_by_labels.keys(),
                title="Attacks",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )
            plt.tight_layout()
            plot_path = Path(cfg.report_root) / f"plot_{dataset}_{row[0]}.pdf"
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
