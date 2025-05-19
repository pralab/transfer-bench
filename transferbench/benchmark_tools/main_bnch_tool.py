r"""Command line interface for running and displaying results of transfer attacks."""

import logging
import sys
from argparse import ArgumentParser
from typing import Optional

import wandb

from .config import OmegaConf, cfg, user_cfg, user_cfg_path
from .report_helpers import collect_results, make_plots, make_tabulars, make_json_summary
from .run_helpers import get_filtered_runs, run_single_scenario

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format=">>> [%(levelname)s] %(asctime)s  \n%(message)s\n<<<",
)
logger = logging.getLogger(__name__)


def parse_args() -> None:
    r"""Parse command line arguments."""
    parser = ArgumentParser(
        description="Command line tool for managing benchamrks of the transfer attack."
    )
    subparser = parser.add_subparsers(dest="command", required=True)
    # Subparsers for getting informations on the runs
    parser_info = subparser.add_parser(
        "display", help="Display informations on the runs."
    )
    parser_info.add_argument(
        dest="status",
        nargs="?",
        choices=[
            "all",
            "running",
            "finished",
            "crashed",
            "killed",
            "failed",
            "missing",
        ],
        default="all",
        type=str,
        help="Information to be displyed",
    )
    parser_info.add_argument("--query", type=str, help="Query to filter displayed runs")
    # Subparser for running the experiments
    parser_run = subparser.add_parser(
        "run",
        help="Run a job from the list of availables.",
    )
    parser_run.add_argument(
        dest="run_ids", nargs="*", type=str, help="ID of the run(s) to be executed."
    )
    parser_run.add_argument(
        "--query",
        type=str,
        help="Lunch the runs selecting them by query. Example of usage.",
    )
    parser_run.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the run, do not resume the previous results.",
    )
    parser_run.add_argument(
        "--device", type=str, default=cfg.default_device, help="Device to be used."
    )
    parser_run.add_argument(
        "--batch-size", type=int, default=20, help="Batch size to be used."
    )
    ## config command, allow user to set configurations
    parser_config = subparser.add_parser(
        "config",
        help="Set the configuration for banchmark tool.",
    )
    parser_config.add_argument(
        "--results-root",
        type=str,
        help="Set root directory for the results.",
    )
    parser_config.add_argument(
        "--project-name",
        type=str,
        help="Set project name.",
    )
    parser_config.add_argument(
        "--project-entity",
        type=str,
        help="Set project entity.",
    )
    parser_report = subparser.add_parser("report", help="Generate a report.")
    parser_report.add_argument(
        "--download",
        action="store_true",
        help="Download the results from Weights & Biases.",
    )

    return parser.parse_args()


def run_batch(run_ids: list[str], batch_size: int, device: str, resume: bool) -> None:
    r"""Run a single scenario."""
    for run_id in run_ids:
        # Check if the run is already running
        if run_id in get_filtered_runs(status="running")["id"].tolist():
            msg = f"Run {run_id} is already running. Skipping..."
            logger.info(msg)
            continue
        try:
            run_single_scenario(run_id, batch_size, device, resume)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected. Exiting...")
            sys.exit(1)
        except Exception as e:
            msg = f"Error while processing run {run_id}\n {e}"
            logger.exception(msg)
            msg = "Do you want to continue with next runs? (y/n)"
            if input(msg).lower() not in ["", "y"]:
                sys.exit(1)


def handle_display(
    status: str = "all",
    query: Optional[str] = None,
) -> None:
    r"""Handle the display subcommand."""
    # Get the information on the runs
    runs = get_filtered_runs(status=status, query=query)
    # Display the runs
    logger.info(runs.to_markdown(index=False))


def handle_runs(
    run_ids: Optional[list[str]],
    query: Optional[str],
    batch_size: int,
    device: str,
    resume: bool,
) -> None:
    r"""Handle the run subcommand."""
    run_ids = run_ids if run_ids is not None else []
    # Get the runs from the query
    if query is not None:
        safe_query = query + " and available == True"
        df_runs = get_filtered_runs(query=safe_query)
        ## Check if finished runs are included and warning the user
        if "finished" in df_runs["status"]:
            logger.info(
                (
                    "The query contains finished run(s). They will be ignored.",
                    "Input them as run_ids arguments if you want to re-run them.",
                )
            )
        safe_query += ' and (status not in ["finished", "running"])'
        safe_query += f' or (id in {run_ids} and status != "running")'
        df_runs = df_runs.query(safe_query)

    else:
        safe_query = f'available == True and id in {run_ids} and status != "running"'
        df_runs = get_filtered_runs(query=safe_query)
    log_msg = f"Processing runs: \n {df_runs.to_markdown(index=False)}"
    logger.info(log_msg)
    run_ids = df_runs["id"].tolist()
    run_batch(run_ids, batch_size, device, resume)


def handle_config(
    **kwargs: Optional[str],
) -> None:
    r"""Handle the config subcommand."""
    # Crete the config file if it does not exist
    cli_cfg = OmegaConf.create(
        {key: value for key, value in kwargs.items() if value is not None}
    )
    # Merge the user config with the default config
    cli_cfg = OmegaConf.merge(user_cfg, cli_cfg)
    if not user_cfg_path.exists():
        user_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cli_cfg, user_cfg_path)
    info_msg = f"Configutation updated: {cli_cfg}"
    logger.info(info_msg)


def handle_report(download: bool = False) -> None:
    r"""Handle the report subcommand."""
    # collect finished runs
    df_results = collect_results(download=download)
    make_tabulars(df_results)[0]
    logger.info("Report generated.")
    make_json_summary(df_results)
    logger.info("JSON generated.")
    make_plots(df_results)
    logger.info("Plots generated.")


def main() -> None:
    r"""Entrypoint to run the script."""
    # Parse the command line arguments
    args = parse_args()
    # Get the command line arguments
    command = args.command
    if command == "config":
        handle_config(
            results_root=args.results_root,
            project_name=args.project_name,
            project_entity=args.project_entity,
        )
        sys.exit()
    elif command == "report":
        handle_report(args.download)
        sys.exit()
    # Login to Weights & Biases-
    wandb.login()
    if command == "display":
        handle_display(
            status=args.status,
            query=args.query,
        )
    elif command == "run":
        handle_runs(
            run_ids=args.run_ids,
            query=args.query,
            batch_size=args.batch_size,
            device=args.device,
            resume=not args.overwrite,
        )
