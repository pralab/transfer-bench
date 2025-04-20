#!/usr/bin/env python3
r"""Command line interface for running and displaying results of transfer attacks."""

import logging
from argparse import ArgumentParser
from typing import Optional

import wandb

from .config import DEFAULT_DEVICE
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
        choices=["all", "running", "finished", "crashed", "failed", "missing"],
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
    parser_run.add_argument("--next", type=str, help="Next available run.")
    parser_run.add_argument(
        "--device", type=str, default=DEFAULT_DEVICE, help="Device to be used."
    )
    parser_run.add_argument(
        "--batch-size", type=int, default=20, help="Batch size to be used."
    )
    return parser.parse_args()


def run_batch(run_ids: list[str], batch_size: int, device: str) -> None:
    r"""Run a single scenario."""
    for run_id in run_ids:
        try:
            run_single_scenario(run_id, batch_size, device)
        except KeyboardInterrupt:  # noqa: PERF203
            logger.info("Keyboard interrupt detected. Exiting...")
            break
        except Exception as e:
            msg = f"Error while processing run {run_id}\n {e}"
            logger.exception(msg)
            # Ask the user if they want to continue
            msg = "Do you want to continue with next runs? (y/n)"
            if input(msg).lower() not in ["", "y"]:
                break


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
    run_ids: Optional[list[str]], query: Optional[str], batch_size: int, device: str
) -> None:
    r"""Handle the run subcommand."""
    run_ids = run_ids if run_ids is not None else []
    # Get the runs from the query
    if query is not None:
        safe_query = query + " and available == True"
        df_runs = get_filtered_runs(query=safe_query)
        ## Check if finished runs are included and warning the user
        if "finished" in df_runs["status"]:
            logger.warning(
                (
                    "The query contains finished run(s), that will be ignored.",
                    "Input them as run_ids arguments if you want to re-run them.",
                )
            )
        safe_query += ' and status in ["missing", "failed", "crashed"]'
        safe_query += " or id in @run_ids"
        df_runs = df_runs.query(safe_query)
        log_msg = f"Processing runs: \n {df_runs.to_markdown(index=False)}"
        logger.info(log_msg)
        run_ids += df_runs["id"].tolist()
    run_batch(run_ids, batch_size, device)


def main() -> None:
    r"""Entrypoint to run the script."""
    # Login to Weights & Biases-
    wandb.login()
    # Parse the command line arguments
    args = parse_args()
    # Get the command line arguments
    command = args.command
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
        )
    else:
        msg = f"Unknown command: {command}"
        logger.error(msg)
