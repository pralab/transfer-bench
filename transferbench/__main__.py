r"""Main CLI interface for the package."""

from typing import Annotated, Optional

import torch
import typer
from evaluations import AttackEval, TransferEval

app = typer.Typer(help="A powerful CLI with transfer and attack commands.")

victim_model_type = Annotated[str, typer.Argument(help="Victim model name.")]
surrogate_model_type = Annotated[
    list[str], typer.Argument(help="Surrogate model name.")
]
batch_size_type = Annotated[str, typer.Argument(help="Batch size.")]
transfer_scenario_type = Annotated[str, typer.Argument(help="Name of the scenario.")]
device_type = Annotated[str, typer.Argument(help="Device for the evaluation.")]

DEFAUlT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@app.command()
def transfer(
    victim: victim_model_type,
    surrogates: surrogate_model_type,
    scenario: Optional[transfer_scenario_type] = None,
    batch_size: batch_size_type = 64,
    device: device_type = DEFAUlT_DEVICE,
) -> None:
    """Evaluate Transferability from surrogates to victim on a specific scenario."""
    typer.echo(
        f"Evaluate transferability of {surrogates} to {victim} on scenario: {scenario}"
    )
    evaluator = TransferEval(
        victim_model=victim,
        surrogate_models=surrogates,
    )
    evaluator.run(batch_size=batch_size, device=device)
    typer.echo("Evaluation completed.")


@app.command()
def attack(
    attack_name: str,
    scenario: str = typer.Option("cnn", help="Name of the scenario."),
    batch_size: int = typer.Option(128, help="Batch size for the evaluation."),
    device: str = typer.Option("cuda", help="Device for the evaluation."),
) -> None:
    """Evaluate Transferability from surrogate to victim on a specific scenario."""
    typer.echo(f"Evaluate transferability of {attack_name} on scenario: {scenario}")
    evaluator = AttackEval(
        attack_name=attack_name,
    )
    if scenario is not None:
        evaluator.set_scenario(scenario)
    result = evaluator.run(batch_size=batch_size, device=device)
    typer.echo("Evaluation completed.")
    typer.echo(f"Result: {result}")


if __name__ == "__main__":
    app()
