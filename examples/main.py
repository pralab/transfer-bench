r"""Main entry point for the TransferBench CLI."""

import argparse

import torch

from transferbench.evaluations import AttackEval, TransferEval

DEFAUlT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def transfer(
    victim: str,
    surrogates: list[str],
    scenario: str,
    batch_size: int = 64,
    device: str | torch.device = DEFAUlT_DEVICE,
):
    """Evaluate Transferability from surrogates to victim on a specific scenario."""
    print(
        f"Evaluate transferability of {surrogates} to {victim} on scenario: {scenario}"
    )
    evaluator = TransferEval(victim_model=victim, surrogate_models=surrogates)
    result = evaluator.run(batch_size=batch_size, device=device)
    print("Evaluation completed.")


def attack(attack_name, scenario="cnn", batch_size=128, device="cuda"):
    """Evaluate Transferability from surrogate to victim on a specific scenario."""
    print(f"Evaluate transferability of {attack_name} on scenario: {scenario}")
    evaluator = AttackEval(attack_name=attack_name)
    if scenario is not None:
        evaluator.set_scenario(scenario)
    result = evaluator.run(batch_size=batch_size, device=device)
    print("Evaluation completed.")
    print(f"Result: {result}")


def main():
    parser = argparse.ArgumentParser(
        description="A powerful CLI with transfer and attack commands."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    transfer_parser = subparsers.add_parser(
        "transfer", help="Evaluate transferability of models."
    )
    transfer_parser.add_argument("victim", type=str, help="Victim model name.")
    transfer_parser.add_argument(
        "surrogates", nargs="+", type=str, help="Surrogate model name(s)."
    )
    transfer_parser.add_argument("--scenario", type=str, help="Name of the scenario.")
    transfer_parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size."
    )
    transfer_parser.add_argument(
        "--device", type=str, default=DEFAUlT_DEVICE, help="Device for the evaluation."
    )
    transfer_parser.set_defaults(
        func=lambda args: transfer(
            args.victim, args.surrogates, args.scenario, args.batch_size, args.device
        )
    )

    attack_parser = subparsers.add_parser(
        "attack", help="Evaluate transferability of an attack."
    )
    attack_parser.add_argument("attack_name", type=str, help="Attack name.")
    attack_parser.add_argument(
        "--scenario", type=str, default="cnn", help="Name of the scenario."
    )
    attack_parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for the evaluation."
    )
    attack_parser.add_argument(
        "--device", type=str, default="cuda", help="Device for the evaluation."
    )
    attack_parser.set_defaults(
        func=lambda args: attack(
            args.attack_name, args.scenario, args.batch_size, args.device
        )
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
