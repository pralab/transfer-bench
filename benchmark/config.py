r"""Configuration for the weight and biases api."""

from pathlib import Path

PROJECT_NAME = "transfer-bench"
PROJECT_ENTITY = "transfer-team"

ALLOWED_SCENARIOS = [
    "omeo-imagenet-inf",
    "etero-imagenet-inf",
    "robust-imagenet-inf",
    "debug",
]

DEFAULT_DEVICE: str = "cuda"
# DEFAULT_DEVICE = "cpu"  # uncomment for testing
RESULTS_ROOT: str = Path("./data")

COLUMNS = [
    "id",
    "status",
    "attack",
    "victim_model",
    "campaign",
    "p",
    "eps",
    "maximum_queries",
    "dataset",
]
