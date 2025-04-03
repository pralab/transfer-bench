from pathlib import Path
from typing import Any

import yaml

# Get the directory of the current config file
CONFIG_DIR = Path(__file__).parent.resolve()


def load_scenario(file_name: str) -> dict[str, Any]:
    """Load and parse a YAML scenario file.

    Args:
        file_name: Name of the YAML file to load (e.g., 'scenarios.yaml')

    Returns:
        Parsed dictionary containing scenario configuration

    Raises:
        FileNotFoundError: If specified file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    yaml_path = CONFIG_DIR / file_name
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
