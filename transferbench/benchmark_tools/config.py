r"""Configuration for the weight and biases api."""

from pathlib import Path

from omegaconf import OmegaConf

from transferbench.types import Config
from transferbench.utils.cache import get_cache_dir

DEFAULT_CFG_PATH = Path(__file__).parent.parent / "config" / "tools" / "defaults.yaml"
user_cfg_path = get_cache_dir() / "config" / "user_config.yaml"

defaults_cfg = OmegaConf.load(DEFAULT_CFG_PATH)
user_cfg = (
    OmegaConf.load(user_cfg_path) if user_cfg_path.exists() else OmegaConf.create()
)

cfg: Config = OmegaConf.merge(defaults_cfg, user_cfg)
