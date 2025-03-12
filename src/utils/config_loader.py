import yaml
from pathlib import Path
import logging


def load_default_config():
    """Loads default settings from YAML file."""
    # Relative path to "config.yaml"
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"

    if not config_path.exists():
        logging.error(f"The configuration file {config_path} does not exist.")
        raise FileNotFoundError(f"The configuration file {config_path} does not exist.")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config.get("default", {})
