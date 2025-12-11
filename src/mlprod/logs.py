from typing import Any

from dotenv import load_dotenv
from pathlib import Path

from rich.logging import RichHandler as _RichHandler
from rich.console import Console

import logging
import logging.config
import os
import yaml

# write the config file location to LOGGING_CONFIG_FILE inside a .env file and this will load it
load_dotenv(override=True)

CWD = Path(os.getcwd())
ENV_LOGGING_CONFIG_FILE_PATH = "LOGGING_CONFIG_FILE"
DEFAULT_LOGGING_CONFIG_FILE_PATH = CWD / "configs" / "logging.yaml"


def setup_logs(
    config_path: Path | None = None,
    default_level: int = logging.INFO,
) -> None:
    """Load and configure logs from environment variables and a YAML file.

    This function checks if the file defined in the LOGGING_CONFIG_FILE environment
    variable exists. This variable defaults to `logging.yaml` file exists. If it does,
    it reads its contents and uses them to configure the logging via `dictConfig`.
    The logging system is then set up to be used throughout the application.

    To avoid multiple calls to this function, it sets and checks the LOG_INITIALIZED
    environment variable to TRUE.

    Args:
        config_path (Path | None): Path to a YAML configuration file containing logging
            settings. If not provided, the function will use the default configuration
            file named "logging.yaml" in the current work directory, if it exists.
            Defaults to None.
        default_level (int): Default log level for basic logging setup if a valid YAML
            configuration file cannot be found or an exception is raised.
            Defaults to INFO.

    """
    # we use an environment variable to test if we already have initialized the loggers
    if os.environ.get("LOG_INITIALIZED", "") == "1":
        return

    if config_path is None:
        # get the position of the logging config file from environment variables...
        LOGGING_CONFIG_FILE = os.environ.get(
            ENV_LOGGING_CONFIG_FILE_PATH,
            # ...if not use a default location
            DEFAULT_LOGGING_CONFIG_FILE_PATH.absolute(),
        )
        config_path = Path(LOGGING_CONFIG_FILE)

    if config_path.exists():
        try:
            # read the content of the logging configuration file
            with open(config_path, "r") as f:
                logging_config: dict[str, dict | Any] = yaml.safe_load(f)

            # create folders that will contains the log files
            for _, handler in logging_config.get("handlers", {}).items():
                if "filename" in handler:
                    path = Path(handler["filename"])
                    path.absolute().parent.mkdir(parents=True, exist_ok=True)

            logging.config.dictConfig(logging_config)

            logging.debug(f"LOGGING_CONFIG_FILE: {LOGGING_CONFIG_FILE}")

        except Exception as e:
            # this means we have not found a valid configuration file
            logging.basicConfig(level=default_level)
            logging.error(f"Logging configuration reading failed: {e}")
            logging.exception(e)

    else:
        # initialize with the given level without using a configuration file
        logging.basicConfig(level=default_level)
        logging.warning(
            f"Logging config path ({config_path}) does not exists, using basic setup"
        )

    os.environ["LOG_INITIALIZED"] = "1"


class RichHandler(_RichHandler):
    """Rich Handler custom configuration."""

    def __init__(
        self,
        rich_tracebacks=True,
        markup=True,
        color_system=None,
        width=200,
        show_path=False,
    ) -> None:
        """Initialize RichHandler with custom console settings."""
        super().__init__(
            rich_tracebacks=rich_tracebacks,
            console=Console(
                markup=markup,
                color_system=color_system,
                width=width,
            ),
            markup=markup,
            show_path=show_path,
        )
