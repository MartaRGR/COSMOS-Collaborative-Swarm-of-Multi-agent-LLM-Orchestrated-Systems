import logging
import pytz
from datetime import datetime


def get_agent_logger(agent_name, log_level=logging.INFO):
    """Consistent formatted logger for agents."""
    logger = logging.getLogger(agent_name)

    # Check if it has already been configured to avoid multiple handlers
    if not logger.hasHandlers():
        handler = logging.StreamHandler()

        class MadridFormatter(logging.Formatter):
            """Custom formatter to convert UTC time to Europe/Madrid time."""
            def formatTime(self, record, datefmt=None):
                dt = datetime.fromtimestamp(record.created, pytz.utc)  # Convertir de UTC a local
                madrid_time = dt.astimezone(pytz.timezone("Europe/Madrid"))
                return madrid_time.strftime(datefmt or "%Y-%m-%d %H:%M:%S")

        formatter = MadridFormatter(
            fmt="[%(asctime)s] [%(levelname)s] [Agent: %(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(log_level)

    return logger
