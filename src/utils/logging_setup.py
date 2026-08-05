"""Shared logging: console + rotating file in outputs/logs/."""

import logging
from logging.handlers import RotatingFileHandler

from config import settings

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        settings.LOGS_DIR / "app.log", maxBytes=2_000_000, backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(console)
    return logger
