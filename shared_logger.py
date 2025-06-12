import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from multiprocessing import current_process

class SharedLogger:
    log_path = os.path.join(os.path.dirname(__file__), 'agents.log')
    #print(f"Logging INFO and above for all agents to: {log_path}")
    LOG_FILE = log_path
    MAX_BYTES = 5 * 1024 * 1024  # 5 MB
    BACKUP_COUNT = 3

    @staticmethod
    def get_logger(name: str = None, console_level_override: str = None) -> logging.Logger:
        logger_name = name or current_process().name
        logger = logging.getLogger(logger_name)
        log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_name, logging.INFO)
        logger.setLevel(log_level)

        if not logger.hasHandlers():
            # Console handler
            ch = logging.StreamHandler(sys.stdout)
            console_level_name = (
                console_level_override or os.getenv("CONSOLE_LOG_LEVEL", "DEBUG")
            ).upper()
            console_level = getattr(logging, console_level_name, logging.DEBUG)
            ch.setLevel(console_level)
            ch.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] [%(processName)s] [%(name)s] %(message)s",
                "%Y-%m-%d %H:%M:%S"
            ))
            logger.addHandler(ch)

            # File handler
            fh = RotatingFileHandler(
                SharedLogger.LOG_FILE,
                maxBytes=SharedLogger.MAX_BYTES,
                backupCount=SharedLogger.BACKUP_COUNT
            )
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] [%(processName)s] [%(name)s] %(message)s",
                "%Y-%m-%d %H:%M:%S"
            ))
            logger.addHandler(fh)

        return logger

    @staticmethod
    def get_log_level_name() -> str:
        return os.getenv("LOG_LEVEL", "INFO").upper()