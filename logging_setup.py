"""
logging_setup.py — Elite Trading Bot v5.3
==========================================
Centralized logging configuration with RotatingFileHandler.

Features:
- RotatingFileHandler: يحتفظ بكل السطور في ملفات متعددة
  - كل ملف max 50MB
  - يحتفظ بـ 10 ملفات backup (= 500MB total)
  - الأقدم بيتحذف تلقائياً لما يتجاوز الـ limit
- Format: timestamp [module] [level] message
- Encoding: UTF-8 (يدعم العربي والـ Unicode)

Usage:
    from logging_setup import setup_logger
    logger = setup_logger("MyModule", Config.LOG_FILE)
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_rotating_handler(
    log_file: str,
    max_bytes: int = 50 * 1024 * 1024,   # 50 MB per file
    backup_count: int = 10,               # 10 backup files = 500MB total
    encoding: str = "utf-8",
) -> RotatingFileHandler:
    """
    Creates a RotatingFileHandler that:
    - Writes to log_file
    - Rotates when file reaches max_bytes
    - Keeps backup_count old files
    - Never loses any log lines

    File naming: bot.log, bot.log.1, bot.log.2, ... bot.log.10
    """
    # Ensure directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding=encoding,
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    return handler


def setup_module_logger(
    name: str,
    log_file: str,
    console_color: str = "\033[92m",  # green
    level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Creates a module-level logger with:
    - RotatingFileHandler (50MB × 10 files)
    - StreamHandler (colored console output)

    Args:
        name: Logger name (e.g., "DataLoader", "Ensemble")
        log_file: Path to log file
        console_color: ANSI color code for console output
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # File handler (rotating)
        fh = setup_rotating_handler(log_file)
        logger.addHandler(fh)

        # Console handler (colored)
        ch = logging.StreamHandler()
        ch.setFormatter(
            logging.Formatter(
                f"{console_color}%(asctime)s\033[0m [%(levelname)s] %(message)s"
            )
        )
        logger.addHandler(ch)

    return logger


def configure_root_logger(
    log_file: str,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
):
    """
    Configures the root logger (used by main.py and train_offline.py).
    Replaces logging.basicConfig() with RotatingFileHandler.

    Args:
        log_file: Path to main log file
        level: Logging level
        fmt: Log format string
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers
    for h in root.handlers[:]:
        root.removeHandler(h)

    formatter = logging.Formatter(fmt)

    # Rotating file handler
    fh = setup_rotating_handler(log_file)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root.addHandler(ch)
