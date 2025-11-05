"""
Logging utilities for correlation engine.
Provides consistent logging with timestamps and automatic truncation of large results.
"""

import logging
from datetime import datetime
from typing import Any, Optional
import json


MAX_LOG_LENGTH = 150  # Maximum characters for large results in logs


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def truncate_large_result(data: Any, max_length: int = MAX_LOG_LENGTH) -> str:
    """
    Truncate large data for logging.

    Args:
        data: Data to potentially truncate
        max_length: Maximum length before truncation

    Returns:
        String representation, truncated if necessary
    """
    if data is None:
        return "None"

    # Convert to string
    if isinstance(data, (dict, list)):
        try:
            data_str = json.dumps(data, indent=None)
        except (TypeError, ValueError):
            data_str = str(data)
    else:
        data_str = str(data)

    # Truncate if too long
    if len(data_str) > max_length:
        return f"{data_str[:max_length]}... (truncated, total length: {len(data_str)})"

    return data_str


class CorrelationLogger:
    """
    Enhanced logger for correlation engine with timestamp and truncation support.
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def info(self, message: str, data: Optional[Any] = None):
        """Log info message with timestamp and optional data."""
        timestamp = get_timestamp()
        if data is not None:
            truncated_data = truncate_large_result(data)
            self.logger.info(f"[{timestamp}] {message}: {truncated_data}")
        else:
            self.logger.info(f"[{timestamp}] {message}")

    def warning(self, message: str, data: Optional[Any] = None):
        """Log warning message with timestamp and optional data."""
        timestamp = get_timestamp()
        if data is not None:
            truncated_data = truncate_large_result(data)
            self.logger.warning(f"[{timestamp}] {message}: {truncated_data}")
        else:
            self.logger.warning(f"[{timestamp}] {message}")

    def error(self, message: str, data: Optional[Any] = None, exc_info: bool = False):
        """Log error message with timestamp and optional data."""
        timestamp = get_timestamp()
        if data is not None:
            truncated_data = truncate_large_result(data)
            self.logger.error(f"[{timestamp}] {message}: {truncated_data}", exc_info=exc_info)
        else:
            self.logger.error(f"[{timestamp}] {message}", exc_info=exc_info)

    def debug(self, message: str, data: Optional[Any] = None):
        """Log debug message with timestamp and optional data."""
        timestamp = get_timestamp()
        if data is not None:
            truncated_data = truncate_large_result(data)
            self.logger.debug(f"[{timestamp}] {message}: {truncated_data}")
        else:
            self.logger.debug(f"[{timestamp}] {message}")

    def critical(self, message: str, data: Optional[Any] = None):
        """Log critical message with timestamp and optional data."""
        timestamp = get_timestamp()
        if data is not None:
            truncated_data = truncate_large_result(data)
            self.logger.critical(f"[{timestamp}] {message}: {truncated_data}")
        else:
            self.logger.critical(f"[{timestamp}] {message}")


def get_logger(name: str) -> CorrelationLogger:
    """
    Get a CorrelationLogger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        CorrelationLogger instance
    """
    return CorrelationLogger(name)


# Convenience functions for backward compatibility
def log_with_timestamp(logger: logging.Logger, level: str, message: str, data: Optional[Any] = None):
    """
    Log a message with timestamp and optional truncated data.

    Args:
        logger: Standard logging.Logger instance
        level: Log level (info, warning, error, debug)
        message: Log message
        data: Optional data to log (will be truncated if large)
    """
    timestamp = get_timestamp()
    log_func = getattr(logger, level.lower(), logger.info)

    if data is not None:
        truncated_data = truncate_large_result(data)
        log_func(f"[{timestamp}] {message}: {truncated_data}")
    else:
        log_func(f"[{timestamp}] {message}")
