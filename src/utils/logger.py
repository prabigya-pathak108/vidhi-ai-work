"""
Centralized Logger Utility for Vidhi AI
Provides consistent logging configuration across all modules
"""
import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        return super().format(record)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get or create a logger with console handler only.
    
    Args:
        name: Logger name (typically __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    
    # Console Handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
        
    return logger


def setup_application_logging(log_level: str = "INFO"):
    """
    Setup application-wide logging configuration (console only)
    Call this once at application startup (in main.py)
    
    Args:
        log_level: Default logging level
    """
    # Configure root logger - console only
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("pinecone").setLevel(logging.WARNING)
    
    root_logger = logging.getLogger()
    root_logger.info("=" * 70)
    root_logger.info("🚀 Vidhi AI Logging System Initialized")
    root_logger.info(f"📝 Log Level: {log_level}")
    root_logger.info(f"📺 Output: Console only")
    root_logger.info("=" * 70)
