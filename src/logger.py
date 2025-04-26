from datetime import datetime
from pathlib import Path
import sys
from typing import Optional

from loguru import logger


class TitanicLogger:
    def __init__(self, logs_path: Optional[str] = None, level: str = "INFO"):
        self.logger = logger
        self.logger.remove()
        
        # Console logging
        self.logger.add(
            sys.stdout,
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
        )
        
        # File logging if path provided
        if logs_path:
            log_file = Path("logs") / Path(logs_path) / f"titanic_{datetime.now().strftime('%Y%m%d')}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.add(
                str(log_file),
                level=level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation="10 MB"
            )

    def debug(self, message: str, *args, **kwargs):
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        self.logger.critical(message, *args, **kwargs)