"""
Logging configuration for the MLOps project
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_format: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Log message format (optional)
    
    Returns:
        Configured logger
    """
    
    # Default log format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance for specific module
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class StructuredLogger:
    """Structured logger for consistent log formatting"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def log_prediction(self, input_data: dict, prediction: float, 
                      processing_time: float, model_version: str):
        """Log prediction request"""
        self.logger.info(
            "Prediction made",
            extra={
                'event_type': 'prediction',
                'input_data': input_data,
                'prediction': prediction,
                'processing_time_ms': processing_time,
                'model_version': model_version
            }
        )
    
    def log_error(self, error_type: str, error_message: str, 
                  input_data: dict = None):
        """Log error occurrence"""
        self.logger.error(
            f"Error occurred: {error_type}",
            extra={
                'event_type': 'error',
                'error_type': error_type,
                'error_message': error_message,
                'input_data': input_data
            }
        )
    
    def log_model_load(self, model_path: str, model_version: str, 
                       load_time: float):
        """Log model loading"""
        self.logger.info(
            "Model loaded",
            extra={
                'event_type': 'model_load',
                'model_path': model_path,
                'model_version': model_version,
                'load_time_ms': load_time
            }
        )
    
    def log_health_check(self, status: str, checks: dict):
        """Log health check"""
        self.logger.info(
            f"Health check: {status}",
            extra={
                'event_type': 'health_check',
                'status': status,
                'checks': checks
            }
        )

# Setup default logging
default_logger = setup_logging(
    log_level=os.getenv('LOG_LEVEL', 'INFO'),
    log_file='monitoring/logs/app.log'
)
