"""
Logging Configuration for Brain-Forge

This module provides centralized logging configuration for the Brain-Forge system,
supporting structured logging with performance metrics and multi-device context.
"""

import logging
import logging.handlers
import sys
import json
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import threading
from contextlib import contextmanager

from .config import Config


@dataclass
class LogContext:
    """Structured context for log entries"""
    device_type: Optional[str] = None
    device_id: Optional[str] = None
    processing_stage: Optional[str] = None
    stream_type: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    experiment_id: Optional[str] = None
    latency_ms: Optional[float] = None
    data_size_bytes: Optional[int] = None
    error_code: Optional[str] = None


class ContextualLogger:
    """Logger with structured context support"""
    
    def __init__(self, name: str, config: Optional[Config] = None):
        """
        Initialize contextual logger
        
        Args:
            name: Logger name
            config: System configuration
        """
        self.name = name
        self.config = config or Config()
        self.logger = logging.getLogger(name)
        self._context_local = threading.local()
        
        # Configure logger if not already configured
        if not self.logger.handlers:
            self._setup_logger()
    
    def _setup_logger(self):
        """Set up logger with appropriate handlers and formatters"""
        log_config = self.config.system.logging
        
        # Set level
        level = getattr(logging, log_config.level.upper())
        self.logger.setLevel(level)
        
        # Console handler
        if log_config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_formatter = self._create_console_formatter()
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_config.file_output:
            file_path = Path(log_config.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Rotating file handler to manage log size
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=log_config.max_file_size,
                backupCount=log_config.backup_count
            )
            file_handler.setLevel(level)
            file_formatter = self._create_file_formatter()
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Structured JSON handler for analysis
        if log_config.structured_output:
            json_path = file_path.with_suffix('.jsonl')
            json_handler = logging.handlers.RotatingFileHandler(
                json_path,
                maxBytes=log_config.max_file_size,
                backupCount=log_config.backup_count
            )
            json_handler.setLevel(level)
            json_formatter = self._create_json_formatter()
            json_handler.setFormatter(json_formatter)
            self.logger.addHandler(json_handler)
    
    def _create_console_formatter(self) -> logging.Formatter:
        """Create console formatter with colors and context"""
        format_str = (
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "%(message)s"
        )
        return ColoredFormatter(format_str)
    
    def _create_file_formatter(self) -> logging.Formatter:
        """Create detailed file formatter"""
        format_str = (
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
        )
        return logging.Formatter(format_str)
    
    def _create_json_formatter(self) -> logging.Formatter:
        """Create structured JSON formatter"""
        return JsonFormatter()
    
    @contextmanager
    def context(self, **kwargs):
        """Context manager for adding structured context to logs"""
        # Get current context or create new
        current_context = getattr(self._context_local, 'context', {})
        
        # Merge with new context
        new_context = {**current_context, **kwargs}
        
        # Set context
        self._context_local.context = new_context
        
        try:
            yield
        finally:
            # Restore previous context
            self._context_local.context = current_context
    
    def _get_context(self) -> Dict[str, Any]:
        """Get current logging context"""
        return getattr(self._context_local, 'context', {})
    
    def _log_with_context(self, level: int, message: str, 
                         context: Optional[LogContext] = None,
                         extra: Optional[Dict[str, Any]] = None):
        """Log message with structured context"""
        # Build extra data
        log_extra = {
            'context': self._get_context(),
            'timestamp_iso': datetime.now(timezone.utc).isoformat(),
        }
        
        if context:
            log_extra['log_context'] = asdict(context)
        
        if extra:
            log_extra.update(extra)
        
        # Log the message
        self.logger.log(level, message, extra=log_extra)
    
    def debug(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log debug message"""
        self._log_with_context(logging.DEBUG, message, context, kwargs)
    
    def info(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log info message"""
        self._log_with_context(logging.INFO, message, context, kwargs)
    
    def warning(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log warning message"""
        self._log_with_context(logging.WARNING, message, context, kwargs)
    
    def error(self, message: str, context: Optional[LogContext] = None, 
              exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception details"""
        if exception:
            kwargs['exception_type'] = type(exception).__name__
            kwargs['exception_message'] = str(exception)
            kwargs['traceback'] = traceback.format_exc()
        
        self._log_with_context(logging.ERROR, message, context, kwargs)
    
    def critical(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log critical message"""
        self._log_with_context(logging.CRITICAL, message, context, kwargs)
    
    def performance(self, operation: str, duration_ms: float, 
                   context: Optional[LogContext] = None, **kwargs):
        """Log performance metrics"""
        perf_context = context or LogContext()
        perf_context.latency_ms = duration_ms
        
        message = f"Performance: {operation} completed in {duration_ms:.2f}ms"
        kwargs['operation'] = operation
        kwargs['duration_ms'] = duration_ms
        
        self._log_with_context(logging.INFO, message, perf_context, kwargs)
    
    def hardware_event(self, device_type: str, device_id: str, event: str,
                      success: bool = True, **kwargs):
        """Log hardware-related events"""
        hw_context = LogContext(
            device_type=device_type,
            device_id=device_id
        )
        
        level = logging.INFO if success else logging.WARNING
        message = f"Hardware {event}: {device_type}[{device_id}]"
        
        kwargs['event'] = event
        kwargs['success'] = success
        
        self._log_with_context(level, message, hw_context, kwargs)
    
    def stream_metrics(self, stream_type: str, samples_per_sec: float,
                      buffer_utilization: float, **kwargs):
        """Log streaming metrics"""
        stream_context = LogContext(stream_type=stream_type)
        
        message = (f"Stream metrics: {stream_type} - "
                  f"{samples_per_sec:.1f} samples/sec, "
                  f"{buffer_utilization:.1%} buffer")
        
        kwargs['samples_per_sec'] = samples_per_sec
        kwargs['buffer_utilization'] = buffer_utilization
        
        self._log_with_context(logging.INFO, message, stream_context, kwargs)


class ColoredFormatter(logging.Formatter):
    """Formatter with color support for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with colors"""
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{level_color}{record.levelname}{self.RESET}"
        
        # Format message
        formatted = super().format(record)
        
        # Add context if available
        if hasattr(record, 'context') and record.context:
            context_str = " | ".join(f"{k}={v}" for k, v in record.context.items())
            formatted += f" | {context_str}"
        
        return formatted


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        """Format log record as JSON"""
        log_data = {
            'timestamp': record.created,
            'timestamp_iso': datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add context if available
        if hasattr(record, 'context'):
            log_data['context'] = record.context
        
        if hasattr(record, 'log_context'):
            log_data['log_context'] = record.log_context
        
        # Add exception info if available
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno',
                          'pathname', 'filename', 'module', 'lineno',
                          'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process',
                          'exc_info', 'exc_text', 'stack_info', 'context',
                          'log_context', 'timestamp_iso']:
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


# Global logger instances
_loggers = {}
_logger_lock = threading.Lock()


def get_logger(name: str, config: Optional[Config] = None) -> ContextualLogger:
    """
    Get or create a contextual logger instance
    
    Args:
        name: Logger name
        config: System configuration
        
    Returns:
        ContextualLogger instance
    """
    with _logger_lock:
        if name not in _loggers:
            _loggers[name] = ContextualLogger(name, config)
        return _loggers[name]


def setup_logging(config: Config):
    """
    Set up global logging configuration
    
    Args:
        config: System configuration
    """
    # Set up root logger
    root_logger = logging.getLogger()
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure based on system config
    log_config = config.system.logging
    
    # Set global level
    level = getattr(logging, log_config.level.upper())
    root_logger.setLevel(level)
    
    # Add null handler to prevent "No handlers could be found" warnings
    root_logger.addHandler(logging.NullHandler())


# Performance timing context manager
@contextmanager
def log_performance(logger: ContextualLogger, operation: str, 
                   context: Optional[LogContext] = None):
    """
    Context manager for logging operation performance
    
    Args:
        logger: Logger instance
        operation: Operation description
        context: Optional log context
    """
    start_time = datetime.now()
    
    try:
        yield
    finally:
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        logger.performance(operation, duration_ms, context)


# Convenience functions for common logging patterns
def log_hardware_connection(logger: ContextualLogger, device_type: str, 
                          device_id: str, success: bool = True):
    """Log hardware connection event"""
    logger.hardware_event(device_type, device_id, "connection", success)


def log_stream_start(logger: ContextualLogger, stream_type: str):
    """Log stream start event"""
    with logger.context(stream_type=stream_type):
        logger.info(f"Started {stream_type} data stream")


def log_processing_stage(logger: ContextualLogger, stage: str, 
                        data_size: Optional[int] = None):
    """Log processing stage completion"""
    context = LogContext(processing_stage=stage)
    if data_size:
        context.data_size_bytes = data_size
    
    logger.info(f"Completed processing stage: {stage}", context)
