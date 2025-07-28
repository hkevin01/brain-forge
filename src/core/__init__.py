"""
Core Brain-Forge Components

This module contains the foundational components of the Brain-Forge system:
- Configuration management
- Exception handling  
- Structured logging
"""

from .config import Config, HardwareConfig, ProcessingConfig, SystemConfig
from .exceptions import (
    BrainForgeError,
    HardwareError, 
    DeviceConnectionError,
    StreamingError,
    ProcessingError,
    ConfigurationError,
    ValidationError
)
from .logger import get_logger, ContextualLogger, LogContext

__all__ = [
    # Configuration
    'Config',
    'HardwareConfig', 
    'ProcessingConfig',
    'SystemConfig',
    
    # Exceptions
    'BrainForgeError',
    'HardwareError',
    'DeviceConnectionError', 
    'StreamingError',
    'ProcessingError',
    'ConfigurationError',
    'ValidationError',
    
    # Logging
    'get_logger',
    'ContextualLogger',
    'LogContext'
]
