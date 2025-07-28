"""
Brain-Forge: Advanced Brain Scanning and Simulation Platform

A comprehensive toolkit for multi-modal brain data acquisition,
processing, mapping, and digital brain simulation. Forge the future
of neuroscience and brain-computer interfaces.
"""

__version__ = "0.1.0"
__author__ = "Brain-Forge Team"

# Core imports
from .core.config import Config
from .core.logger import get_logger
from .core.exceptions import BrainForgeError

# Main system components
from .integrated_system import IntegratedBrainSystem

__all__ = [
    "Config",
    "get_logger", 
    "BrainForgeError",
    "IntegratedBrainSystem"
]
