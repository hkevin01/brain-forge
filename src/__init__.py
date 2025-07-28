"""
Brain-Forge: Advanced Multi-Modal Brain-Computer Interface System

A comprehensive neuroscience platform integrating OPM magnetometry, 
Kernel optical imaging, and accelerometer arrays for real-time brain 
data acquisition, processing, and analysis.

Key Features:
- Multi-modal brain data acquisition (OMP helmet 306 channels, Kernel Flow/Flux)
- Real-time signal processing with <100ms latency 
- Advanced neural signal compression (5-10x ratios)
- Brain mapping with Harvard-Oxford atlas integration
- Neural simulation with Brian2/NEST integration
- Pattern transfer learning capabilities

Hardware Support:
- OPM Helmet: 306-channel magnetometer array
- Kernel Flow: Real-time hemodynamic imaging
- Kernel Flux: Neuron speed measurement  
- Brown's Accelo-hat: 3-axis motion tracking

Author: Brain-Forge Development Team
License: TBD
Version: 0.1.0-dev
"""

__version__ = "0.1.0-dev"
__author__ = "Brain-Forge Development Team"

# Core system imports
from .core.config import Config
from .core.exceptions import BrainForgeError
from .core.logger import get_logger

# Main system class
from .integrated_system import IntegratedBrainSystem

# Key processing components
from .processing import (
    RealTimeFilter,
    WaveletCompressor, 
    ArtifactRemover,
    FeatureExtractor,
    RealTimeProcessor
)

# Streaming management
from .acquisition.stream_manager import StreamManager

# Specialized tools
from .specialized_tools import EEGNotebooksIntegration

__all__ = [
    # Core
    'Config',
    'BrainForgeError', 
    'get_logger',
    
    # Main system
    'IntegratedBrainSystem',
    
    # Processing
    'RealTimeFilter',
    'WaveletCompressor',
    'ArtifactRemover', 
    'FeatureExtractor',
    'RealTimeProcessor',
    
    # Acquisition
    'StreamManager',
    
    # Tools
    'EEGNotebooksIntegration',
    
    # Metadata
    '__version__',
    '__author__'
]

# Package level configuration
import sys
import warnings

# Configure warnings for scientific computing
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# Minimum Python version check
if sys.version_info < (3, 8):
    raise ImportError("Brain-Forge requires Python 3.8 or higher")

# Initialize default logger
_default_logger = get_logger('brain-forge')
_default_logger.info(f"Brain-Forge v{__version__} initialized")

def get_system_info():
    """Get system information and capabilities"""
    import platform
    import pkg_resources
    
    info = {
        'version': __version__,
        'python_version': sys.version,
        'platform': platform.platform(),
        'dependencies': {}
    }
    
    # Check key dependencies
    key_deps = [
        'numpy', 'scipy', 'scikit-learn', 'mne', 'nilearn',
        'brian2', 'nest', 'pylsl', 'pywavelets'
    ]
    
    for dep in key_deps:
        try:
            version = pkg_resources.get_distribution(dep).version
            info['dependencies'][dep] = version
        except pkg_resources.DistributionNotFound:
            info['dependencies'][dep] = 'Not installed'
    
    return info

def check_hardware_support():
    """Check hardware interface availability"""
    hardware_status = {
        'lsl_available': False,
        'brian2_available': False,
        'nest_available': False,
        'mne_available': False
    }
    
    try:
        import pylsl
        hardware_status['lsl_available'] = True
    except ImportError:
        pass
    
    try:
        import brian2
        hardware_status['brian2_available'] = True
    except ImportError:
        pass
        
    try:
        import nest
        hardware_status['nest_available'] = True  
    except ImportError:
        pass
        
    try:
        import mne
        hardware_status['mne_available'] = True
    except ImportError:
        pass
    
    return hardware_status

# Export convenience functions
__all__.extend(['get_system_info', 'check_hardware_support'])
