"""
Real-time Processing Pipeline for Brain-Forge

This module implements advanced signal processing capabilities for multi-modal
brain data, including filtering, compression, feature extraction, and artifact removal.

The processing pipeline is organized into modular components:
- filters: Digital filtering and frequency analysis
- compression: Wavelet-based data compression
- artifacts: ICA and statistical artifact removal
- features: Spectral and temporal feature extraction
- realtime: Real-time processing orchestration
"""

# Import core processing classes and functions
from .filters import (
    RealTimeFilter,
    FilterBank,
    create_standard_filter_bank,
    ProcessingParameters
)

from .compression import (
    WaveletCompressor,
    CompressionManager,
    create_compressor
)

from .artifacts import (
    ICArtifactRemover,
    StatisticalArtifactRemover,
    HybridArtifactRemover,
    create_artifact_remover
)

from .features import (
    SpectralFeatureExtractor,
    TemporalFeatureExtractor,
    ComprehensiveFeatureExtractor,
    create_feature_extractor
)

from .realtime import (
    RealTimeProcessor,
    test_processor
)

__all__ = [
    # Core parameters
    'ProcessingParameters',

    # Filtering
    'RealTimeFilter',
    'FilterBank',
    'create_standard_filter_bank',

    # Compression
    'WaveletCompressor',
    'CompressionManager',
    'create_compressor',

    # Artifact removal
    'ICArtifactRemover',
    'StatisticalArtifactRemover',
    'HybridArtifactRemover',
    'create_artifact_remover',

    # Feature extraction
    'SpectralFeatureExtractor',
    'TemporalFeatureExtractor',
    'ComprehensiveFeatureExtractor',
    'create_feature_extractor',

    # Real-time processing
    'RealTimeProcessor',
    'test_processor'
]
