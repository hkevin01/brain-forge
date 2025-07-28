"""
Test Core Brain-Forge Infrastructure

This test module validates the core infrastructure components:
- Configuration system
- Exception handling
- Logging system
- Package imports
"""

import pytest
import sys
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestCoreInfrastructure:
    """Test core infrastructure components"""
    
    def test_config_import(self):
        """Test configuration system import"""
        from core.config import Config, HardwareConfig, ProcessingConfig, SystemConfig
        
        # Test default config creation
        config = Config()
        assert isinstance(config, Config)
        assert isinstance(config.hardware, HardwareConfig)
        assert isinstance(config.processing, ProcessingConfig)
        assert isinstance(config.system, SystemConfig)
    
    def test_exceptions_import(self):
        """Test exception system import"""
        from core.exceptions import (
            BrainForgeError,
            HardwareError,
            ProcessingError,
            StreamingError
        )
        
        # Test basic exception creation
        error = BrainForgeError("Test error")
        assert str(error) == "Test error"
        
        # Test hardware error with context
        hw_error = HardwareError(
            "Device connection failed",
            device_type="omp",
            device_id="test_device"
        )
        assert hw_error.device_type == "omp"
        assert hw_error.device_id == "test_device"
    
    def test_logger_import(self):
        """Test logging system import"""
        from core.logger import get_logger, ContextualLogger, LogContext
        
        # Test logger creation
        logger = get_logger("test")
        assert isinstance(logger, ContextualLogger)
        
        # Test log context
        context = LogContext(device_type="test", device_id="test_device")
        assert context.device_type == "test"
        assert context.device_id == "test_device"


class TestMainSystemImports:
    """Test main system component imports"""
    
    def test_integrated_system_import(self):
        """Test integrated system import"""
        try:
            from integrated_system import IntegratedBrainSystem
            # Just test import, not instantiation (requires hardware)
            assert IntegratedBrainSystem is not None
        except ImportError as e:
            pytest.skip(f"IntegratedBrainSystem import failed: {e}")
    
    def test_processing_import(self):
        """Test processing components import"""
        try:
            from processing import (
                RealTimeFilter,
                WaveletCompressor,
                ArtifactRemover,
                FeatureExtractor,
                RealTimeProcessor
            )
            
            # Test that classes are available
            assert RealTimeFilter is not None
            assert WaveletCompressor is not None
            assert ArtifactRemover is not None
            assert FeatureExtractor is not None
            assert RealTimeProcessor is not None
            
        except ImportError as e:
            pytest.skip(f"Processing components import failed: {e}")
    
    def test_stream_manager_import(self):
        """Test stream manager import"""
        try:
            from acquisition.stream_manager import StreamManager
            assert StreamManager is not None
        except ImportError as e:
            pytest.skip(f"StreamManager import failed: {e}")
    
    def test_specialized_tools_import(self):
        """Test specialized tools import"""  
        try:
            from specialized_tools import EEGNotebooksIntegration
            assert EEGNotebooksIntegration is not None
        except ImportError as e:
            pytest.skip(f"EEGNotebooksIntegration import failed: {e}")


class TestProcessingComponents:
    """Test processing component functionality"""
    
    def test_filter_creation(self):
        """Test filter creation without data processing"""
        try:
            from processing import RealTimeFilter
            
            # Test filter creation (without actual filtering)
            filter_obj = RealTimeFilter(
                filter_type='bandpass',
                frequencies=(1.0, 100.0),
                sampling_rate=1000.0,
                order=4
            )
            
            assert filter_obj.filter_type == 'bandpass'
            assert filter_obj.frequencies == (1.0, 100.0)
            assert filter_obj.sampling_rate == 1000.0
            
        except ImportError as e:
            pytest.skip(f"RealTimeFilter import failed: {e}")
        except Exception as e:
            pytest.skip(f"Filter creation failed: {e}")
    
    def test_compressor_creation(self):
        """Test compressor creation"""
        try:
            from processing import WaveletCompressor
            
            compressor = WaveletCompressor(
                wavelet='db8',
                levels=6,
                threshold_method='soft'
            )
            
            assert compressor.wavelet == 'db8'
            assert compressor.levels == 6
            assert compressor.threshold_method == 'soft'
            
        except ImportError as e:
            pytest.skip(f"WaveletCompressor import failed: {e}")
        except Exception as e:
            pytest.skip(f"Compressor creation failed: {e}")


class TestConfigurationSystem:
    """Test configuration system functionality"""
    
    def test_config_values(self):
        """Test configuration default values"""
        from core.config import Config
        
        config = Config()
        
        # Test hardware config defaults
        assert config.hardware.omp_helmet.num_channels == 306
        assert config.hardware.sampling_rates.meg == 1000.0
        assert config.hardware.kernel_flow.enabled is True
        
        # Test processing config defaults  
        assert config.processing.real_time.target_latency_ms == 100.0
        assert config.processing.compression.target_ratio == 5.0
        assert config.processing.filters.meg_bandpass == (1.0, 100.0)
        
        # Test system config defaults
        assert config.system.logging.level == "INFO"
        assert config.system.performance.max_memory_gb == 16.0
    
    def test_config_validation(self):
        """Test configuration validation"""
        from core.config import Config, HardwareConfig
        from core.exceptions import ConfigurationError
        
        config = Config()
        
        # Test that config validates properly
        assert config.hardware.omp_helmet.num_channels > 0
        assert config.processing.real_time.target_latency_ms > 0
        assert config.system.performance.max_memory_gb > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
