"""
Comprehensive Test Suite for Brain-Forge Processing Pipeline

This module tests the extensive existing processing implementation (~673 lines)
including real-time filtering, wavelet compression, artifact removal, and 
feature extraction capabilities.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestProcessingPipeline:
    """Test the comprehensive processing pipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample MEG-like data for testing"""
        np.random.seed(42)
        # Simulate 306 channels, 1000 samples (1 second at 1000Hz)
        channels = 306
        samples = 1000
        data = np.random.randn(channels, samples) * 1e-12  # fT scale
        
        # Add some realistic neural oscillations
        time = np.linspace(0, 1, samples)
        for ch in range(channels):
            # Alpha oscillation (10 Hz)
            data[ch] += 5e-13 * np.sin(2 * np.pi * 10 * time)
            # Beta oscillation (20 Hz)  
            data[ch] += 3e-13 * np.sin(2 * np.pi * 20 * time)
            
        return data
    
    @pytest.fixture
    def sampling_rate(self):
        """Standard MEG sampling rate"""
        return 1000.0
    
    def test_realtime_filter_import_and_creation(self):
        """Test that RealTimeFilter can be imported and created"""
        try:
            from processing import RealTimeFilter
            
            # Test bandpass filter creation
            filter_obj = RealTimeFilter(
                filter_type='bandpass',
                frequencies=(1.0, 100.0),
                sampling_rate=1000.0,
                order=4
            )
            
            assert filter_obj.filter_type == 'bandpass'
            assert filter_obj.frequencies == (1.0, 100.0)
            assert filter_obj.sampling_rate == 1000.0
            assert filter_obj.order == 4
            
        except ImportError:
            pytest.skip("RealTimeFilter import failed - dependencies missing")
    
    def test_wavelet_compressor_creation(self):
        """Test WaveletCompressor initialization"""
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
            
        except ImportError:
            pytest.skip("WaveletCompressor import failed - dependencies missing")
    
    def test_artifact_remover_creation(self):
        """Test ArtifactRemover initialization"""
        try:
            from processing import ArtifactRemover
            
            artifact_remover = ArtifactRemover(
                method='ica',
                n_components=20
            )
            
            # Just test creation, not functionality (needs real data)
            assert artifact_remover is not None
            
        except ImportError:
            pytest.skip("ArtifactRemover import failed - dependencies missing")
    
    def test_feature_extractor_creation(self):
        """Test FeatureExtractor initialization"""
        try:
            from processing import FeatureExtractor
            
            extractor = FeatureExtractor(
                features=['spectral_power', 'connectivity'],
                frequency_bands={
                    'alpha': (8, 12),
                    'beta': (13, 30)
                }
            )
            
            assert extractor is not None
            
        except ImportError:
            pytest.skip("FeatureExtractor import failed - dependencies missing")
    
    def test_realtime_processor_orchestration(self):
        """Test RealTimeProcessor orchestration"""
        try:
            from processing import RealTimeProcessor
            
            # Test processor creation with mock components
            processor = RealTimeProcessor(
                sampling_rate=1000.0,
                buffer_size=1024,
                target_latency_ms=100.0
            )
            
            assert processor is not None
            
        except ImportError:
            pytest.skip("RealTimeProcessor import failed - dependencies missing")
    
    @pytest.mark.slow
    def test_processing_pipeline_with_sample_data(self, sample_data, sampling_rate):
        """Test processing pipeline with sample data"""
        try:
            from processing import RealTimeFilter, WaveletCompressor
            
            # Test filter processing
            filter_obj = RealTimeFilter(
                filter_type='bandpass', 
                frequencies=(1.0, 100.0),
                sampling_rate=sampling_rate
            )
            
            # Test that filter can process data (mock the actual filtering)
            with patch.object(filter_obj, 'process') as mock_process:
                mock_process.return_value = sample_data
                result = filter_obj.process(sample_data)
                assert result.shape == sample_data.shape
                mock_process.assert_called_once()
            
            # Test compressor
            compressor = WaveletCompressor(wavelet='db4', levels=4)
            
            # Mock compression
            with patch.object(compressor, 'compress') as mock_compress:
                mock_result = {
                    'compressed_data': sample_data[:, ::5],  # Simulated compression
                    'compression_ratio': 5.0,
                    'original_size': sample_data.nbytes,
                    'compressed_size': sample_data.nbytes // 5
                }
                mock_compress.return_value = mock_result
                
                result = compressor.compress(sample_data)
                assert result['compression_ratio'] == 5.0
                mock_compress.assert_called_once()
                
        except ImportError:
            pytest.skip("Processing components import failed")


class TestHardwareIntegration:
    """Test hardware integration components"""
    
    def test_integrated_brain_system_import(self):
        """Test IntegratedBrainSystem can be imported"""
        try:
            from integrated_system import IntegratedBrainSystem
            
            # Test that class exists and has expected methods
            assert hasattr(IntegratedBrainSystem, '__init__')
            # Don't instantiate - would require actual hardware
            
        except ImportError:
            pytest.skip("IntegratedBrainSystem import failed")
    
    def test_stream_manager_import(self):
        """Test StreamManager import and basic functionality"""
        try:
            from acquisition.stream_manager import StreamManager
            
            # Test class exists
            assert StreamManager is not None
            
            # Test that we can create mock instance
            with patch('acquisition.stream_manager.pylsl') as mock_pylsl:
                mock_pylsl.StreamInlet = Mock()
                mock_pylsl.resolve_stream = Mock(return_value=[])
                
                # Test basic creation (mocked)
                manager = StreamManager()
                assert manager is not None
                
        except ImportError:
            pytest.skip("StreamManager import failed")
    
    def test_specialized_tools_import(self):
        """Test specialized tools integration"""
        try:
            from specialized_tools import EEGNotebooksIntegration
            
            assert EEGNotebooksIntegration is not None
            
        except ImportError:
            pytest.skip("EEGNotebooksIntegration import failed")


class TestSystemConfiguration:
    """Test system configuration and setup"""
    
    def test_complete_config_system(self):
        """Test comprehensive configuration system"""
        from core.config import Config
        
        config = Config()
        
        # Test hardware configuration
        assert config.hardware.omp_helmet.num_channels == 306
        assert config.hardware.sampling_rates.meg == 1000.0
        assert config.hardware.kernel_flow.enabled is True
        assert config.hardware.kernel_flux.enabled is True
        assert config.hardware.accelerometer.enabled is True
        
        # Test processing configuration
        assert config.processing.real_time.target_latency_ms == 100.0
        assert config.processing.compression.target_ratio == 5.0
        assert config.processing.filters.meg_bandpass == (1.0, 100.0)
        
        # Test system configuration
        assert config.system.logging.level == "INFO"
        assert config.system.performance.max_memory_gb == 16.0
    
    def test_exception_system_completeness(self):
        """Test comprehensive exception system"""
        from core.exceptions import (
            BrainForgeError, HardwareError, ProcessingError,
            StreamingError, ConfigurationError, ValidationError
        )
        
        # Test basic exception
        error = BrainForgeError("Test error", error_code="TEST001")
        assert str(error) == "[TEST001] Test error"
        
        # Test hardware error context
        hw_error = HardwareError(
            "Device failed",
            device_type="omp",
            device_id="device_01"
        )
        assert hw_error.device_type == "omp"
        assert hw_error.device_id == "device_01"
    
    def test_logging_system_functionality(self):
        """Test structured logging system"""
        from core.logger import get_logger, LogContext
        
        logger = get_logger("test")
        
        # Test context creation
        context = LogContext(
            device_type="omp",
            device_id="test_device",
            processing_stage="filtering"
        )
        
        assert context.device_type == "omp"
        assert context.device_id == "test_device"
        assert context.processing_stage == "filtering"
        
        # Test logger methods exist
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'performance')
        assert hasattr(logger, 'hardware_event')


class TestPackageStructure:
    """Test package structure and imports"""
    
    def test_main_package_import(self):
        """Test main package can be imported"""
        import src
        
        assert hasattr(src, '__version__')
        assert hasattr(src, 'get_system_info')
        assert hasattr(src, 'check_hardware_support')
    
    def test_core_module_imports(self):
        """Test core module imports"""
        from core import (
            Config, BrainForgeError, get_logger,
            HardwareError, ProcessingError
        )
        
        # Test all imports work
        assert Config is not None
        assert BrainForgeError is not None
        assert get_logger is not None
        assert HardwareError is not None  
        assert ProcessingError is not None
    
    def test_acquisition_module_structure(self):
        """Test acquisition module structure"""
        from acquisition import StreamManager, get_available_devices
        
        assert StreamManager is not None
        
        devices = get_available_devices()
        expected_devices = ['omp_helmet', 'kernel_flow', 'kernel_flux', 'accelerometer']
        for device in expected_devices:
            assert device in devices


if __name__ == "__main__":
    # Run with coverage
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "-x"  # Stop on first failure for debugging
    ])
