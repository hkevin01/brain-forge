"""
Comprehensive Unit Tests for Core Brain-Forge Exceptions

This test module provides complete coverage of the exception system,
testing all exception classes, their initialization, context handling,
and hierarchy relationships.
"""

import pytest
import sys
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.exceptions import (
    # Base exception
    BrainForgeError,
    
    # Hardware exceptions
    HardwareError,
    DeviceConnectionError,
    DeviceCalibrationError,
    SensorError,
    
    # Streaming exceptions
    StreamingError,
    BufferOverflowError,
    SynchronizationError,
    
    # Processing exceptions
    ProcessingError,
    FilteringError,
    CompressionError,
    FeatureExtractionError,
    ArtifactRemovalError,
    
    # Configuration exceptions
    ConfigurationError,
    ValidationError,
    
    # Visualization exceptions
    VisualizationError,
    
    # Storage exceptions
    StorageError,
    
    # API exceptions
    APIError,
    AuthenticationError,
    RateLimitError,
    
    # Timeout exception
    TimeoutError,
)


class TestBrainForgeBaseError:
    """Test the base BrainForgeError class"""
    
    def test_basic_initialization(self):
        """Test basic error initialization"""
        error = BrainForgeError("Test error message")
        assert str(error) == "Test error message"
        assert error.context == {}
        assert error.timestamp is not None
    
    def test_initialization_with_context(self):
        """Test error initialization with context"""
        context = {"component": "test", "value": 42}
        error = BrainForgeError("Test error", context=context)
        
        assert str(error) == "Test error"
        assert error.context == context
        assert error.timestamp is not None
    
    def test_add_context(self):
        """Test adding context to existing error"""
        error = BrainForgeError("Test error")
        error.add_context("key1", "value1")
        error.add_context("key2", 123)
        
        assert error.context["key1"] == "value1"
        assert error.context["key2"] == 123
    
    def test_get_detailed_message(self):
        """Test detailed message generation"""
        error = BrainForgeError("Test error")
        error.add_context("component", "processor")
        error.add_context("stage", "filtering")
        
        detailed_msg = error.get_detailed_message()
        assert "Test error" in detailed_msg
        assert "component: processor" in detailed_msg
        assert "stage: filtering" in detailed_msg


class TestHardwareErrors:
    """Test hardware-related exception classes"""
    
    def test_hardware_error_basic(self):
        """Test basic HardwareError functionality"""
        error = HardwareError("Hardware failure", device_type="omp")
        assert str(error) == "Hardware failure"
        assert error.device_type == "omp"
        assert error.device_id is None
    
    def test_hardware_error_with_device_id(self):
        """Test HardwareError with device ID"""
        error = HardwareError(
            "Device malfunction", 
            device_type="kernel_optical",
            device_id="device_001"
        )
        assert error.device_type == "kernel_optical"
        assert error.device_id == "device_001"
    
    def test_device_connection_error(self):
        """Test DeviceConnectionError functionality"""
        error = DeviceConnectionError(
            "Connection failed",
            device_type="accelerometer",
            connection_type="usb"
        )
        assert error.device_type == "accelerometer"
        assert error.connection_type == "usb"
        assert isinstance(error, HardwareError)
    
    def test_device_calibration_error(self):
        """Test DeviceCalibrationError functionality"""
        error = DeviceCalibrationError(
            "Calibration out of range",
            calibration_type="gain",
            expected_value=1.0,
            actual_value=1.5
        )
        assert error.calibration_type == "gain"
        assert error.expected_value == 1.0
        assert error.actual_value == 1.5
        assert isinstance(error, HardwareError)
    
    def test_sensor_error(self):
        """Test SensorError functionality"""
        error = SensorError(
            "Sensor reading invalid",
            sensor_id="MEG_001",
            channel=42
        )
        assert error.sensor_id == "MEG_001"
        assert error.channel == 42
        assert isinstance(error, HardwareError)


class TestStreamingErrors:
    """Test streaming-related exception classes"""
    
    def test_streaming_error_basic(self):
        """Test basic StreamingError functionality"""
        error = StreamingError("Stream failed", stream_type="meg")
        assert str(error) == "Stream failed"
        assert error.stream_type == "meg"
    
    def test_buffer_overflow_error(self):
        """Test BufferOverflowError functionality"""
        error = BufferOverflowError(
            "Buffer overflow detected",
            buffer_size=1000,
            overflow_amount=250
        )
        assert error.buffer_size == 1000
        assert error.overflow_amount == 250
        assert isinstance(error, StreamingError)
    
    def test_synchronization_error(self):
        """Test SynchronizationError functionality"""
        devices = ["omp_helmet", "kernel_optical"]
        error = SynchronizationError(
            "Devices out of sync",
            devices=devices,
            time_offset=0.005
        )
        assert error.devices == devices
        assert error.time_offset == 0.005
        assert isinstance(error, StreamingError)


class TestProcessingErrors:
    """Test processing-related exception classes"""
    
    def test_processing_error_basic(self):
        """Test basic ProcessingError functionality"""
        error = ProcessingError("Processing failed", processing_stage="filtering")
        assert str(error) == "Processing failed"
        assert error.processing_stage == "filtering"
    
    def test_filtering_error(self):
        """Test FilteringError functionality"""
        error = FilteringError(
            "Filter parameters invalid",
            filter_type="bandpass",
            cutoff_freq=50.0
        )
        assert error.filter_type == "bandpass"
        assert error.cutoff_freq == 50.0
        assert isinstance(error, ProcessingError)
    
    def test_compression_error(self):
        """Test CompressionError functionality"""
        error = CompressionError(
            "Compression failed",
            compression_algorithm="wavelet",
            compression_ratio=5.0
        )
        assert error.compression_algorithm == "wavelet"
        assert error.compression_ratio == 5.0
        assert isinstance(error, ProcessingError)
    
    def test_feature_extraction_error(self):
        """Test FeatureExtractionError functionality"""
        error = FeatureExtractionError(
            "Feature extraction failed",
            feature_type="spectral",
            extraction_method="PCA"
        )
        assert error.feature_type == "spectral"
        assert error.extraction_method == "PCA"
        assert isinstance(error, ProcessingError)
    
    def test_artifact_removal_error(self):
        """Test ArtifactRemovalError functionality"""
        error = ArtifactRemovalError(
            "Artifact removal failed",
            artifact_type="motion",
            removal_method="ICA"
        )
        assert error.artifact_type == "motion"
        assert error.removal_method == "ICA"
        assert isinstance(error, ProcessingError)


class TestConfigurationErrors:
    """Test configuration-related exception classes"""
    
    def test_configuration_error(self):
        """Test ConfigurationError functionality"""
        error = ConfigurationError(
            "Invalid configuration",
            config_section="hardware",
            config_key="sampling_rate"
        )
        assert error.config_section == "hardware"
        assert error.config_key == "sampling_rate"
    
    def test_validation_error(self):
        """Test ValidationError functionality"""
        error = ValidationError(
            "Validation failed",
            field_name="channels",
            expected_type="int"
        )
        assert error.field_name == "channels"
        assert error.expected_type == "int"
        assert isinstance(error, ConfigurationError)


class TestVisualizationErrors:
    """Test visualization-related exception classes"""
    
    def test_visualization_error(self):
        """Test VisualizationError functionality"""
        error = VisualizationError(
            "Rendering failed",
            plot_type="3d_brain",
            rendering_backend="mayavi"
        )
        assert error.plot_type == "3d_brain"
        assert error.rendering_backend == "mayavi"
        assert isinstance(error, BrainForgeError)


class TestStorageErrors:
    """Test storage-related exception classes"""
    
    def test_storage_error(self):
        """Test StorageError functionality"""
        error = StorageError(
            "File write failed",
            file_path="/path/to/file.h5",
            operation="write"
        )
        assert error.file_path == "/path/to/file.h5"
        assert error.operation == "write"
        assert isinstance(error, BrainForgeError)


class TestAPIErrors:
    """Test API-related exception classes"""
    
    def test_api_error_basic(self):
        """Test basic APIError functionality"""
        error = APIError(
            "API request failed",
            endpoint="/api/v1/scan",
            status_code=500
        )
        assert error.endpoint == "/api/v1/scan"
        assert error.status_code == 500
    
    def test_authentication_error(self):
        """Test AuthenticationError functionality"""
        error = AuthenticationError(
            "Authentication failed",
            auth_method="token",
            user_id="user123"
        )
        assert error.auth_method == "token"
        assert error.user_id == "user123"
        assert isinstance(error, APIError)
    
    def test_rate_limit_error(self):
        """Test RateLimitError functionality"""
        error = RateLimitError(
            "Rate limit exceeded",
            limit=100,
            retry_after=60
        )
        assert error.limit == 100
        assert error.retry_after == 60
        assert isinstance(error, APIError)


class TestTimeoutError:
    """Test timeout exception class"""
    
    def test_timeout_error(self):
        """Test TimeoutError functionality"""
        error = TimeoutError(
            "Operation timed out",
            timeout_duration=30.0,
            operation="brain_scan"
        )
        assert error.timeout_duration == 30.0
        assert error.operation == "brain_scan"
        assert isinstance(error, BrainForgeError)


class TestExceptionHierarchy:
    """Test exception class inheritance hierarchy"""
    
    def test_hardware_exception_hierarchy(self):
        """Test hardware exception inheritance"""
        device_error = DeviceConnectionError("Connection failed")
        sensor_error = SensorError("Sensor failed")
        calib_error = DeviceCalibrationError("Calibration failed")
        
        # All should be instances of HardwareError and BrainForgeError
        for error in [device_error, sensor_error, calib_error]:
            assert isinstance(error, HardwareError)
            assert isinstance(error, BrainForgeError)
            assert isinstance(error, Exception)
    
    def test_streaming_exception_hierarchy(self):
        """Test streaming exception inheritance"""
        buffer_error = BufferOverflowError("Buffer overflow")
        sync_error = SynchronizationError("Sync lost")
        
        for error in [buffer_error, sync_error]:
            assert isinstance(error, StreamingError)
            assert isinstance(error, BrainForgeError)
            assert isinstance(error, Exception)
    
    def test_processing_exception_hierarchy(self):
        """Test processing exception inheritance"""
        filter_error = FilteringError("Filter failed")
        compress_error = CompressionError("Compression failed")
        feature_error = FeatureExtractionError("Feature extraction failed")
        artifact_error = ArtifactRemovalError("Artifact removal failed")
        
        for error in [filter_error, compress_error, feature_error, artifact_error]:
            assert isinstance(error, ProcessingError)
            assert isinstance(error, BrainForgeError)
            assert isinstance(error, Exception)
    
    def test_api_exception_hierarchy(self):
        """Test API exception inheritance"""
        auth_error = AuthenticationError("Auth failed")
        rate_error = RateLimitError("Rate limited")
        
        for error in [auth_error, rate_error]:
            assert isinstance(error, APIError)
            assert isinstance(error, BrainForgeError)
            assert isinstance(error, Exception)


class TestExceptionContextualInformation:
    """Test contextual information handling in exceptions"""
    
    def test_error_context_preservation(self):
        """Test that context is preserved through exception hierarchy"""
        context = {"module": "test", "function": "test_func", "line": 42}
        
        # Test with different exception types
        errors = [
            HardwareError("Hardware error", context=context),
            ProcessingError("Processing error", context=context),
            StreamingError("Streaming error", context=context),
        ]
        
        for error in errors:
            assert error.context == context
            detailed_msg = error.get_detailed_message()
            assert "module: test" in detailed_msg
            assert "function: test_func" in detailed_msg
            assert "line: 42" in detailed_msg
    
    def test_error_timestamp(self):
        """Test that all errors have timestamps"""
        import time
        
        start_time = time.time()
        error = BrainForgeError("Test error")
        end_time = time.time()
        
        assert error.timestamp is not None
        assert start_time <= error.timestamp <= end_time
    
    def test_multiple_context_additions(self):
        """Test adding multiple context items"""
        error = BrainForgeError("Test error")
        
        error.add_context("step", "initialization")
        error.add_context("device", "omp_helmet")
        error.add_context("attempt", 3)
        
        assert error.context["step"] == "initialization"
        assert error.context["device"] == "omp_helmet"
        assert error.context["attempt"] == 3
        
        # Test that detailed message includes all context
        detailed_msg = error.get_detailed_message()
        assert "step: initialization" in detailed_msg
        assert "device: omp_helmet" in detailed_msg
        assert "attempt: 3" in detailed_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
