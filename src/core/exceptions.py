"""
Custom Exception Classes for Brain-Forge

This module defines all custom exceptions used throughout the Brain-Forge system,
providing specific error types for different components and failure modes.
"""

from typing import Optional, Any, Dict, List


class BrainForgeError(Exception):
    """Base exception class for all Brain-Forge errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize Brain-Forge error
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary of additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        """String representation of the error"""
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        return base_msg


class HardwareError(BrainForgeError):
    """Exceptions related to hardware interface failures"""
    
    def __init__(self, message: str, device_type: Optional[str] = None,
                 device_id: Optional[str] = None, **kwargs):
        """
        Initialize hardware error
        
        Args:
            message: Error message
            device_type: Type of hardware device (omp, kernel, accelerometer)
            device_id: Specific device identifier
        """
        super().__init__(message, **kwargs)
        self.device_type = device_type
        self.device_id = device_id


class DeviceConnectionError(HardwareError):
    """Device connection and communication failures"""
    
    def __init__(self, message: str, device_type: Optional[str] = None,
                 connection_type: Optional[str] = None, **kwargs):
        """
        Initialize device connection error
        
        Args:
            message: Error message
            device_type: Type of device (omp, kernel, accel)
            connection_type: Connection method (usb, wireless, ethernet)
        """
        super().__init__(message, device_type=device_type, **kwargs)
        self.connection_type = connection_type


class DeviceCalibrationError(HardwareError):
    """Device calibration failures"""
    
    def __init__(self, message: str, calibration_type: Optional[str] = None,
                 expected_value: Optional[float] = None,
                 actual_value: Optional[float] = None, **kwargs):
        """
        Initialize device calibration error
        
        Args:
            message: Error message
            calibration_type: Type of calibration (gain, offset, noise)
            expected_value: Expected calibration value
            actual_value: Actual measured value
        """
        super().__init__(message, **kwargs)
        self.calibration_type = calibration_type
        self.expected_value = expected_value
        self.actual_value = actual_value


class SensorError(HardwareError):
    """Individual sensor failures"""
    
    def __init__(self, message: str, sensor_id: Optional[str] = None,
                 channel: Optional[int] = None, **kwargs):
        """
        Initialize sensor error
        
        Args:
            message: Error message
            sensor_id: Identifier of the problematic sensor
            channel: Channel number if applicable
        """
        super().__init__(message, **kwargs)
        self.sensor_id = sensor_id
        self.channel = channel


class StreamingError(BrainForgeError):
    """Data streaming and acquisition errors"""
    
    def __init__(self, message: str, stream_type: Optional[str] = None,
                 **kwargs):
        """
        Initialize streaming error
        
        Args:
            message: Error message
            stream_type: Type of data stream (meg, optical, accel)
        """
        super().__init__(message, **kwargs)
        self.stream_type = stream_type


class BufferOverflowError(StreamingError):
    """Data buffer overflow errors"""
    
    def __init__(self, message: str, buffer_size: Optional[int] = None,
                 overflow_amount: Optional[int] = None, **kwargs):
        """
        Initialize buffer overflow error
        
        Args:
            message: Error message
            buffer_size: Maximum buffer capacity
            overflow_amount: Amount of data that caused overflow
        """
        super().__init__(message, **kwargs)
        self.buffer_size = buffer_size
        self.overflow_amount = overflow_amount


class SynchronizationError(StreamingError):
    """Multi-device synchronization errors"""
    
    def __init__(self, message: str, devices: Optional[List[str]] = None,
                 time_offset: Optional[float] = None, **kwargs):
        """
        Initialize synchronization error
        
        Args:
            message: Error message
            devices: List of devices that lost sync
            time_offset: Time offset in seconds
        """
        super().__init__(message, **kwargs)
        self.devices = devices or []
        self.time_offset = time_offset


class ProcessingError(BrainForgeError):
    """Signal processing and analysis errors"""
    
    def __init__(self, message: str, processing_stage: Optional[str] = None,
                 **kwargs):
        """
        Initialize processing error
        
        Args:
            message: Error message
            processing_stage: Stage where error occurred (filter, compress, etc.)
        """
        super().__init__(message, **kwargs)
        self.processing_stage = processing_stage


class FilteringError(ProcessingError):
    """Digital filtering errors"""
    
    def __init__(self, message: str, filter_type: Optional[str] = None,
                 cutoff_freq: Optional[float] = None, **kwargs):
        """
        Initialize filtering error
        
        Args:
            message: Error message
            filter_type: Type of filter (lowpass, highpass, bandpass)
            cutoff_freq: Filter cutoff frequency
        """
        super().__init__(message, **kwargs)
        self.filter_type = filter_type
        self.cutoff_freq = cutoff_freq


class CompressionError(ProcessingError):
    """Data compression errors"""
    
    def __init__(self, message: str, compression_algorithm: Optional[str] = None,
                 compression_ratio: Optional[float] = None, **kwargs):
        """
        Initialize compression error
        
        Args:
            message: Error message
            compression_algorithm: Algorithm used (wavelet, neural_lz, etc.)
            compression_ratio: Target compression ratio
        """
        super().__init__(message, **kwargs)
        self.compression_algorithm = compression_algorithm
        self.compression_ratio = compression_ratio


class FeatureExtractionError(ProcessingError):
    """Feature extraction errors"""
    
    def __init__(self, message: str, feature_type: Optional[str] = None,
                 extraction_method: Optional[str] = None, **kwargs):
        """
        Initialize feature extraction error
        
        Args:
            message: Error message
            feature_type: Type of feature (spectral, temporal, spatial)
            extraction_method: Method used (PCA, ICA, transformers)
        """
        super().__init__(message, **kwargs)
        self.feature_type = feature_type
        self.extraction_method = extraction_method


class ArtifactRemovalError(ProcessingError):
    """Artifact removal errors"""
    
    def __init__(self, message: str, artifact_type: Optional[str] = None,
                 removal_method: Optional[str] = None, **kwargs):
        """
        Initialize artifact removal error
        
        Args:
            message: Error message
            artifact_type: Type of artifact (motion, eye, muscle)
            removal_method: Method used (ICA, regression, filtering)
        """
        super().__init__(message, **kwargs)
        self.artifact_type = artifact_type
        self.removal_method = removal_method


class ConfigurationError(BrainForgeError):
    """Configuration and setup errors"""
    
    def __init__(self, message: str, config_section: Optional[str] = None,
                 config_key: Optional[str] = None, **kwargs):
        """
        Initialize configuration error
        
        Args:
            message: Error message
            config_section: Configuration section (hardware, processing, system)
            config_key: Specific configuration key
        """
        super().__init__(message, **kwargs)
        self.config_section = config_section
        self.config_key = config_key


class ValidationError(BrainForgeError):
    """Data validation and format errors"""
    
    def __init__(self, message: str, data_type: Optional[str] = None,
                 expected_format: Optional[str] = None, **kwargs):
        """
        Initialize validation error
        
        Args:
            message: Error message
            data_type: Type of data being validated
            expected_format: Expected data format
        """
        super().__init__(message, **kwargs)
        self.data_type = data_type
        self.expected_format = expected_format


class SimulationError(BrainForgeError):
    """Neural simulation errors"""
    
    def __init__(self, message: str, simulation_type: Optional[str] = None,
                 **kwargs):
        """
        Initialize simulation error
        
        Args:
            message: Error message
            simulation_type: Type of simulation (brian2, nest, etc.)
        """
        super().__init__(message, **kwargs)
        self.simulation_type = simulation_type


class ModelError(BrainForgeError):
    """Machine learning model errors"""
    
    def __init__(self, message: str, model_type: Optional[str] = None,
                 **kwargs):
        """
        Initialize model error
        
        Args:
            message: Error message
            model_type: Type of model (neural network, classifier, etc.)
        """
        super().__init__(message, **kwargs)
        self.model_type = model_type


class VisualizationError(BrainForgeError):
    """Visualization and plotting errors"""
    
    def __init__(self, message: str, plot_type: Optional[str] = None,
                 rendering_backend: Optional[str] = None, **kwargs):
        """
        Initialize visualization error
        
        Args:
            message: Error message
            plot_type: Type of plot (3d_brain, connectivity, time_series)
            rendering_backend: Backend used (mayavi, plotly, matplotlib)
        """
        super().__init__(message, **kwargs)
        self.plot_type = plot_type
        self.rendering_backend = rendering_backend


class StorageError(BrainForgeError):
    """Data storage and I/O errors"""
    
    def __init__(self, message: str, file_path: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        """
        Initialize storage error
        
        Args:
            message: Error message
            file_path: Path to problematic file
            operation: Storage operation (read, write, delete)
        """
        super().__init__(message, **kwargs)
        self.file_path = file_path
        self.operation = operation


class APIError(BrainForgeError):
    """API and external interface errors"""
    
    def __init__(self, message: str, endpoint: Optional[str] = None,
                 status_code: Optional[int] = None, **kwargs):
        """
        Initialize API error
        
        Args:
            message: Error message
            endpoint: API endpoint
            status_code: HTTP status code if applicable
        """
        super().__init__(message, **kwargs)
        self.endpoint = endpoint
        self.status_code = status_code


class AuthenticationError(APIError):
    """Authentication and authorization errors"""
    
    def __init__(self, message: str, auth_method: Optional[str] = None,
                 user_id: Optional[str] = None, **kwargs):
        """
        Initialize authentication error
        
        Args:
            message: Error message
            auth_method: Authentication method (token, basic, oauth)
            user_id: User identifier if available
        """
        super().__init__(message, **kwargs)
        self.auth_method = auth_method
        self.user_id = user_id


class RateLimitError(APIError):
    """API rate limiting errors"""
    
    def __init__(self, message: str, limit: Optional[int] = None,
                 retry_after: Optional[int] = None, **kwargs):
        """
        Initialize rate limit error
        
        Args:
            message: Error message
            limit: Rate limit (requests per time period)
            retry_after: Seconds to wait before retry
        """
        super().__init__(message, **kwargs)
        self.limit = limit
        self.retry_after = retry_after


class TimeoutError(BrainForgeError):
    """Operation timeout errors"""
    
    def __init__(self, message: str, timeout_duration: Optional[float] = None,
                 operation: Optional[str] = None, **kwargs):
        """
        Initialize timeout error
        
        Args:
            message: Error message
            timeout_duration: Timeout duration in seconds
            operation: Operation that timed out
        """
        super().__init__(message, **kwargs)
        self.timeout_duration = timeout_duration
        self.operation = operation


class MemoryError(BrainForgeError):
    """Memory allocation and management errors"""
    
    def __init__(self, message: str, requested_size: Optional[int] = None,
                 available_size: Optional[int] = None, **kwargs):
        """
        Initialize memory error
        
        Args:
            message: Error message
            requested_size: Requested memory size in bytes
            available_size: Available memory size in bytes
        """
        super().__init__(message, **kwargs)
        self.requested_size = requested_size
        self.available_size = available_size


class PerformanceError(BrainForgeError):
    """Performance and real-time constraint violations"""
    
    def __init__(self, message: str, target_latency: Optional[float] = None,
                 actual_latency: Optional[float] = None, **kwargs):
        """
        Initialize performance error
        
        Args:
            message: Error message
            target_latency: Target processing latency in seconds
            actual_latency: Actual processing latency in seconds
        """
        super().__init__(message, **kwargs)
        self.target_latency = target_latency
        self.actual_latency = actual_latency


# Error code constants for programmatic error handling
class ErrorCodes:
    """Error code constants for programmatic handling"""
    
    # Hardware error codes
    HARDWARE_CONNECTION_FAILED = "HW001"
    HARDWARE_CALIBRATION_FAILED = "HW002"
    SENSOR_MALFUNCTION = "HW003"
    DEVICE_NOT_FOUND = "HW004"
    
    # Streaming error codes
    STREAM_BUFFER_OVERFLOW = "ST001"
    STREAM_SYNC_FAILED = "ST002"
    STREAM_CONNECTION_LOST = "ST003"
    
    # Processing error codes
    PROCESSING_FILTER_FAILED = "PR001"
    PROCESSING_COMPRESSION_FAILED = "PR002"
    PROCESSING_FEATURE_EXTRACTION_FAILED = "PR003"
    PROCESSING_ARTIFACT_REMOVAL_FAILED = "PR004"
    
    # Configuration error codes
    CONFIG_INVALID_PARAMETER = "CF001"
    CONFIG_MISSING_REQUIRED = "CF002"
    CONFIG_FILE_NOT_FOUND = "CF003"
    
    # Validation error codes
    VALIDATION_INVALID_FORMAT = "VL001"
    VALIDATION_OUT_OF_RANGE = "VL002"
    VALIDATION_MISSING_DATA = "VL003"
    
    # Performance error codes
    PERFORMANCE_LATENCY_EXCEEDED = "PF001"
    PERFORMANCE_MEMORY_EXCEEDED = "PF002"
    PERFORMANCE_THROUGHPUT_LOW = "PF003"


def create_error_with_context(error_class: type, message: str, 
                            context: Dict[str, Any]) -> BrainForgeError:
    """
    Create an error instance with additional context information
    
    Args:
        error_class: Exception class to instantiate
        message: Error message
        context: Dictionary of contextual information
        
    Returns:
        Configured error instance
    """
    return error_class(message, details=context)


def handle_hardware_error(device_type: str, device_id: str, 
                         error: Exception) -> HardwareError:
    """
    Convert generic exception to hardware error with context
    
    Args:
        device_type: Type of hardware device
        device_id: Device identifier
        error: Original exception
        
    Returns:
        HardwareError with device context
    """
    return HardwareError(
        message=f"Hardware error in {device_type} device {device_id}: {str(error)}",
        device_type=device_type,
        device_id=device_id,
        details={'original_error': str(error), 'error_type': type(error).__name__}
    )


def handle_processing_error(stage: str, error: Exception) -> ProcessingError:
    """
    Convert generic exception to processing error with context
    
    Args:
        stage: Processing stage where error occurred
        error: Original exception
        
    Returns:
        ProcessingError with stage context
    """
    return ProcessingError(
        message=f"Processing error in {stage}: {str(error)}",
        processing_stage=stage,
        details={'original_error': str(error), 'error_type': type(error).__name__}
    )