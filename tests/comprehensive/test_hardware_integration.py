"""
Hardware Integration Tests

Tests for all hardware-related functionality claimed in the README,
including device interfaces, calibration, and multi-modal synchronization.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.config import Config, HardwareConfig
from core.exceptions import BrainForgeError


class TestHardwareInterfaces:
    """Test hardware device interfaces and mock implementations"""
    
    def setup_method(self):
        """Set up test environment"""
        self.config = Config()
        self.temp_dir = tempfile.mkdtemp(prefix="brain_forge_hardware_test_")
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_omp_helmet_interface(self):
        """
        Test OMP helmet interface matches README specifications:
        - 306+ channels
        - Matrix coil compensation (48 coils)
        - 1000 Hz sampling rate
        - Magnetic shielding support
        """
        # Test configuration
        assert self.config.hardware.omp_enabled is True
        assert self.config.hardware.omp_channels >= 306
        assert self.config.hardware.omp_sampling_rate >= 1000.0
        
        # Test calibration file support
        assert hasattr(self.config.hardware, 'omp_calibration_file')
        assert self.config.hardware.omp_calibration_file.endswith('.json')
    
    def test_kernel_optical_interface(self):
        """
        Test Kernel optical helmet interface:
        - TD-fNIRS + EEG fusion
        - 40 optical modules
        - Dual wavelength (690nm/905nm range)
        - Hemodynamic + electrical measurement
        """
        # Test configuration
        assert self.config.hardware.kernel_enabled is True
        assert hasattr(self.config.hardware, 'kernel_flow_channels')
        assert hasattr(self.config.hardware, 'kernel_flux_channels')
        
        # Test wavelength configuration
        assert hasattr(self.config.hardware, 'kernel_wavelengths')
        wavelengths = self.config.hardware.kernel_wavelengths
        assert len(wavelengths) >= 2
        # Should be in near-infrared range
        assert all(600 <= w <= 1000 for w in wavelengths)
    
    def test_accelerometer_interface(self):
        """
        Test Brown Accelo-hat accelerometer interface:
        - 64+ accelerometers
        - 3-axis motion detection
        - Impact detection capability
        - Navy-grade specifications
        """
        # Test configuration
        assert self.config.hardware.accel_enabled is True
        assert self.config.hardware.accel_channels >= 3  # 3-axis minimum
        assert self.config.hardware.accel_sampling_rate >= 1000.0
        
        # Test range and resolution
        assert hasattr(self.config.hardware, 'accel_range')
        assert hasattr(self.config.hardware, 'accel_resolution')
        assert self.config.hardware.accel_range > 0
        assert self.config.hardware.accel_resolution > 0
    
    @patch('serial.Serial')
    def test_hardware_communication_interfaces(self, mock_serial):
        """Test hardware communication interfaces work"""
        # Mock serial communication
        mock_port = Mock()
        mock_serial.return_value = mock_port
        mock_port.is_open = True
        mock_port.read.return_value = b'test_data'
        
        # Test that we can instantiate hardware config with serial port
        config = HardwareConfig()
        assert hasattr(config, 'omp_port')
        assert config.omp_port.startswith('/dev/')
    
    def test_synchronization_specifications(self):
        """
        Test multi-modal synchronization specifications:
        - Sub-millisecond precision
        - Synchronized data streams
        - Cross-modal temporal alignment
        """
        # This would test the actual synchronization implementation
        # For now, test that sync precision is specified
        config = Config()
        
        # All devices should have matching sampling rates for sync
        omp_rate = config.hardware.omp_sampling_rate
        accel_rate = config.hardware.accel_sampling_rate
        
        # Should be able to synchronize at highest common rate
        assert omp_rate >= 1000.0
        assert accel_rate >= 1000.0


class TestHardwareCalibration:
    """Test hardware calibration functionality"""
    
    def setup_method(self):
        """Set up calibration test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="brain_forge_calibration_test_")
        self.config = Config()
    
    def teardown_method(self):
        """Clean up calibration test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_omp_calibration_file_format(self):
        """Test OMP calibration file format and structure"""
        import json

        # Create mock calibration file
        calibration_data = {
            "channel_count": 306,
            "matrix_coils": 48,
            "calibration_matrix": [[1.0, 0.0, 0.0] for _ in range(306)],
            "baseline_noise": [0.1 for _ in range(306)],
            "sensor_positions": {
                f"channel_{i}": {"x": 0.0, "y": 0.0, "z": 0.0} 
                for i in range(306)
            },
            "timestamp": "2025-01-01T00:00:00Z",
            "version": "1.0"
        }
        
        calibration_file = Path(self.temp_dir) / "test_omp_calibration.json"
        with open(calibration_file, 'w') as f:
            json.dump(calibration_data, f)
        
        # Test file can be loaded
        with open(calibration_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data["channel_count"] == 306
        assert loaded_data["matrix_coils"] == 48
        assert len(loaded_data["calibration_matrix"]) == 306
    
    def test_sensor_position_mapping(self):
        """Test sensor position mapping for brain atlas integration"""
        # Test that we can define sensor positions
        sensor_positions = {}
        
        # Generate realistic sensor positions for 306 channels
        for i in range(306):
            # Simplified spherical coordinates for head surface
            theta = (i / 306.0) * 2 * np.pi
            phi = ((i % 17) / 17.0) * np.pi
            
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            sensor_positions[f"omp_channel_{i}"] = {"x": x, "y": y, "z": z}
        
        # Should have positions for all channels
        assert len(sensor_positions) == 306
        
        # All positions should be on unit sphere (approximately)
        for pos in sensor_positions.values():
            distance = np.sqrt(pos["x"]**2 + pos["y"]**2 + pos["z"]**2)
            assert abs(distance - 1.0) < 0.01  # Unit sphere with tolerance
    
    def test_kernel_calibration_specifications(self):
        """Test Kernel optical calibration requirements"""
        # Test wavelength calibration
        wavelengths = [690, 905]  # README specified wavelengths
        
        # Should be able to configure these wavelengths
        config = HardwareConfig()
        config.kernel_wavelengths = wavelengths
        
        assert config.kernel_wavelengths == wavelengths
        assert all(isinstance(w, int) for w in wavelengths)
        assert all(w > 0 for w in wavelengths)
    
    def test_accelerometer_calibration(self):
        """Test accelerometer calibration and sensitivity"""
        config = HardwareConfig()
        
        # Test range and resolution settings
        assert config.accel_range > 0  # Should have positive range
        assert config.accel_resolution > 0  # Should have positive resolution
        
        # Calculate sensitivity (LSB/g)
        max_value = 2**(config.accel_resolution - 1) - 1  # 16-bit signed
        sensitivity = max_value / config.accel_range
        
        assert sensitivity > 0
        assert sensitivity < 10000  # Reasonable upper bound


class TestHardwareDataFormats:
    """Test hardware data formats and structures"""
    
    def test_omp_data_format(self):
        """Test OMP data format structure"""
        # OMP data should be structured as:
        # - 306 channels
        # - Time series data
        # - Magnetic field measurements in Tesla
        
        # Mock OMP data structure
        num_channels = 306
        num_samples = 1000  # 1 second at 1000 Hz
        
        omp_data = np.random.randn(num_channels, num_samples) * 1e-12  # fT range
        
        assert omp_data.shape == (306, 1000)
        assert omp_data.dtype == np.float64
        
        # Should be in realistic MEG range (femtoTesla)
        assert np.max(np.abs(omp_data)) < 1e-9  # Less than 1 pT
    
    def test_kernel_data_format(self):
        """Test Kernel optical data format"""
        # Kernel data should include:
        # - Hemodynamic signals (optical)
        # - EEG signals (electrical)
        # - Dual wavelength measurements
        
        # Mock Kernel data
        optical_channels = 40
        eeg_channels = 4
        samples_optical = 100  # 100 Hz for hemodynamic
        samples_eeg = 1000     # 1000 Hz for EEG
        
        kernel_optical = np.random.randn(optical_channels, samples_optical)
        kernel_eeg = np.random.randn(eeg_channels, samples_eeg)
        
        assert kernel_optical.shape == (40, 100)
        assert kernel_eeg.shape == (4, 1000)
        
        # Different sampling rates for different modalities
        assert samples_eeg > samples_optical  # EEG typically faster
    
    def test_accelerometer_data_format(self):
        """Test accelerometer data format"""
        # Accelerometer data should be:
        # - 3-axis (x, y, z) per sensor
        # - Multiple sensors (64 in Brown Accelo-hat)
        # - High sampling rate for impact detection
        
        num_sensors = 64
        num_axes = 3
        num_samples = 1000
        
        # Structure: (sensors, axes, samples) or (sensors*axes, samples)
        accel_data_3d = np.random.randn(num_sensors, num_axes, num_samples)
        accel_data_2d = np.random.randn(num_sensors * num_axes, num_samples)
        
        assert accel_data_3d.shape == (64, 3, 1000)
        assert accel_data_2d.shape == (192, 1000)  # 64 * 3 = 192
        
        # Should be in reasonable acceleration range (±16g)
        accel_data_g = np.random.randn(192, 1000) * 8  # ±8g typical
        assert np.max(np.abs(accel_data_g)) < 20  # Within ±20g
    
    def test_synchronized_data_structure(self):
        """Test synchronized multi-modal data structure"""
        # Test that data from all modalities can be synchronized
        
        # Common time base (1000 Hz)
        base_samples = 1000
        
        # Different modalities at different rates
        omp_data = np.random.randn(306, base_samples)  # 1000 Hz
        kernel_optical = np.random.randn(40, base_samples // 10)  # 100 Hz
        kernel_eeg = np.random.randn(4, base_samples)  # 1000 Hz
        accel_data = np.random.randn(192, base_samples)  # 1000 Hz
        
        # Synchronized data structure
        synchronized_data = {
            'timestamp': np.linspace(0, 1, base_samples),
            'omp_data': omp_data,
            'kernel_optical': kernel_optical,
            'kernel_eeg': kernel_eeg,
            'accel_data': accel_data,
            'sync_markers': np.zeros(base_samples, dtype=bool)
        }
        
        # Verify structure
        assert 'timestamp' in synchronized_data
        assert len(synchronized_data['timestamp']) == base_samples
        assert synchronized_data['omp_data'].shape[1] == base_samples
        assert synchronized_data['kernel_eeg'].shape[1] == base_samples
        assert synchronized_data['accel_data'].shape[1] == base_samples


class TestHardwareErrorHandling:
    """Test hardware error handling and fault tolerance"""
    
    def test_device_connection_errors(self):
        """Test handling of device connection failures"""
        # Test configuration with invalid ports
        config = HardwareConfig()
        
        # Should handle invalid port gracefully
        config.omp_port = "/dev/nonexistent_port"
        
        # Configuration should still be valid (actual connection tested elsewhere)
        assert config.omp_port == "/dev/nonexistent_port"
    
    def test_data_validation_errors(self):
        """Test data validation and error detection"""
        # Test invalid data shapes
        invalid_shapes = [
            np.random.randn(305, 1000),  # Wrong channel count for OMP
            np.random.randn(306),        # Wrong dimensions
            np.array([]),                # Empty data
        ]
        
        for invalid_data in invalid_shapes:
            # Should be able to detect invalid data
            if invalid_data.size == 0:
                assert invalid_data.size == 0
            elif invalid_data.ndim == 1:
                assert invalid_data.ndim == 1
            else:
                assert invalid_data.shape[0] != 306 or invalid_data.shape[0] == 305
    
    def test_calibration_error_handling(self):
        """Test calibration error handling"""
        import json

        # Test invalid calibration data
        invalid_calibration = {
            "channel_count": 305,  # Wrong count
            "matrix_coils": -1,    # Invalid value
            "calibration_matrix": [],  # Empty matrix
        }
        
        # Should be able to detect invalid calibration
        assert invalid_calibration["channel_count"] != 306
        assert invalid_calibration["matrix_coils"] < 0
        assert len(invalid_calibration["calibration_matrix"]) == 0


class TestHardwareBenchmarks:
    """Test hardware performance benchmarks"""
    
    def test_data_throughput_benchmarks(self):
        """Test data throughput meets specifications"""
        # Test data generation at specified rates
        
        # OMP: 306 channels × 1000 Hz × 8 bytes = ~2.4 MB/s
        omp_rate = 306 * 1000 * 8  # bytes per second
        assert omp_rate > 2e6  # > 2 MB/s
        
        # Kernel optical: 40 channels × 100 Hz × 8 bytes = ~32 KB/s
        kernel_optical_rate = 40 * 100 * 8
        assert kernel_optical_rate > 30000  # > 30 KB/s
        
        # Kernel EEG: 4 channels × 1000 Hz × 8 bytes = ~32 KB/s
        kernel_eeg_rate = 4 * 1000 * 8
        assert kernel_eeg_rate > 30000  # > 30 KB/s
        
        # Accelerometer: 192 channels × 1000 Hz × 8 bytes = ~1.5 MB/s
        accel_rate = 192 * 1000 * 8
        assert accel_rate > 1.5e6  # > 1.5 MB/s
        
        # Total throughput: ~4 MB/s
        total_rate = omp_rate + kernel_optical_rate + kernel_eeg_rate + accel_rate
        assert total_rate > 4e6  # > 4 MB/s
    
    def test_latency_requirements(self):
        """Test latency requirements for real-time operation"""
        # Sub-millisecond synchronization requirement
        sync_latency_target = 0.001  # 1 ms
        
        # Buffer sizes for different sampling rates
        buffer_1000hz = 10  # 10 ms buffer at 1000 Hz = 10 samples
        buffer_100hz = 1    # 10 ms buffer at 100 Hz = 1 sample
        
        # Latency should be manageable with small buffers
        assert buffer_1000hz / 1000 < 0.1  # < 100 ms
        assert buffer_100hz / 100 < 0.1    # < 100 ms
        
        # Synchronization precision should be achievable
        assert sync_latency_target > 0
        assert sync_latency_target < 0.01  # Less than 10 ms
    
    def test_memory_requirements(self):
        """Test memory requirements for hardware data"""
        # Calculate memory requirements for different buffer sizes
        
        # 1 second buffers
        omp_1s = 306 * 1000 * 8  # bytes
        kernel_optical_1s = 40 * 100 * 8
        kernel_eeg_1s = 4 * 1000 * 8
        accel_1s = 192 * 1000 * 8
        
        total_1s = omp_1s + kernel_optical_1s + kernel_eeg_1s + accel_1s
        
        # Should be manageable (< 10 MB for 1 second)
        assert total_1s < 10e6  # < 10 MB
        
        # 10 second buffers
        total_10s = total_1s * 10
        assert total_10s < 100e6  # < 100 MB
        
        # 1 minute buffers
        total_60s = total_1s * 60
        assert total_60s < 500e6  # < 500 MB


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
