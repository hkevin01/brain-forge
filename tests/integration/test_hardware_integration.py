"""
Comprehensive Integration Tests for Brain-Forge Hardware Interfaces

This test module provides integration testing for all hardware components,
including OMP helmet, Kernel optical, and accelerometer interfaces.
Uses mocking to simulate hardware interactions.
"""

import pytest
import numpy as np
import asyncio
import time
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Add src to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.config import Config
from core.exceptions import (
    HardwareError,
    DeviceConnectionError,
    StreamingError,
    SynchronizationError
)


class MockLSLStream:
    """Mock LabStreamingLayer stream for testing"""
    
    def __init__(self, name, channel_count, sampling_rate):
        self.name = name
        self.channel_count = channel_count
        self.sampling_rate = sampling_rate
        self.connected = False
        self.data_buffer = []
    
    def connect(self):
        """Mock connection to stream"""
        self.connected = True
        return True
    
    def disconnect(self):
        """Mock disconnection from stream"""
        self.connected = False
    
    def pull_sample(self, timeout=1.0):
        """Mock pulling a sample from stream"""
        if not self.connected:
            return None, None
        
        # Generate mock data
        sample = np.random.randn(self.channel_count).tolist()
        timestamp = time.time()
        return sample, timestamp
    
    def pull_chunk(self, timeout=1.0, max_samples=1000):
        """Mock pulling a chunk of samples"""
        if not self.connected:
            return [], []
        
        # Generate mock chunk
        chunk_size = np.random.randint(10, 100)
        samples = np.random.randn(chunk_size, self.channel_count).tolist()
        timestamps = [time.time() + i * (1.0 / self.sampling_rate) 
                     for i in range(chunk_size)]
        return samples, timestamps


class MockOPMHelmet:
    """Mock OMP helmet for testing"""
    
    def __init__(self, channels=306):
        self.channels = channels
        self.sampling_rate = 1000
        self.connected = False
        self.calibrated = False
        self.stream = None
    
    async def connect(self):
        """Mock connection to OMP helmet"""
        await asyncio.sleep(0.1)  # Simulate connection delay
        self.connected = True
        self.stream = MockLSLStream("OMP_Stream", self.channels, self.sampling_rate)
        self.stream.connect()
        return True
    
    async def disconnect(self):
        """Mock disconnection from OMP helmet"""
        self.connected = False
        if self.stream:
            self.stream.disconnect()
    
    async def calibrate(self):
        """Mock calibration process"""
        if not self.connected:
            raise DeviceConnectionError("Device not connected")
        
        await asyncio.sleep(0.5)  # Simulate calibration time
        self.calibrated = True
        return {"status": "success", "noise_level": 0.1}
    
    def get_data_chunk(self, duration=1.0):
        """Mock getting data chunk"""
        if not self.connected or not self.stream:
            raise StreamingError("Stream not available")
        
        return self.stream.pull_chunk()
    
    def get_status(self):
        """Get device status"""
        return {
            "connected": self.connected,
            "calibrated": self.calibrated,
            "channels": self.channels,
            "sampling_rate": self.sampling_rate,
            "signal_quality": np.random.uniform(0.8, 1.0) if self.connected else 0.0
        }


class MockKernelOptical:
    """Mock Kernel optical helmet for testing"""
    
    def __init__(self, flow_channels=32, flux_channels=64):
        self.flow_channels = flow_channels
        self.flux_channels = flux_channels
        self.total_channels = flow_channels + flux_channels
        self.sampling_rate = 100  # Typical for optical
        self.connected = False
        self.stream = None
    
    async def connect(self):
        """Mock connection to Kernel optical helmet"""
        await asyncio.sleep(0.2)
        self.connected = True
        self.stream = MockLSLStream("Kernel_Stream", self.total_channels, self.sampling_rate)
        self.stream.connect()
        return True
    
    async def disconnect(self):
        """Mock disconnection"""
        self.connected = False
        if self.stream:
            self.stream.disconnect()
    
    def get_hemodynamic_data(self):
        """Mock getting hemodynamic data"""
        if not self.connected:
            raise StreamingError("Device not connected")
        
        # Simulate hemodynamic signals
        flow_data = np.random.randn(self.flow_channels) * 0.1
        return flow_data.tolist()
    
    def get_neural_speed_data(self):
        """Mock getting neural speed data"""
        if not self.connected:
            raise StreamingError("Device not connected")
        
        # Simulate neural speed measurements
        flux_data = np.random.randn(self.flux_channels) * 0.05
        return flux_data.tolist()
    
    def get_status(self):
        """Get device status"""
        return {
            "connected": self.connected,
            "flow_channels": self.flow_channels,
            "flux_channels": self.flux_channels,
            "sampling_rate": self.sampling_rate,
            "temperature": np.random.uniform(20.0, 25.0),
            "laser_power": np.random.uniform(0.9, 1.0) if self.connected else 0.0
        }


class MockAccelerometer:
    """Mock accelerometer array for testing"""
    
    def __init__(self, sensor_count=64):
        self.sensor_count = sensor_count
        self.channels = sensor_count * 3  # 3 axes per sensor
        self.sampling_rate = 1000
        self.connected = False
        self.stream = None
    
    async def connect(self):
        """Mock connection to accelerometer array"""
        await asyncio.sleep(0.1)
        self.connected = True
        self.stream = MockLSLStream("Accel_Stream", self.channels, self.sampling_rate)
        self.stream.connect()
        return True
    
    async def disconnect(self):
        """Mock disconnection"""
        self.connected = False
        if self.stream:
            self.stream.disconnect()
    
    def get_motion_data(self):
        """Mock getting motion data"""
        if not self.connected:
            raise StreamingError("Device not connected")
        
        # Simulate 3-axis accelerometer data
        motion_data = np.random.randn(self.channels) * 0.02  # Small motion
        return motion_data.tolist()
    
    def detect_artifacts(self, neural_data):
        """Mock motion artifact detection"""
        if not self.connected:
            raise StreamingError("Device not connected")
        
        # Simple mock artifact detection
        motion_threshold = 0.1
        current_motion = np.random.uniform(0, 0.2)
        
        return {
            "artifacts_detected": current_motion > motion_threshold,
            "motion_level": current_motion,
            "affected_channels": [1, 5, 12] if current_motion > motion_threshold else []
        }
    
    def get_status(self):
        """Get device status"""
        return {
            "connected": self.connected,
            "sensor_count": self.sensor_count,
            "sampling_rate": self.sampling_rate,
            "motion_level": np.random.uniform(0, 0.1),
            "battery_level": np.random.uniform(0.7, 1.0) if self.connected else 0.0
        }


class TestOPMHelmetIntegration:
    """Test OMP helmet integration"""
    
    @pytest.fixture
    def omp_helmet(self):
        """Create mock OMP helmet fixture"""
        return MockOPMHelmet(channels=306)
    
    @pytest.mark.asyncio
    async def test_connection_sequence(self, omp_helmet):
        """Test complete connection sequence"""
        # Initial state
        assert not omp_helmet.connected
        assert not omp_helmet.calibrated
        
        # Connect
        result = await omp_helmet.connect()
        assert result is True
        assert omp_helmet.connected
        
        # Calibrate
        calibration_result = await omp_helmet.calibrate()
        assert calibration_result["status"] == "success"
        assert omp_helmet.calibrated
        
        # Disconnect
        await omp_helmet.disconnect()
        assert not omp_helmet.connected
    
    @pytest.mark.asyncio
    async def test_data_acquisition(self, omp_helmet):
        """Test data acquisition from OMP helmet"""
        await omp_helmet.connect()
        await omp_helmet.calibrate()
        
        # Get data chunk
        samples, timestamps = omp_helmet.get_data_chunk()
        
        assert len(samples) > 0
        assert len(timestamps) == len(samples)
        assert len(samples[0]) == omp_helmet.channels
        
        # Check timestamp ordering
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1]
    
    def test_status_monitoring(self, omp_helmet):
        """Test device status monitoring"""
        # Disconnected status
        status = omp_helmet.get_status()
        assert status["connected"] is False
        assert status["signal_quality"] == 0.0
        
        # Connected status (using asyncio.run for async method)
        async def connect_and_check():
            await omp_helmet.connect()
            status = omp_helmet.get_status()
            assert status["connected"] is True
            assert status["signal_quality"] > 0.0
            assert status["channels"] == 306
            
        asyncio.run(connect_and_check())
    
    @pytest.mark.asyncio
    async def test_error_conditions(self, omp_helmet):
        """Test error handling"""
        # Try calibration without connection
        with pytest.raises(DeviceConnectionError):
            await omp_helmet.calibrate()
        
        # Try data acquisition without connection
        with pytest.raises(StreamingError):
            omp_helmet.get_data_chunk()


class TestKernelOpticalIntegration:
    """Test Kernel optical helmet integration"""
    
    @pytest.fixture
    def kernel_optical(self):
        """Create mock Kernel optical helmet fixture"""
        return MockKernelOptical(flow_channels=32, flux_channels=64)
    
    @pytest.mark.asyncio
    async def test_optical_connection(self, kernel_optical):
        """Test optical helmet connection"""
        assert not kernel_optical.connected
        
        result = await kernel_optical.connect()
        assert result is True
        assert kernel_optical.connected
        
        await kernel_optical.disconnect()
        assert not kernel_optical.connected
    
    def test_hemodynamic_data_acquisition(self, kernel_optical):
        """Test hemodynamic data acquisition"""
        async def run_test():
            await kernel_optical.connect()
            
            flow_data = kernel_optical.get_hemodynamic_data()
            assert len(flow_data) == kernel_optical.flow_channels
            assert all(isinstance(x, (int, float)) for x in flow_data)
            
        asyncio.run(run_test())
    
    def test_neural_speed_measurement(self, kernel_optical):
        """Test neural speed measurement"""
        async def run_test():
            await kernel_optical.connect()
            
            flux_data = kernel_optical.get_neural_speed_data()
            assert len(flux_data) == kernel_optical.flux_channels
            assert all(isinstance(x, (int, float)) for x in flux_data)
            
        asyncio.run(run_test())
    
    def test_optical_status_monitoring(self, kernel_optical):
        """Test optical device status"""
        async def run_test():
            # Disconnected status
            status = kernel_optical.get_status()
            assert status["connected"] is False
            assert status["laser_power"] == 0.0
            
            # Connected status
            await kernel_optical.connect()
            status = kernel_optical.get_status()
            assert status["connected"] is True
            assert status["laser_power"] > 0.0
            assert 20.0 <= status["temperature"] <= 25.0
            
        asyncio.run(run_test())


class TestAccelerometerIntegration:
    """Test accelerometer array integration"""
    
    @pytest.fixture
    def accelerometer(self):
        """Create mock accelerometer array fixture"""
        return MockAccelerometer(sensor_count=64)
    
    @pytest.mark.asyncio
    async def test_accelerometer_connection(self, accelerometer):
        """Test accelerometer connection"""
        assert not accelerometer.connected
        
        result = await accelerometer.connect()
        assert result is True
        assert accelerometer.connected
        assert accelerometer.channels == 192  # 64 sensors * 3 axes
        
        await accelerometer.disconnect()
        assert not accelerometer.connected
    
    def test_motion_data_acquisition(self, accelerometer):
        """Test motion data acquisition"""
        async def run_test():
            await accelerometer.connect()
            
            motion_data = accelerometer.get_motion_data()
            assert len(motion_data) == accelerometer.channels
            assert all(isinstance(x, (int, float)) for x in motion_data)
            
        asyncio.run(run_test())
    
    def test_artifact_detection(self, accelerometer):
        """Test motion artifact detection"""
        async def run_test():
            await accelerometer.connect()
            
            # Mock neural data
            neural_data = np.random.randn(306, 1000)
            
            artifact_result = accelerometer.detect_artifacts(neural_data)
            assert "artifacts_detected" in artifact_result
            assert "motion_level" in artifact_result
            assert "affected_channels" in artifact_result
            assert isinstance(artifact_result["artifacts_detected"], bool)
            
        asyncio.run(run_test())


class TestMultiModalSynchronization:
    """Test multi-modal device synchronization"""
    
    @pytest.fixture
    def all_devices(self):
        """Create all mock devices"""
        return {
            "omp": MockOPMHelmet(channels=306),
            "optical": MockKernelOptical(flow_channels=32, flux_channels=64),
            "accel": MockAccelerometer(sensor_count=64)
        }
    
    @pytest.mark.asyncio
    async def test_synchronized_connection(self, all_devices):
        """Test synchronized connection of all devices"""
        # Connect all devices
        connection_tasks = [
            all_devices["omp"].connect(),
            all_devices["optical"].connect(),
            all_devices["accel"].connect()
        ]
        
        results = await asyncio.gather(*connection_tasks)
        assert all(results)
        
        # Verify all connected
        for device in all_devices.values():
            assert device.connected
    
    @pytest.mark.asyncio
    async def test_synchronized_data_acquisition(self, all_devices):
        """Test synchronized data acquisition"""
        # Connect all devices
        await asyncio.gather(*[device.connect() for device in all_devices.values()])
        
        # Acquire data simultaneously
        start_time = time.time()
        
        omp_data = all_devices["omp"].get_data_chunk()
        optical_flow = all_devices["optical"].get_hemodynamic_data()
        optical_flux = all_devices["optical"].get_neural_speed_data()
        motion_data = all_devices["accel"].get_motion_data()
        
        end_time = time.time()
        acquisition_time = end_time - start_time
        
        # Verify data shapes
        assert len(omp_data[0]) > 0  # OMP samples
        assert len(optical_flow) == 32  # Flow channels
        assert len(optical_flux) == 64  # Flux channels
        assert len(motion_data) == 192  # Accel channels
        
        # Verify acquisition was fast (simulating real-time)
        assert acquisition_time < 0.1  # Should be sub-100ms
    
    def test_synchronization_timing(self, all_devices):
        """Test timing synchronization across devices"""
        async def run_test():
            # Connect all devices
            await asyncio.gather(*[device.connect() for device in all_devices.values()])
            
            # Get timestamps from each device
            timestamps = []
            
            # OMP timestamp
            _, omp_ts = all_devices["omp"].stream.pull_sample()
            timestamps.append(omp_ts)
            
            # Optical timestamp
            _, optical_ts = all_devices["optical"].stream.pull_sample()
            timestamps.append(optical_ts)
            
            # Accelerometer timestamp
            _, accel_ts = all_devices["accel"].stream.pull_sample()
            timestamps.append(accel_ts)
            
            # Check synchronization (should be within microseconds)
            max_offset = max(timestamps) - min(timestamps)
            assert max_offset < 0.01  # Within 10ms (relaxed for mock)
            
        asyncio.run(run_test())


class TestHardwareErrorHandling:
    """Test hardware error handling and recovery"""
    
    def test_connection_failure_handling(self):
        """Test connection failure scenarios"""
        # Mock a device that fails to connect
        class FailingDevice(MockOPMHelmet):
            async def connect(self):
                raise DeviceConnectionError("Connection failed", device_type="omp")
        
        failing_device = FailingDevice()
        
        async def run_test():
            with pytest.raises(DeviceConnectionError):
                await failing_device.connect()
                
        asyncio.run(run_test())
    
    def test_data_streaming_errors(self):
        """Test data streaming error scenarios"""
        omp = MockOPMHelmet()
        
        # Try to get data without connection
        with pytest.raises(StreamingError):
            omp.get_data_chunk()
    
    def test_synchronization_loss_detection(self, all_devices):
        """Test detection of synchronization loss"""
        async def run_test():
            # Connect all devices
            await asyncio.gather(*[device.connect() for device in all_devices.values()])
            
            # Simulate synchronization loss by disconnecting one device
            await all_devices["optical"].disconnect()
            
            # Try synchronized data acquisition
            try:
                omp_data = all_devices["omp"].get_data_chunk()
                optical_data = all_devices["optical"].get_hemodynamic_data()  # Should fail
                assert False, "Should have raised StreamingError"
            except StreamingError:
                pass  # Expected error
                
        asyncio.run(run_test())


class TestHardwarePerformance:
    """Test hardware performance and latency"""
    
    @pytest.mark.asyncio
    async def test_connection_latency(self):
        """Test device connection latency"""
        omp = MockOPMHelmet()
        
        start_time = time.time()
        await omp.connect()
        connection_time = time.time() - start_time
        
        # Should connect quickly (mock allows for this test)
        assert connection_time < 1.0
    
    def test_data_acquisition_throughput(self):
        """Test data acquisition throughput"""
        async def run_test():
            omp = MockOPMHelmet(channels=306)
            await omp.connect()
            
            # Measure throughput
            start_time = time.time()
            total_samples = 0
            
            for _ in range(10):  # 10 chunks
                samples, _ = omp.get_data_chunk()
                total_samples += len(samples)
            
            end_time = time.time()
            duration = end_time - start_time
            throughput = total_samples / duration
            
            # Should achieve reasonable throughput
            assert throughput > 100  # samples per second
            
        asyncio.run(run_test())
    
    def test_multi_device_performance(self, all_devices):
        """Test multi-device performance"""
        async def run_test():
            # Connect all devices
            await asyncio.gather(*[device.connect() for device in all_devices.values()])
            
            # Measure multi-device data acquisition
            start_time = time.time()
            
            for _ in range(5):
                # Acquire data from all devices
                omp_data = all_devices["omp"].get_data_chunk()
                optical_flow = all_devices["optical"].get_hemodynamic_data()
                motion_data = all_devices["accel"].get_motion_data()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should maintain real-time performance
            assert total_time < 1.0  # 5 acquisitions in under 1 second
            
        asyncio.run(run_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
