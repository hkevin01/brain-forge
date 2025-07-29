#!/usr/bin/env python3
"""
Brain-Forge Mock Hardware Development Framework

This demo showcases comprehensive mock hardware interfaces for development
without physical devices, addressing the concern about hardware partnerships
still being "in development."

Key Features Demonstrated:
- Abstract base classes for extensibility
- Realistic mock implementations for all three modalities
- Hardware abstraction layer
- Development workflow without physical hardware
- Partnership readiness validation
"""

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import sleep, time
from typing import Dict, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.config import BrainForgeConfig
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SensorCapabilities:
    """Hardware sensor capability specification"""
    name: str
    channels: int
    sampling_rate: float
    dynamic_range: Tuple[float, float]
    sensitivity: float
    latency_ms: float
    data_format: str


class BrainSensorInterface(ABC):
    """Abstract base class for all brain sensor interfaces"""
    
    def __init__(self, capabilities: SensorCapabilities):
        self.capabilities = capabilities
        self.is_connected = False
        self.is_streaming = False
        self._stream_buffer = []
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize hardware connection"""
        pass
        
    @abstractmethod
    def start_acquisition(self) -> None:
        """Start data acquisition"""
        pass
        
    @abstractmethod
    def stop_acquisition(self) -> None:
        """Stop data acquisition"""
        pass
        
    @abstractmethod
    def get_data_stream(self) -> Iterator[np.ndarray]:
        """Get real-time data stream"""
        pass
        
    @abstractmethod
    def calibrate(self) -> bool:
        """Perform hardware calibration"""
        pass
        
    @abstractmethod
    def check_connection_quality(self) -> Dict[str, float]:
        """Check connection and signal quality"""
        pass


class MockOPMHelmet(BrainSensorInterface):
    """Mock OPM helmet interface for development without NIBIB hardware"""
    
    def __init__(self):
        capabilities = SensorCapabilities(
            name="Mock NIBIB OPM Helmet",
            channels=306,
            sampling_rate=1000.0,
            dynamic_range=(-50e-9, 50e-9),  # ±50 nT
            sensitivity=10e-15,  # 10 fT/√Hz
            latency_ms=1.0,
            data_format="MEG_TESLA"
        )
        super().__init__(capabilities)
        self._noise_level = 15e-15  # Realistic noise floor
        self._calibration_matrix = np.eye(306)  # Identity for now
        
    def initialize(self) -> bool:
        """Initialize mock OPM helmet"""
        try:
            logger.info(f"Initializing {self.capabilities.name}...")
            sleep(2.0)  # Simulate initialization time
            
            # Simulate hardware checks
            self._perform_hardware_checks()
            
            self.is_connected = True
            logger.info(f"✓ OPM helmet initialized: {self.capabilities.channels} channels")
            return True
            
        except Exception as e:
            logger.error(f"OPM initialization failed: {e}")
            return False
            
    def _perform_hardware_checks(self) -> None:
        """Simulate hardware diagnostic checks"""
        logger.info("  - Checking magnetometer arrays...")
        sleep(0.5)
        logger.info("  - Calibrating field compensation...")
        sleep(0.8)
        logger.info("  - Validating noise levels...")
        sleep(0.3)
        
    def start_acquisition(self) -> None:
        """Start mock MEG data acquisition"""
        if not self.is_connected:
            raise RuntimeError("OPM helmet not initialized")
        self.is_streaming = True
        logger.info("OPM helmet streaming started")
        
    def stop_acquisition(self) -> None:
        """Stop mock MEG data acquisition"""
        self.is_streaming = False
        logger.info("OPM helmet streaming stopped")
        
    def get_data_stream(self) -> Iterator[np.ndarray]:
        """Generate realistic MEG signals"""
        if not self.is_streaming:
            yield np.zeros((self.capabilities.channels, 1))
            return
            
        # Generate realistic MEG patterns
        t = np.linspace(0, 1.0/self.capabilities.sampling_rate, 1)
        
        # Simulate alpha rhythm (8-12 Hz) in occipital channels
        alpha_channels = list(range(280, 306))  # Posterior channels
        alpha_signal = 50e-15 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
        
        # Simulate mu rhythm (8-13 Hz) in sensorimotor channels
        mu_channels = list(range(100, 130))  # Central channels
        mu_signal = 30e-15 * np.sin(2 * np.pi * 11 * t)  # 11 Hz mu
        
        # Create full channel data
        data = np.zeros((self.capabilities.channels, len(t)))
        
        # Add alpha rhythm
        for ch in alpha_channels:
            data[ch] = alpha_signal + self._noise_level * np.random.randn(len(t))
            
        # Add mu rhythm
        for ch in mu_channels:
            data[ch] = mu_signal + self._noise_level * np.random.randn(len(t))
            
        # Add noise to all other channels
        noise_channels = [i for i in range(306) if i not in alpha_channels + mu_channels]
        for ch in noise_channels:
            data[ch] = self._noise_level * np.random.randn(len(t))
            
        yield data
        
    def calibrate(self) -> bool:
        """Perform mock OPM calibration"""
        logger.info("Performing OPM calibration...")
        sleep(3.0)  # Simulate calibration time
        
        # Mock calibration matrix computation
        self._calibration_matrix = np.eye(306) + 0.01 * np.random.randn(306, 306)
        
        logger.info("✓ OPM calibration completed")
        return True
        
    def check_connection_quality(self) -> Dict[str, float]:
        """Check mock OPM connection quality"""
        return {
            'signal_quality': 0.85 + 0.1 * np.random.random(),
            'noise_level': self._noise_level * (1 + 0.2 * np.random.random()),
            'channel_connectivity': 0.98,
            'field_compensation': 0.92
        }


class MockKernelOpticalHelmet(BrainSensorInterface):
    """Mock Kernel optical helmet for development without Kernel hardware"""
    
    def __init__(self, helmet_type: str = "Flow2"):
        self.helmet_type = helmet_type
        
        capabilities = SensorCapabilities(
            name=f"Mock Kernel {helmet_type} Helmet",
            channels=64 if helmet_type == "Flow2" else 32,
            sampling_rate=100.0,  # Typical for fNIRS
            dynamic_range=(0.0, 1.0),  # Normalized optical density
            sensitivity=1e-4,  # Optical density units
            latency_ms=10.0,
            data_format="FNIRS_OD"
        )
        super().__init__(capabilities)
        
        # Wavelengths for different helmet types
        self.wavelengths = [690, 830, 905] if helmet_type == "Flow2" else [760, 850]
        
    def initialize(self) -> bool:
        """Initialize mock Kernel helmet"""
        try:
            logger.info(f"Initializing {self.capabilities.name}...")
            sleep(1.5)
            
            self._perform_optical_checks()
            
            self.is_connected = True
            logger.info(f"✓ Kernel {self.helmet_type} initialized: {self.capabilities.channels} channels")
            return True
            
        except Exception as e:
            logger.error(f"Kernel initialization failed: {e}")
            return False
            
    def _perform_optical_checks(self) -> None:
        """Simulate optical system checks"""
        logger.info("  - Checking laser sources...")
        sleep(0.4)
        logger.info("  - Calibrating photodetectors...")
        sleep(0.6)
        logger.info("  - Validating optode coupling...")
        sleep(0.5)
        
    def start_acquisition(self) -> None:
        """Start mock optical data acquisition"""
        if not self.is_connected:
            raise RuntimeError("Kernel helmet not initialized")
        self.is_streaming = True
        logger.info(f"Kernel {self.helmet_type} streaming started")
        
    def stop_acquisition(self) -> None:
        """Stop mock optical data acquisition"""
        self.is_streaming = False
        logger.info(f"Kernel {self.helmet_type} streaming stopped")
        
    def get_data_stream(self) -> Iterator[np.ndarray]:
        """Generate realistic fNIRS/TD-NIRS signals"""
        if not self.is_streaming:
            yield np.zeros((self.capabilities.channels, 1))
            return
            
        t = np.linspace(0, 1.0/self.capabilities.sampling_rate, 1)
        
        # Generate hemodynamic response patterns
        data = np.zeros((self.capabilities.channels, len(t)))
        
        # Simulate task-related activation in frontal channels
        active_channels = list(range(0, 16))  # Frontal cortex
        for ch in active_channels:
            # Hemodynamic response with realistic timing
            hrf_amplitude = 0.02 + 0.01 * np.random.random()
            baseline_drift = 0.001 * np.sin(2 * np.pi * 0.01 * t)  # Slow drift
            noise = 0.005 * np.random.randn(len(t))
            
            data[ch] = hrf_amplitude * self._generate_hrf(t) + baseline_drift + noise
            
        # Add noise to remaining channels
        for ch in range(16, self.capabilities.channels):
            data[ch] = 0.003 * np.random.randn(len(t))
            
        yield data
        
    def _generate_hrf(self, t: np.ndarray) -> np.ndarray:
        """Generate realistic hemodynamic response function"""
        # Double gamma HRF
        a1, a2 = 6, 16
        b1, b2 = 1, 1
        c = 1/6
        
        hrf = (t**(a1-1) * np.exp(-t/b1) / (b1**a1 * np.math.gamma(a1)) - 
               c * t**(a2-1) * np.exp(-t/b2) / (b2**a2 * np.math.gamma(a2)))
        
        return hrf
        
    def calibrate(self) -> bool:
        """Perform mock optical calibration"""
        logger.info(f"Performing {self.helmet_type} optical calibration...")
        sleep(2.0)
        
        logger.info("✓ Optical calibration completed")
        return True
        
    def check_connection_quality(self) -> Dict[str, float]:
        """Check mock optical connection quality"""
        return {
            'signal_quality': 0.82 + 0.15 * np.random.random(),
            'optode_coupling': 0.88 + 0.1 * np.random.random(),
            'laser_stability': 0.95,
            'detector_snr': 45 + 10 * np.random.random()
        }


class MockAccelerometerArray(BrainSensorInterface):
    """Mock accelerometer array for development without Brown's hardware"""
    
    def __init__(self):
        capabilities = SensorCapabilities(
            name="Mock Brown Accelo-hat",
            channels=192,  # 64 sensors × 3 axes
            sampling_rate=1000.0,
            dynamic_range=(-16.0, 16.0),  # ±16g
            sensitivity=0.001,  # 1 mg resolution
            latency_ms=1.0,
            data_format="ACCELERATION_G"
        )
        super().__init__(capabilities)
        self.n_sensors = 64
        
    def initialize(self) -> bool:
        """Initialize mock accelerometer array"""
        try:
            logger.info(f"Initializing {self.capabilities.name}...")
            sleep(1.0)
            
            self._perform_motion_checks()
            
            self.is_connected = True
            logger.info(f"✓ Accelerometer array initialized: {self.n_sensors} sensors")
            return True
            
        except Exception as e:
            logger.error(f"Accelerometer initialization failed: {e}")
            return False
            
    def _perform_motion_checks(self) -> None:
        """Simulate accelerometer system checks"""
        logger.info("  - Checking sensor connectivity...")
        sleep(0.3)
        logger.info("  - Calibrating orientation...")
        sleep(0.4)
        logger.info("  - Testing motion detection...")
        sleep(0.3)
        
    def start_acquisition(self) -> None:
        """Start mock motion data acquisition"""
        if not self.is_connected:
            raise RuntimeError("Accelerometer array not initialized")
        self.is_streaming = True
        logger.info("Accelerometer array streaming started")
        
    def stop_acquisition(self) -> None:
        """Stop mock motion data acquisition"""
        self.is_streaming = False
        logger.info("Accelerometer array streaming stopped")
        
    def get_data_stream(self) -> Iterator[np.ndarray]:
        """Generate realistic motion data"""
        if not self.is_streaming:
            yield np.zeros((self.capabilities.channels, 1))
            return
            
        # Generate realistic head motion patterns
        data = np.zeros((self.capabilities.channels, 1))
        
        # Simulate natural head movements
        for sensor in range(self.n_sensors):
            base_idx = sensor * 3
            
            # Small random movements (typical resting state)
            motion_amplitude = 0.02 + 0.01 * np.random.random()
            
            # X, Y, Z accelerations
            data[base_idx] = motion_amplitude * np.random.randn()  # X
            data[base_idx + 1] = motion_amplitude * np.random.randn()  # Y
            data[base_idx + 2] = 1.0 + 0.1 * motion_amplitude * np.random.randn()  # Z (gravity)
            
        yield data
        
    def calibrate(self) -> bool:
        """Perform mock accelerometer calibration"""
        logger.info("Performing accelerometer calibration...")
        sleep(1.5)
        
        logger.info("✓ Accelerometer calibration completed")
        return True
        
    def check_connection_quality(self) -> Dict[str, float]:
        """Check mock accelerometer connection quality"""
        return {
            'signal_quality': 0.90 + 0.08 * np.random.random(),
            'sensor_connectivity': 0.96,
            'motion_sensitivity': 0.88,
            'noise_floor': 0.001 * (1 + 0.1 * np.random.random())
        }


class HardwareAbstractionLayer:
    """Hardware abstraction layer for unified device management"""
    
    def __init__(self):
        self.devices: Dict[str, BrainSensorInterface] = {}
        self.sync_master = None
        
    def register_device(self, name: str, device: BrainSensorInterface) -> None:
        """Register a brain sensor device"""
        self.devices[name] = device
        logger.info(f"Device registered: {name}")
        
    def initialize_all_devices(self) -> bool:
        """Initialize all registered devices"""
        logger.info("=== Initializing All Hardware Devices ===")
        
        success_count = 0
        for name, device in self.devices.items():
            if device.initialize():
                success_count += 1
            else:
                logger.error(f"Failed to initialize {name}")
                
        success = success_count == len(self.devices)
        
        if success:
            logger.info(f"✓ All {len(self.devices)} devices initialized successfully")
        else:
            logger.warning(f"Only {success_count}/{len(self.devices)} devices initialized")
            
        return success
        
    def start_synchronized_acquisition(self) -> bool:
        """Start synchronized data acquisition across all devices"""
        if not all(device.is_connected for device in self.devices.values()):
            logger.error("Not all devices are connected")
            return False
            
        logger.info("Starting synchronized acquisition...")
        
        # Start all devices simultaneously
        for name, device in self.devices.items():
            try:
                device.start_acquisition()
                logger.info(f"  ✓ {name} streaming")
            except Exception as e:
                logger.error(f"  ✗ {name} failed: {e}")
                return False
                
        logger.info("✓ All devices streaming in sync")
        return True
        
    def stop_all_acquisition(self) -> None:
        """Stop data acquisition on all devices"""
        logger.info("Stopping all data acquisition...")
        
        for name, device in self.devices.items():
            if device.is_streaming:
                device.stop_acquisition()
                logger.info(f"  ✓ {name} stopped")
                
    def get_system_status(self) -> Dict[str, Dict]:
        """Get comprehensive system status"""
        status = {}
        
        for name, device in self.devices.items():
            device_status = {
                'connected': device.is_connected,
                'streaming': device.is_streaming,
                'capabilities': device.capabilities,
                'quality_metrics': device.check_connection_quality() if device.is_connected else {}
            }
            status[name] = device_status
            
        return status


class PartnershipReadinessValidator:
    """Validate readiness for hardware partnerships"""
    
    def __init__(self, hal: HardwareAbstractionLayer):
        self.hal = hal
        
    def validate_integration_readiness(self) -> Dict[str, bool]:
        """Validate system readiness for hardware partnerships"""
        logger.info("=== Partnership Readiness Validation ===")
        
        results = {}
        
        # Test 1: Device Interface Compliance
        results['interface_compliance'] = self._test_interface_compliance()
        
        # Test 2: Multi-device Synchronization
        results['synchronization'] = self._test_synchronization()
        
        # Test 3: Error Handling
        results['error_handling'] = self._test_error_handling()
        
        # Test 4: Performance Requirements
        results['performance'] = self._test_performance_requirements()
        
        # Test 5: Data Format Compatibility
        results['data_formats'] = self._test_data_format_compatibility()
        
        overall_readiness = all(results.values())
        
        logger.info("=== Validation Results ===")
        for test, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"{test.replace('_', ' ').title()}: {status}")
            
        logger.info(f"Overall Partnership Readiness: {'✓ READY' if overall_readiness else '✗ NOT READY'}")
        
        return results
        
    def _test_interface_compliance(self) -> bool:
        """Test interface compliance with hardware requirements"""
        logger.info("Testing interface compliance...")
        
        required_methods = ['initialize', 'start_acquisition', 'stop_acquisition', 
                          'get_data_stream', 'calibrate', 'check_connection_quality']
        
        for name, device in self.hal.devices.items():
            for method in required_methods:
                if not hasattr(device, method):
                    logger.error(f"{name} missing required method: {method}")
                    return False
                    
        logger.info("  ✓ All devices implement required interface")
        return True
        
    def _test_synchronization(self) -> bool:
        """Test multi-device synchronization capabilities"""
        logger.info("Testing synchronization capabilities...")
        
        try:
            success = self.hal.start_synchronized_acquisition()
            if success:
                sleep(1.0)  # Let it run briefly
                self.hal.stop_all_acquisition()
                logger.info("  ✓ Synchronization test passed")
                return True
            else:
                logger.error("  ✗ Synchronization test failed")
                return False
        except Exception as e:
            logger.error(f"  ✗ Synchronization error: {e}")
            return False
            
    def _test_error_handling(self) -> bool:
        """Test error handling capabilities"""
        logger.info("Testing error handling...")
        
        # Test initialization without connection
        test_device = MockOPMHelmet()
        try:
            test_device.start_acquisition()  # Should raise error
            logger.error("  ✗ No error raised for invalid operation")
            return False
        except RuntimeError:
            logger.info("  ✓ Proper error handling confirmed")
            return True
        except Exception as e:
            logger.error(f"  ✗ Unexpected error type: {e}")
            return False
            
    def _test_performance_requirements(self) -> bool:
        """Test performance requirements"""
        logger.info("Testing performance requirements...")
        
        # Test latency requirements
        start_time = time()
        for device in self.hal.devices.values():
            if device.is_connected:
                quality = device.check_connection_quality()
        end_time = time()
        
        latency_ms = (end_time - start_time) * 1000
        
        if latency_ms < 100:  # Realistic target
            logger.info(f"  ✓ Quality check latency: {latency_ms:.1f}ms")
            return True
        else:
            logger.error(f"  ✗ Quality check too slow: {latency_ms:.1f}ms")
            return False
            
    def _test_data_format_compatibility(self) -> bool:
        """Test data format compatibility"""
        logger.info("Testing data format compatibility...")
        
        expected_formats = {'MEG_TESLA', 'FNIRS_OD', 'ACCELERATION_G'}
        actual_formats = {device.capabilities.data_format for device in self.hal.devices.values()}
        
        if expected_formats.issubset(actual_formats):
            logger.info("  ✓ All required data formats supported")
            return True
        else:
            missing = expected_formats - actual_formats
            logger.error(f"  ✗ Missing data formats: {missing}")
            return False


def main():
    """Main demo showcasing mock hardware development framework"""
    logger.info("=== Brain-Forge Mock Hardware Development Framework ===")
    
    # Create hardware abstraction layer
    hal = HardwareAbstractionLayer()
    
    # Register mock devices
    hal.register_device("omp_helmet", MockOPMHelmet())
    hal.register_device("kernel_optical", MockKernelOpticalHelmet("Flow2"))
    hal.register_device("accelerometer", MockAccelerometerArray())
    
    try:
        # Initialize all devices
        if not hal.initialize_all_devices():
            logger.error("System initialization failed")
            return
            
        # Test synchronized acquisition
        if hal.start_synchronized_acquisition():
            logger.info("🚀 Multi-modal acquisition running...")
            
            # Simulate data collection
            sleep(5.0)
            
            # Get system status
            status = hal.get_system_status()
            logger.info("=== System Status ===")
            for device_name, device_status in status.items():
                quality = device_status['quality_metrics']
                logger.info(f"{device_name}: Quality={quality.get('signal_quality', 0):.2f}")
                
            hal.stop_all_acquisition()
            
        # Validate partnership readiness
        validator = PartnershipReadinessValidator(hal)
        readiness_results = validator.validate_integration_readiness()
        
        logger.info("\n=== Mock Hardware Framework Demo Complete ===")
        logger.info("✓ Abstract interfaces defined")
        logger.info("✓ All three modalities implemented as mocks")
        logger.info("✓ Hardware abstraction layer functional")
        logger.info("✓ Partnership readiness validated")
        logger.info("\nNext steps:")
        logger.info("  1. Share interfaces with hardware partners")
        logger.info("  2. Replace mock implementations with real hardware")
        logger.info("  3. Validate integration with actual devices")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
