"""
Comprehensive Test Suite Configuration and Runner for Brain-Forge

This module provides centralized test configuration, fixtures, and
utilities for running the complete Brain-Forge test suite.
"""

import pytest
import sys
import logging
from pathlib import Path
import numpy as np
import asyncio
from typing import Dict, List, Any
import tempfile

# Add src to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.config import Config, HardwareConfig, ProcessingConfig, SystemConfig
from core.exceptions import BrainForgeError
from core.logger import get_logger


# Configure test logging
logging.basicConfig(level=logging.INFO)
test_logger = get_logger("brain_forge_tests")


class TestEnvironment:
    """Test environment configuration and utilities"""
    
    def __init__(self):
        self.temp_dir = None
        self.config = None
        self.mock_data = {}
        
    def setup_test_environment(self):
        """Set up the test environment"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp(prefix="brain_forge_test_")
        
        # Create test configuration
        self.config = self.create_test_config()
        
        # Generate mock data
        self.mock_data = self.generate_mock_data()
        
        test_logger.info(f"Test environment setup complete: {self.temp_dir}")
        
    def create_test_config(self):
        """Create test-specific configuration"""
        config = Config()
        
        # Hardware configuration for testing
        config.hardware = HardwareConfig(
            omp_channels=306,
            optical_channels=96,
            accelerometer_count=64,
            sampling_rate=1000,
            calibration_enabled=True,
            sync_precision=0.00001
        )
        
        # Processing configuration for testing
        config.processing = ProcessingConfig(
            filter_low=1.0,
            filter_high=100.0,
            compression_algorithm="wavelet",
            compression_quality="high",
            artifact_removal_enabled=True,
            real_time_threshold=0.001,
            buffer_size=10000
        )
        
        # System configuration for testing
        config.system = SystemConfig(
            log_level="DEBUG",
            debug_mode=True,
            performance_monitoring=True,
            max_memory_gb=8,  # Reduced for testing
            num_workers=2,    # Reduced for testing
            gpu_enabled=False  # Disabled for CI/testing
        )
        
        return config
    
    def generate_mock_data(self):
        """Generate mock data for testing"""
        np.random.seed(42)  # Reproducible random data
        
        mock_data = {
            # OMP helmet data (306 channels, 1000 Hz)
            "omp_data": {
                "single_sample": np.random.randn(306),
                "short_chunk": np.random.randn(306, 100),    # 100ms
                "medium_chunk": np.random.randn(306, 1000),  # 1s
                "long_chunk": np.random.randn(306, 10000),   # 10s
            },
            
            # Kernel optical data (96 channels, 100 Hz)
            "optical_data": {
                "single_sample": np.random.randn(96),
                "short_chunk": np.random.randn(96, 10),      # 100ms
                "medium_chunk": np.random.randn(96, 100),    # 1s
                "long_chunk": np.random.randn(96, 1000),     # 10s
            },
            
            # Accelerometer data (192 channels, 1000 Hz)
            "accel_data": {
                "single_sample": np.random.randn(192),
                "short_chunk": np.random.randn(192, 100),    # 100ms
                "medium_chunk": np.random.randn(192, 1000),  # 1s
                "long_chunk": np.random.randn(192, 10000),   # 10s
            },
            
            # Multi-modal synchronized data
            "synchronized_data": {
                "duration_1s": {
                    "omp": np.random.randn(306, 1000),
                    "optical": np.random.randn(96, 100),
                    "accel": np.random.randn(192, 1000),
                    "timestamps": np.linspace(0, 1, 1000)
                }
            }
        }
        
        return mock_data
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.temp_dir and Path(self.temp_dir).exists():
            import shutil
            shutil.rmtree(self.temp_dir)
            test_logger.info(f"Cleaned up test environment: {self.temp_dir}")


# Global test environment instance
test_env = TestEnvironment()


@pytest.fixture(scope="session", autouse=True)
def setup_test_session():
    """Set up test session"""
    test_env.setup_test_environment()
    yield test_env
    test_env.cleanup_test_environment()


@pytest.fixture
def test_config():
    """Provide test configuration"""
    return test_env.config


@pytest.fixture
def mock_data():
    """Provide mock data for tests"""
    return test_env.mock_data


@pytest.fixture
def temp_dir():
    """Provide temporary directory for tests"""
    return test_env.temp_dir


class MockHardwareFixtures:
    """Mock hardware fixtures for testing"""
    
    @staticmethod
    @pytest.fixture
    def mock_omp_helmet():
        """Create mock OMP helmet"""
        class MockOMP:
            def __init__(self):
                self.channels = 306
                self.sampling_rate = 1000
                self.connected = False
                self.calibrated = False
            
            async def connect(self):
                self.connected = True
                return True
            
            async def disconnect(self):
                self.connected = False
                return True
            
            async def calibrate(self):
                if not self.connected:
                    raise BrainForgeError("Device not connected")
                self.calibrated = True
                return {"status": "success"}
            
            def get_data_sample(self):
                if not self.connected:
                    raise BrainForgeError("Device not connected")
                return np.random.randn(self.channels)
        
        return MockOMP()
    
    @staticmethod
    @pytest.fixture
    def mock_kernel_optical():
        """Create mock Kernel optical helmet"""
        class MockKernel:
            def __init__(self):
                self.flow_channels = 32
                self.flux_channels = 64
                self.total_channels = 96
                self.sampling_rate = 100
                self.connected = False
            
            async def connect(self):
                self.connected = True
                return True
            
            async def disconnect(self):
                self.connected = False
                return True
            
            def get_hemodynamic_data(self):
                if not self.connected:
                    raise BrainForgeError("Device not connected")
                return np.random.randn(self.flow_channels)
            
            def get_neural_speed_data(self):
                if not self.connected:
                    raise BrainForgeError("Device not connected")
                return np.random.randn(self.flux_channels)
        
        return MockKernel()
    
    @staticmethod
    @pytest.fixture
    def mock_accelerometer():
        """Create mock accelerometer array"""
        class MockAccel:
            def __init__(self):
                self.sensor_count = 64
                self.channels = 192  # 64 sensors * 3 axes
                self.sampling_rate = 1000
                self.connected = False
            
            async def connect(self):
                self.connected = True
                return True
            
            async def disconnect(self):
                self.connected = False
                return True
            
            def get_motion_data(self):
                if not self.connected:
                    raise BrainForgeError("Device not connected")
                return np.random.randn(self.channels)
        
        return MockAccel()


# Performance test fixtures
class PerformanceTestFixtures:
    """Performance test fixtures and utilities"""
    
    @staticmethod
    def benchmark_function(func, *args, **kwargs):
        """Benchmark a function execution"""
        import time
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        return {
            "result": result,
            "execution_time": execution_time,
            "function_name": func.__name__
        }
    
    @staticmethod
    async def benchmark_async_function(func, *args, **kwargs):
        """Benchmark an async function execution"""
        import time
        
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        return {
            "result": result,
            "execution_time": execution_time,
            "function_name": func.__name__
        }
    
    @staticmethod
    @pytest.fixture
    def performance_metrics():
        """Provide performance metrics collection"""
        metrics = {
            "latency_measurements": [],
            "throughput_measurements": [],
            "memory_usage": [],
            "cpu_usage": []
        }
        
        def add_latency(measurement):
            metrics["latency_measurements"].append(measurement)
        
        def add_throughput(measurement):
            metrics["throughput_measurements"].append(measurement)
        
        def add_memory_usage(measurement):
            metrics["memory_usage"].append(measurement)
        
        def add_cpu_usage(measurement):
            metrics["cpu_usage"].append(measurement)
        
        def get_summary():
            return {
                "avg_latency": np.mean(metrics["latency_measurements"]) if metrics["latency_measurements"] else 0,
                "max_latency": np.max(metrics["latency_measurements"]) if metrics["latency_measurements"] else 0,
                "avg_throughput": np.mean(metrics["throughput_measurements"]) if metrics["throughput_measurements"] else 0,
                "peak_memory": np.max(metrics["memory_usage"]) if metrics["memory_usage"] else 0,
                "avg_cpu": np.mean(metrics["cpu_usage"]) if metrics["cpu_usage"] else 0
            }
        
        metrics["add_latency"] = add_latency
        metrics["add_throughput"] = add_throughput
        metrics["add_memory_usage"] = add_memory_usage
        metrics["add_cpu_usage"] = add_cpu_usage
        metrics["get_summary"] = get_summary
        
        return metrics


# Test markers for categorizing tests
pytest_plugins = ["pytest_asyncio"]

# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "hardware: mark test as requiring hardware"
    )
    config.addinivalue_line(
        "markers", "mock: mark test as using mocked components"
    )


# Test collection and reporting
def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Add slow marker to long-running tests
        if "long_duration" in item.name or "stability" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Add performance marker to performance tests
        if "performance" in str(item.fspath) or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)
        
        # Add integration marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add unit marker to unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)


def pytest_runtest_setup(item):
    """Set up individual test runs"""
    test_logger.info(f"Running test: {item.name}")


def pytest_runtest_teardown(item, nextitem):
    """Tear down individual test runs"""
    test_logger.info(f"Completed test: {item.name}")


# Test result reporting
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Custom terminal summary"""
    if hasattr(terminalreporter, 'stats'):
        passed = len(terminalreporter.stats.get('passed', []))
        failed = len(terminalreporter.stats.get('failed', []))
        errors = len(terminalreporter.stats.get('error', []))
        skipped = len(terminalreporter.stats.get('skipped', []))
        
        total = passed + failed + errors + skipped
        
        print("\n" + "="*50)
        print("Brain-Forge Test Suite Summary")
        print("="*50)
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Errors: {errors}")
        print(f"Skipped: {skipped}")
        
        if total > 0:
            success_rate = (passed / total) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        print("="*50)


# Test data validation utilities
class TestDataValidator:
    """Utilities for validating test data"""
    
    @staticmethod
    def validate_neural_data(data, expected_channels=None, expected_samples=None):
        """Validate neural data structure"""
        assert isinstance(data, np.ndarray), "Data must be numpy array"
        assert data.ndim == 2, "Data must be 2D (channels x samples)"
        
        if expected_channels:
            assert data.shape[0] == expected_channels, f"Expected {expected_channels} channels, got {data.shape[0]}"
        
        if expected_samples:
            assert data.shape[1] == expected_samples, f"Expected {expected_samples} samples, got {data.shape[1]}"
        
        # Check for reasonable data ranges
        assert not np.any(np.isnan(data)), "Data contains NaN values"
        assert not np.any(np.isinf(data)), "Data contains infinite values"
        
        return True
    
    @staticmethod
    def validate_processing_result(result):
        """Validate processing pipeline result"""
        required_keys = ["filtered_data", "compressed_data", "features"]
        
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
        
        # Validate filtered data
        assert isinstance(result["filtered_data"], np.ndarray)
        
        # Validate compressed data
        assert isinstance(result["compressed_data"], np.ndarray)
        assert result["compressed_data"].size < result["filtered_data"].size
        
        # Validate features
        assert isinstance(result["features"], dict)
        feature_keys = ["mean", "std", "power_bands"]
        for key in feature_keys:
            assert key in result["features"], f"Missing feature: {key}"
        
        return True
    
    @staticmethod
    def validate_system_status(status):
        """Validate system status structure"""
        required_keys = ["system_status", "devices", "processing_pipeline"]
        
        for key in required_keys:
            assert key in status, f"Missing status key: {key}"
        
        # Validate device status
        assert isinstance(status["devices"], dict)
        for device_name, device_info in status["devices"].items():
            assert "status" in device_info
            assert device_info["status"] in ["connected", "disconnected", "error"]
        
        return True


# Test utilities for async testing
class AsyncTestUtilities:
    """Utilities for async testing"""
    
    @staticmethod
    def run_async_test(async_func, *args, **kwargs):
        """Run async test function"""
        return asyncio.run(async_func(*args, **kwargs))
    
    @staticmethod
    async def wait_for_condition(condition_func, timeout=5.0, interval=0.1):
        """Wait for a condition to become true"""
        import asyncio
        
        start_time = asyncio.get_event_loop().time()
        
        while True:
            if condition_func():
                return True
            
            current_time = asyncio.get_event_loop().time()
            if current_time - start_time > timeout:
                return False
            
            await asyncio.sleep(interval)
    
    @staticmethod
    async def simulate_real_time_data_stream(duration, sampling_rate, channels):
        """Simulate real-time data stream"""
        samples_per_chunk = int(sampling_rate * 0.1)  # 100ms chunks
        total_chunks = int(duration / 0.1)
        
        for chunk_num in range(total_chunks):
            data_chunk = np.random.randn(channels, samples_per_chunk)
            timestamp = chunk_num * 0.1
            
            yield data_chunk, timestamp
            await asyncio.sleep(0.1)  # Simulate real-time delay


if __name__ == "__main__":
    # Run the complete test suite
    pytest.main([
        "tests/",
        "-v",
        "--tb=short",
        "--maxfail=10",
        "-x"  # Stop on first failure for development
    ])
