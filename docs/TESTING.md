# Testing Guide

## Table of Contents
- [Testing Philosophy](#testing-philosophy)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Writing Tests](#writing-tests)
- [Performance Testing](#performance-testing)
- [Hardware Testing](#hardware-testing)
- [Continuous Integration](#continuous-integration)

## Testing Philosophy

Brain-Forge follows a comprehensive testing strategy that ensures reliability, performance, and correctness across all system components. Our testing approach is built on the following principles:

### Testing Pyramid

```
    ┌─────────────────┐
    │   E2E Tests     │  ← Integration & System Tests
    │     (5%)        │
    ├─────────────────┤
    │ Integration     │  ← Component Integration Tests
    │   Tests (15%)   │
    ├─────────────────┤
    │   Unit Tests    │  ← Individual Function/Class Tests
    │     (80%)       │
    └─────────────────┘
```

### Quality Gates

- **Unit Test Coverage**: >90%
- **Integration Test Coverage**: >85%
- **Performance Benchmarks**: <100ms processing latency
- **Hardware Tests**: All devices must pass connection tests
- **Code Quality**: Static analysis with zero critical issues

## Test Structure

### Directory Organization

```
tests/
├── unit/                      # Unit tests
│   ├── test_signal_processing/
│   ├── test_hardware/
│   ├── test_visualization/
│   ├── test_data_management/
│   └── test_api/
├── integration/               # Integration tests
│   ├── test_data_pipeline/
│   ├── test_hardware_integration/
│   ├── test_api_integration/
│   └── test_gui_integration/
├── performance/               # Performance benchmarks
│   ├── test_latency/
│   ├── test_throughput/
│   └── test_memory_usage/
├── hardware/                  # Hardware-specific tests
│   ├── test_omp_helmet/
│   ├── test_kernel_optical/
│   └── test_accelerometer/
├── e2e/                      # End-to-end tests
│   ├── test_complete_workflow/
│   ├── test_gui_workflows/
│   └── test_api_workflows/
├── fixtures/                  # Test data and utilities
│   ├── sample_data/
│   ├── mock_devices/
│   └── test_configs/
└── conftest.py               # Pytest configuration
```

### Test Configuration

```python
# conftest.py
import pytest
import numpy as np
from brain_forge import BrainForge
from brain_forge.testing import MockDevice, SampleDataGenerator

@pytest.fixture(scope="session")
def brain_forge_instance():
    """Create BrainForge instance for testing"""
    config = TestConfig()
    bf = BrainForge(config=config, testing_mode=True)
    yield bf
    bf.cleanup()

@pytest.fixture
def sample_eeg_data():
    """Generate sample EEG data for testing"""
    generator = SampleDataGenerator()
    return generator.create_eeg_data(
        channels=306,
        duration=10.0,
        sampling_rate=1000,
        noise_level=0.1
    )

@pytest.fixture
def mock_omp_helmet():
    """Mock OPM helmet device"""
    device = MockDevice(
        device_type="omp_helmet",
        channels=306,
        sampling_rate=1000
    )
    yield device
    device.disconnect()
```

## Running Tests

### Quick Test Commands

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only
pytest tests/performance/             # Performance tests only

# Run tests with coverage
pytest --cov=brain_forge --cov-report=html

# Run tests in parallel
pytest -n auto                       # Auto-detect CPU cores
pytest -n 4                          # Use 4 processes

# Run specific test files
pytest tests/unit/test_signal_processing/test_filters.py
pytest tests/integration/test_data_pipeline/test_realtime_processing.py
```

### Verbose Testing

```bash
# Verbose output with test names
pytest -v

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Run tests matching pattern
pytest -k "test_filter"
pytest -k "not slow"
```

### Testing with Hardware

```bash
# Skip hardware tests (default)
pytest

# Include hardware tests (requires connected devices)
pytest --hardware

# Test specific hardware
pytest --hardware --device=omp_helmet
pytest --hardware --device=kernel_optical
```

## Test Categories

### Unit Tests

Test individual components in isolation with mocked dependencies.

```python
# tests/unit/test_signal_processing/test_filters.py
import pytest
import numpy as np
from brain_forge.processing.filters import ButterworthFilter

class TestButterworthFilter:
    def test_filter_initialization(self):
        """Test filter can be initialized with valid parameters"""
        filter_obj = ButterworthFilter(
            low_cut=0.1,
            high_cut=100,
            fs=1000,
            order=4
        )
        assert filter_obj.low_cut == 0.1
        assert filter_obj.high_cut == 100
        assert filter_obj.fs == 1000
        assert filter_obj.order == 4

    def test_filter_invalid_parameters(self):
        """Test filter raises error with invalid parameters"""
        with pytest.raises(ValueError):
            ButterworthFilter(low_cut=-1, high_cut=100, fs=1000)

        with pytest.raises(ValueError):
            ButterworthFilter(low_cut=100, high_cut=50, fs=1000)

    def test_bandpass_filtering(self, sample_eeg_data):
        """Test bandpass filtering functionality"""
        filter_obj = ButterworthFilter(
            low_cut=8, high_cut=12, fs=1000, filter_type='bandpass'
        )

        filtered_data = filter_obj.apply(sample_eeg_data)

        # Check output shape matches input
        assert filtered_data.shape == sample_eeg_data.shape

        # Check frequency content (simplified test)
        assert np.std(filtered_data) < np.std(sample_eeg_data)

    @pytest.mark.parametrize("filter_type,low_cut,high_cut", [
        ("lowpass", None, 40),
        ("highpass", 0.1, None),
        ("bandpass", 8, 12),
        ("bandstop", 58, 62)
    ])
    def test_filter_types(self, sample_eeg_data, filter_type, low_cut, high_cut):
        """Test different filter types"""
        filter_obj = ButterworthFilter(
            low_cut=low_cut,
            high_cut=high_cut,
            fs=1000,
            filter_type=filter_type
        )

        filtered_data = filter_obj.apply(sample_eeg_data)
        assert filtered_data.shape == sample_eeg_data.shape
        assert not np.array_equal(filtered_data, sample_eeg_data)
```

### Integration Tests

Test component interactions and data flow between modules.

```python
# tests/integration/test_data_pipeline/test_realtime_processing.py
import pytest
import asyncio
from brain_forge.realtime import RealTimeProcessor
from brain_forge.hardware import MockDevice

class TestRealTimeProcessing:
    @pytest.mark.asyncio
    async def test_realtime_data_flow(self, brain_forge_instance):
        """Test complete real-time data processing pipeline"""
        # Setup mock device
        device = MockDevice("omp_helmet", channels=306, sampling_rate=1000)

        # Create real-time processor
        processor = RealTimeProcessor(
            buffer_size=1000,
            overlap=100,
            processing_steps=['butterworth_filter', 'artifact_removal']
        )

        # Track processed chunks
        processed_chunks = []

        async def data_callback(chunk):
            processed_chunks.append(chunk)

        processor.add_callback(data_callback)

        # Start processing
        await processor.start(device)

        # Simulate data acquisition
        for i in range(10):
            data_chunk = device.generate_data_chunk(1000)
            await processor.process_chunk(data_chunk)

        await processor.stop()

        # Verify processing occurred
        assert len(processed_chunks) == 10
        assert all(chunk.shape[1] == 1000 for chunk in processed_chunks)

    @pytest.mark.asyncio
    async def test_processing_latency(self, brain_forge_instance):
        """Test real-time processing meets latency requirements"""
        import time

        processor = RealTimeProcessor(buffer_size=1000)
        device = MockDevice("omp_helmet", channels=306, sampling_rate=1000)

        latencies = []

        async def measure_latency(chunk):
            end_time = time.perf_counter()
            latency = (end_time - chunk.metadata['timestamp']) * 1000  # ms
            latencies.append(latency)

        processor.add_callback(measure_latency)
        await processor.start(device)

        # Process 100 chunks
        for i in range(100):
            start_time = time.perf_counter()
            data_chunk = device.generate_data_chunk(1000)
            data_chunk.metadata['timestamp'] = start_time
            await processor.process_chunk(data_chunk)

        await processor.stop()

        # Verify latency requirements
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)

        assert avg_latency < 50, f"Average latency {avg_latency}ms exceeds 50ms"
        assert max_latency < 100, f"Max latency {max_latency}ms exceeds 100ms"
```

### Performance Tests

Benchmark critical system performance metrics.

```python
# tests/performance/test_latency/test_processing_latency.py
import pytest
import time
import numpy as np
from brain_forge.processing import SignalProcessor

class TestProcessingLatency:
    @pytest.mark.performance
    def test_filter_processing_latency(self, sample_eeg_data):
        """Test signal filtering latency"""
        processor = SignalProcessor()
        processor.add_butterworth_filter(0.1, 100, 1000)

        # Warm up
        for _ in range(10):
            processor.process(sample_eeg_data[:, :1000])

        # Measure processing time
        latencies = []
        for _ in range(100):
            start_time = time.perf_counter()
            result = processor.process(sample_eeg_data[:, :1000])
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # ms

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        # Performance assertions
        assert avg_latency < 10, f"Average latency {avg_latency}ms exceeds 10ms"
        assert p95_latency < 20, f"95th percentile latency {p95_latency}ms exceeds 20ms"

    @pytest.mark.performance
    @pytest.mark.parametrize("channels,samples", [
        (64, 1000),
        (128, 1000),
        (256, 1000),
        (306, 1000),
    ])
    def test_scalability_by_channels(self, channels, samples):
        """Test processing scalability with channel count"""
        data = np.random.randn(channels, samples)
        processor = SignalProcessor()
        processor.add_butterworth_filter(0.1, 100, 1000)

        start_time = time.perf_counter()
        result = processor.process(data)
        end_time = time.perf_counter()

        latency = (end_time - start_time) * 1000
        latency_per_channel = latency / channels

        # Should scale roughly linearly with channels
        assert latency_per_channel < 0.1, f"Per-channel latency {latency_per_channel}ms too high"
```

### Hardware Tests

Test hardware device integration (requires actual devices).

```python
# tests/hardware/test_omp_helmet/test_connection.py
import pytest
from brain_forge.hardware import OMPHelmet

@pytest.mark.hardware
@pytest.mark.skipif(not pytest.config.getoption("--hardware"),
                   reason="Hardware testing disabled")
class TestOMPHelmetHardware:
    def test_device_connection(self):
        """Test OPM helmet connection"""
        helmet = OMPHelmet()

        # Test connection
        assert helmet.connect(), "Failed to connect to OPM helmet"
        assert helmet.is_connected(), "Device not reporting as connected"

        # Test device info
        info = helmet.get_device_info()
        assert info['channels'] == 306
        assert info['sampling_rate'] >= 1000
        assert 'firmware_version' in info

        # Cleanup
        helmet.disconnect()
        assert not helmet.is_connected()

    @pytest.mark.hardware
    def test_data_acquisition(self):
        """Test data acquisition from OPM helmet"""
        helmet = OMPHelmet()
        helmet.connect()

        try:
            # Start acquisition
            helmet.start_acquisition()
            assert helmet.is_acquiring()

            # Acquire data for 1 second
            data = helmet.acquire_data(duration=1.0)

            # Verify data properties
            assert data.shape[0] == 306  # channels
            assert data.shape[1] >= 1000  # samples (at least 1 second at 1kHz)
            assert not np.isnan(data).any(), "Data contains NaN values"
            assert np.std(data) > 0, "Data appears to be flat/zero"

        finally:
            helmet.stop_acquisition()
            helmet.disconnect()

    @pytest.mark.hardware
    @pytest.mark.slow
    def test_long_acquisition(self):
        """Test long-duration data acquisition stability"""
        helmet = OMPHelmet()
        helmet.connect()

        try:
            helmet.start_acquisition()

            # Acquire data for 30 seconds in chunks
            total_samples = 0
            for i in range(30):
                data_chunk = helmet.acquire_data(duration=1.0)
                total_samples += data_chunk.shape[1]

                # Check for data quality issues
                assert not np.isnan(data_chunk).any()
                assert np.std(data_chunk) > 0

            # Verify total acquisition time
            expected_samples = 30 * 1000  # 30 seconds at 1kHz
            assert abs(total_samples - expected_samples) < 100  # Allow small variance

        finally:
            helmet.stop_acquisition()
            helmet.disconnect()
```

## Writing Tests

### Test Writing Guidelines

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **AAA Pattern**: Arrange, Act, Assert
3. **Single Responsibility**: One test should test one thing
4. **Deterministic**: Tests should produce consistent results
5. **Isolated**: Tests should not depend on other tests

### Example Test Template

```python
import pytest
import numpy as np
from brain_forge.module import ComponentUnderTest

class TestComponentUnderTest:
    """Test suite for ComponentUnderTest"""

    def setup_method(self):
        """Setup run before each test method"""
        self.component = ComponentUnderTest()

    def teardown_method(self):
        """Cleanup run after each test method"""
        if hasattr(self.component, 'cleanup'):
            self.component.cleanup()

    def test_basic_functionality(self):
        """Test basic functionality works as expected"""
        # Arrange
        input_data = np.random.randn(10, 100)
        expected_shape = (10, 100)

        # Act
        result = self.component.process(input_data)

        # Assert
        assert result.shape == expected_shape
        assert isinstance(result, np.ndarray)
        assert not np.isnan(result).any()

    def test_edge_case_empty_input(self):
        """Test behavior with empty input"""
        # Arrange
        empty_input = np.array([])

        # Act & Assert
        with pytest.raises(ValueError):
            self.component.process(empty_input)

    @pytest.mark.parametrize("input_size", [1, 10, 100, 1000])
    def test_different_input_sizes(self, input_size):
        """Test component with different input sizes"""
        # Arrange
        input_data = np.random.randn(input_size)

        # Act
        result = self.component.process(input_data)

        # Assert
        assert len(result) == input_size
```

### Mock Usage

```python
# Using pytest-mock
def test_with_mocked_device(mocker):
    """Test using mocked hardware device"""
    # Mock the hardware device
    mock_device = mocker.Mock()
    mock_device.acquire_data.return_value = np.random.randn(306, 1000)
    mock_device.is_connected.return_value = True

    # Test component that uses the device
    processor = DataProcessor(device=mock_device)
    result = processor.process_real_time()

    # Verify mock was called correctly
    mock_device.acquire_data.assert_called_once()
    assert result is not None

# Using context manager for temporary mocking
from unittest.mock import patch

def test_with_patched_function():
    """Test with patched function"""
    with patch('brain_forge.hardware.detect_devices') as mock_detect:
        mock_detect.return_value = ['omp_helmet', 'kernel_optical']

        from brain_forge import BrainForge
        bf = BrainForge()
        devices = bf.get_available_devices()

        assert len(devices) == 2
        mock_detect.assert_called_once()
```

## Performance Testing

### Benchmark Framework

```python
# tests/performance/conftest.py
import pytest
import time
import psutil
import numpy as np

@pytest.fixture
def performance_monitor():
    """Monitor system performance during tests"""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.start_cpu = None

        def start(self):
            self.start_time = time.perf_counter()
            self.start_memory = psutil.virtual_memory().used
            self.start_cpu = psutil.cpu_percent()

        def stop(self):
            end_time = time.perf_counter()
            end_memory = psutil.virtual_memory().used
            end_cpu = psutil.cpu_percent()

            return {
                'duration': end_time - self.start_time,
                'memory_delta': end_memory - self.start_memory,
                'cpu_usage': max(end_cpu, self.start_cpu)
            }

    return PerformanceMonitor()

@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing"""
    return np.random.randn(306, 100000)  # ~4.5 GB of data
```

### Memory Usage Tests

```python
def test_memory_efficiency(performance_monitor, large_dataset):
    """Test memory usage stays within bounds"""
    performance_monitor.start()

    # Process large dataset
    processor = SignalProcessor()
    result = processor.process(large_dataset)

    metrics = performance_monitor.stop()

    # Memory usage should not exceed 2x input size
    input_size = large_dataset.nbytes
    max_allowed_memory = input_size * 2

    assert metrics['memory_delta'] < max_allowed_memory
    assert metrics['duration'] < 10.0  # Should complete in 10 seconds
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=brain_forge --cov-report=xml

    - name: Run integration tests
      run: |
        pytest tests/integration/ -v

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  performance:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run performance tests
      run: |
        pytest tests/performance/ -v --benchmark-only

    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
```

### Test Report Generation

```bash
# Generate HTML coverage report
pytest --cov=brain_forge --cov-report=html

# Generate performance benchmark report
pytest tests/performance/ --benchmark-html=benchmark_report.html

# Generate test report with junit XML
pytest --junitxml=test_report.xml
```

### Quality Gates

```python
# pytest.ini
[tool:pytest]
minversion = 6.0
addopts =
    -ra
    --strict-markers
    --disable-warnings
    --cov=brain_forge
    --cov-report=term-missing:skip-covered
    --cov-fail-under=90
    --tb=short
testpaths = tests
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    hardware: marks tests as requiring hardware
    performance: marks tests as performance benchmarks
    integration: marks tests as integration tests
```

---

This testing framework ensures Brain-Forge maintains high quality, performance, and reliability standards throughout development and deployment.
