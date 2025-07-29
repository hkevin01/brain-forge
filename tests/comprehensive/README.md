# Brain-Forge Comprehensive Test Suite

This test suite provides comprehensive validation that the Brain-Forge implementation matches all claims and examples in the project README.md file.

## Overview

The test suite is organized into 6 main categories that systematically verify all documented functionality:

### 1. README Claims Validation (`test_readme_claims.py`)
- **Purpose**: Validates core system claims from README
- **Coverage**: Multi-modal support, real-time processing, compression ratios, latency targets
- **Key Tests**: 
  - System initialization and configuration
  - Multi-modal hardware integration claims
  - Processing pipeline functionality  
  - Performance targets validation

### 2. Hardware Integration (`test_hardware_integration.py`)
- **Purpose**: Tests hardware interface and multi-modal support
- **Coverage**: OMP magnetometry, Kernel optical, Brown accelerometer integration
- **Key Tests**:
  - Individual hardware module interfaces
  - Multi-modal synchronization
  - Data acquisition pipelines
  - Hardware-specific configuration

### 3. Processing Validation (`test_processing_validation.py`) 
- **Purpose**: Validates signal processing algorithms and pipeline
- **Coverage**: Filtering, compression, artifact removal, feature extraction
- **Key Tests**:
  - Wavelet-based neural compression
  - ICA artifact removal
  - Real-time filtering performance
  - Multi-modal feature extraction

### 4. Configuration System (`test_configuration_system.py`)
- **Purpose**: Tests configuration management and validation
- **Coverage**: YAML/JSON loading, environment overrides, validation
- **Key Tests**:
  - Configuration loading from multiple sources
  - Environment variable overrides
  - Configuration validation and error handling
  - Default configuration behavior

### 5. Performance Benchmarks (`test_performance_benchmarks.py`)
- **Purpose**: Validates performance claims and benchmarks
- **Coverage**: Latency targets, throughput claims, memory usage
- **Key Tests**:
  - <100ms processing latency validation
  - 2-10x compression ratio verification
  - Memory usage benchmarks
  - Concurrent processing performance

### 6. Integration Workflows (`test_integration_workflows.py`)
- **Purpose**: End-to-end integration and example validation
- **Coverage**: Complete workflows, README examples, use cases
- **Key Tests**:
  - Full processing pipeline execution
  - README code example validation
  - Multi-modal data flow testing
  - Error handling and recovery

## Running the Tests

### Quick Start
```bash
# Run all comprehensive tests
python tests/comprehensive/run_comprehensive_tests.py

# Or use pytest directly
pytest tests/comprehensive/ -v
```

### Test Runner Options
```bash
# Quick validation only (fast smoke test)
python tests/comprehensive/run_comprehensive_tests.py --quick

# Validate README examples only  
python tests/comprehensive/run_comprehensive_tests.py --examples

# Check installation requirements only
python tests/comprehensive/run_comprehensive_tests.py --install

# Run full comprehensive suite (default)
python tests/comprehensive/run_comprehensive_tests.py --all
```

### Individual Test Categories
```bash
# README claims validation
pytest tests/comprehensive/test_readme_claims.py -v

# Hardware integration tests
pytest tests/comprehensive/test_hardware_integration.py -v

# Processing pipeline tests
pytest tests/comprehensive/test_processing_validation.py -v

# Configuration system tests
pytest tests/comprehensive/test_configuration_system.py -v

# Performance benchmarks
pytest tests/comprehensive/test_performance_benchmarks.py -v

# Integration workflows
pytest tests/comprehensive/test_integration_workflows.py -v
```

### Test Filtering by Markers
```bash
# Run only performance tests
pytest tests/comprehensive/ -m performance -v

# Run only tests that don't require hardware
pytest tests/comprehensive/ -m "not requires_hardware" -v

# Run only README claim validation tests
pytest tests/comprehensive/ -m readme_claims -v

# Skip slow tests
pytest tests/comprehensive/ -m "not slow" -v
```

## Test Coverage

### README Claims Covered
- ✅ Multi-modal BCI system support (OMP + Kernel + Brown)
- ✅ 306+ channel OMP magnetometry
- ✅ Real-time processing with <100ms latency
- ✅ 2-10x compression ratios
- ✅ Wavelet-based neural compression
- ✅ ICA artifact removal
- ✅ Multi-modal synchronization
- ✅ Configuration management (YAML/JSON)
- ✅ System requirements validation
- ✅ Performance benchmarks
- ✅ Installation procedures
- ✅ Code examples validation

### Test Statistics
- **Total Test Methods**: ~80+ individual test methods
- **Test Lines of Code**: ~2000+ lines
- **Coverage Areas**: 6 major functional areas
- **Mock Usage**: Extensive mocking for hardware interfaces
- **Performance Tests**: Latency, throughput, memory benchmarks
- **Error Testing**: Comprehensive error condition coverage

## Test Strategy

### Normal Cases
- Valid input data processing
- Standard configuration loading
- Typical hardware operation scenarios
- Expected performance ranges

### Edge Cases  
- Boundary value testing (min/max channels, rates)
- Empty/minimal data inputs
- Configuration edge cases
- Resource limit scenarios

### Error Conditions
- Invalid input data
- Missing configuration files
- Hardware connection failures
- Resource exhaustion scenarios

### Integration Testing
- Multi-modal data flow
- End-to-end processing pipelines
- Configuration system integration
- Hardware abstraction layer testing

## Dependencies

### Required for Testing
```
pytest>=6.0.0
pytest-asyncio>=0.18.0
pytest-mock>=3.6.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
pywavelets>=1.1.0
```

### Test-Specific Dependencies
```
unittest.mock (built-in)
asyncio (built-in)  
pathlib (built-in)
time (built-in)
```

## Expected Results

When all tests pass, you can be confident that:

1. **README Accuracy**: All documented functionality is implemented and working
2. **Performance Claims**: Latency and throughput targets are met
3. **Hardware Support**: Multi-modal hardware integration functions correctly
4. **Configuration**: System configuration works as documented
5. **Examples**: All README code examples execute successfully
6. **Error Handling**: System handles error conditions gracefully

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure src/ is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**Missing Dependencies**
```bash
# Install test dependencies
pip install -r requirements-test.txt
```

**Hardware Tests Failing**
```bash
# Skip hardware-dependent tests
pytest tests/comprehensive/ -m "not requires_hardware" -v
```

### Performance Test Failures
- Check system resources (CPU, memory)
- Ensure no other intensive processes running
- Consider adjusting performance thresholds for your hardware

### Configuration Test Failures  
- Verify config files exist in expected locations
- Check file permissions
- Validate YAML/JSON syntax

## Contributing

When adding new functionality to Brain-Forge:

1. Update the corresponding test file
2. Add new test methods for new features
3. Update README claims tests if documentation changes
4. Ensure all tests pass before submitting changes

## Test Maintenance

The test suite should be updated when:
- README claims change
- New features are added
- Performance targets are modified
- Configuration options change
- Hardware support is expanded

This ensures the tests continue to accurately validate that the implementation matches the documentation.
