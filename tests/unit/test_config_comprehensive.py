"""
Comprehensive Unit Tests for Core Brain-Forge Configuration

This test module provides complete coverage of the configuration system,
testing all configuration classes, validation, loading/saving, and
default values.
"""

import pytest
import tempfile
import yaml
import sys
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.config import (
    Config,
    HardwareConfig,
    ProcessingConfig,
    SystemConfig
)
from core.exceptions import ConfigurationError, ValidationError


class TestHardwareConfig:
    """Test HardwareConfig dataclass"""
    
    def test_default_initialization(self):
        """Test default hardware configuration values"""
        config = HardwareConfig()
        
        assert config.omp_channels == 128
        assert config.optical_channels == 64
        assert config.accelerometer_count == 12
        assert config.sampling_rate == 1000
        assert config.calibration_enabled is True
        assert config.sync_precision == 0.00001  # 10μs
    
    def test_custom_initialization(self):
        """Test custom hardware configuration values"""
        config = HardwareConfig(
            omp_channels=306,
            optical_channels=104,
            accelerometer_count=64,
            sampling_rate=2000,
            calibration_enabled=False,
            sync_precision=0.000001
        )
        
        assert config.omp_channels == 306
        assert config.optical_channels == 104
        assert config.accelerometer_count == 64
        assert config.sampling_rate == 2000
        assert config.calibration_enabled is False
        assert config.sync_precision == 0.000001
    
    def test_validation_valid_config(self):
        """Test validation with valid configuration"""
        config = HardwareConfig()
        assert config.validate() is True
    
    def test_validation_invalid_channels(self):
        """Test validation with invalid channel counts"""
        config = HardwareConfig(omp_channels=0)
        assert config.validate() is False
        
        config = HardwareConfig(optical_channels=-5)
        assert config.validate() is False
    
    def test_validation_invalid_sampling_rate(self):
        """Test validation with invalid sampling rate"""
        config = HardwareConfig(sampling_rate=0)
        assert config.validate() is False
        
        config = HardwareConfig(sampling_rate=-100)
        assert config.validate() is False


class TestProcessingConfig:
    """Test ProcessingConfig dataclass"""
    
    def test_default_initialization(self):
        """Test default processing configuration values"""
        config = ProcessingConfig()
        
        assert config.filter_low == 1.0
        assert config.filter_high == 100.0
        assert config.compression_algorithm == "wavelet"
        assert config.compression_quality == "high"
        assert config.artifact_removal_enabled is True
        assert config.real_time_threshold == 0.001
        assert config.buffer_size == 10000
    
    def test_custom_initialization(self):
        """Test custom processing configuration values"""
        config = ProcessingConfig(
            filter_low=0.5,
            filter_high=200.0,
            compression_algorithm="neural_lz",
            compression_quality="ultra_high",
            artifact_removal_enabled=False,
            real_time_threshold=0.0005,
            buffer_size=50000
        )
        
        assert config.filter_low == 0.5
        assert config.filter_high == 200.0
        assert config.compression_algorithm == "neural_lz"
        assert config.compression_quality == "ultra_high"
        assert config.artifact_removal_enabled is False
        assert config.real_time_threshold == 0.0005
        assert config.buffer_size == 50000
    
    def test_validation_valid_config(self):
        """Test validation with valid configuration"""
        config = ProcessingConfig()
        assert config.validate() is True
    
    def test_validation_invalid_filter_range(self):
        """Test validation with invalid filter range"""
        config = ProcessingConfig(filter_low=50.0, filter_high=10.0)
        assert config.validate() is False
    
    def test_validation_invalid_compression_algorithm(self):
        """Test validation with invalid compression algorithm"""
        config = ProcessingConfig(compression_algorithm="invalid_algorithm")
        assert config.validate() is False
    
    def test_validation_invalid_quality(self):
        """Test validation with invalid quality setting"""
        config = ProcessingConfig(compression_quality="invalid_quality")
        assert config.validate() is False


class TestSystemConfig:
    """Test SystemConfig dataclass"""
    
    def test_default_initialization(self):
        """Test default system configuration values"""
        config = SystemConfig()
        
        assert config.log_level == "INFO"
        assert config.debug_mode is False
        assert config.performance_monitoring is True
        assert config.max_memory_gb == 16
        assert config.num_workers == 4
        assert config.gpu_enabled is True
    
    def test_custom_initialization(self):
        """Test custom system configuration values"""
        config = SystemConfig(
            log_level="DEBUG",
            debug_mode=True,
            performance_monitoring=False,
            max_memory_gb=32,
            num_workers=8,
            gpu_enabled=False
        )
        
        assert config.log_level == "DEBUG"
        assert config.debug_mode is True
        assert config.performance_monitoring is False
        assert config.max_memory_gb == 32
        assert config.num_workers == 8
        assert config.gpu_enabled is False
    
    def test_validation_valid_config(self):
        """Test validation with valid configuration"""
        config = SystemConfig()
        assert config.validate() is True
    
    def test_validation_invalid_log_level(self):
        """Test validation with invalid log level"""
        config = SystemConfig(log_level="INVALID")
        assert config.validate() is False
    
    def test_validation_invalid_memory(self):
        """Test validation with invalid memory setting"""
        config = SystemConfig(max_memory_gb=0)
        assert config.validate() is False
    
    def test_validation_invalid_workers(self):
        """Test validation with invalid worker count"""
        config = SystemConfig(num_workers=0)
        assert config.validate() is False


class TestMainConfig:
    """Test main Config class"""
    
    def test_default_initialization(self):
        """Test default configuration initialization"""
        config = Config()
        
        assert isinstance(config.hardware, HardwareConfig)
        assert isinstance(config.processing, ProcessingConfig)
        assert isinstance(config.system, SystemConfig)
        assert config.config_file is None
    
    def test_initialization_with_file(self):
        """Test configuration initialization with file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'hardware': {
                    'omp_channels': 256,
                    'sampling_rate': 2000
                },
                'processing': {
                    'filter_low': 0.5,
                    'compression_algorithm': 'neural_lz'
                },
                'system': {
                    'log_level': 'DEBUG',
                    'debug_mode': True
                }
            }
            yaml.dump(config_data, f)
            config_file = f.name
        
        config = Config(config_file)
        
        assert config.hardware.omp_channels == 256
        assert config.hardware.sampling_rate == 2000
        assert config.processing.filter_low == 0.5
        assert config.processing.compression_algorithm == 'neural_lz'
        assert config.system.log_level == 'DEBUG'
        assert config.system.debug_mode is True
        
        # Clean up
        Path(config_file).unlink()
    
    def test_load_from_dict(self):
        """Test loading configuration from dictionary"""
        config_dict = {
            'hardware': {
                'omp_channels': 512,
                'optical_channels': 128
            },
            'processing': {
                'compression_quality': 'ultra_high',
                'buffer_size': 20000
            },
            'system': {
                'num_workers': 16,
                'gpu_enabled': False
            }
        }
        
        config = Config()
        config.load_from_dict(config_dict)
        
        assert config.hardware.omp_channels == 512
        assert config.hardware.optical_channels == 128
        assert config.processing.compression_quality == 'ultra_high'
        assert config.processing.buffer_size == 20000
        assert config.system.num_workers == 16
        assert config.system.gpu_enabled is False
    
    def test_save_to_file(self):
        """Test saving configuration to file"""
        config = Config()
        config.hardware.omp_channels = 512
        config.processing.compression_algorithm = "neural_lz"
        config.system.log_level = "DEBUG"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_file = f.name
        
        config.save_to_file(config_file)
        
        # Verify file was saved correctly
        assert Path(config_file).exists()
        
        with open(config_file, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data['hardware']['omp_channels'] == 512
        assert saved_data['processing']['compression_algorithm'] == 'neural_lz'
        assert saved_data['system']['log_level'] == 'DEBUG'
        
        # Clean up
        Path(config_file).unlink()
    
    def test_to_dict(self):
        """Test converting configuration to dictionary"""
        config = Config()
        config.hardware.omp_channels = 256
        config.processing.filter_low = 0.5
        config.system.debug_mode = True
        
        config_dict = config.to_dict()
        
        assert config_dict['hardware']['omp_channels'] == 256
        assert config_dict['processing']['filter_low'] == 0.5
        assert config_dict['system']['debug_mode'] is True
    
    def test_validate_config_valid(self):
        """Test validation with valid configuration"""
        config = Config()
        assert config.validate_config() is True
    
    def test_validate_config_invalid_hardware(self):
        """Test validation with invalid hardware configuration"""
        config = Config()
        config.hardware.omp_channels = -1
        
        with pytest.raises(ValidationError):
            config.validate_config()
    
    def test_validate_config_invalid_processing(self):
        """Test validation with invalid processing configuration"""
        config = Config()
        config.processing.filter_low = 100.0
        config.processing.filter_high = 10.0
        
        with pytest.raises(ValidationError):
            config.validate_config()
    
    def test_validate_config_invalid_system(self):
        """Test validation with invalid system configuration"""
        config = Config()
        config.system.max_memory_gb = -5
        
        with pytest.raises(ValidationError):
            config.validate_config()
    
    def test_get_hardware_summary(self):
        """Test hardware configuration summary"""
        config = Config()
        summary = config.get_hardware_summary()
        
        assert 'Total Channels' in summary
        assert 'Sampling Rate' in summary
        assert 'Calibration' in summary
        assert 'Sync Precision' in summary
    
    def test_get_processing_summary(self):
        """Test processing configuration summary"""
        config = Config()
        summary = config.get_processing_summary()
        
        assert 'Filter Range' in summary
        assert 'Compression' in summary
        assert 'Real-time Threshold' in summary
        assert 'Buffer Size' in summary
    
    def test_get_system_summary(self):
        """Test system configuration summary"""
        config = Config()
        summary = config.get_system_summary()
        
        assert 'Log Level' in summary
        assert 'Debug Mode' in summary
        assert 'Max Memory' in summary
        assert 'Workers' in summary


class TestConfigurationValidation:
    """Test configuration validation logic"""
    
    def test_hardware_channel_validation(self):
        """Test hardware channel count validation"""
        config = HardwareConfig()
        
        # Valid channel counts
        config.omp_channels = 128
        config.optical_channels = 64
        assert config.validate() is True
        
        # Invalid channel counts
        config.omp_channels = 0
        assert config.validate() is False
        
        config.omp_channels = 128
        config.optical_channels = -10
        assert config.validate() is False
    
    def test_processing_filter_validation(self):
        """Test processing filter validation"""
        config = ProcessingConfig()
        
        # Valid filter range
        config.filter_low = 1.0
        config.filter_high = 100.0
        assert config.validate() is True
        
        # Invalid filter range (low > high)
        config.filter_low = 50.0
        config.filter_high = 10.0
        assert config.validate() is False
        
        # Invalid negative frequencies
        config.filter_low = -1.0
        config.filter_high = 100.0
        assert config.validate() is False
    
    def test_system_resource_validation(self):
        """Test system resource validation"""
        config = SystemConfig()
        
        # Valid resource settings
        config.max_memory_gb = 16
        config.num_workers = 4
        assert config.validate() is True
        
        # Invalid memory
        config.max_memory_gb = 0
        assert config.validate() is False
        
        # Invalid worker count
        config.max_memory_gb = 16
        config.num_workers = -1
        assert config.validate() is False


class TestConfigurationFileHandling:
    """Test configuration file loading and saving"""
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file"""
        with pytest.raises(ConfigurationError):
            Config("/nonexistent/path/config.yaml")
    
    def test_load_invalid_yaml(self):
        """Test loading invalid YAML file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [[[")
            config_file = f.name
        
        with pytest.raises(ConfigurationError):
            Config(config_file)
        
        # Clean up
        Path(config_file).unlink()
    
    def test_load_partial_config(self):
        """Test loading partial configuration (missing sections)"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'hardware': {
                    'omp_channels': 256
                }
                # Missing processing and system sections
            }
            yaml.dump(config_data, f)
            config_file = f.name
        
        config = Config(config_file)
        
        # Should use defaults for missing sections
        assert config.hardware.omp_channels == 256
        assert config.processing.filter_low == 1.0  # default
        assert config.system.log_level == "INFO"  # default
        
        # Clean up
        Path(config_file).unlink()
    
    def test_save_to_readonly_location(self):
        """Test saving to read-only location"""
        config = Config()
        
        # Try to save to a location that doesn't exist
        with pytest.raises(ConfigurationError):
            config.save_to_file("/readonly/path/config.yaml")


class TestConfigurationEdgeCases:
    """Test configuration edge cases and error conditions"""
    
    def test_extremely_large_values(self):
        """Test configuration with extremely large values"""
        config = HardwareConfig(
            omp_channels=1000000,
            sampling_rate=1000000
        )
        # Should still validate (no upper limits defined)
        assert config.validate() is True
    
    def test_boundary_values(self):
        """Test configuration with boundary values"""
        # Minimum valid values
        config = ProcessingConfig(
            filter_low=0.1,
            filter_high=0.2,
            real_time_threshold=0.0001
        )
        assert config.validate() is True
        
        # Zero threshold (edge case)
        config.real_time_threshold = 0.0
        assert config.validate() is True
    
    def test_string_case_sensitivity(self):
        """Test string configuration case sensitivity"""
        config = SystemConfig(log_level="debug")  # lowercase
        assert config.validate() is False  # Should be uppercase
        
        config.log_level = "DEBUG"
        assert config.validate() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
