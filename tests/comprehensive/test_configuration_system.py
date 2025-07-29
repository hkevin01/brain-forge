"""
Configuration System Tests

Tests for configuration management system claims from the README,
including file format support, environment overrides, and validation.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.config import (
    Config,
    HardwareConfig,
    ProcessingConfig,
    SystemConfig,
    TransferLearningConfig,
)
from core.exceptions import BrainForgeError


class TestConfigurationSystemClaims:
    """Test configuration system claims from README"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="brain_forge_config_test_")
        self.original_env = dict(os.environ)  # Save original environment
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil

        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        
        # Clean up temp directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestConfigurationFileFormats(TestConfigurationSystemClaims):
    """Test configuration file format support"""
    
    def test_yaml_configuration_support(self):
        """
        README shows YAML configuration examples
        Test that YAML files can be loaded correctly
        """
        # Create test YAML configuration
        yaml_config = {
            'hardware': {
                'omp_enabled': True,
                'omp_channels': 306,
                'omp_sampling_rate': 1000.0,
                'kernel_enabled': True,
                'kernel_wavelengths': [690, 905],
                'accel_enabled': True,
                'accel_channels': 3
            },
            'processing': {
                'filter_low': 1.0,
                'filter_high': 100.0,
                'compression_enabled': True,
                'compression_ratio': 5.0,
                'artifact_removal_enabled': True
            },
            'system': {
                'log_level': 'INFO',
                'gpu_enabled': True,
                'processing_threads': 4
            }
        }
        
        # Write YAML file
        yaml_file = Path(self.temp_dir) / "test_config.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_config, f)
        
        # Load configuration
        config = Config(config_file=str(yaml_file))
        
        # Verify YAML values were loaded
        assert config.hardware.omp_enabled is True
        assert config.hardware.omp_channels == 306
        assert config.hardware.omp_sampling_rate == 1000.0
        assert config.hardware.kernel_wavelengths == [690, 905]
        assert config.processing.filter_low == 1.0
        assert config.processing.compression_ratio == 5.0
        assert config.system.log_level == 'INFO'
    
    def test_json_configuration_support(self):
        """
        Test that JSON configuration files are supported
        """
        # Create test JSON configuration
        json_config = {
            'hardware': {
                'omp_channels': 512,
                'kernel_flow_channels': 64,
                'accel_range': 16
            },
            'processing': {
                'compression_algorithm': 'wavelet',
                'wavelet_type': 'db8',
                'ica_components': 25
            },
            'system': {
                'max_memory_usage': '32GB',
                'gpu_device': 'cuda:1'
            }
        }
        
        # Write JSON file
        json_file = Path(self.temp_dir) / "test_config.json"
        with open(json_file, 'w') as f:
            json.dump(json_config, f, indent=2)
        
        # Load configuration
        config = Config(config_file=str(json_file))
        
        # Verify JSON values were loaded
        assert config.hardware.omp_channels == 512
        assert config.processing.compression_algorithm == 'wavelet'
        assert config.processing.wavelet_type == 'db8'
        assert config.system.max_memory_usage == '32GB'
    
    def test_unsupported_file_format_error(self):
        """
        Test that unsupported file formats raise appropriate errors
        """
        # Create file with unsupported extension
        txt_file = Path(self.temp_dir) / "config.txt"
        with open(txt_file, 'w') as f:
            f.write("some config data")
        
        # Should raise error for unsupported format
        with pytest.raises((ValueError, BrainForgeError)):
            Config(config_file=str(txt_file))
    
    def test_missing_configuration_file_handling(self):
        """
        Test handling of missing configuration files
        """
        nonexistent_file = Path(self.temp_dir) / "nonexistent.yaml"
        
        # Should handle missing file gracefully (use defaults)
        config = Config(config_file=str(nonexistent_file))
        
        # Should still have default values
        assert config.hardware.omp_channels >= 306
        assert config.processing.filter_low > 0
        assert config.system.processing_threads > 0
    
    def test_malformed_configuration_file_error(self):
        """
        Test handling of malformed configuration files
        """
        # Create malformed YAML file
        malformed_yaml = Path(self.temp_dir) / "malformed.yaml"
        with open(malformed_yaml, 'w') as f:
            f.write("invalid: yaml: content: [unclosed")
        
        # Should raise error for malformed file
        with pytest.raises((yaml.YAMLError, BrainForgeError)):
            Config(config_file=str(malformed_yaml))


class TestEnvironmentVariableOverrides(TestConfigurationSystemClaims):
    """Test environment variable override functionality"""
    
    def test_hardware_environment_overrides(self):
        """
        Test that hardware configuration can be overridden by environment variables
        """
        # Set environment variables with Brain-Forge prefix
        env_vars = {
            'BRAIN_FORGE_HARDWARE_OMP_CHANNELS': '512',
            'BRAIN_FORGE_HARDWARE_OMP_SAMPLING_RATE': '2000.0',
            'BRAIN_FORGE_HARDWARE_KERNEL_ENABLED': 'false',
            'BRAIN_FORGE_HARDWARE_ACCEL_RANGE': '32'
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        # Create configuration
        config = Config()
        
        # Verify environment overrides were applied
        assert config.hardware.omp_channels == 512
        assert config.hardware.omp_sampling_rate == 2000.0
        assert config.hardware.kernel_enabled is False
        assert config.hardware.accel_range == 32
    
    def test_processing_environment_overrides(self):
        """
        Test processing configuration environment overrides
        """
        env_vars = {
            'BRAIN_FORGE_PROCESSING_FILTER_LOW': '2.0',
            'BRAIN_FORGE_PROCESSING_FILTER_HIGH': '80.0',
            'BRAIN_FORGE_PROCESSING_COMPRESSION_ENABLED': 'false',
            'BRAIN_FORGE_PROCESSING_COMPRESSION_RATIO': '8.0',
            'BRAIN_FORGE_PROCESSING_WAVELET_TYPE': 'db4'
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        config = Config()
        
        # Verify processing overrides
        assert config.processing.filter_low == 2.0
        assert config.processing.filter_high == 80.0
        assert config.processing.compression_enabled is False
        assert config.processing.compression_ratio == 8.0
        assert config.processing.wavelet_type == 'db4'
    
    def test_system_environment_overrides(self):
        """
        Test system configuration environment overrides
        """
        env_vars = {
            'BRAIN_FORGE_SYSTEM_LOG_LEVEL': 'DEBUG',
            'BRAIN_FORGE_SYSTEM_GPU_ENABLED': 'false',
            'BRAIN_FORGE_SYSTEM_PROCESSING_THREADS': '8',
            'BRAIN_FORGE_SYSTEM_MAX_MEMORY_USAGE': '64GB'
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        config = Config()
        
        # Verify system overrides
        assert config.system.log_level == 'DEBUG'
        assert config.system.gpu_enabled is False
        assert config.system.processing_threads == 8
        assert config.system.max_memory_usage == '64GB'
    
    def test_boolean_environment_variable_parsing(self):
        """
        Test correct parsing of boolean environment variables
        """
        # Test different boolean representations
        boolean_tests = [
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('1', True),
            ('yes', True),
            ('on', True),
            ('false', False),
            ('False', False),
            ('FALSE', False),
            ('0', False),
            ('no', False),
            ('off', False)
        ]
        
        for env_value, expected_bool in boolean_tests:
            os.environ['BRAIN_FORGE_HARDWARE_OMP_ENABLED'] = env_value
            
            config = Config()
            assert config.hardware.omp_enabled == expected_bool
    
    def test_list_environment_variable_parsing(self):
        """
        Test parsing of list environment variables
        """
        # Set list environment variable
        os.environ['BRAIN_FORGE_HARDWARE_KERNEL_WAVELENGTHS'] = '650,780,850,905'
        
        config = Config()
        
        # Should parse as list of strings (conversion handled by application)
        expected_wavelengths = ['650', '780', '850', '905']
        assert config.hardware.kernel_wavelengths == expected_wavelengths
    
    def test_environment_prefix_configuration(self):
        """
        Test that environment prefix is configurable and correct
        """
        config = Config()
        
        # Verify prefix is as documented
        assert config.get_env_prefix() == 'BRAIN_FORGE_'
        
        # Test with custom prefix would go here if supported
        # For now, verify the standard prefix works
        os.environ['BRAIN_FORGE_HARDWARE_OMP_CHANNELS'] = '128'
        
        config.reload()
        assert config.hardware.omp_channels == 128


class TestConfigurationValidation(TestConfigurationSystemClaims):
    """Test configuration validation functionality"""
    
    def test_hardware_configuration_validation(self):
        """
        Test validation of hardware configuration parameters
        """
        config = Config()
        
        # Test invalid channel count
        config.hardware.omp_channels = -1
        with pytest.raises((ValueError, BrainForgeError)):
            config._validate_config()
        
        # Test invalid sampling rate
        config.hardware.omp_channels = 306  # Reset to valid
        config.hardware.omp_sampling_rate = -1000.0
        with pytest.raises((ValueError, BrainForgeError)):
            config._validate_config()
        
        # Test zero sampling rate
        config.hardware.omp_sampling_rate = 0.0
        with pytest.raises((ValueError, BrainForgeError)):
            config._validate_config()
    
    def test_processing_configuration_validation(self):
        """
        Test validation of processing configuration parameters
        """
        config = Config()
        
        # Test invalid filter frequencies (high < low)
        config.processing.filter_low = 100.0
        config.processing.filter_high = 50.0
        with pytest.raises((ValueError, BrainForgeError)):
            config._validate_config()
        
        # Test invalid compression ratio
        config.processing.filter_low = 1.0  # Reset to valid
        config.processing.filter_high = 100.0
        config.processing.compression_ratio = 0.5  # Less than 1.0
        with pytest.raises((ValueError, BrainForgeError)):
            config._validate_config()
        
        # Test negative compression ratio
        config.processing.compression_ratio = -2.0
        with pytest.raises((ValueError, BrainForgeError)):
            config._validate_config()
    
    def test_system_configuration_validation(self):
        """
        Test validation of system configuration parameters
        """
        config = Config()
        
        # Test invalid thread count
        config.system.processing_threads = -1
        with pytest.raises((ValueError, BrainForgeError)):
            config._validate_config()
        
        # Test zero thread count
        config.system.processing_threads = 0
        with pytest.raises((ValueError, BrainForgeError)):
            config._validate_config()
        
        # Test invalid buffer size
        config.system.processing_threads = 4  # Reset to valid
        config.system.buffer_size = -100
        with pytest.raises((ValueError, BrainForgeError)):
            config._validate_config()
    
    def test_valid_configuration_passes_validation(self):
        """
        Test that valid configuration passes validation without errors
        """
        config = Config()
        
        # Default configuration should be valid
        try:
            config._validate_config()
        except Exception as e:
            pytest.fail(f"Valid configuration failed validation: {e}")
        
        # Test with reasonable custom values
        config.hardware.omp_channels = 512
        config.hardware.omp_sampling_rate = 2000.0
        config.processing.filter_low = 0.5
        config.processing.filter_high = 200.0
        config.processing.compression_ratio = 8.0
        config.system.processing_threads = 8
        config.system.buffer_size = 5000
        
        try:
            config._validate_config()
        except Exception as e:
            pytest.fail(f"Valid custom configuration failed validation: {e}")


class TestConfigurationPersistence(TestConfigurationSystemClaims):
    """Test configuration saving and loading functionality"""
    
    def test_save_configuration_to_yaml(self):
        """
        Test saving configuration to YAML file
        """
        config = Config()
        
        # Modify some values
        config.hardware.omp_channels = 512
        config.processing.compression_ratio = 7.5
        config.system.log_level = 'DEBUG'
        
        # Save to YAML file
        yaml_file = Path(self.temp_dir) / "saved_config.yaml"
        config.save_to_file(str(yaml_file))
        
        # Verify file was created
        assert yaml_file.exists()
        
        # Load and verify contents
        with open(yaml_file, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data['hardware']['omp_channels'] == 512
        assert saved_data['processing']['compression_ratio'] == 7.5
        assert saved_data['system']['log_level'] == 'DEBUG'
    
    def test_save_configuration_to_json(self):
        """
        Test saving configuration to JSON file
        """
        config = Config()
        
        # Modify some values
        config.hardware.kernel_flow_channels = 128
        config.processing.wavelet_type = 'haar'
        config.system.gpu_enabled = False
        
        # Save to JSON file
        json_file = Path(self.temp_dir) / "saved_config.json"
        config.save_to_file(str(json_file))
        
        # Verify file was created
        assert json_file.exists()
        
        # Load and verify contents
        with open(json_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['hardware']['kernel_flow_channels'] == 128
        assert saved_data['processing']['wavelet_type'] == 'haar'
        assert saved_data['system']['gpu_enabled'] is False
    
    def test_configuration_roundtrip(self):
        """
        Test that configuration can be saved and loaded back identically
        """
        # Create configuration with specific values
        original_config = Config()
        original_config.hardware.omp_channels = 256
        original_config.hardware.omp_sampling_rate = 1500.0
        original_config.processing.filter_low = 0.5
        original_config.processing.filter_high = 150.0
        original_config.processing.compression_ratio = 6.0
        original_config.system.processing_threads = 6
        
        # Save configuration
        config_file = Path(self.temp_dir) / "roundtrip_config.yaml"
        original_config.save_to_file(str(config_file))
        
        # Load configuration
        loaded_config = Config(config_file=str(config_file))
        
        # Verify values match
        assert loaded_config.hardware.omp_channels == 256
        assert loaded_config.hardware.omp_sampling_rate == 1500.0
        assert loaded_config.processing.filter_low == 0.5
        assert loaded_config.processing.filter_high == 150.0
        assert loaded_config.processing.compression_ratio == 6.0
        assert loaded_config.system.processing_threads == 6
    
    def test_configuration_reload(self):
        """
        Test configuration reload functionality
        """
        # Create initial configuration file
        initial_config = {
            'hardware': {'omp_channels': 306},
            'processing': {'compression_ratio': 5.0}
        }
        
        config_file = Path(self.temp_dir) / "reload_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(initial_config, f)
        
        # Load configuration
        config = Config(config_file=str(config_file))
        assert config.hardware.omp_channels == 306
        assert config.processing.compression_ratio == 5.0
        
        # Modify configuration file
        modified_config = {
            'hardware': {'omp_channels': 512},
            'processing': {'compression_ratio': 8.0}
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(modified_config, f)
        
        # Reload configuration
        config.reload()
        
        # Verify changes were loaded
        assert config.hardware.omp_channels == 512
        assert config.processing.compression_ratio == 8.0


class TestConfigurationDataclasses(TestConfigurationSystemClaims):
    """Test configuration dataclass functionality"""
    
    def test_hardware_config_dataclass(self):
        """
        Test HardwareConfig dataclass functionality
        """
        # Test default values
        hw_config = HardwareConfig()
        
        assert hw_config.omp_enabled is True
        assert hw_config.omp_channels >= 306
        assert hw_config.omp_sampling_rate > 0
        assert hw_config.kernel_enabled is True
        assert hw_config.accel_enabled is True
        
        # Test value modification
        hw_config.omp_channels = 512
        hw_config.omp_sampling_rate = 2000.0
        
        assert hw_config.omp_channels == 512
        assert hw_config.omp_sampling_rate == 2000.0
    
    def test_processing_config_dataclass(self):
        """
        Test ProcessingConfig dataclass functionality
        """
        proc_config = ProcessingConfig()
        
        # Test default values
        assert proc_config.filter_low > 0
        assert proc_config.filter_high > proc_config.filter_low
        assert proc_config.compression_enabled is True
        assert proc_config.compression_ratio > 1.0
        assert proc_config.artifact_removal_enabled is True
        
        # Test frequency bands structure
        assert 'delta' in proc_config.frequency_bands
        assert 'theta' in proc_config.frequency_bands
        assert 'alpha' in proc_config.frequency_bands
        assert 'beta' in proc_config.frequency_bands
        assert 'gamma' in proc_config.frequency_bands
        
        # Verify frequency band ranges
        for band, (low, high) in proc_config.frequency_bands.items():
            assert low >= 0
            assert high > low
    
    def test_system_config_dataclass(self):
        """
        Test SystemConfig dataclass functionality
        """
        sys_config = SystemConfig()
        
        # Test default values
        assert sys_config.processing_threads > 0
        assert sys_config.buffer_size > 0
        assert sys_config.log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        assert sys_config.processing_latency_target > 0
        assert sys_config.max_processing_delay > sys_config.processing_latency_target
    
    def test_transfer_learning_config_dataclass(self):
        """
        Test TransferLearningConfig dataclass functionality
        """
        tl_config = TransferLearningConfig()
        
        # Test nested dataclass structure
        assert hasattr(tl_config, 'pattern_extraction')
        assert hasattr(tl_config.pattern_extraction, 'current_subject_id')
        assert hasattr(tl_config.pattern_extraction, 'frequency_bands')
        
        # Test default values
        assert tl_config.transfer_threshold > 0
        assert tl_config.adaptation_learning_rate > 0
        assert tl_config.max_adaptation_iterations > 0
        
        # Test pattern extraction config
        pe_config = tl_config.pattern_extraction
        assert pe_config.spatial_filters > 0
        assert pe_config.pattern_quality_threshold > 0
        assert pe_config.extraction_window_size > 0
        assert 0 <= pe_config.overlap_ratio <= 1


class TestConfigurationIntegration(TestConfigurationSystemClaims):
    """Test configuration system integration with other components"""
    
    def test_config_string_representation(self):
        """
        Test configuration string representation for debugging
        """
        config = Config()
        
        config_str = str(config)
        
        # Should contain key configuration sections
        assert 'Config(' in config_str
        assert 'hardware=' in config_str
        assert 'processing=' in config_str
        assert 'system=' in config_str
        
        # Should be properly formatted
        assert config_str.endswith(')')
    
    def test_config_with_multiple_sources(self):
        """
        Test configuration loading from multiple sources (file + env)
        """
        # Create configuration file
        file_config = {
            'hardware': {'omp_channels': 306},
            'processing': {'compression_ratio': 5.0},
            'system': {'log_level': 'INFO'}
        }
        
        config_file = Path(self.temp_dir) / "multi_source_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(file_config, f)
        
        # Set environment override
        os.environ['BRAIN_FORGE_HARDWARE_OMP_CHANNELS'] = '512'
        os.environ['BRAIN_FORGE_SYSTEM_LOG_LEVEL'] = 'DEBUG'
        
        # Load configuration
        config = Config(config_file=str(config_file))
        
        # Environment should override file values
        assert config.hardware.omp_channels == 512  # From environment
        assert config.processing.compression_ratio == 5.0  # From file
        assert config.system.log_level == 'DEBUG'  # From environment
    
    def test_config_directory_loading(self):
        """
        Test loading configuration from directory with multiple files
        """
        config_dir = Path(self.temp_dir) / "config_dir"
        config_dir.mkdir()
        
        # Create multiple config files
        hardware_config = {'hardware': {'omp_channels': 256, 'kernel_enabled': True}}
        processing_config = {'processing': {'filter_low': 0.5, 'compression_ratio': 6.0}}
        
        with open(config_dir / "hardware.yaml", 'w') as f:
            yaml.dump(hardware_config, f)
        
        with open(config_dir / "processing.yaml", 'w') as f:
            yaml.dump(processing_config, f)
        
        # Load from directory
        config = Config(config_dir=str(config_dir))
        
        # Should load values from both files
        assert config.hardware.omp_channels == 256
        assert config.hardware.kernel_enabled is True
        assert config.processing.filter_low == 0.5
        assert config.processing.compression_ratio == 6.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
