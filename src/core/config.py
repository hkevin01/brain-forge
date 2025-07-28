"""
Configuration Management System for Brain-Forge

This module provides centralized configuration management for all Brain-Forge
components, including hardware interfaces, processing parameters, and system settings.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging


@dataclass
class HardwareConfig:
    """Hardware-specific configuration parameters"""
    # OMP Helmet configuration
    omp_enabled: bool = True
    omp_channels: int = 306
    omp_sampling_rate: float = 1000.0
    omp_port: str = "/dev/ttyUSB0"
    omp_calibration_file: str = "omp_calibration.json"
    
    # Kernel Optical configuration
    kernel_enabled: bool = True
    kernel_flow_channels: int = 32
    kernel_flux_channels: int = 64
    kernel_sampling_rate: float = 100.0
    kernel_wavelengths: list = field(default_factory=lambda: [650, 850])
    
    # Accelerometer configuration
    accel_enabled: bool = True
    accel_channels: int = 3
    accel_sampling_rate: float = 1000.0
    accel_range: int = 16  # ±16g
    accel_resolution: int = 16  # 16-bit


@dataclass
class ProcessingConfig:
    """Signal processing configuration parameters"""
    # Filtering parameters
    filter_low: float = 1.0
    filter_high: float = 100.0
    notch_freq: float = 60.0
    filter_order: int = 4
    
    # Compression parameters
    compression_enabled: bool = True
    compression_ratio: float = 5.0
    compression_algorithm: str = "wavelet"  # "wavelet", "fft", "pca"
    wavelet_type: str = "db8"
    
    # Artifact removal
    artifact_removal_enabled: bool = True
    ica_components: int = 20
    artifact_threshold: float = 3.0
    
    # Feature extraction
    frequency_bands: Dict[str, list] = field(default_factory=lambda: {
        'delta': [1, 4],
        'theta': [4, 8],
        'alpha': [8, 12],
        'beta': [12, 30],
        'gamma': [30, 100]
    })


@dataclass
class SystemConfig:
    """System-level configuration parameters"""
    # Performance settings
    max_memory_usage: str = "16GB"
    processing_threads: int = 4
    gpu_enabled: bool = True
    gpu_device: str = "cuda:0"
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "brain_forge.log"
    log_max_size: str = "100MB"
    log_backup_count: int = 5
    
    # Data storage
    data_directory: str = "data/"
    temp_directory: str = "temp/"
    results_directory: str = "results/"
    compression_enabled: bool = True
    
    # Real-time processing
    buffer_size: int = 1000  # samples
    processing_latency_target: float = 0.001  # 1ms
    max_processing_delay: float = 0.1  # 100ms


class Config:
    """
    Centralized configuration management system for Brain-Forge
    
    Handles loading configuration from files, environment variables,
    and provides defaults for all system components.
    """
    
    def __init__(self, config_file: Optional[str] = None, config_dir: Optional[str] = None):
        """
        Initialize configuration system
        
        Args:
            config_file: Path to specific configuration file
            config_dir: Directory containing configuration files
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Configuration components
        self.hardware = HardwareConfig()
        self.processing = ProcessingConfig()
        self.system = SystemConfig()
        
        # Configuration file paths
        self.config_dir = Path(config_dir) if config_dir else Path("configs")
        self.config_file = config_file
        
        # Environment overrides
        self.env_prefix = "BRAIN_FORGE_"
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration from files and environment variables"""
        try:
            # Load from configuration files
            if self.config_file:
                self._load_from_file(self.config_file)
            else:
                self._load_default_configs()
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            # Validate configuration
            self._validate_config()
            
            self.logger.info("Configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_from_file(self, config_file: str) -> None:
        """Load configuration from a specific file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            self.logger.warning(f"Configuration file not found: {config_file}")
            return
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            # Update configuration from loaded data
            self._update_from_dict(config_data)
            
            self.logger.info(f"Configuration loaded from: {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration file {config_file}: {e}")
            raise
    
    def _load_default_configs(self) -> None:
        """Load default configuration files from config directory"""
        default_files = [
            "default.yaml",
            "hardware.yaml", 
            "processing.yaml",
            "system.yaml"
        ]
        
        for config_file in default_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                self._load_from_file(str(config_path))
    
    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary data"""
        if 'hardware' in config_data:
            self._update_dataclass(self.hardware, config_data['hardware'])
        
        if 'processing' in config_data:
            self._update_dataclass(self.processing, config_data['processing'])
        
        if 'system' in config_data:
            self._update_dataclass(self.system, config_data['system'])
    
    def _update_dataclass(self, target_config: Any, source_data: Dict[str, Any]) -> None:
        """Update dataclass instance with dictionary values"""
        for key, value in source_data.items():
            if hasattr(target_config, key):
                setattr(target_config, key, value)
            else:
                self.logger.warning(f"Unknown configuration parameter: {key}")
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides"""
        # Hardware overrides
        self._apply_env_to_dataclass(self.hardware, "HARDWARE_")
        
        # Processing overrides
        self._apply_env_to_dataclass(self.processing, "PROCESSING_")
        
        # System overrides
        self._apply_env_to_dataclass(self.system, "SYSTEM_")
    
    def _apply_env_to_dataclass(self, target_config: Any, section_prefix: str) -> None:
        """Apply environment variables to a specific dataclass"""
        full_prefix = self.env_prefix + section_prefix
        
        for key, value in os.environ.items():
            if key.startswith(full_prefix):
                config_key = key[len(full_prefix):].lower()
                
                if hasattr(target_config, config_key):
                    # Convert string environment variable to appropriate type
                    current_value = getattr(target_config, config_key)
                    converted_value = self._convert_env_value(value, type(current_value))
                    setattr(target_config, config_key, converted_value)
    
    def _convert_env_value(self, env_value: str, target_type: type) -> Any:
        """Convert environment variable string to target type"""
        if target_type == bool:
            return env_value.lower() in ('true', '1', 'yes', 'on')
        elif target_type == int:
            return int(env_value)
        elif target_type == float:
            return float(env_value)
        elif target_type == list:
            return env_value.split(',')
        else:
            return env_value
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        # Validate hardware configuration
        if self.hardware.omp_channels <= 0:
            raise ValueError("OMP channels must be positive")
        if self.hardware.omp_sampling_rate <= 0:
            raise ValueError("OMP sampling rate must be positive")
        
        # Validate processing configuration
        if self.processing.filter_low >= self.processing.filter_high:
            raise ValueError("Low-pass filter frequency must be less than high-pass")
        if self.processing.compression_ratio <= 1.0:
            raise ValueError("Compression ratio must be greater than 1.0")
        
        # Validate system configuration
        if self.system.processing_threads <= 0:
            raise ValueError("Processing threads must be positive")
        if self.system.buffer_size <= 0:
            raise ValueError("Buffer size must be positive")
    
    def save_to_file(self, config_file: str) -> None:
        """Save current configuration to file"""
        config_data = {
            'hardware': self._dataclass_to_dict(self.hardware),
            'processing': self._dataclass_to_dict(self.processing),
            'system': self._dataclass_to_dict(self.system)
        }
        
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                elif config_path.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            self.logger.info(f"Configuration saved to: {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_file}: {e}")
            raise
    
    def _dataclass_to_dict(self, dataclass_instance: Any) -> Dict[str, Any]:
        """Convert dataclass instance to dictionary"""
        result = {}
        for field_name, field_value in dataclass_instance.__dict__.items():
            result[field_name] = field_value
        return result
    
    def get_env_prefix(self) -> str:
        """Get environment variable prefix"""
        return self.env_prefix
    
    def reload(self) -> None:
        """Reload configuration from files"""
        self._load_configuration()
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return (f"Config(\n"
                f"  hardware={self.hardware}\n"
                f"  processing={self.processing}\n"
                f"  system={self.system}\n"
                f")")