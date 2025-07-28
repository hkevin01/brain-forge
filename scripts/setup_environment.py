#!/usr/bin/env python3
"""
Environment Setup Script for Brain-Forge Platform

This script sets up the development and runtime environment for the Brain-Forge
brain-computer interface system, including dependency installation, configuration
file creation, and environment validation.

Usage:
    python scripts/setup_environment.py --mode=development
    python scripts/setup_environment.py --mode=production --gpu
    python scripts/setup_environment.py --validate-only
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import logging
import platform

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Environment configuration
ENVIRONMENT_CONFIGS = {
    'development': {
        'name': 'Development Environment',
        'requirements_files': ['base.txt', 'dev.txt', 'visualization.txt'],
        'optional_requirements': ['gpu.txt', 'hardware.txt'],
        'config_template': 'development.yaml',
        'create_test_data': True,
        'enable_debugging': True
    },
    'production': {
        'name': 'Production Environment', 
        'requirements_files': ['base.txt', 'visualization.txt'],
        'optional_requirements': ['gpu.txt', 'hardware.txt'],
        'config_template': 'production.yaml',
        'create_test_data': False,
        'enable_debugging': False
    }
}


class EnvironmentSetup:
    """Brain-Forge environment setup and validation system."""
    
    def __init__(self, mode: str = 'development', gpu_support: bool = False,
                 validate_only: bool = False):
        self.mode = mode
        self.gpu_support = gpu_support
        self.validate_only = validate_only
        self.config = ENVIRONMENT_CONFIGS.get(mode, ENVIRONMENT_CONFIGS['development'])
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Paths
        self.project_root = project_root
        self.requirements_dir = self.project_root / "requirements"
        self.configs_dir = self.project_root / "configs"
        self.data_dir = self.project_root / "data"
    
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        self.logger.info("Checking Python version...")
        
        current_version = sys.version_info
        min_version = (3, 8)
        
        if current_version < min_version:
            self.logger.error(
                f"Python {min_version[0]}.{min_version[1]}+ required. "
                f"Current: {current_version.major}.{current_version.minor}"
            )
            return False
        
        self.logger.info(f"Python version OK: {current_version.major}.{current_version.minor}")
        return True
    
    def install_python_packages(self) -> bool:
        """Install required Python packages."""
        if self.validate_only:
            self.logger.info("Skipping package installation (validate-only mode)")
            return True
            
        self.logger.info(f"Installing packages for {self.config['name']}...")
        
        # Base requirements
        for req_file in self.config['requirements_files']:
            req_path = self.requirements_dir / req_file
            if req_path.exists():
                self.logger.info(f"Installing packages from {req_file}...")
                try:
                    subprocess.run([
                        sys.executable, '-m', 'pip', 'install', '-r', str(req_path)
                    ], check=True)
                    self.logger.info(f"  ✓ Installed packages from {req_file}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"  ✗ Failed to install from {req_file}: {e}")
                    return False
        
        return True
    
    def setup_data_directories(self) -> bool:
        """Set up data directories."""
        self.logger.info("Setting up data directories...")
        
        directories = [
            self.data_dir / "test_datasets",
            self.data_dir / "brain_atlases", 
            self.data_dir / "calibration_files"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"  ✓ Created directory: {directory.name}")
        
        return True
    
    def validate_installation(self) -> Dict[str, bool]:
        """Validate the installation."""
        self.logger.info("Validating installation...")
        
        validation_results = {}
        
        # Check core modules
        core_modules = ['numpy', 'scipy', 'matplotlib']
        
        for module in core_modules:
            try:
                __import__(module)
                self.logger.info(f"  ✓ {module}: Import successful")
                validation_results[f"import_{module}"] = True
            except ImportError as e:
                self.logger.error(f"  ✗ {module}: Import failed - {e}")
                validation_results[f"import_{module}"] = False
        
        return validation_results
    
    def run_setup(self) -> bool:
        """Run the complete environment setup."""
        self.logger.info(f"Setting up {self.config['name']}...")
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Install packages
        if not self.install_python_packages():
            return False
        
        # Set up directories
        if not self.setup_data_directories():
            return False
        
        # Validate installation
        validation_results = self.validate_installation()
        
        # Check if validation passed
        critical_validations = ['import_numpy', 'import_scipy', 'import_matplotlib']
        success = all(validation_results.get(val, False) for val in critical_validations)
        
        if success:
            self.logger.info("✓ Environment setup completed successfully!")
        else:
            self.logger.error("✗ Environment setup completed with errors!")
        
        return success


def main():
    """Main entry point for environment setup."""
    parser = argparse.ArgumentParser(description="Brain-Forge Environment Setup")
    parser.add_argument("--mode", default="development",
                       choices=['development', 'production'],
                       help="Environment mode")
    parser.add_argument("--gpu", action="store_true",
                       help="Enable GPU support")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate existing installation")
    
    args = parser.parse_args()
    
    setup = EnvironmentSetup(
        mode=args.mode,
        gpu_support=args.gpu,
        validate_only=args.validate_only
    )
    
    try:
        success = setup.run_setup()
        return 0 if success else 1
    except Exception as e:
        print(f"Setup failed with error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
