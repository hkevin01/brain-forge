#!/usr/bin/env python3
"""
Brain-Forge Project Structure Validator and Creator

This script ensures the Brain-Forge project has the correct directory structure
and creates any missing directories or essential files.

Usage:
    python scripts/validate_project_structure.py [--create-missing] [--verbose]

Options:
    --create-missing    Create missing directories and files
    --verbose          Show detailed output
    --fix-permissions  Fix file permissions
    --check-only       Only check structure, don't create anything
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ProjectStructureValidator:
    """Validates and creates Brain-Forge project structure"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.missing_dirs = []
        self.missing_files = []
        self.created_items = []
        
        # Define the expected project structure
        self.expected_structure = {
            # Root level files
            'files': [
                'README.md',
                'LICENSE',
                'pyproject.toml',
                '.gitignore',
            ],
            
            # Directory structure
            'directories': {
                'requirements': [
                    'base.txt',
                    'dev.txt', 
                    'gpu.txt',
                    'visualization.txt',
                    'hardware.txt'
                ],
                'docs': [
                    'api/',
                    'tutorials/',
                    'architecture.md',
                    'getting-started.md',
                    'project_structure.md',
                    'project_plan.md'
                ],
                'src': {
                    '__init__.py': None,
                    'core': [
                        '__init__.py',
                        'config.py',
                        'exceptions.py',
                        'logger.py'
                    ],
                    'acquisition': [
                        '__init__.py',
                        'opm_helmet.py',
                        'kernel_optical.py',
                        'accelerometer.py',
                        'stream_manager.py',
                        'synchronization.py'
                    ],
                    'processing': [
                        '__init__.py',
                        'preprocessing.py',
                        'compression.py',
                        'feature_extraction.py',
                        'signal_analysis.py'
                    ],
                    'mapping': [
                        '__init__.py',
                        'brain_atlas.py',
                        'connectivity.py',
                        'spatial_mapping.py',
                        'functional_networks.py'
                    ],
                    'simulation': [
                        '__init__.py',
                        'neural_models.py',
                        'brain_simulator.py',
                        'dynamics.py',
                        'plasticity.py'
                    ],
                    'transfer': [
                        '__init__.py',
                        'pattern_extraction.py',
                        'feature_mapping.py',
                        'neural_encoding.py',
                        'transfer_learning.py'
                    ],
                    'visualization': [
                        '__init__.py',
                        'real_time_plots.py',
                        'brain_viewer.py',
                        'network_graphs.py',
                        'dashboard.py'
                    ],
                    'api': [
                        '__init__.py',
                        'rest_api.py',
                        'websocket_server.py',
                        'cli.py'
                    ],
                    'hardware': {
                        '__init__.py': None,
                        'device_drivers': ['__init__.py'],
                        'calibration': ['__init__.py'],
                        'interfaces': ['__init__.py']
                    },
                    'ml': {
                        '__init__.py': None,
                        'models': ['__init__.py'],
                        'training': ['__init__.py'],
                        'inference': ['__init__.py']
                    },
                    'utils': [
                        '__init__.py',
                        'data_io.py',
                        'math_utils.py',
                        'validation.py'
                    ]
                },
                'tests': {
                    '__init__.py': None,
                    'unit': ['__init__.py'],
                    'integration': ['__init__.py'],
                    'hardware': ['__init__.py'],
                    'performance': ['__init__.py']
                },
                'examples': [
                    'quick_start.py',
                    'full_pipeline_demo.py',
                    'real_time_monitoring.py',
                    'jupyter_notebooks/'
                ],
                'scripts': [
                    'setup_environment.py',
                    'download_test_data.py',
                    'benchmark_performance.py',
                    'calibrate_hardware.py'
                ],
                'data': {
                    'test_datasets': [],
                    'brain_atlases': [],
                    'calibration_files': []
                },
                'configs': [
                    'default.yaml',
                    'development.yaml',
                    'production.yaml',
                    'hardware_profiles/'
                ],
                'docker': [
                    'Dockerfile',
                    'docker-compose.yml',
                    'requirements.txt'
                ],
                '.github': {
                    'workflows': [
                        'ci.yml',
                        'docs.yml',
                        'release.yml'
                    ],
                    'ISSUE_TEMPLATE': []
                },
                'tasksync': [
                    'log.md',
                    'tasks.md'
                ]
            }
        }
    
    def validate_structure(self) -> Tuple[bool, Dict]:
        """
        Validate the project structure
        
        Returns:
            Tuple of (is_valid, report)
        """
        logger.info(f"Validating project structure at: {self.project_root}")
        
        report = {
            'valid': True,
            'missing_directories': [],
            'missing_files': [],
            'unexpected_items': [],
            'summary': {}
        }
        
        # Check root files
        for file_name in self.expected_structure['files']:
            file_path = self.project_root / file_name
            if not file_path.exists():
                report['missing_files'].append(str(file_path))
                report['valid'] = False
        
        # Check directory structure
        self._validate_directories(
            self.expected_structure['directories'],
            self.project_root,
            report
        )
        
        # Generate summary
        report['summary'] = {
            'total_missing_dirs': len(report['missing_directories']),
            'total_missing_files': len(report['missing_files']),
            'total_issues': len(report['missing_directories']) + len(report['missing_files'])
        }
        
        return report['valid'], report
    
    def _validate_directories(self, structure: Dict, base_path: Path, report: Dict):
        """Recursively validate directory structure"""
        for item_name, item_content in structure.items():
            item_path = base_path / item_name
            
            if isinstance(item_content, dict):
                # It's a directory with subdirectories
                if not item_path.exists():
                    report['missing_directories'].append(str(item_path))
                    report['valid'] = False
                else:
                    # Check contents
                    self._validate_directories(item_content, item_path, report)
            
            elif isinstance(item_content, list):
                # It's a directory with files
                if not item_path.exists():
                    report['missing_directories'].append(str(item_path))
                    report['valid'] = False
                else:
                    # Check files in directory
                    for file_name in item_content:
                        if file_name.endswith('/'):
                            # It's a subdirectory
                            subdir_path = item_path / file_name.rstrip('/')
                            if not subdir_path.exists():
                                report['missing_directories'].append(str(subdir_path))
                                report['valid'] = False
                        else:
                            # It's a file
                            file_path = item_path / file_name
                            if not file_path.exists():
                                report['missing_files'].append(str(file_path))
                                report['valid'] = False
            
            elif item_content is None:
                # It's a single file in a directory
                if not item_path.exists():
                    report['missing_files'].append(str(item_path))
                    report['valid'] = False
    
    def create_missing_structure(self, report: Dict) -> bool:
        """
        Create missing directories and files
        
        Args:
            report: Validation report from validate_structure()
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Creating missing project structure...")
        
        try:
            # Create missing directories
            for dir_path in report['missing_directories']:
                path = Path(dir_path)
                path.mkdir(parents=True, exist_ok=True)
                self.created_items.append(f"Created directory: {dir_path}")
                logger.info(f"Created directory: {dir_path}")
            
            # Create missing files
            for file_path in report['missing_files']:
                path = Path(file_path)
                
                # Ensure parent directory exists
                path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create file with appropriate content
                content = self._get_file_template(path)
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.created_items.append(f"Created file: {file_path}")
                logger.info(f"Created file: {file_path}")
            
            logger.info(f"Successfully created {len(self.created_items)} items")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create project structure: {e}")
            return False
    
    def _get_file_template(self, file_path: Path) -> str:
        """Get appropriate template content for different file types"""
        file_name = file_path.name
        file_suffix = file_path.suffix
        parent_name = file_path.parent.name
        
        # Python __init__.py files
        if file_name == '__init__.py':
            return f'"""Brain-Forge {parent_name} module"""\n'
        
        # Python module files
        elif file_suffix == '.py':
            return f'''"""
Brain-Forge {file_name.replace('.py', '')} module

This module is part of the Brain-Forge brain-computer interface platform.
"""

from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class {self._to_class_name(file_name.replace('.py', ''))}:
    """Brain-Forge {file_name.replace('.py', '')} implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{{__name__}}.{{self.__class__.__name__}}")
        self.logger.info(f"Initialized {{self.__class__.__name__}}")
    
    def placeholder_method(self) -> None:
        """Placeholder method - implement functionality here"""
        pass


# Example usage
if __name__ == "__main__":
    instance = {self._to_class_name(file_name.replace('.py', ''))}()
    print(f"{{instance.__class__.__name__}} initialized successfully")
'''
        
        # YAML configuration files
        elif file_suffix in ['.yaml', '.yml']:
            if 'default' in file_name:
                return '''# Brain-Forge Default Configuration

system:
  name: "brain-forge"
  version: "0.1.0"
  debug: false
  log_level: "INFO"

hardware:
  omp_helmet:
    enabled: true
    channels: 306
    sampling_rate: 1000
    filter_range: [1, 100]
  
  kernel_optical:
    enabled: true
    flow_channels: 52
    flux_channels: 52
    sampling_rate: 10
  
  accelerometer:
    enabled: true
    axes: 3
    sampling_rate: 1000

processing:
  real_time: true
  compression:
    enabled: true
    algorithm: "wavelet"
    compression_ratio: 5
  
  filtering:
    notch_filter: 60  # Hz
    bandpass: [1, 100]  # Hz

simulation:
  neurons: 100000
  timestep: 0.1  # ms
  duration: 1000  # ms
  plasticity: true

visualization:
  real_time_plots: true
  brain_viewer: true
  update_rate: 30  # fps
'''
            else:
                return f'# Brain-Forge {file_name} Configuration\n# TODO: Add configuration options\n'
        
        # Requirements files
        elif file_name.endswith('.txt') and 'requirements' in str(file_path):
            return self._get_requirements_content(file_name)
        
        # Markdown files
        elif file_suffix == '.md':
            if file_name == 'README.md':
                return '''# Brain-Forge

🧠 **Advanced Brain Scanning and Simulation Platform**

Brain-Forge is a comprehensive toolkit for multi-modal brain data acquisition, processing, mapping, and digital brain simulation. Forge the future of neuroscience and brain-computer interfaces.

## Features

- Multi-modal brain data acquisition (OMP, Kernel optical, accelerometer)
- Real-time signal processing and compression
- Advanced brain mapping and connectivity analysis
- Neural simulation and digital brain creation
- Brain pattern transfer and learning protocols
- Interactive visualization and monitoring

## Quick Start

```bash
pip install -r requirements/base.txt
python examples/quick_start.py
```

## Hardware Support

- OMP Helmet (MEG-like sensors)
- Kernel Flow/Flux Optical Helmets
- Brown Accelerometer
- Real-time synchronization across devices

## Documentation

See [docs/](docs/) for comprehensive documentation.

## License

MIT License
'''
            elif file_name == 'getting-started.md':
                return '''# Getting Started with Brain-Forge

## Installation

1. Clone the repository
2. Create virtual environment
3. Install dependencies
4. Run examples

## Basic Usage

TODO: Add usage examples
'''
            else:
                return f'# {file_name.replace(".md", "").replace("_", " ").title()}\n\nTODO: Add content\n'
        
        # License file
        elif file_name == 'LICENSE':
            return '''MIT License

Copyright (c) 2025 Brain-Forge Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
        
        # pyproject.toml
        elif file_name == 'pyproject.toml':
            return '''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "brain-forge"
version = "0.1.0"
description = "Advanced brain scanning, mapping, and simulation platform"
authors = [{name = "Brain-Forge Team", email = "contact@brain-forge.org"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
dependencies = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "mne>=1.0.0",
    "nilearn>=0.8.0",
    "brian2>=2.4.0",
    "pylsl>=1.14.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=6.2.0",
    "black>=21.0.0",
    "flake8>=4.0.0",
    "mypy>=0.910"
]
gpu = [
    "cupy>=9.0.0",
    "torch>=1.10.0"
]
viz = [
    "mayavi>=4.7.0",
    "plotly>=5.0.0"
]

[project.scripts]
brain-forge = "src.api.cli:main"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
'''
        
        # .gitignore
        elif file_name == '.gitignore':
            return '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data files
*.h5
*.hdf5
*.mat
*.fif
*.edf
data/raw/
data/processed/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Config
.env
*.secret

# Brain-Forge specific
calibration_files/
test_datasets/
'''
        
        # Default empty file
        else:
            return f'# {file_name}\n# TODO: Add content\n'
    
    def _get_requirements_content(self, file_name: str) -> str:
        """Get content for requirements files"""
        if file_name == 'base.txt':
            return '''# Core scientific libraries
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Neuroimaging & brain analysis
mne>=1.0.0
nilearn>=0.8.0
dipy>=1.4.0
nibabel>=3.2.0

# Neural simulation
brian2>=2.4.0

# Real-time processing
pylsl>=1.14.0
pywavelets>=1.1.0

# Visualization
matplotlib>=3.5.0
pyvista>=0.32.0

# Utilities
pyyaml>=6.0
click>=8.0.0
tqdm>=4.62.0
h5py>=3.4.0
'''
        elif file_name == 'dev.txt':
            return '''# Development dependencies
-r base.txt

# Testing
pytest>=6.2.0
pytest-cov>=3.0.0
pytest-asyncio>=0.18.0

# Code quality
black>=21.0.0
flake8>=4.0.0
mypy>=0.910
pre-commit>=2.15.0

# Documentation
sphinx>=4.0.0
sphinx-rtd-theme>=1.0.0
'''
        elif file_name == 'gpu.txt':
            return '''# GPU acceleration dependencies
-r base.txt

# CUDA support
cupy>=9.0.0
numba>=0.54.0

# Deep learning
torch>=1.10.0
torchvision>=0.11.0
tensorflow>=2.7.0
'''
        elif file_name == 'visualization.txt':
            return '''# Visualization dependencies
-r base.txt

# Advanced plotting
plotly>=5.0.0
seaborn>=0.11.0
mayavi>=4.7.0
vtk>=9.0.0

# Interactive dashboards
streamlit>=1.2.0
dash>=2.0.0
jupyter>=1.0.0
'''
        elif file_name == 'hardware.txt':
            return '''# Hardware interface dependencies
-r base.txt

# Hardware interfaces
pyserial>=3.5
pyusb>=1.2.0
bleak>=0.13.0

# Parallel processing
joblib>=1.1.0
'''
        else:
            return '# Requirements file\n'
    
    def _to_class_name(self, snake_case: str) -> str:
        """Convert snake_case to PascalCase"""
        return ''.join(word.capitalize() for word in snake_case.split('_'))
    
    def print_report(self, report: Dict, verbose: bool = False):
        """Print validation report"""
        print("\n" + "="*60)
        print("BRAIN-FORGE PROJECT STRUCTURE VALIDATION REPORT")
        print("="*60)
        
        if report['valid']:
            print("✅ Project structure is VALID")
        else:
            print("❌ Project structure has ISSUES")
        
        print(f"\nSummary:")
        print(f"  Missing directories: {report['summary']['total_missing_dirs']}")
        print(f"  Missing files: {report['summary']['total_missing_files']}")
        print(f"  Total issues: {report['summary']['total_issues']}")
        
        if verbose or not report['valid']:
            if report['missing_directories']:
                print(f"\nMissing directories ({len(report['missing_directories'])}):")
                for dir_path in sorted(report['missing_directories']):
                    print(f"  📁 {dir_path}")
            
            if report['missing_files']:
                print(f"\nMissing files ({len(report['missing_files'])}):")
                for file_path in sorted(report['missing_files']):
                    print(f"  📄 {file_path}")
        
        if self.created_items:
            print(f"\nCreated items ({len(self.created_items)}):")
            for item in self.created_items:
                print(f"  ✨ {item}")
        
        print("="*60)
    
    def check_permissions(self) -> List[str]:
        """Check and fix file permissions"""
        issues = []
        
        # Check if we can write to the project directory
        if not os.access(self.project_root, os.W_OK):
            issues.append(f"No write access to project root: {self.project_root}")
        
        return issues


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Validate and create Brain-Forge project structure",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--project-root',
        type=str,
        default='.',
        help='Project root directory (default: current directory)'
    )
    
    parser.add_argument(
        '--create-missing',
        action='store_true',
        help='Create missing directories and files'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check structure, don\'t create anything'
    )
    
    parser.add_argument(
        '--fix-permissions',
        action='store_true',
        help='Check and report permission issues'
    )
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = ProjectStructureValidator(args.project_root)
    
    # Check permissions if requested
    if args.fix_permissions:
        permission_issues = validator.check_permissions()
        if permission_issues:
            logger.error("Permission issues found:")
            for issue in permission_issues:
                logger.error(f"  {issue}")
            return 1
    
    # Validate structure
    is_valid, report = validator.validate_structure()
    
    # Create missing items if requested and not in check-only mode
    if args.create_missing and not args.check_only and not is_valid:
        success = validator.create_missing_structure(report)
        if not success:
            logger.error("Failed to create missing structure")
            return 1
        
        # Re-validate after creation
        is_valid, report = validator.validate_structure()
    
    # Print report
    validator.print_report(report, args.verbose)
    
    # Return appropriate exit code
    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
