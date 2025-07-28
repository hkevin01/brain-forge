#!/usr/bin/env python3
"""
Quick Brain-Forge Structure Setup Script

This script quickly creates the essential Brain-Forge project structure.
Run this script from the project root directory.

Usage:
    python scripts/quick_setup.py
"""

import os
from pathlib import Path


def create_brain_forge_structure():
    """Create the complete Brain-Forge project structure"""
    
    project_root = Path.cwd()
    print(f"🧠 Setting up Brain-Forge project in: {project_root}")
    
    # Define the complete structure
    structure = {
        # Root directories
        'requirements': None,
        'docs/api': None,
        'docs/tutorials': None,
        'src': None,
        'src/core': None,
        'src/acquisition': None,
        'src/processing': None,
        'src/mapping': None,
        'src/simulation': None,
        'src/transfer': None,
        'src/visualization': None,
        'src/api': None,
        'src/hardware/device_drivers': None,
        'src/hardware/calibration': None,
        'src/hardware/interfaces': None,
        'src/ml/models': None,
        'src/ml/training': None,
        'src/ml/inference': None,
        'src/utils': None,
        'tests/unit': None,
        'tests/integration': None,
        'tests/hardware': None,
        'tests/performance': None,
        'examples/jupyter_notebooks': None,
        'scripts': None,
        'data/test_datasets': None,
        'data/brain_atlases': None,
        'data/calibration_files': None,
        'configs/hardware_profiles': None,
        'docker': None,
        '.github/workflows': None,
        '.github/ISSUE_TEMPLATE': None,
        'tasksync': None
    }
    
    # Core files that need content
    files = {
        'README.md': '''# Brain-Forge

🧠 **Advanced Brain Scanning and Simulation Platform**

Brain-Forge is a comprehensive toolkit for multi-modal brain data acquisition, processing, mapping, and digital brain simulation.

## Features

- Multi-modal brain data acquisition (OMP, Kernel optical, accelerometer)
- Real-time signal processing and compression
- Advanced brain mapping and connectivity analysis
- Neural simulation and digital brain creation

## Quick Start

```bash
pip install -r requirements/base.txt
python examples/quick_start.py
```

## Documentation

See [docs/](docs/) for comprehensive documentation.
''',
        
        'LICENSE': '''MIT License

Copyright (c) 2025 Brain-Forge Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
''',
        
        'pyproject.toml': '''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "brain-forge"
version = "0.1.0"
description = "Advanced brain scanning and simulation platform"
authors = [{name = "Brain-Forge Team"}]
license = {text = "MIT"}
dependencies = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "mne>=1.0.0",
    "nilearn>=0.8.0",
    "brian2>=2.4.0",
    "pylsl>=1.14.0"
]

[tool.setuptools]
package-dir = {"" = "src"}
''',
        
        '.gitignore': '''# Python
__pycache__/
*.py[cod]
*.so
build/
dist/
*.egg-info/

# Virtual environments
venv/
env/

# IDE
.vscode/
.idea/

# Data files
*.h5
*.hdf5
data/raw/
data/processed/

# Logs
*.log

# OS
.DS_Store
''',
        
        'requirements/base.txt': '''# Core dependencies
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
mne>=1.0.0
nilearn>=0.8.0
brian2>=2.4.0
pylsl>=1.14.0
matplotlib>=3.5.0
pyvista>=0.32.0
pyyaml>=6.0
click>=8.0.0
tqdm>=4.62.0
''',
        
        'requirements/dev.txt': '''# Development dependencies
-r base.txt
pytest>=6.2.0
black>=21.0.0
flake8>=4.0.0
mypy>=0.910
''',
        
        'requirements/gpu.txt': '''# GPU dependencies
-r base.txt
cupy>=9.0.0
torch>=1.10.0
tensorflow>=2.7.0
''',
        
        'requirements/visualization.txt': '''# Visualization dependencies
-r base.txt
plotly>=5.0.0
seaborn>=0.11.0
streamlit>=1.2.0
''',
        
        'requirements/hardware.txt': '''# Hardware dependencies
-r base.txt
pyserial>=3.5
pyusb>=1.2.0
bleak>=0.13.0
''',
        
        'configs/default.yaml': '''# Brain-Forge Configuration
system:
  name: "brain-forge"
  version: "0.1.0"
  debug: false

hardware:
  omp_helmet:
    enabled: true
    channels: 306
    sampling_rate: 1000
  
  kernel_optical:
    enabled: true
    channels: 104
    sampling_rate: 10
  
  accelerometer:
    enabled: true
    axes: 3
    sampling_rate: 1000

processing:
  compression:
    enabled: true
    algorithm: "wavelet"
    ratio: 5

simulation:
  neurons: 100000
  timestep: 0.1
  duration: 1000
''',
        
        'docs/getting-started.md': '''# Getting Started with Brain-Forge

## Installation

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements/base.txt`

## Basic Usage

```python
from src.integrated_system import IntegratedBrainSystem

# Initialize system
brain_system = IntegratedBrainSystem()

# Run brain transfer protocol
result = await brain_system.run_brain_transfer_protocol("subject_001")
```

## Documentation

See the `docs/` directory for detailed documentation.
'''
    }
    
    # Create directories
    print("📁 Creating directories...")
    created_dirs = 0
    for path_str in structure.keys():
        path = project_root / path_str
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {path_str}")
            created_dirs += 1
    
    # Create __init__.py files in Python packages
    print("🐍 Creating __init__.py files...")
    python_dirs = [
        'src', 'src/core', 'src/acquisition', 'src/processing', 'src/mapping',
        'src/simulation', 'src/transfer', 'src/visualization', 'src/api',
        'src/hardware', 'src/hardware/device_drivers', 'src/hardware/calibration',
        'src/hardware/interfaces', 'src/ml', 'src/ml/models', 'src/ml/training',
        'src/ml/inference', 'src/utils', 'tests', 'tests/unit', 'tests/integration',
        'tests/hardware', 'tests/performance'
    ]
    
    init_files = 0
    for dir_path in python_dirs:
        init_file = project_root / dir_path / '__init__.py'
        if not init_file.exists():
            with open(init_file, 'w') as f:
                f.write(f'"""Brain-Forge {dir_path.split("/")[-1]} module"""\n')
            init_files += 1
    
    # Create files with content
    print("📄 Creating files...")
    created_files = 0
    for file_path, content in files.items():
        full_path = project_root / file_path
        if not full_path.exists():
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"   Created: {file_path}")
            created_files += 1
    
    # Create example Python files
    example_files = {
        'examples/quick_start.py': '''#!/usr/bin/env python3
"""Brain-Forge Quick Start Example"""

print("🧠 Welcome to Brain-Forge!")
print("Setting up brain scanning system...")

# TODO: Add actual implementation
print("✅ Brain-Forge initialized successfully!")
''',
        
        'scripts/setup_environment.py': '''#!/usr/bin/env python3
"""Environment setup script for Brain-Forge"""

import subprocess
import sys

def setup_environment():
    """Set up the Brain-Forge development environment"""
    print("🧠 Setting up Brain-Forge environment...")
    
    # Install requirements
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements/base.txt"])
        print("✅ Base requirements installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False
    
    print("✅ Environment setup complete!")
    return True

if __name__ == "__main__":
    setup_environment()
'''
    }
    
    for file_path, content in example_files.items():
        full_path = project_root / file_path
        if not full_path.exists():
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            created_files += 1
    
    # Summary
    print("\n" + "="*50)
    print("🎉 Brain-Forge Project Structure Created!")
    print("="*50)
    print(f"📁 Directories created: {created_dirs}")
    print(f"🐍 __init__.py files: {init_files}")
    print(f"📄 Files created: {created_files}")
    print("\n🚀 Next steps:")
    print("1. python -m venv venv")
    print("2. source venv/bin/activate  # or venv\\Scripts\\activate on Windows")
    print("3. pip install -r requirements/base.txt")
    print("4. python examples/quick_start.py")
    print("\n📚 Documentation: docs/getting-started.md")
    print("="*50)


if __name__ == "__main__":
    create_brain_forge_structure()
