#!/usr/bin/env python3
"""
Brain-Forge Development Environment Setup

Automated setup script for Brain-Forge development environment.
Handles virtual environment creation, dependency installation, and configuration.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(command, check=True, capture_output=False):
    """Run shell command with error handling"""
    print(f"Running: {command}")
    result = subprocess.run(
        command, 
        shell=True, 
        check=check, 
        capture_output=capture_output,
        text=True
    )
    if capture_output:
        return result.stdout.strip()
    return result.returncode == 0

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    print("\n🏗️  Setting up virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("⚠️  Virtual environment already exists")
        response = input("Do you want to recreate it? (y/N): ")
        if response.lower() == 'y':
            shutil.rmtree(venv_path)
        else:
            print("✅ Using existing virtual environment")
            return True
    
    if run_command(f"{sys.executable} -m venv venv"):
        print("✅ Virtual environment created")
        return True
    else:
        print("❌ Failed to create virtual environment")
        return False

def activate_venv():
    """Get activation command for virtual environment"""
    if os.name == 'nt':  # Windows
        return "venv\\Scripts\\activate"
    else:  # Unix/Linux/macOS
        return "source venv/bin/activate"

def install_dependencies():
    """Install project dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Get Python executable in venv
    if os.name == 'nt':
        python_exe = "venv\\Scripts\\python.exe"
        pip_exe = "venv\\Scripts\\pip.exe"
    else:
        python_exe = "venv/bin/python"
        pip_exe = "venv/bin/pip"
    
    # Upgrade pip first
    if not run_command(f"{python_exe} -m pip install --upgrade pip"):
        print("❌ Failed to upgrade pip")
        return False
    
    # Install base requirements
    requirements_files = [
        "requirements/base.txt",
        "requirements/dev.txt"
    ]
    
    for req_file in requirements_files:
        if Path(req_file).exists():
            print(f"📋 Installing {req_file}...")
            if not run_command(f"{pip_exe} install -r {req_file}"):
                print(f"❌ Failed to install {req_file}")
                return False
        else:
            print(f"⚠️  {req_file} not found, skipping")
    
    # Install package in editable mode
    if Path("setup.py").exists() or Path("pyproject.toml").exists():
        print("📦 Installing Brain-Forge in editable mode...")
        if not run_command(f"{pip_exe} install -e ."):
            print("❌ Failed to install Brain-Forge package")
            return False
    
    print("✅ Dependencies installed successfully")
    return True

def setup_configuration():
    """Set up configuration files"""
    print("\n⚙️  Setting up configuration...")
    
    # Create .env file from template
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_example.exists() and not env_file.exists():
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
        print("   Please review and customize .env file for your environment")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  No .env.example found")
    
    # Create necessary directories
    directories = [
        "data",
        "temp", 
        "results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print(f"✅ Created directories: {', '.join(directories)}")
    
    return True

def verify_installation():
    """Verify installation by importing Brain-Forge"""
    print("\n🧪 Verifying installation...")
    
    if os.name == 'nt':
        python_exe = "venv\\Scripts\\python.exe"
    else:
        python_exe = "venv/bin/python"
    
    # Test basic imports
    test_imports = [
        "import sys; print(f'Python: {sys.version}')",
        "import numpy; print(f'NumPy: {numpy.__version__}')",
        "import scipy; print(f'SciPy: {scipy.__version__}')",
        "import fastapi; print(f'FastAPI: {fastapi.__version__}')",
    ]
    
    for test_import in test_imports:
        try:
            result = run_command(
                f"{python_exe} -c \"{test_import}\"", 
                capture_output=True
            )
            if result:
                print(f"✅ {result}")
        except:
            print(f"⚠️  Failed: {test_import}")
    
    # Test Brain-Forge specific imports
    try:
        bf_test = "import sys; sys.path.insert(0, 'src'); from core.config import Config; print('Brain-Forge: Core modules importable')"
        result = run_command(f"{python_exe} -c \"{bf_test}\"", capture_output=True)
        if result:
            print(f"✅ {result}")
    except:
        print("⚠️  Brain-Forge core modules may have import issues")
    
    return True

def setup_git_hooks():
    """Set up pre-commit hooks"""
    print("\n🔧 Setting up git hooks...")
    
    if os.name == 'nt':
        python_exe = "venv\\Scripts\\python.exe"
    else:
        python_exe = "venv/bin/python"
    
    if Path(".pre-commit-config.yaml").exists():
        if run_command(f"{python_exe} -m pre_commit install"):
            print("✅ Pre-commit hooks installed")
        else:
            print("⚠️  Failed to install pre-commit hooks")
    else:
        print("⚠️  No .pre-commit-config.yaml found")
    
    return True

def print_next_steps():
    """Print next steps for user"""
    print("\n" + "="*60)
    print("🎉 Brain-Forge Development Environment Setup Complete!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. Activate the virtual environment:")
    print(f"   {activate_venv()}")
    
    print("\n2. Review and customize configuration:")
    print("   edit .env")
    
    print("\n3. Test the API server:")
    print("   python src/api/rest_api.py")
    
    print("\n4. Run validation tests:")
    print("   python validation/test_project_completion.py")
    
    print("\n5. Start development:")
    print("   - API docs: http://localhost:8000/docs")
    print("   - Health check: http://localhost:8000/health")
    
    print("\n📚 Documentation:")
    print("   - README.md - Project overview")
    print("   - docs/api/rest_api.md - API documentation")
    print("   - examples/ - Usage examples")
    
    print("\n🆘 Support:")
    print("   - Check logs in logs/brain_forge.log")
    print("   - Run tools/test_api_functionality.py for diagnostics")

def main():
    """Main setup function"""
    print("🧠 Brain-Forge Development Environment Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    # Setup steps
    setup_steps = [
        ("Creating virtual environment", create_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Setting up configuration", setup_configuration),
        ("Verifying installation", verify_installation),
        ("Setting up git hooks", setup_git_hooks),
    ]
    
    for step_name, step_func in setup_steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            print(f"❌ Setup failed at: {step_name}")
            return False
    
    print_next_steps()
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
