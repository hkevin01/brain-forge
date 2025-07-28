#!/usr/bin/env python3
"""
Simple import test to diagnose Brain-Forge module issues
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_core_imports():
    """Test core module imports"""
    try:
        print("Testing core.config import...")
        from core.config import Config
        print("✅ core.config imported successfully")
        
        print("Testing core.exceptions import...")
        from core.exceptions import BrainForgeError
        print("✅ core.exceptions imported successfully")
        
        print("Testing core.logger import...")
        from core.logger import get_logger
        print("✅ core.logger imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Core import failed: {e}")
        return False

def test_processing_imports():
    """Test processing module imports"""
    try:
        print("Testing processing import...")
        import processing
        print("✅ processing imported successfully")
        return True
    except Exception as e:
        print(f"❌ Processing import failed: {e}")
        return False

def test_config_creation():
    """Test config object creation"""
    try:
        print("Testing Config creation...")
        from core.config import Config
        config = Config()
        print(f"✅ Config created: {type(config)}")
        print(f"   Hardware enabled: OMP={config.hardware.omp_enabled}")
        return True
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return False

if __name__ == '__main__':
    print("🧠 Brain-Forge Import Diagnostics")
    print("=" * 40)
    
    success = True
    success &= test_core_imports()
    success &= test_processing_imports()
    success &= test_config_creation()
    
    print("=" * 40)
    if success:
        print("🎉 All imports successful!")
    else:
        print("⚠️  Import failures detected")
    
    sys.exit(0 if success else 1)
