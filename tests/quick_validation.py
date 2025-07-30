#!/usr/bin/env python3
"""
Quick Brain-Forge System Validation
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def validate_basic_imports():
    """Test basic scientific imports"""
    try:
        import numpy as np
        import scipy
        import matplotlib.pyplot as plt
        print("✅ Basic scientific libraries working")
        return True
    except ImportError as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def validate_brain_forge_imports():
    """Test Brain-Forge core imports"""
    try:
        from core.config import Config
        from core.logger import get_logger
        from core.exceptions import BrainForgeError
        print("✅ Brain-Forge core modules working")
        return True
    except ImportError as e:
        print(f"❌ Brain-Forge imports failed: {e}")
        return False

def validate_processing_system():
    """Test processing system imports"""
    try:
        from processing import RealTimeFilter, WaveletCompressor
        print("✅ Processing system working")
        return True
    except ImportError as e:
        print(f"❌ Processing system failed: {e}")
        return False

def main():
    """Run validation tests"""
    print("🧠 Brain-Forge System Validation")
    print("=" * 40)
    
    tests = [
        validate_basic_imports,
        validate_brain_forge_imports,
        validate_processing_system
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 40)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All systems operational!")
        return 0
    else:
        print("⚠️  Some systems need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
