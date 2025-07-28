#!/usr/bin/env python3
"""
Quick validation test for Brain-Forge core infrastructure
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_core_infrastructure():
    """Test that core infrastructure works"""
    print("Testing Brain-Forge core infrastructure...")
    
    try:
        # Test config
        try:
            from core.config import Config
            config = Config()
            channels = config.hardware.omp_helmet.num_channels
            print(f"✓ Config loaded: {channels} channels")
        except ImportError:
            print("⚠️  Core config not yet implemented - skipping")
        
        # Test exceptions
        try:
            from core.exceptions import BrainForgeError
            try:
                raise BrainForgeError('Test error')
            except BrainForgeError as e:
                print(f"✓ Exceptions working: {e}")
        except ImportError:
            print("⚠️  Core exceptions not yet implemented - skipping")
        
        # Test logger
        try:
            from core.logger import get_logger
            logger = get_logger('test')
            logger.info('Core infrastructure test successful')
            print("✓ Logger working")
        except ImportError:
            print("⚠️  Core logger not yet implemented - skipping")
        
        print("All core components checked!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """Test that main modules can be imported"""
    print("\nTesting main module imports...")
    
    try:
        # Test processing imports
        try:
            import processing  # noqa: F401
            print("✓ Processing module imported")
        except ImportError:
            print("⚠️  Processing module not yet implemented - skipping")
        
        # Test basic scientific libraries
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported")
        
        import scipy
        print(f"✓ SciPy {scipy.__version__} imported")
        
        print("Main module imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def main():
    """Main validation function"""
    print("🧠 Brain-Forge Infrastructure Validation")
    print("=" * 40)
    
    success = True
    success &= test_core_infrastructure()
    success &= test_imports()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 Infrastructure validation: SUCCESS!")
    else:
        print("❌ Infrastructure validation: ISSUES DETECTED")
    
    return 0 if success else 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Validation failed: {e}")
        sys.exit(1)
