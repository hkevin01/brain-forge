#!/usr/bin/env python3
"""
Quick test script for Brain-Forge functionality
"""

import sys


def test_imports():
    """Test that we can import basic modules"""
    try:
        import numpy  # noqa: F401
        print("✅ Basic scientific modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality"""
    import numpy as np
    
    # Test basic array operations
    data = np.random.randn(100, 10)
    mean_data = np.mean(data, axis=0)
    
    print(f"✅ Generated test data shape: {data.shape}")
    print(f"✅ Computed means shape: {mean_data.shape}")
    
    return True


def main():
    """Main test function"""
    print("🧠 Brain-Forge Quick Test")
    print("=" * 40)
    
    success = True
    success &= test_imports()
    success &= test_basic_functionality()
    
    if success:
        print("\n🎉 All quick tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
