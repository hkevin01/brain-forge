#!/usr/bin/env python3
"""
Quick validation test for Brain-Forge core infrastructure
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_core_infrastructure():
    """Test that core infrastructure works"""
    print("Testing Brain-Forge core infrastructure...")
    
    try:
        # Test config
        from core.config import Config
        config = Config()
        print(f"✓ Config loaded: {config.hardware.omp_helmet.num_channels} channels")
        
        # Test exceptions  
        from core.exceptions import BrainForgeError
        try:
            raise BrainForgeError('Test error')
        except BrainForgeError as e:
            print(f"✓ Exceptions working: {e}")
        
        # Test logger
        from core.logger import get_logger
        logger = get_logger('test')
        logger.info('Core infrastructure test successful')
        print("✓ Logger working")
        
        print("All core components functional!")
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
        from processing import RealTimeFilter, WaveletCompressor
        print("✓ Processing modules imported")
        
        # Test stream manager
        from acquisition.stream_manager import StreamManager
        print("✓ Stream manager imported")
        
        # Test integrated system
        from integrated_system import IntegratedBrainSystem
        print("✓ Integrated system imported")
        
        print("All main modules can be imported!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_core_infrastructure()
    success2 = test_imports()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Brain-Forge infrastructure is functional.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)
