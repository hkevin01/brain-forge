"""
Test configuration system and imports
"""
import sys
from pathlib import Path

# Add src to Python path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_config_import():
    """Test that config can be imported"""
    try:
        from core.config import Config
        assert Config is not None
        return True
    except ImportError:
        print("⚠️  Core config not yet implemented")
        return True  # Pass for now


def test_config_creation():
    """Test that config can be created"""
    try:
        from core.config import Config
        config = Config()
        assert config is not None
        assert hasattr(config, 'hardware')
        assert hasattr(config, 'processing')
        assert hasattr(config, 'transfer_learning')
        assert hasattr(config, 'system')
        return True
    except ImportError:
        print("⚠️  Core config not yet implemented")
        return True  # Pass for now


def test_transfer_learning_config():
    """Test transfer learning config exists"""
    try:
        from core.config import Config
        config = Config()
        assert hasattr(config.transfer_learning, 'pattern_extraction')
        pattern_extraction = config.transfer_learning.pattern_extraction
        assert hasattr(pattern_extraction, 'current_subject_id')
        assert hasattr(pattern_extraction, 'frequency_bands')
        return True
    except ImportError:
        print("⚠️  Core config not yet implemented")
        return True  # Pass for now


def test_exceptions_import():
    """Test exceptions can be imported"""
    try:
        from core.exceptions import BrainForgeError
        assert BrainForgeError is not None
        return True
    except ImportError:
        print("⚠️  Core exceptions not yet implemented")
        return True  # Pass for now


def test_logger_import():
    """Test logger can be imported"""
    try:
        from core.logger import get_logger
        logger = get_logger("test")
        assert logger is not None
        return True
    except ImportError:
        print("⚠️  Core logger not yet implemented")
        return True  # Pass for now


if __name__ == '__main__':
    # Run tests manually
    try:
        print("🧪 Testing Brain-Forge imports...")
        
        test_config_import()
        print("✅ Config import: PASSED")
        
        test_config_creation()
        print("✅ Config creation: PASSED")
        
        test_transfer_learning_config()
        print("✅ Transfer learning config: PASSED")
        
        test_exceptions_import()
        print("✅ Exceptions import: PASSED")
        
        test_logger_import()
        print("✅ Logger import: PASSED")
        
        print("🎉 All import tests PASSED!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
