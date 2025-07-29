#!/usr/bin/env python3
"""
Quick Integration Test

Simple test to verify third-party library integration is working.
Run this to check if real libraries are available.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test if we can import all modules"""
    print("Testing imports...")
    
    try:
        from specialized_tools import BrainDecodeIntegration
        print("✅ BrainDecodeIntegration imported")
    except Exception as e:
        print(f"❌ BrainDecodeIntegration: {e}")
    
    try:
        from specialized_tools import NeuroKit2Integration  
        print("✅ NeuroKit2Integration imported")
    except Exception as e:
        print(f"❌ NeuroKit2Integration: {e}")

def test_library_availability():
    """Test if third-party libraries are available"""
    print("\nTesting library availability...")
    
    # Test Braindecode
    try:
        import braindecode
        print(f"✅ Braindecode v{braindecode.__version__}")
    except ImportError:
        print("⚠️  Braindecode not installed")
    
    # Test NeuroKit2
    try:
        import neurokit2 as nk
        print(f"✅ NeuroKit2 v{nk.__version__}")
    except ImportError:
        print("⚠️  NeuroKit2 not installed")
    
    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch v{torch.__version__}")
    except ImportError:
        print("⚠️  PyTorch not installed")

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from specialized_tools import BrainDecodeIntegration
        bd = BrainDecodeIntegration()
        model = bd.create_eegnet_classifier(n_channels=32, n_classes=2)
        print(f"✅ EEGNet model created: {model['name']}")
        print(f"   Framework: {model['framework']}")
    except Exception as e:
        print(f"❌ EEGNet test failed: {e}")
    
    try:
        from specialized_tools import NeuroKit2Integration
        import numpy as np
        
        nk2 = NeuroKit2Integration()
        test_signal = np.random.randn(1000)
        result = nk2.process_eeg_signals(test_signal, sampling_rate=250)
        print(f"✅ EEG processing completed")
        print(f"   Framework: {result['framework']}")
        print(f"   Quality score: {result['quality_score']:.3f}")
    except Exception as e:
        print(f"❌ NeuroKit2 test failed: {e}")

if __name__ == "__main__":
    print("🧠 Brain-Forge Third-Party Integration Test")
    print("=" * 50)
    
    test_imports()
    test_library_availability()
    test_basic_functionality()
    
    print("\n" + "=" * 50)
    print("🎉 Integration test completed!")
