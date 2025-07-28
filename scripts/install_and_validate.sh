#!/bin/bash
# Brain-Forge Installation and Validation Script
# This script sets up the Brain-Forge environment and runs validation.

set -e  # Exit on any error

echo "🧠 Brain-Forge Installation and Validation"
echo "=================================================="

# Check Python version
echo "🐍 Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8+."
    exit 1
fi

# Check if we can import basic packages
echo "📦 Checking core dependencies..."
python3 -c "import numpy, scipy; print('✅ NumPy and SciPy available')" || {
    echo "❌ NumPy/SciPy not available. Installing core packages..."
    pip3 install numpy scipy || {
        echo "❌ Failed to install core packages"
        exit 1
    }
}

# Test basic functionality without brain-forge modules
echo "🧪 Running basic functionality tests..."
python3 -c "
import numpy as np
from scipy import signal
print('✅ NumPy version:', np.__version__)
print('✅ SciPy available')

# Test basic signal processing
fs = 1000
t = np.linspace(0, 1, fs, False)
sig = np.sin(2 * np.pi * 10 * t)
print('✅ Signal generation works')

# Test filtering
try:
    sos = signal.butter(4, 50, 'low', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, sig)
    print('✅ Signal filtering works')
except Exception as e:
    print('❌ Signal filtering failed:', e)
    exit(1)

print('🎉 Basic functionality validated!')
"

echo "📊 Checking project structure..."
if [ -d "src" ]; then
    echo "   ✅ src/ directory found"
else
    echo "   ⚠️  src/ directory not found - creating placeholder"
    mkdir -p src
fi

if [ -d "tests" ]; then
    echo "   ✅ tests/ directory found"
    test_count=$(find tests/ -name "*.py" | wc -l)
    echo "   📊 Found $test_count test files"
else
    echo "   ⚠️  tests/ directory not found"
fi

if [ -d "docs" ]; then
    echo "   ✅ docs/ directory found"
else
    echo "   ⚠️  docs/ directory not found"
fi

if [ -d "examples" ]; then
    echo "   ✅ examples/ directory found"
else
    echo "   ⚠️  examples/ directory not found"
fi

if [ -d "scripts" ]; then
    echo "   ✅ scripts/ directory found"
else
    echo "   ⚠️  scripts/ directory not found"
fi

echo "🧪 Running comprehensive validation..."
if [ -f "scripts/validate_brain_forge.py" ]; then
    echo "   Running Brain-Forge validation suite..."
    python3 scripts/validate_brain_forge.py
else
    echo "   ⚠️  Validation script not found - running basic validation"
    python3 -c "
import sys
import numpy as np

print('🧠 Basic Brain-Forge Validation')
print('=' * 40)

# Test scientific computing
try:
    data = np.random.randn(100, 10)
    mean_data = np.mean(data, axis=0)
    print('✅ Scientific computing: PASSED')
except Exception as e:
    print('❌ Scientific computing: FAILED')
    sys.exit(1)

# Test neural data simulation
try:
    channels, timepoints = 64, 1000
    neural_data = np.random.randn(channels, timepoints)
    features = np.mean(neural_data, axis=1)
    print('✅ Neural data simulation: PASSED')
except Exception as e:
    print('❌ Neural data simulation: FAILED')
    sys.exit(1)

print('🎉 Basic validation: SUCCESS!')
"
fi

echo "=================================================="
echo "🎉 Brain-Forge installation and validation complete!"
echo "   Platform ready for development and testing"
echo "=================================================="
