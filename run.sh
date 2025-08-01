#!/bin/bash
# Brain-Forge GUI Runner Script
# Comprehensive script to setup and run the complete Brain-Forge BCI Demo GUI
set -e

echo "🧠 Brain-Forge BCI Platform - GUI Runner"
echo "========================================"
echo "Starting comprehensive Brain-Computer Interface demonstration..."
echo ""

# Check if we're in the correct directory
if [ ! -f "demo-gui/package.json" ]; then
    echo "❌ Error: Must run from Brain-Forge root directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected: /path/to/brain-forge/"
    echo ""
    echo "💡 Please navigate to the Brain-Forge root directory and run:"
    echo "   cd /path/to/brain-forge"
    echo "   bash run.sh"
    exit 1
fi

echo "📂 Working directory: $(pwd)"
echo "🎯 Target: demo-gui application"
echo ""

# Change to demo-gui directory
cd demo-gui

echo "🔍 Checking GUI project structure..."
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found in demo-gui directory"
    exit 1
fi

if [ ! -d "src" ]; then
    echo "❌ Error: src directory not found"
    exit 1
fi

echo "✅ GUI project structure verified"
echo ""

# Check if node_modules exists, if not install dependencies
if [ ! -d "node_modules" ]; then
    echo "📦 Installing npm dependencies..."
    echo "   This may take a few minutes..."
    npm install
    echo "✅ Dependencies installed successfully"
    echo ""
else
    echo "✅ Dependencies already installed"
    echo ""
fi

# Run type check
echo "🔧 Running TypeScript type check..."
npm run type-check
echo "✅ Type check passed"
echo ""

# Check if build directory exists, if not create production build
if [ ! -d "dist" ]; then
    echo "🏗️  Creating production build..."
    npm run build
    echo "✅ Production build created"
    echo ""
else
    echo "✅ Production build already exists"
    echo ""
fi

echo "🚀 Starting Brain-Forge GUI Development Server..."
echo ""
echo "🌐 Server will be available at: http://localhost:3000"
echo "🧠 Features available:"
echo "   • Real-time 3D brain visualization"
echo "   • Multi-modal data acquisition simulation"
echo "   • Advanced signal processing display"
echo "   • Professional neuroscience interface"
echo ""
echo "⭐ Press Ctrl+C to stop the server"
echo ""
echo "🎉 Launching Brain-Forge BCI Demo GUI..."
echo "================================================="

# Start the development server
npm run dev
