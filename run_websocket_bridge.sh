#!/bin/bash

# Brain-Forge WebSocket Bridge Runner
# Connects React demo GUI to Python backend for real data

echo "🔗 Starting Brain-Forge WebSocket Bridge..."
echo "=============================================="

# Check if websockets is installed
python3 -c "import websockets" 2>/dev/null || {
    echo "❌ websockets not found. Installing..."
    pip install websockets
}

# Check if numpy is installed
python3 -c "import numpy" 2>/dev/null || {
    echo "❌ numpy not found. Installing..."
    pip install numpy
}

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Launch WebSocket bridge
echo "🚀 Starting WebSocket Bridge Server..."
echo "📡 React GUI can connect to: ws://localhost:8765"
echo "⏹️  Press Ctrl+C to stop the bridge"
echo ""

python3 websocket_bridge.py

echo ""
echo "🛑 WebSocket bridge stopped."
