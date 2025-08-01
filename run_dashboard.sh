#!/bin/bash

# Brain-Forge Streamlit Dashboard Runner
# This script launches the production scientific dashboard

echo "🧠 Starting Brain-Forge Scientific Dashboard..."
echo "======================================================"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit
fi

# Check if we're in the right directory
if [ ! -f "src/streamlit_app.py" ]; then
    echo "❌ Error: streamlit_app.py not found in src/"
    echo "   Please run this script from the Brain-Forge root directory"
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Launch Streamlit dashboard
echo "🚀 Launching Brain-Forge Scientific Dashboard..."
echo "📡 Dashboard will be available at: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the dashboard"
echo ""

cd src
streamlit run streamlit_app.py --server.port 8501 --server.address localhost --theme.base dark

echo ""
echo "🛑 Dashboard stopped."
