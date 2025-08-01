#!/bin/bash
# Brain-Forge GUI Quick Runner
# Simple script to directly run the GUI without setup checks
set -e

echo "🧠 Brain-Forge BCI Platform - Quick Start"
echo "========================================"

# Navigate to demo-gui directory
cd "$(dirname "$0")/demo-gui"

echo "🚀 Starting Brain-Forge GUI..."
echo "🌐 Will be available at: http://localhost:3000"
echo ""

# Start development server directly
npm run dev
