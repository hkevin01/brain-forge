#!/bin/bash
# GUI Setup and Integration Script for Brain-Forge Demo
set -e

echo "🧠 Brain-Forge GUI Setup - Final Integration Phase"
echo "================================================="

# Change to demo-gui directory
cd /home/kevin/Projects/brain-forge/demo-gui

echo "📦 Installing npm dependencies..."
npm install

echo "🔧 Running type check..."
npm run type-check

echo "🏗️ Building application..."
npm run build

echo "✅ GUI Setup Complete - Ready for development server"
echo "🚀 Run 'npm run dev' to start development server"
