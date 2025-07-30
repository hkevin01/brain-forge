#!/bin/bash
# Project Restructuring Script for Brain-Forge
# Phase 1: Moving files to appropriate directories

echo "🚀 Starting Brain-Forge Project Restructuring..."

# Move completion and status reports to reports/
echo "📊 Moving reports to reports/ directory..."
mv COMPLETION_REPORT.md reports/ 2>/dev/null
mv FINAL_COMPLETION_STATUS.md reports/ 2>/dev/null
mv FINAL_SUCCESS_REPORT.md reports/ 2>/dev/null
mv IMPLEMENTATION_COMPLETE.md reports/ 2>/dev/null
mv PROJECT_CLEANUP_SUMMARY.md reports/ 2>/dev/null
mv PROJECT_COMPLETION_REPORT.md reports/ 2>/dev/null
mv PROJECT_PROGRESS_TRACKER.md reports/ 2>/dev/null
mv PROJECT_REORGANIZATION_COMPLETE.md reports/ 2>/dev/null
mv PROJECT_STATUS.md reports/ 2>/dev/null

# Move validation scripts to validation/
echo "🔍 Moving validation scripts to validation/ directory..."
mv validate_*.py validation/ 2>/dev/null
mv quick_validation.py validation/ 2>/dev/null
mv quick_test.py validation/ 2>/dev/null
mv test_basic_functionality.py validation/ 2>/dev/null
mv test_core_imports.py validation/ 2>/dev/null
mv test_imports.py validation/ 2>/dev/null
mv test_project_completion.py validation/ 2>/dev/null

# Move cleanup and utility scripts to tools/
echo "🛠️ Moving tools and utilities to tools/ directory..."
mv cleanup_*.py tools/ 2>/dev/null
mv cleanup_*.sh tools/ 2>/dev/null
mv brain_forge_complete.py tools/ 2>/dev/null
mv final_cleanup.py tools/ 2>/dev/null
mv demonstrate_completion.sh tools/ 2>/dev/null
mv install_and_validate.sh tools/ 2>/dev/null
mv run_tests.py tools/ 2>/dev/null
mv run_validation.py tools/ 2>/dev/null

# Move legacy/cleanup instructions to archive/
echo "📦 Moving legacy files to archive/ directory..."
mv CLEANUP_INSTRUCTIONS.md archive/ 2>/dev/null
mv .delete_brain_forge_complete.py archive/ 2>/dev/null

echo "✅ Phase 1 restructuring complete!"
