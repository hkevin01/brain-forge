#!/usr/bin/env python3
"""
Project Test Runner

Unified test runner for the Brain-Forge project.
Runs all test suites and provides comprehensive reporting.
"""

import subprocess
import sys
from pathlib import Path


def run_all_tests():
    """Run all project tests"""
    project_root = Path(__file__).parent.parent
    
    print("🧠 Brain-Forge Test Runner")
    print("=" * 50)
    
    # Run pytest on tests directory
    tests_dir = project_root / "tests"
    if tests_dir.exists():
        print("Running main test suite...")
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(tests_dir), "-v"
        ], capture_output=True, text=True)
        
        print("Test Results:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
    
    # Run validation tests
    validation_dir = project_root / "validation"
    if validation_dir.exists():
        print("\nRunning validation tests...")
        validation_script = validation_dir / "test_project_completion.py"
        if validation_script.exists():
            result = subprocess.run([
                sys.executable, str(validation_script)
            ], capture_output=True, text=True)
            
            print("Validation Results:")
            print(result.stdout)
            if result.stderr:
                print("Validation Errors:")
                print(result.stderr)
    
    print("\n✅ Test execution complete!")

if __name__ == "__main__":
    run_all_tests()
