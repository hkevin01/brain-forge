#!/usr/bin/env python3
"""
Comprehensive Test Runner for Brain-Forge

This script provides a comprehensive test runner for the Brain-Forge
system, including unit tests, integration tests, performance tests,
and coverage reporting.
"""

import sys
import subprocess
import argparse
import time
from pathlib import Path
import json


def run_command(command, capture_output=True):
    """Run a command and return the result"""
    print(f"Running: {' '.join(command)}")
    
    if capture_output:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        return result.returncode, result.stdout, result.stderr
    else:
        result = subprocess.run(
            command,
            cwd=Path(__file__).parent.parent
        )
        return result.returncode, "", ""


def run_unit_tests():
    """Run unit tests"""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    command = [
        sys.executable, "-m", "pytest",
        "tests/unit/",
        "-v",
        "--tb=short",
        "-m", "unit",
        "--junitxml=test-results/unit-tests.xml"
    ]
    
    returncode, stdout, stderr = run_command(command, capture_output=False)
    
    if returncode == 0:
        print("✅ Unit tests PASSED")
    else:
        print("❌ Unit tests FAILED")
        if stderr:
            print(f"Error output:\n{stderr}")
    
    return returncode == 0


def run_integration_tests():
    """Run integration tests"""
    print("\n" + "="*60)
    print("RUNNING INTEGRATION TESTS")
    print("="*60)
    
    command = [
        sys.executable, "-m", "pytest",
        "tests/integration/",
        "-v",
        "--tb=short",
        "-m", "integration",
        "--junitxml=test-results/integration-tests.xml"
    ]
    
    returncode, stdout, stderr = run_command(command, capture_output=False)
    
    if returncode == 0:
        print("✅ Integration tests PASSED")
    else:
        print("❌ Integration tests FAILED")
        if stderr:
            print(f"Error output:\n{stderr}")
    
    return returncode == 0


def run_performance_tests():
    """Run performance tests"""
    print("\n" + "="*60)
    print("RUNNING PERFORMANCE TESTS")
    print("="*60)
    
    command = [
        sys.executable, "-m", "pytest",
        "tests/performance/",
        "-v",
        "--tb=short",
        "-m", "performance",
        "--junitxml=test-results/performance-tests.xml"
    ]
    
    returncode, stdout, stderr = run_command(command, capture_output=False)
    
    if returncode == 0:
        print("✅ Performance tests PASSED")
    else:
        print("❌ Performance tests FAILED")
        if stderr:
            print(f"Error output:\n{stderr}")
    
    return returncode == 0


def run_coverage_analysis():
    """Run test coverage analysis"""
    print("\n" + "="*60)
    print("RUNNING COVERAGE ANALYSIS")
    print("="*60)
    
    # Install coverage if not available
    try:
        import coverage
    except ImportError:
        print("Installing coverage...")
        subprocess.run([sys.executable, "-m", "pip", "install", "coverage"])
    
    # Run tests with coverage
    command = [
        sys.executable, "-m", "coverage", "run",
        "--source=src",
        "-m", "pytest",
        "tests/unit/",
        "tests/integration/",
        "-v"
    ]
    
    returncode, stdout, stderr = run_command(command)
    
    if returncode == 0:
        # Generate coverage report
        print("\nGenerating coverage report...")
        
        # Console report
        run_command([sys.executable, "-m", "coverage", "report", "-m"])
        
        # HTML report
        run_command([sys.executable, "-m", "coverage", "html", 
                    "--directory=test-results/coverage-html"])
        
        # XML report
        run_command([sys.executable, "-m", "coverage", "xml", 
                    "-o", "test-results/coverage.xml"])
        
        print("✅ Coverage analysis completed")
        print("📊 Coverage reports generated in test-results/")
        return True
    else:
        print("❌ Coverage analysis FAILED")
        return False


def run_linting():
    """Run code linting"""
    print("\n" + "="*60)
    print("RUNNING CODE LINTING")
    print("="*60)
    
    # Check if flake8 is available
    try:
        import flake8
    except ImportError:
        print("Installing flake8...")
        subprocess.run([sys.executable, "-m", "pip", "install", "flake8"])
    
    command = [
        sys.executable, "-m", "flake8",
        "src/",
        "tests/",
        "--max-line-length=88",
        "--ignore=E203,W503",
        "--output-file=test-results/flake8-report.txt"
    ]
    
    returncode, stdout, stderr = run_command(command)
    
    if returncode == 0:
        print("✅ Linting PASSED")
        return True
    else:
        print("❌ Linting found issues")
        print(f"Check test-results/flake8-report.txt for details")
        return False


def run_type_checking():
    """Run type checking with mypy"""
    print("\n" + "="*60)
    print("RUNNING TYPE CHECKING")
    print("="*60)
    
    # Check if mypy is available
    try:
        import mypy
    except ImportError:
        print("Installing mypy...")
        subprocess.run([sys.executable, "-m", "pip", "install", "mypy"])
    
    command = [
        sys.executable, "-m", "mypy",
        "src/",
        "--ignore-missing-imports",
        "--show-error-codes",
        "--output", "test-results/mypy-report.txt"
    ]
    
    returncode, stdout, stderr = run_command(command)
    
    if returncode == 0:
        print("✅ Type checking PASSED")
        return True
    else:
        print("❌ Type checking found issues")
        print(f"Check test-results/mypy-report.txt for details")
        return False


def setup_test_environment():
    """Set up test environment"""
    print("Setting up test environment...")
    
    # Create test results directory
    test_results_dir = Path("test-results")
    test_results_dir.mkdir(exist_ok=True)
    
    # Install test dependencies
    dependencies = [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "psutil>=5.8.0",
    ]
    
    print("Installing test dependencies...")
    for dep in dependencies:
        subprocess.run([
            sys.executable, "-m", "pip", "install", dep
        ], capture_output=True)
    
    print("✅ Test environment setup complete")


def generate_test_report(results):
    """Generate comprehensive test report"""
    print("\n" + "="*60)
    print("GENERATING TEST REPORT")
    print("="*60)
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "summary": {
            "total_test_suites": len(results),
            "passed_suites": sum(1 for r in results.values() if r),
            "failed_suites": sum(1 for r in results.values() if not r),
        }
    }
    
    # Calculate overall success
    overall_success = all(results.values())
    report["overall_success"] = overall_success
    
    # Save JSON report
    with open("test-results/test-report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"Test Suites Run: {report['summary']['total_test_suites']}")
    print(f"Passed: {report['summary']['passed_suites']}")
    print(f"Failed: {report['summary']['failed_suites']}")
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Brain-Forge is ready for deployment")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("🔧 Please fix issues before deployment")
    
    print(f"\n📄 Detailed report saved to: test-results/test-report.json")
    
    return overall_success


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Brain-Forge Test Runner")
    parser.add_argument("--unit", action="store_true", 
                       help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", 
                       help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", 
                       help="Run performance tests only")
    parser.add_argument("--coverage", action="store_true", 
                       help="Run coverage analysis")
    parser.add_argument("--lint", action="store_true", 
                       help="Run linting")
    parser.add_argument("--type-check", action="store_true", 
                       help="Run type checking")
    parser.add_argument("--all", action="store_true", 
                       help="Run all tests and checks")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick test suite (unit tests only)")
    parser.add_argument("--setup", action="store_true", 
                       help="Setup test environment only")
    
    args = parser.parse_args()
    
    # Setup test environment
    setup_test_environment()
    
    if args.setup:
        print("Test environment setup complete!")
        return 0
    
    results = {}
    
    start_time = time.time()
    
    print("\n🧠 Brain-Forge Comprehensive Test Suite")
    print("="*60)
    
    # Determine which tests to run
    run_all = args.all or not any([
        args.unit, args.integration, args.performance, 
        args.coverage, args.lint, args.type_check, args.quick
    ])
    
    if args.quick or args.unit or run_all:
        results["unit_tests"] = run_unit_tests()
    
    if args.integration or run_all:
        results["integration_tests"] = run_integration_tests()
    
    if args.performance or run_all:
        results["performance_tests"] = run_performance_tests()
    
    if args.coverage or run_all:
        results["coverage"] = run_coverage_analysis()
    
    if args.lint or run_all:
        results["linting"] = run_linting()
    
    if args.type_check or run_all:
        results["type_checking"] = run_type_checking()
    
    # Skip other tests if quick mode
    if args.quick and not run_all:
        print("\n⚡ Quick test mode - skipping integration and performance tests")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n⏱️ Total test execution time: {total_time:.2f} seconds")
    
    # Generate comprehensive report
    overall_success = generate_test_report(results)
    
    # Return appropriate exit code
    return 0 if overall_success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
