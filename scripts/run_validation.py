#!/usr/bin/env python3
"""
Brain-Forge Test Runner

Comprehensive test runner for the Brain-Forge neuroscience platform.
Executes all validation tests and provides detailed reporting.
"""

import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any


def run_command(cmd: List[str], timeout: int = 60) -> Dict[str, Any]:
    """Run a command and return results"""
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent
        )
        end_time = time.time()
        
        return {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'duration': end_time - start_time
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': f'Command timed out after {timeout} seconds',
            'duration': timeout
        }
    except Exception as e:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'duration': 0
        }


def check_code_structure():
    """Check project code structure"""
    print("🔍 Checking code structure...")
    
    project_root = Path(__file__).parent.parent
    required_dirs = ['src', 'tests', 'docs', 'examples', 'scripts']
    
    results = {
        'success': True,
        'missing_dirs': [],
        'found_dirs': []
    }
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            results['found_dirs'].append(dir_name)
            print(f"  ✅ {dir_name}/ directory found")
        else:
            results['missing_dirs'].append(dir_name)
            results['success'] = False
            print(f"  ❌ {dir_name}/ directory missing")
    
    return results


def analyze_codebase_metrics():
    """Analyze codebase metrics"""
    print("📊 Analyzing codebase metrics...")
    
    project_root = Path(__file__).parent.parent
    
    try:
        python_files = list(project_root.rglob('*.py'))
        total_lines = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                    total_lines += lines
            except Exception:
                continue
        
        metrics = {
            'python_files': len(python_files),
            'total_lines': total_lines
        }
        
        print(f"  📁 Python files: {metrics['python_files']}")
        print(f"  📝 Total lines: {metrics['total_lines']}")
        
        return metrics
        
    except Exception as e:
        return {'error': str(e)}


def run_pytest_tests():
    """Run pytest tests if available"""
    print("🧪 Running validation tests...")
    
    project_root = Path(__file__).parent.parent
    test_files = list(project_root.glob('tests/test_*.py'))
    
    if not test_files:
        print("  ⚠️  No test files found in tests/ directory")
        return {
            'success': True,
            'total_tests': 0,
            'passed_tests': 0,
            'message': 'No tests to run'
        }
    
    # Try to run tests manually since pytest might not be available
    passed_tests = 0
    total_tests = len(test_files)
    
    for test_file in test_files:
        print(f"  🔍 Running {test_file.name}...")
        result = run_command([sys.executable, str(test_file)])
        if result['success']:
            passed_tests += 1
            print(f"    ✅ PASSED")
        else:
            print(f"    ❌ FAILED: {result['stderr'][:100]}")
    
    success = passed_tests == total_tests
    
    return {
        'success': success,
        'total_tests': total_tests,
        'passed_tests': passed_tests
    }


def run_comprehensive_validation():
    """Run comprehensive Brain-Forge platform validation"""
    print("🧠 Brain-Forge Comprehensive Platform Validation")
    print("=" * 50)
    
    start_time = time.time()
    
    # 1. Check code structure
    structure_results = check_code_structure()
    
    # 2. Analyze codebase
    metrics = analyze_codebase_metrics()
    
    # 3. Run validation tests
    test_results = run_pytest_tests()
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Summary report
    print("\n" + "=" * 50)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 50)
    
    print(f"⏱️  Total validation time: {total_duration:.2f} seconds")
    
    # Structure results
    structure_success = structure_results['success']
    structure_icon = "✅" if structure_success else "❌"
    status = 'PASSED' if structure_success else 'FAILED'
    print(f"{structure_icon} Code structure validation: {status}")
    
    # Test results
    if 'error' not in test_results:
        test_success = test_results['success']
        test_icon = "✅" if test_success else "❌"
        passed = test_results['passed_tests']
        total = test_results['total_tests']
        status = 'PASSED' if test_success else 'FAILED'
        print(f"{test_icon} Test validation: {status} ({passed}/{total} tests)")
    else:
        print("❌ Test validation: ERROR")
    
    # Metrics summary
    if 'error' not in metrics:
        lines = metrics['total_lines']
        files = metrics['python_files']
        print(f"📊 Codebase metrics: {lines} lines across {files} files")
    
    # Overall status
    overall_success = (
        structure_success and
        (test_results.get('success', False) if 'error' not in test_results
         else False)
    )
    
    print("\n" + "=" * 50)
    if overall_success:
        print("🎉 BRAIN-FORGE PLATFORM VALIDATION: SUCCESS!")
        print("   The neuroscience platform is ready for deployment.")
    else:
        print("⚠️  BRAIN-FORGE PLATFORM VALIDATION: ISSUES DETECTED")
        print("   Review the above results and address any failures.")
    print("=" * 50)
    
    return {
        'overall_success': overall_success,
        'structure_results': structure_results,
        'test_results': test_results,
        'metrics': metrics,
        'duration': total_duration
    }


if __name__ == '__main__':
    try:
        results = run_comprehensive_validation()
        exit_code = 0 if results['overall_success'] else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        sys.exit(1)
