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
            cwd=Path(__file__).parent
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


def run_pytest_tests() -> Dict[str, Any]:
    """Run pytest validation tests"""
    print("🧪 Running Brain-Forge validation tests...")
    
    test_files = [
        'test_core_infrastructure.py',
        'test_processing_validation.py', 
        'test_hardware_validation.py',
        'test_streaming_validation.py'
    ]
    
    results = {}
    total_tests = 0
    passed_tests = 0
    
    for test_file in test_files:
        print(f"  Running {test_file}...")
        
        result = run_command(['python', '-m', 'pytest', test_file, '-v'])
        results[test_file] = result
        
        if result['success']:
            # Count tests from pytest output
            stdout = result['stdout']
            if 'passed' in stdout:
                try:
                    # Extract test counts from pytest output
                    lines = stdout.split('\n')
                    for line in lines:
                        if 'passed' in line and ('failed' in line or 'error' in line):
                            # Format like "5 passed, 2 failed"
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == 'passed':
                                    passed_tests += int(parts[i-1])
                                if part in ['failed', 'error']:
                                    total_tests += int(parts[i-1])
                        elif line.strip().endswith('passed'):
                            # Format like "5 passed"
                            parts = line.split()
                            if len(parts) >= 2:
                                passed_tests += int(parts[0])
                    
                    total_tests += passed_tests
                except:
                    pass
            
            print(f"    ✅ {test_file} passed ({result['duration']:.2f}s)")
        else:
            print(f"    ❌ {test_file} failed ({result['duration']:.2f}s)")
            if result['stderr']:
                print(f"       Error: {result['stderr'][:200]}...")
    
    return {
        'results': results,
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success': all(r['success'] for r in results.values())
    }


def check_code_structure() -> Dict[str, Any]:
    """Check code structure and imports"""
    print("🏗️  Checking code structure...")
    
    # Check if main modules can be imported
    import_tests = [
        'src.core.config',
        'src.core.exceptions', 
        'src.core.logger',
        'src.processing',
        'src.hardware',
        'src.streaming',
        'src.transfer.pattern_extraction'
    ]
    
    results = {}
    
    for module in import_tests:
        try:
            result = run_command(['python', '-c', f'import {module}; print("OK")'])
            results[module] = result['success']
            
            if result['success']:
                print(f"    ✅ {module}")
            else:
                print(f"    ❌ {module}: {result['stderr'][:100]}...")
                
        except Exception as e:
            results[module] = False
            print(f"    ❌ {module}: {str(e)[:100]}...")
    
    return {
        'results': results,
        'success': all(results.values())
    }


def analyze_codebase_metrics() -> Dict[str, Any]:
    """Analyze codebase metrics"""
    print("📊 Analyzing codebase metrics...")
    
    src_path = Path('src')
    if not src_path.exists():
        return {'error': 'src directory not found'}
    
    metrics = {
        'total_files': 0,
        'total_lines': 0,
        'python_files': 0,
        'test_files': 0,
        'modules': []
    }
    
    for file_path in src_path.rglob('*.py'):
        metrics['total_files'] += 1
        metrics['python_files'] += 1
        
        if 'test_' in file_path.name:
            metrics['test_files'] += 1
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                metrics['total_lines'] += lines
                
                metrics['modules'].append({
                    'name': str(file_path.relative_to(src_path)),
                    'lines': lines
                })
        except Exception:
            pass
    
    # Sort modules by line count
    metrics['modules'].sort(key=lambda x: x['lines'], reverse=True)
    
    print(f"    📁 Total files: {metrics['total_files']}")
    print(f"    🐍 Python files: {metrics['python_files']}")
    print(f"    🧪 Test files: {metrics['test_files']}")
    print(f"    📄 Total lines: {metrics['total_lines']}")
    print(f"    🏆 Largest modules:")
    
    for module in metrics['modules'][:5]:
        print(f"       {module['name']}: {module['lines']} lines")
    
    return metrics


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
    print(f"{structure_icon} Code structure validation: {'PASSED' if structure_success else 'FAILED'}")
    
    # Test results
    if 'error' not in test_results:
        test_success = test_results['success']
        test_icon = "✅" if test_success else "❌"
        passed = test_results['passed_tests']
        total = test_results['total_tests']
        print(f"{test_icon} Test validation: {'PASSED' if test_success else 'FAILED'} ({passed}/{total} tests)")
    else:
        print("❌ Test validation: ERROR")
    
    # Metrics summary
    if 'error' not in metrics:
        print(f"📊 Codebase metrics: {metrics['total_lines']} lines across {metrics['python_files']} files")
    
    # Overall status
    overall_success = (
        structure_success and 
        (test_results.get('success', False) if 'error' not in test_results else False)
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
