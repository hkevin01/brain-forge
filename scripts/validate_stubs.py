#!/usr/bin/env python3
"""Brain-Forge Validation - Check for actual stub implementations"""

import os
import re
import sys
from pathlib import Path


def analyze_file(file_path):
    """Analyze a Python file for actual stub implementations"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Look for actual stub patterns (not comments or exception handlers)
        actual_stubs = []
        
        # Pattern 1: Functions that only raise NotImplementedError
        pattern1 = re.compile(r'def\s+(\w+)\([^)]*\):[^{]*raise\s+NotImplementedError', re.DOTALL)
        for match in pattern1.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            actual_stubs.append(f"Line {line_num}: Function '{match.group(1)}' raises NotImplementedError")
        
        # Pattern 2: Functions with only pass (excluding exception handlers)
        lines = content.split('\n')
        in_function = False
        current_function = ""
        function_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Function definition
            if re.match(r'def\s+(\w+)', stripped):
                if in_function and len(function_lines) == 1 and function_lines[0].strip() == 'pass':
                    # Previous function was just pass
                    actual_stubs.append(f"Line {i}: Function '{current_function}' only contains 'pass'")
                
                match = re.match(r'def\s+(\w+)', stripped)
                current_function = match.group(1) if match else "unknown"
                in_function = True
                function_lines = []
            
            elif in_function:
                if stripped and not stripped.startswith('"""') and not stripped.startswith('#'):
                    # Skip docstrings and comments
                    if stripped.startswith('def ') or stripped.startswith('class '):
                        # New function/class started, check previous
                        if len(function_lines) == 1 and function_lines[0].strip() == 'pass':
                            actual_stubs.append(f"Line {i}: Function '{current_function}' only contains 'pass'")
                        in_function = False
                    else:
                        function_lines.append(stripped)
        
        # Count actual code (not comments or empty lines)
        code_lines = [line for line in lines 
                     if line.strip() and 
                     not line.strip().startswith('#') and
                     not line.strip().startswith('"""') and
                     not line.strip().startswith("'''")]
        
        return {
            'total_lines': len(lines),
            'code_lines': len(code_lines),
            'actual_stubs': actual_stubs,
            'is_substantial': len(code_lines) > 10  # Has real implementation
        }
        
    except Exception as e:
        return {'error': str(e), 'actual_stubs': [], 'is_substantial': False}


def main():
    """Run the validation"""
    root = Path(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
    src_path = root / 'src'
    
    print("🧠 BRAIN-FORGE STUB DETECTION & VALIDATION")
    print("=" * 70)
    
    if not src_path.exists():
        print("❌ ERROR: src/ directory not found!")
        return
    
    # Key modules to validate
    key_modules = {
        'integrated_system.py': 'Main BCI integration system',
        'core/exceptions.py': 'Exception handling system', 
        'core/config.py': 'Configuration management',
        'processing/__init__.py': 'Real-time processing pipeline',
        'visualization/brain_visualization.py': '3D brain visualization',
        'api/rest_api.py': 'REST API interface',
        'specialized_tools.py': 'Specialized neuroscience tools'
    }
    
    print("\n📊 KEY MODULE ANALYSIS:")
    total_actual_stubs = 0
    substantial_modules = 0
    
    for module, description in key_modules.items():
        module_path = src_path / module
        if module_path.exists():
            result = analyze_file(module_path)
            
            if 'error' not in result:
                stubs = result['actual_stubs']
                lines = result['code_lines']
                is_substantial = result['is_substantial']
                
                total_actual_stubs += len(stubs)
                if is_substantial:
                    substantial_modules += 1
                
                # Status indicator
                if len(stubs) == 0 and is_substantial:
                    status = "✅ COMPLETE"
                elif len(stubs) == 0:
                    status = "⚠️ MINIMAL"
                else:
                    status = f"❌ {len(stubs)} STUBS"
                
                print(f"   {module}")
                print(f"      {lines} code lines - {status}")
                
                # Show actual stub details
                for stub in stubs:
                    print(f"      🔍 {stub}")
                    
            else:
                print(f"   {module}: ❌ ERROR - {result['error']}")
        else:
            print(f"   {module}: ❌ MISSING")
    
    # Scan all files for comprehensive analysis
    print(f"\n📁 COMPREHENSIVE SCAN:")
    all_files = list(src_path.rglob('*.py'))
    total_files = len(all_files)
    files_with_stubs = 0
    total_code_lines = 0
    
    for py_file in all_files:
        result = analyze_file(py_file)
        if 'error' not in result:
            total_code_lines += result['code_lines']
            if result['actual_stubs']:
                files_with_stubs += 1
    
    print(f"   📄 Total Python files: {total_files}")
    print(f"   📝 Total code lines: {total_code_lines:,}")
    print(f"   ✅ Substantial modules: {substantial_modules}/{len(key_modules)}")
    print(f"   🔍 Files with actual stubs: {files_with_stubs}")
    print(f"   ❌ Total actual stub functions: {total_actual_stubs}")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    
    completion_score = 0
    if substantial_modules >= 6:  # Most key modules substantial
        completion_score += 40
    if total_actual_stubs == 0:  # No actual stubs
        completion_score += 30
    if total_code_lines > 2000:  # Substantial codebase
        completion_score += 20
    if files_with_stubs == 0:  # No files have stubs
        completion_score += 10
    
    if completion_score >= 90:
        status = "🎉 PRODUCTION READY"
        assessment = "Fully implemented with comprehensive functionality"
    elif completion_score >= 75:
        status = "✅ NEARLY COMPLETE"
        assessment = "Well-developed with minimal remaining work"
    elif completion_score >= 50:
        status = "🔶 SUBSTANTIAL PROGRESS"
        assessment = "Good foundation but needs completion work"
    else:
        status = "🚧 IN DEVELOPMENT"
        assessment = "Early stage with significant work remaining"
    
    print(f"   Status: {status}")
    print(f"   Score: {completion_score}/100")
    print(f"   Assessment: {assessment}")
    
    if total_actual_stubs == 0:
        print(f"\n🎊 EXCELLENT: No actual stub implementations detected!")
        print(f"   The codebase appears to have real, working implementations.")
    else:
        print(f"\n⚠️  Found {total_actual_stubs} actual stub implementations that need work.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
