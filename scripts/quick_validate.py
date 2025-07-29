#!/usr/bin/env python3
"""Brain-Forge Validation Script - Detect stub code and validate completion"""

import os
import re
import sys
from pathlib import Path


def scan_file_for_stubs(file_path):
    """Scan a file for stub patterns"""
    patterns = [
        r'raise\s+NotImplementedError',
        r'#\s*TODO',
        r'#\s*FIXME',
        r'pass\s*$',
        r'def\s+\w+\([^)]*\):\s*pass\s*$'
    ]
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        stubs = []
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                line_num = content[:match.start()].count('\n') + 1
                stubs.append((line_num, match.group().strip()))
        
        lines = content.split('\n')
        code_lines = [line for line in lines 
                     if line.strip() and not line.strip().startswith('#')]
        
        return {
            'total_lines': len(lines),
            'code_lines': len(code_lines),
            'stubs': stubs
        }
    except Exception as e:
        return {'error': str(e)}


def main():
    """Run validation"""
    root = Path(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
    src_path = root / 'src'
    
    print("🧠 BRAIN-FORGE VALIDATION REPORT")
    print("=" * 60)
    
    if not src_path.exists():
        print("❌ ERROR: src/ directory not found!")
        return
    
    # Core modules to check
    core_modules = {
        'integrated_system.py': 'Main BCI system',
        'core/exceptions.py': 'Exception handling',
        'processing/__init__.py': 'Processing pipeline',
        'api/rest_api.py': 'REST API'
    }
    
    total_files = 0
    total_lines = 0
    total_stubs = 0
    
    print("\n📊 CORE MODULE STATUS:")
    for module, desc in core_modules.items():
        module_path = src_path / module
        if module_path.exists():
            result = scan_file_for_stubs(module_path)
            if 'error' not in result:
                stubs = result['stubs']
                lines = result['code_lines']
                total_files += 1
                total_lines += lines
                total_stubs += len(stubs)
                
                status = "✅" if len(stubs) == 0 else f"⚠️ {len(stubs)} stubs"
                print(f"   {module}: {lines} lines - {status}")
                
                if stubs:
                    for line_num, stub in stubs[:2]:
                        print(f"      Line {line_num}: {stub}")
            else:
                print(f"   {module}: ❌ ERROR - {result['error']}")
        else:
            print(f"   {module}: ❌ MISSING")
    
    # Scan all Python files
    print(f"\n📁 SCANNING ALL PYTHON FILES...")
    all_py_files = list(src_path.rglob('*.py'))
    all_stubs = 0
    
    for py_file in all_py_files:
        result = scan_file_for_stubs(py_file)
        if 'error' not in result:
            all_stubs += len(result['stubs'])
    
    print(f"   Total Python files: {len(all_py_files)}")
    print(f"   Total stub locations: {all_stubs}")
    
    # Calculate completion
    completion = max(0, 100 - (all_stubs * 5))
    
    if completion >= 95:
        status = "PRODUCTION READY ✅"
    elif completion >= 85:
        status = "NEAR COMPLETE 🔶"
    else:
        status = "IN DEVELOPMENT 🚧"
    
    print(f"\n🎯 ASSESSMENT:")
    print(f"   Completion: {completion:.0f}% - {status}")
    
    if all_stubs == 0:
        print("   🎉 NO STUB CODE DETECTED!")
    else:
        print(f"   ⚠️  {all_stubs} stub locations need implementation")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
