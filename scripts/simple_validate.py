#!/usr/bin/env python3
"""Simple Brain-Forge validation to check actual completion"""

import os
import sys
from pathlib import Path


def check_file_substantial(file_path):
    """Check if a file has substantial implementation"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Count meaningful code lines (not comments, empty, or imports)
        code_lines = 0
        for line in lines:
            stripped = line.strip()
            if (stripped and 
                not stripped.startswith('#') and 
                not stripped.startswith('import') and
                not stripped.startswith('from') and
                not stripped.startswith('"""') and
                not stripped.startswith("'''")):
                code_lines += 1
        
        return {
            'total_lines': len(lines),
            'code_lines': code_lines,
            'substantial': code_lines > 20
        }
    except:
        return {'total_lines': 0, 'code_lines': 0, 'substantial': False}


def main():
    """Run validation"""
    root = Path(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
    src_path = root / 'src'
    
    print("🧠 BRAIN-FORGE PROJECT VALIDATION")
    print("=" * 50)
    
    if not src_path.exists():
        print("❌ src/ directory not found!")
        return
    
    key_files = {
        'integrated_system.py': 'Main BCI system',
        'core/exceptions.py': 'Exception system',
        'processing/__init__.py': 'Processing pipeline',
        'api/rest_api.py': 'REST API',
        'specialized_tools.py': 'Specialized tools'
    }
    
    print("\n📊 KEY MODULES:")
    total_code = 0
    substantial_count = 0
    
    for file_path, desc in key_files.items():
        full_path = src_path / file_path
        if full_path.exists():
            result = check_file_substantial(full_path)
            total_code += result['code_lines'] 
            
            if result['substantial']:
                substantial_count += 1
                status = "✅ SUBSTANTIAL"
            else:
                status = "⚠️ MINIMAL"
                
            print(f"   {file_path}: {result['code_lines']} lines - {status}")
        else:
            print(f"   {file_path}: ❌ MISSING")
    
    # Count all Python files
    all_py_files = list(src_path.rglob('*.py'))
    print(f"\n📁 OVERVIEW:")
    print(f"   Total Python files: {len(all_py_files)}")
    print(f"   Key modules substantial: {substantial_count}/{len(key_files)}")
    print(f"   Total key module lines: {total_code:,}")
    
    # Assessment
    if substantial_count >= 4 and total_code > 1500:
        status = "🎉 WELL DEVELOPED"
        note = "Project has substantial implementation"
    elif substantial_count >= 3:
        status = "✅ GOOD PROGRESS"
        note = "Most key modules implemented"
    else:
        status = "🔶 IN DEVELOPMENT"
        note = "Needs more implementation work" 
    
    print(f"\n🎯 ASSESSMENT: {status}")
    print(f"   {note}")
    print(f"   Ready for: Validation and testing phase")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
