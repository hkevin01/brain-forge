#!/usr/bin/env python3
"""
Brain-Forge Project Completion Validation Script

This script examines the project plan against the current source code
implementation to detect stub code and validate completion claims.
"""

import os
import re
import sys
from pathlib import Path


def scan_for_stubs(file_path):
    """Scan a Python file for stub patterns"""
    stub_patterns = [
        r'raise\s+NotImplementedError',
        r'#\s*TODO',
        r'#\s*FIXME',
        r'pass\s*$',
        r'def\s+\w+\([^)]*\):\s*pass\s*$'
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        stubs = []
        for i, pattern in enumerate(stub_patterns):
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                stubs.append({
                    'pattern': pattern,
                    'line': line_num,
                    'match': match.group().strip()
                })
        
        lines = content.split('\n')
        metrics = {
            'total_lines': len(lines),
            'code_lines': len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
            'comment_lines': len([l for l in lines if l.strip().startswith('#')]),
            'stubs': stubs
        }
        
        return metrics
        
    except Exception as e:
        return {'error': str(e)}


def validate_core_modules(src_path):
    """Check core module implementation status"""
    core_modules = {
        'integrated_system.py': 'Main BCI system',
        'core/config.py': 'Configuration management',
        'core/exceptions.py': 'Exception handling',
        'core/logger.py': 'Logging system',
        'processing/__init__.py': 'Processing pipeline',
        'visualization/brain_visualization.py': '3D visualization',
        'api/rest_api.py': 'REST API interface'
    }
    
    results = {}
    for module, description in core_modules.items():
        module_path = src_path / module
        if module_path.exists():
            analysis = scan_for_stubs(module_path)
            results[module] = {
                'exists': True,
                'description': description,
                'lines': analysis.get('code_lines', 0),
                'stubs': len(analysis.get('stubs', [])),
                'analysis': analysis
            }
        else:
            results[module] = {
                'exists': False,
                'description': description,
                'lines': 0,
                'stubs': 0
            }
    
    return results


def scan_directory(directory):
    """Scan directory for Python files"""
    results = {}
    if not directory.exists():
        return results
    
    for py_file in directory.rglob('*.py'):
        relative_path = str(py_file.relative_to(directory.parent.parent))
        results[relative_path] = scan_for_stubs(py_file)
    
    return results


def main():
    """Main validation function"""
    project_root = Path(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
    src_path = project_root / 'src'
    
    print("=" * 80)
    print("🧠 BRAIN-FORGE PROJECT VALIDATION REPORT")
    print("=" * 80)
    print()
    
    # Validate source directory exists
    if not src_path.exists():
        print("❌ ERROR: src/ directory not found!")
        return
    
    # Scan all source files
    print("📁 Scanning source files...")
    src_files = scan_directory(src_path)
    
    # Count overall metrics
    total_files = len(src_files)
    total_lines = sum(f.get('code_lines', 0) for f in src_files.values())
    total_stubs = sum(len(f.get('stubs', [])) for f in src_files.values())
    files_with_stubs = sum(1 for f in src_files.values() if f.get('stubs', []))
    
    print(f"   Found {total_files} Python files")
    print(f"   Total code lines: {total_lines:,}")
    print(f"   Files with stubs: {files_with_stubs}")
    print(f"   Total stub locations: {total_stubs}")
    print()
    
    # Validate core modules
    print("📊 CORE MODULE STATUS:")
    core_results = validate_core_modules(src_path)
    
    for module, status in core_results.items():
        if status['exists']:
            stub_info = f"({status['stubs']} stubs)" if status['stubs'] > 0 else "✅"
            print(f"   {module}: {status['lines']} lines {stub_info}")
        else:
            print(f"   {module}: ❌ MISSING")
    print()
    
    # Show stub details if found
    if total_stubs > 0:
        print("⚠️  STUB CODE LOCATIONS:")
        for file_path, analysis in src_files.items():
            stubs = analysis.get('stubs', [])
            if stubs:
                print(f"   📄 {file_path}:")
                for stub in stubs[:3]:  # Show first 3 stubs per file
                    print(f"      Line {stub['line']}: {stub['match']}")
                if len(stubs) > 3:
                    print(f"      ... and {len(stubs) - 3} more")
        print()
    
    # Calculate completion percentage
    completion = max(0, 100 - (total_stubs * 10) - (files_with_stubs * 2))
    status = ("PRODUCTION READY" if completion >= 95 else
              "NEAR COMPLETE" if completion >= 85 else
              "IN DEVELOPMENT" if completion >= 70 else
              "EARLY STAGE")
    
    print("🎯 OVERALL ASSESSMENT:")
    print(f"   Completion: {completion:.1f}% - {status}")
    
    if total_stubs == 0:
        print("   ✅ NO STUB CODE DETECTED - Ready for deployment!")
    else:
        print(f"   ⚠️  {total_stubs} stub locations need implementation")
    
    print()
    print("=" * 80)
    
    # Export summary
    summary = {
        'total_files': total_files,
        'total_lines': total_lines,
        'total_stubs': total_stubs,
        'files_with_stubs': files_with_stubs,
        'completion_percentage': completion,
        'status': status,
        'core_modules': core_results,
        'detailed_analysis': src_files
    }
    
    import json
    with open(project_root / 'validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📄 Summary exported to: validation_summary.json")


if __name__ == "__main__":
    main()
