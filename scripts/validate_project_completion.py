#!/usr/bin/env python3
"""
Brain-Forge Project Completion Validation Script

This script examines the project plan against the current source code
implementation to detect stub code and validate completion claims.

Author: Brain-Forge Development Team
Date: January 2025
"""

import os
import ast
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any


class ProjectValidator:
    """Comprehensive project validation and stub detection"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_path = self.project_root / "src"
        self.tests_path = self.project_root / "tests"
        self.docs_path = self.project_root / "docs"
        
        # Stub patterns to detect
        self.stub_patterns = [
            r'raise\s+NotImplementedError',
            r'#\s*TODO',
            r'#\s*FIXME',
            r'#\s*STUB',
            r'pass\s*$',
            r'pass\s*#.*stub',
            r'pass\s*#.*TODO',
            r'def\s+\w+\([^)]*\):\s*pass\s*$',
            r'def\s+\w+\([^)]*\):\s*#.*not implemented',
            r'def\s+\w+\([^)]*\):\s*""".*"""[\s\n]*pass\s*$',
        ]
        
        # Compiled regex patterns
        self.compiled_patterns = [re.compile(pattern, re.MULTILINE | re.IGNORECASE) 
                                 for pattern in self.stub_patterns]
        
        # Results storage
        self.results = {
            'stub_files': [],
            'stub_functions': [],
            'empty_files': [],
            'missing_modules': [],
            'implementation_status': {},
            'test_coverage': {},
            'file_stats': {},
            'overall_metrics': {}
        }
    
    def scan_file_for_stubs(self, file_path: Path) -> Dict[str, Any]:
        """Scan a single Python file for stubs and incomplete implementation"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return {'type': 'empty', 'issues': ['File is empty'], 'content': ''}
            
            issues = []
            stub_locations = []
            
            # Check for stub patterns
            for i, pattern in enumerate(self.compiled_patterns):
                matches = pattern.finditer(content)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append(f"Stub pattern {i+1} at line {line_num}: {match.group().strip()}")
                    stub_locations.append({
                        'pattern': self.stub_patterns[i],
                        'line': line_num,
                        'match': match.group().strip()
                    })
            
            # Parse AST to find function/class definitions
            try:
                tree = ast.parse(content, filename=str(file_path))
                functions = []
                classes = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_body = ast.get_source_segment(content, node) if hasattr(ast, 'get_source_segment') else None
                        functions.append({
                            'name': node.name,
                            'line': node.lineno,
                            'is_stub': self._is_stub_function(node, content)
                        })
                    elif isinstance(node, ast.ClassDef):
                        classes.append({
                            'name': node.name,
                            'line': node.lineno,
                            'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                        })
            except SyntaxError as e:
                issues.append(f"Syntax error: {e}")
                functions = []
                classes = []
            
            # Calculate metrics
            lines = content.split('\n')
            total_lines = len(lines)
            code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            comment_lines = len([line for line in lines if line.strip().startswith('#')])
            
            return {
                'type': 'analyzed',
                'issues': issues,
                'stub_locations': stub_locations,
                'functions': functions,
                'classes': classes,
                'metrics': {
                    'total_lines': total_lines,
                    'code_lines': code_lines,
                    'comment_lines': comment_lines,
                    'blank_lines': total_lines - code_lines - comment_lines
                }
            }
            
        except Exception as e:
            return {'type': 'error', 'issues': [f"Error reading file: {e}"], 'content': ''}
    
    def _is_stub_function(self, node: ast.FunctionDef, content: str) -> bool:
        """Check if a function definition is a stub"""
        # Check if function body contains only pass, docstring, or raise NotImplementedError
        if len(node.body) == 1:
            stmt = node.body[0]
            if isinstance(stmt, ast.Pass):
                return True
            elif isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
                if isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == 'NotImplementedError':
                    return True
        elif len(node.body) == 2:
            # Check for docstring + pass/raise
            if (isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
                stmt = node.body[1]
                if isinstance(stmt, ast.Pass):
                    return True
                elif isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
                    if isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == 'NotImplementedError':
                        return True
        return False
    
    def scan_directory(self, directory: Path) -> Dict[str, Any]:
        """Recursively scan directory for Python files and analyze them"""
        results = {}
        
        if not directory.exists():
            return {'error': f"Directory {directory} does not exist"}
        
        for py_file in directory.rglob('*.py'):
            relative_path = py_file.relative_to(self.project_root)
            results[str(relative_path)] = self.scan_file_for_stubs(py_file)
        
        return results
    
    def validate_core_modules(self) -> Dict[str, Any]:
        """Validate that core modules are properly implemented"""
        core_modules = {
            'core/config.py': 'Configuration management system',
            'core/exceptions.py': 'Custom exception classes',
            'core/logger.py': 'Logging system',
            'acquisition/stream_manager.py': 'Data acquisition',
            'processing/__init__.py': 'Data processing pipeline',
            'visualization/brain_visualization.py': '3D brain visualization',
            'api/rest_api.py': 'REST API interface',
            'integrated_system.py': 'Main BCI system',
        }
        
        validation_results = {}
        
        for module_path, description in core_modules.items():
            full_path = self.src_path / module_path
            if full_path.exists():
                analysis = self.scan_file_for_stubs(full_path)
                validation_results[module_path] = {
                    'exists': True,
                    'description': description,
                    'analysis': analysis,
                    'stub_count': len(analysis.get('stub_locations', [])),
                    'function_count': len(analysis.get('functions', [])),
                    'class_count': len(analysis.get('classes', [])),
                    'lines': analysis.get('metrics', {}).get('code_lines', 0)
                }
            else:
                validation_results[module_path] = {
                    'exists': False,
                    'description': description,
                    'analysis': {'type': 'missing'},
                    'stub_count': 0,
                    'function_count': 0,
                    'class_count': 0,
                    'lines': 0
                }
        
        return validation_results
    
    def analyze_test_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage and completeness"""
        test_results = {}
        
        if not self.tests_path.exists():
            return {'error': 'Tests directory not found'}
        
        # Scan test directories
        test_dirs = ['unit', 'integration', 'performance', 'assurance']
        
        for test_dir in test_dirs:
            test_path = self.tests_path / test_dir
            if test_path.exists():
                test_results[test_dir] = self.scan_directory(test_path)
            else:
                test_results[test_dir] = {'error': f'Test directory {test_dir} not found'}
        
        # Analyze conftest.py
        conftest_path = self.tests_path / 'conftest.py'
        if conftest_path.exists():
            test_results['conftest'] = self.scan_file_for_stubs(conftest_path)
        
        return test_results
    
    def compare_with_project_plan(self) -> Dict[str, Any]:
        """Compare current implementation with project plan claims"""
        plan_comparison = {
            'claimed_complete': [],
            'actually_complete': [],
            'claimed_incomplete': [],
            'actually_incomplete': [],
            'discrepancies': []
        }
        
        # Read project plan if it exists
        project_plan_path = self.project_root / 'project_plan.md'
        tasks_md_path = self.project_root / 'tasksync' / 'tasks.md'
        
        claimed_status = {}
        
        # Parse tasks.md for claimed completion status
        if tasks_md_path.exists():
            try:
                with open(tasks_md_path, 'r', encoding='utf-8') as f:
                    tasks_content = f.read()
                
                # Look for completion claims
                completion_patterns = [
                    r'✅\s*COMPLETED[^:]*:\s*([^\n]+)',
                    r'✅\s*IMPLEMENTED[^:]*:\s*([^\n]+)',
                    r'Status:\s*(\d+)%\+?\s*COMPLETE',
                    r'Brain-Forge\s+is\s+(\d+)%\+?\s*COMPLETE',
                ]
                
                for pattern in completion_patterns:
                    matches = re.finditer(pattern, tasks_content, re.IGNORECASE)
                    for match in matches:
                        plan_comparison['claimed_complete'].append(match.group(1) if match.groups() else match.group(0))
                
            except Exception as e:
                plan_comparison['discrepancies'].append(f"Error reading tasks.md: {e}")
        
        return plan_comparison
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        # Scan all source files
        src_analysis = self.scan_directory(self.src_path)
        
        # Validate core modules
        core_validation = self.validate_core_modules()
        
        # Analyze tests
        test_analysis = self.analyze_test_coverage()
        
        # Compare with project plan
        plan_comparison = self.compare_with_project_plan()
        
        # Calculate overall metrics
        total_files = len(src_analysis)
        files_with_stubs = sum(1 for analysis in src_analysis.values() 
                              if analysis.get('stub_locations', []))
        total_stub_locations = sum(len(analysis.get('stub_locations', [])) 
                                  for analysis in src_analysis.values())
        
        total_functions = sum(len(analysis.get('functions', [])) 
                             for analysis in src_analysis.values())
        stub_functions = sum(len([f for f in analysis.get('functions', []) if f.get('is_stub', False)]) 
                            for analysis in src_analysis.values())
        
        total_lines = sum(analysis.get('metrics', {}).get('total_lines', 0) 
                         for analysis in src_analysis.values())
        code_lines = sum(analysis.get('metrics', {}).get('code_lines', 0) 
                        for analysis in src_analysis.values())
        
        # Determine overall completion status
        completion_percentage = max(0, 100 - (total_stub_locations * 10) - (stub_functions * 5))
        completion_status = "PRODUCTION READY" if completion_percentage >= 95 else \
                          "NEAR COMPLETE" if completion_percentage >= 85 else \
                          "IN DEVELOPMENT" if completion_percentage >= 70 else \
                          "EARLY STAGE"
        
        summary = {
            'validation_timestamp': __import__('datetime').datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'overall_metrics': {
                'total_python_files': total_files,
                'total_code_lines': code_lines,
                'total_lines': total_lines,
                'files_with_stubs': files_with_stubs,
                'total_stub_locations': total_stub_locations,
                'total_functions': total_functions,
                'stub_functions': stub_functions,
                'completion_percentage': completion_percentage,
                'completion_status': completion_status
            },
            'source_analysis': src_analysis,
            'core_module_validation': core_validation,
            'test_analysis': test_analysis,
            'plan_comparison': plan_comparison,
            'recommendations': self._generate_recommendations(
                total_stub_locations, stub_functions, files_with_stubs, completion_percentage
            )
        }
        
        return summary
    
    def _generate_recommendations(self, stub_locations: int, stub_functions: int, 
                                 files_with_stubs: int, completion_percentage: float) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if stub_locations > 0:
            recommendations.append(f"Remove {stub_locations} stub location(s) found in source code")
        
        if stub_functions > 0:
            recommendations.append(f"Implement {stub_functions} stub function(s)")
        
        if files_with_stubs > 0:
            recommendations.append(f"Address stub code in {files_with_stubs} file(s)")
        
        if completion_percentage < 85:
            recommendations.append("Project requires significant additional development")
        elif completion_percentage < 95:
            recommendations.append("Project is near completion, focus on remaining stubs and testing")
        else:
            recommendations.append("Project appears production-ready, perform final validation")
        
        return recommendations
    
    def export_detailed_report(self, output_path: str) -> bool:
        """Export detailed validation report to JSON file"""
        try:
            summary = self.generate_summary_report()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error exporting report: {e}")
            return False
    
    def print_summary(self):
        """Print a human-readable summary to console"""
        summary = self.generate_summary_report()
        metrics = summary['overall_metrics']
        
        print("=" * 80)
        print("🧠 BRAIN-FORGE PROJECT VALIDATION REPORT")
        print("=" * 80)
        print()
        print(f"📊 OVERALL METRICS:")
        print(f"   Total Python Files: {metrics['total_python_files']}")
        print(f"   Total Code Lines: {metrics['total_code_lines']:,}")
        print(f"   Total Functions: {metrics['total_functions']}")
        print(f"   Completion Status: {metrics['completion_status']} ({metrics['completion_percentage']:.1f}%)")
        print()
        
        if metrics['total_stub_locations'] > 0:
            print(f"⚠️  STUB CODE DETECTED:")
            print(f"   Files with stubs: {metrics['files_with_stubs']}")
            print(f"   Total stub locations: {metrics['total_stub_locations']}")
            print(f"   Stub functions: {metrics['stub_functions']}")
            print()
        else:
            print("✅ NO STUB CODE DETECTED")
            print()
        
        print("📁 CORE MODULE STATUS:")
        for module, status in summary['core_module_validation'].items():
            if status['exists']:
                stub_status = f"({status['stub_count']} stubs)" if status['stub_count'] > 0 else "✅"
                print(f"   {module}: {status['lines']} lines, {status['function_count']} functions {stub_status}")
            else:
                print(f"   {module}: ❌ MISSING")
        print()
        
        if summary['recommendations']:
            print("📋 RECOMMENDATIONS:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"   {i}. {rec}")
            print()
        
        print("=" * 80)

def main():
    """Main execution function"""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = os.getcwd()
    
    # Initialize validator
    validator = ProjectValidator(project_root)
    
    # Print summary to console
    validator.print_summary()
    
    # Export detailed report
    report_path = os.path.join(project_root, 'validation_report.json')
    if validator.export_detailed_report(report_path):
        print(f"📄 Detailed report exported to: {report_path}")
    else:
        print("❌ Failed to export detailed report")

if __name__ == "__main__":
    main()
