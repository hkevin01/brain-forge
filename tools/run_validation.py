#!/usr/bin/env python3
"""
Project Validation Runner

Runs comprehensive validation checks on the Brain-Forge project.
"""

import sys
from pathlib import Path


def run_validation():
    """Run project validation"""
    project_root = Path(__file__).parent.parent
    
    print("🔍 Brain-Forge Validation Runner")
    print("=" * 50)
    
    # Import and run validation tests
    sys.path.insert(0, str(project_root / "validation"))
    
    try:
        from test_project_completion import TestProjectCompletion
        
        test_class = TestProjectCompletion()
        test_class.setup_method()
        
        print("Running comprehensive project validation...")
        
        # Run all validation tests
        validation_methods = [
            'test_all_documented_features_exist',
            'test_all_examples_in_readme_work', 
            'test_all_project_goals_met',
            'test_code_documentation_alignment',
            'test_performance_benchmarks_met',
            'test_hardware_interfaces_functional',
            'test_api_endpoints_working',
            'test_deployment_readiness'
        ]
        
        passed = 0
        failed = 0
        
        for method_name in validation_methods:
            try:
                method = getattr(test_class, method_name)
                method()
                print(f"✅ {method_name}: PASSED")
                passed += 1
            except Exception as e:
                print(f"❌ {method_name}: FAILED - {str(e)}")
                failed += 1
        
        print(f"\n📊 Validation Summary:")
        print(f"   Passed: {passed}")
        print(f"   Failed: {failed}")
        print(f"   Success Rate: {passed/(passed+failed)*100:.1f}%")
        
        if failed == 0:
            print("\n🎉 All validations passed!")
        else:
            print(f"\n⚠️  {failed} validations failed - see details above")
            
    except ImportError as e:
        print(f"❌ Could not import validation tests: {e}")
        print("Make sure validation/test_project_completion.py exists")

if __name__ == "__main__":
    run_validation()
