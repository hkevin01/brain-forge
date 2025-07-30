#!/usr/bin/env python3
"""
Run Brain-Forge Root Directory Cleanup
Execute the comprehensive cleanup script
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the root directory cleanup"""
    project_root = Path(__file__).parent
    cleanup_script = project_root / "validate_tests.py"
    
    print("🧹 Starting Brain-Forge Root Directory Cleanup...")
    print("=" * 50)
    
    if not cleanup_script.exists():
        print("❌ Cleanup script not found: validate_tests.py")
        return 1
    
    try:
        # Run the comprehensive cleanup script
        result = subprocess.run([sys.executable, str(cleanup_script)], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("✅ Root directory cleanup completed successfully!")
            print("\n🎯 Next steps:")
            print("1. Review the organized directory structure")
            print("2. Verify all files are in their correct locations")
            print("3. Update any hardcoded file paths in scripts")
            print("4. Run tests to ensure nothing is broken")
            return 0
        else:
            print(f"\n❌ Cleanup failed with return code: {result.returncode}")
            return result.returncode
            
    except Exception as e:
        print(f"❌ Error running cleanup: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
