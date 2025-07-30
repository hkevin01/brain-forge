#!/usr/bin/env python3
"""
Project File Cleanup Tool

Removes empty scattered files from the root directory and organizes the project structure.
This is part of Phase 1 of the comprehensive project restructuring.
"""

import os
import shutil
from pathlib import Path


def cleanup_empty_files():
    """Remove empty files from the root directory"""
    project_root = Path(__file__).parent.parent
    
    print("🧹 Brain-Forge Project File Cleanup")
    print("=" * 50)
    
    # List of empty files to remove
    empty_files_to_remove = [
        "validate_infrastructure.py",
        "validate_brain_forge.py", 
        "validate_implementation.py",
        "validate_tests.py",
        "validate_completion.py",
        "test_basic_functionality.py",
        "test_core_imports.py",
        "test_imports.py",
        "cleanup_scattered_files.py",
        "final_cleanup.py",
        "quick_test.py",
        "quick_validation.py",
        "run_tests.py",
        "run_validation.py",
        "brain_forge_complete.py",
        "cleanup_original_files.sh",
        "demonstrate_completion.sh",
        "install_and_validate.sh",
        "COMPLETION_REPORT.md",
        "FINAL_COMPLETION_STATUS.md",
        "FINAL_SUCCESS_REPORT.md",
        "IMPLEMENTATION_COMPLETE.md",
        "PROJECT_CLEANUP_SUMMARY.md",
        "PROJECT_REORGANIZATION_COMPLETE.md",
        "PROJECT_STATUS.md",
        ".delete_brain_forge_complete.py"
    ]
    
    # Move important files first (if they have content)
    important_files = [
        ("test_project_completion.py", "archive/test_project_completion_original.py"),
    ]
    
    for src_file, dest_file in important_files:
        src_path = project_root / src_file
        dest_path = project_root / dest_file
        
        if src_path.exists():
            try:
                # Check if file has content
                if src_path.stat().st_size > 0:
                    dest_path.parent.mkdir(exist_ok=True)
                    shutil.move(str(src_path), str(dest_path))
                    print(f"📦 Archived: {src_file} → {dest_file}")
                else:
                    src_path.unlink()
                    print(f"🗑️  Removed empty: {src_file}")
            except Exception as e:
                print(f"❌ Error handling {src_file}: {e}")
    
    # Remove empty files
    removed_count = 0
    for file_name in empty_files_to_remove:
        file_path = project_root / file_name
        if file_path.exists():
            try:
                # Double-check if file is actually empty
                if file_path.stat().st_size == 0:
                    file_path.unlink()
                    print(f"🗑️  Removed empty: {file_name}")
                    removed_count += 1
                else:
                    print(f"⚠️  Skipped non-empty: {file_name}")
            except Exception as e:
                print(f"❌ Error removing {file_name}: {e}")
        else:
            print(f"ℹ️  Not found: {file_name}")
    
    print(f"\n✅ Cleanup complete!")
    print(f"   Files removed: {removed_count}")
    print(f"   Files archived: {len([f for f in important_files if (project_root / f[0]).exists()])}")
    
    # Show remaining root directory contents
    print(f"\n📂 Remaining root directory contents:")
    for item in sorted(project_root.iterdir()):
        if item.is_file() and not item.name.startswith('.'):
            size = item.stat().st_size
            if size == 0:
                print(f"   📄 {item.name} (empty - consider removal)")
            else:
                print(f"   📄 {item.name} ({size:,} bytes)")
        elif item.is_dir() and not item.name.startswith('.'):
            print(f"   📁 {item.name}/")

def show_project_organization():
    """Show the new organized project structure"""
    project_root = Path(__file__).parent.parent
    
    print(f"\n🏗️  New Project Organization:")
    print("=" * 50)
    
    important_dirs = [
        ("src/", "Source code - core Brain-Forge implementation"),
        ("tests/", "Unit tests and integration tests"),
        ("validation/", "Project completion validation tests"),
        ("tools/", "Utility scripts and development tools"), 
        ("reports/", "Project reports and documentation"),
        ("archive/", "Historical files and legacy reports"),
        ("docs/", "Documentation and design specifications"),
        ("configs/", "Configuration files and templates"),
        ("examples/", "Usage examples and demonstrations"),
        ("scripts/", "Build and deployment scripts")
    ]
    
    for dir_name, description in important_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.rglob("*")))
            print(f"   📁 {dir_name:<12} - {description} ({file_count} files)")
        else:
            print(f"   📁 {dir_name:<12} - {description} (not present)")

if __name__ == "__main__":
    import sys
    sys.path.append('/home/kevin/Projects/brain-forge/tools')
    cleanup_empty_files()
    show_project_organization()
    
    print("\n🎯 Phase 1 of project restructuring complete!")
    print("   Next steps:")
    print("   • Phase 2: Code modernization") 
    print("   • Phase 3: Documentation enhancement")
    print("   • Phase 4: Configuration improvements")
    print("   • Phase 5: Testing/CI-CD integration")
    print("   • Phase 6: Final organization standards")
