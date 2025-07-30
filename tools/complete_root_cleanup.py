#!/usr/bin/env python3
"""
Complete Brain-Forge Root Directory Cleanup
Move remaining files to proper locations and remove empty files
"""

import os
import shutil
from pathlib import Path


def main():
    """Complete the root directory cleanup"""
    project_root = Path("/home/kevin/Projects/brain-forge")
    
    print("=== Completing Brain-Forge Root Directory Cleanup ===\n")
    
    # Files to move to docs/reports/
    report_files = [
        "COMPLETION_REPORT.md",
        "FINAL_COMPLETION_STATUS.md", 
        "FINAL_SUCCESS_REPORT.md",
        "IMPLEMENTATION_COMPLETE.md",
        "PROJECT_CLEANUP_SUMMARY.md",
        "PROJECT_COMPLETION_REPORT.md", 
        "PROJECT_PROGRESS_TRACKER.md",
        "PROJECT_REORGANIZATION_COMPLETE.md",
        "PROJECT_STATUS.md",
    ]
    
    # Files to move to docs/project/
    project_files = [
        "CLEANUP_INSTRUCTIONS.md",
        "USAGE.md",
    ]
    
    # Files to move to tests/
    test_files = [
        "test_basic_functionality.py",
        "test_core_imports.py", 
        "test_imports.py",
        "test_project_completion.py",
        "quick_test.py",
    ]
    
    # Files to move to tools/validation/
    validation_files = [
        "validate_brain_forge.py",
        "validate_completion.py",
        "validate_implementation.py", 
        "validate_infrastructure.py",
        "run_validation.py",
    ]
    
    # Files to move to tools/development/
    dev_files = [
        "brain_forge_complete.py",
        "run_tests.py",
    ]
    
    # Files to move to scripts/setup/
    setup_files = [
        "install_and_validate.sh",
        "restructure_project.sh",
    ]
    
    # Files to move to scripts/cleanup/
    cleanup_files = [
        "cleanup_scattered_files.py",
        "final_cleanup.py",
        "cleanup_original_files.sh",
    ]
    
    # Files to move to scripts/demo/
    demo_files = [
        "quick_validation.py",
        "demonstrate_completion.sh",
    ]
    
    # Files to move to config/linting/
    config_files = [
        ".mypy.ini",
        ".pre-commit-config.yaml",
        ".pylintrc",
        "pytest.ini",
        "pyproject.toml",
    ]
    
    # Files to archive
    archive_files = [
        ".delete_brain_forge_complete.py",
    ]
    
    # Move files by category
    moves = [
        ("Reports", report_files, "docs/reports/"),
        ("Project Docs", project_files, "docs/project/"),
        ("Tests", test_files, "tests/"),
        ("Validation", validation_files, "tools/validation/"),
        ("Development", dev_files, "tools/development/"),
        ("Setup Scripts", setup_files, "scripts/setup/"),
        ("Cleanup Scripts", cleanup_files, "scripts/cleanup/"),
        ("Demo Scripts", demo_files, "scripts/demo/"),
        ("Config Files", config_files, "config/linting/"),
        ("Archive", archive_files, "archive/scripts/"),
    ]
    
    moved_count = 0
    
    for category, files, target_dir in moves:
        print(f"{category}:")
        target_path = project_root / target_dir
        target_path.mkdir(parents=True, exist_ok=True)
        
        for filename in files:
            source_path = project_root / filename
            dest_path = target_path / filename
            
            if source_path.exists():
                if source_path.stat().st_size > 0:
                    shutil.move(str(source_path), str(dest_path))
                    print(f"  ✓ Moved: {filename}")
                    moved_count += 1
                else:
                    source_path.unlink()
                    print(f"  ✓ Removed empty: {filename}")
            else:
                print(f"  - Not found: {filename}")
        print()
    
    # Remove .flake8 from root (already moved)
    root_flake8 = project_root / ".flake8"
    if root_flake8.exists():
        root_flake8.unlink()
        print("✓ Removed duplicate .flake8 from root")
    
    print(f"=== Cleanup Complete! ===")
    print(f"Files moved: {moved_count}")
    print("Root directory is now clean and organized!")
    
    # Show final clean root structure
    print(f"\nFinal root directory contents:")
    root_items = sorted([item.name for item in project_root.iterdir() 
                        if not item.name.startswith('.') or item.name in ['.env.example', '.gitignore']])
    for item in root_items:
        print(f"  {item}")

if __name__ == "__main__":
    main()
