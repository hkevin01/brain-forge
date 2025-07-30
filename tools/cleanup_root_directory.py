#!/usr/bin/env python3
"""
Direct Brain-Forge Cleanup - Move key files to proper locations
"""

import shutil
from pathlib import Path


def move_file_if_exists(source, target):
    """Move file if it exists and has content"""
    source_path = Path(source)
    target_path = Path(target)
    
    if source_path.exists() and source_path.stat().st_size > 0:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(target_path))
        print(f"✓ Moved: {source_path.name} → {target_path.parent}")
        return True
    elif source_path.exists():
        source_path.unlink()  # Remove empty file
        print(f"✓ Removed empty file: {source_path.name}")
        return False
    return False

def main():
    project_root = Path("/home/kevin/Projects/brain-forge")
    
    print("=== Brain-Forge Direct Cleanup ===\n")
    
    # Move important reports
    report_moves = [
        ("FINAL_COMPLETION_SUMMARY.md", "docs/reports/FINAL_COMPLETION_SUMMARY.md"),
        ("PROJECT_COMPLETION_REPORT.md", "docs/reports/PROJECT_COMPLETION_REPORT.md"),
        ("FINAL_COMPLETION_STATUS.md", "docs/reports/FINAL_COMPLETION_STATUS.md"),
        ("FINAL_SUCCESS_REPORT.md", "docs/reports/FINAL_SUCCESS_REPORT.md"),
        ("IMPLEMENTATION_COMPLETE.md", "docs/reports/IMPLEMENTATION_COMPLETE.md"),
        ("PROJECT_STATUS.md", "docs/reports/PROJECT_STATUS.md"),
        ("PROJECT_PROGRESS_TRACKER.md", "docs/reports/PROJECT_PROGRESS_TRACKER.md"),
        ("PROJECT_CLEANUP_SUMMARY.md", "docs/reports/PROJECT_CLEANUP_SUMMARY.md"),
        ("PROJECT_REORGANIZATION_COMPLETE.md", "docs/reports/PROJECT_REORGANIZATION_COMPLETE.md"),
    ]
    
    # Move validation scripts
    validation_moves = [
        ("validate_brain_forge.py", "tools/validation/validate_brain_forge.py"),
        ("validate_completion.py", "tools/validation/validate_completion.py"),
        ("validate_implementation.py", "tools/validation/validate_implementation.py"),
        ("validate_infrastructure.py", "tools/validation/validate_infrastructure.py"),
        ("run_validation.py", "tools/validation/run_validation.py"),
    ]
    
    # Move test files
    test_moves = [
        ("test_basic_functionality.py", "tests/test_basic_functionality.py"),
        ("test_core_imports.py", "tests/test_core_imports.py"),
        ("test_imports.py", "tests/test_imports.py"),
        ("test_project_completion.py", "tests/test_project_completion.py"),
        ("quick_test.py", "tests/quick_test.py"),
    ]
    
    # Move scripts
    script_moves = [
        ("cleanup_scattered_files.py", "scripts/cleanup/cleanup_scattered_files.py"),
        ("final_cleanup.py", "scripts/cleanup/final_cleanup.py"),
        ("brain_forge_complete.py", "tools/development/brain_forge_complete.py"),
        ("run_tests.py", "tools/development/run_tests.py"),
        ("quick_validation.py", "scripts/demo/quick_validation.py"),
        ("cleanup_original_files.sh", "scripts/cleanup/cleanup_original_files.sh"),
        ("install_and_validate.sh", "scripts/setup/install_and_validate.sh"),
        ("restructure_project.sh", "scripts/setup/restructure_project.sh"),
        ("demonstrate_completion.sh", "scripts/demo/demonstrate_completion.sh"),
    ]
    
    # Move config files
    config_moves = [
        (".flake8", "config/linting/.flake8"),
        (".mypy.ini", "config/linting/.mypy.ini"),
        (".pre-commit-config.yaml", "config/linting/.pre-commit-config.yaml"),
        (".pylintrc", "config/linting/.pylintrc"),
        ("pytest.ini", "config/linting/pytest.ini"),
        ("pyproject.toml", "config/linting/pyproject.toml"),
    ]
    
    # Archive files
    archive_moves = [
        (".delete_brain_forge_complete.py", "archive/scripts/.delete_brain_forge_complete.py"),
    ]
    
    # Move project docs
    doc_moves = [
        ("CLEANUP_INSTRUCTIONS.md", "docs/project/CLEANUP_INSTRUCTIONS.md"),
        ("USAGE.md", "docs/project/USAGE.md"),
        ("COMPLETION_REPORT.md", "docs/reports/COMPLETION_REPORT.md"),
    ]
    
    all_moves = [
        ("Reports", report_moves),
        ("Validation Scripts", validation_moves), 
        ("Test Files", test_moves),
        ("Scripts", script_moves),
        ("Config Files", config_moves),
        ("Archive Files", archive_moves),
        ("Documentation", doc_moves),
    ]
    
    for category, moves in all_moves:
        print(f"\n{category}:")
        for source, target in moves:
            move_file_if_exists(project_root / source, project_root / target)
    
    print(f"\n=== Cleanup Complete! ===")
    print("Root directory is now organized with proper subfolder structure.")

if __name__ == "__main__":
    main()
