#!/usr/bin/env python3
"""
Brain-Forge Root Directory Cleanup - Direct Execution
Simple script to immediately clean up the root directory
"""

import shutil
from pathlib import Path


def move_file_safely(source_path, target_path):
    """Move file safely, creating directories as needed"""
    if source_path.exists() and source_path.stat().st_size > 0:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(target_path))
        print(f"✓ Moved: {source_path.name} → {target_path.parent.name}/")
        return True
    elif source_path.exists():
        source_path.unlink()  # Remove empty file
        print(f"✓ Removed empty: {source_path.name}")
    return False

def main():
    """Execute root directory cleanup"""
    project_root = Path("/home/kevin/Projects/brain-forge")
    
    print("=== Brain-Forge Root Directory Cleanup ===\n")
    
    # Create necessary directories
    directories = [
        "docs/reports",
        "docs/project", 
        "scripts/setup",
        "scripts/cleanup",
        "scripts/demo",
        "tools/validation",
        "tools/development",
        "config/linting",
        "archive/scripts"
    ]
    
    print("Creating directory structure...")
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Ensured: {dir_path}/")
    
    print("\nMoving files to appropriate locations...")
    
    # Define file moves
    file_moves = [
        # Reports
        ("COMPLETION_REPORT.md", "docs/reports/"),
        ("FINAL_COMPLETION_STATUS.md", "docs/reports/"),
        ("FINAL_COMPLETION_SUMMARY.md", "docs/reports/"),
        ("FINAL_SUCCESS_REPORT.md", "docs/reports/"),
        ("IMPLEMENTATION_COMPLETE.md", "docs/reports/"),
        ("PROJECT_CLEANUP_SUMMARY.md", "docs/reports/"),
        ("PROJECT_COMPLETION_REPORT.md", "docs/reports/"),
        ("PROJECT_PROGRESS_TRACKER.md", "docs/reports/"),
        ("PROJECT_REORGANIZATION_COMPLETE.md", "docs/reports/"),
        ("PROJECT_STATUS.md", "docs/reports/"),
        
        # Project docs
        ("CLEANUP_INSTRUCTIONS.md", "docs/project/"),
        ("USAGE.md", "docs/project/"),
        
        # Scripts
        ("install_and_validate.sh", "scripts/setup/"),
        ("restructure_project.sh", "scripts/setup/"),
        ("cleanup_original_files.sh", "scripts/cleanup/"),
        ("cleanup_scattered_files.py", "scripts/cleanup/"),
        ("final_cleanup.py", "scripts/cleanup/"),
        ("demonstrate_completion.sh", "scripts/demo/"),
        ("quick_test.py", "scripts/demo/"),
        ("quick_validation.py", "scripts/demo/"),
        
        # Validation tools
        ("validate_brain_forge.py", "tools/validation/"),
        ("validate_completion.py", "tools/validation/"),
        ("validate_implementation.py", "tools/validation/"),
        ("validate_infrastructure.py", "tools/validation/"),
        ("run_validation.py", "tools/validation/"),
        
        # Development tools
        ("run_tests.py", "tools/development/"),
        ("brain_forge_complete.py", "tools/development/"),
        
        # Test files
        ("test_basic_functionality.py", "tests/"),
        ("test_core_imports.py", "tests/"),
        ("test_imports.py", "tests/"),
        ("test_project_completion.py", "tests/"),
        
        # Config files
        (".flake8", "config/linting/"),
        (".mypy.ini", "config/linting/"),
        (".pre-commit-config.yaml", "config/linting/"),
        (".pylintrc", "config/linting/"),
        ("pytest.ini", "config/linting/"),
        ("pyproject.toml", "config/linting/"),
        
        # Archive
        (".delete_brain_forge_complete.py", "archive/scripts/"),
    ]
    
    moved_count = 0
    for filename, target_dir in file_moves:
        source_path = project_root / filename
        target_path = project_root / target_dir / filename
        
        if move_file_safely(source_path, target_path):
            moved_count += 1
    
    print(f"\n=== Cleanup Complete! ===")
    print(f"Files moved: {moved_count}")
    print("Root directory is now clean and organized!")
    
    # Show current root contents
    print(f"\nCurrent root directory contents:")
    essential_files = [
        "README.md", "LICENSE", "CHANGELOG.md", "CONTRIBUTING.md", 
        "CODE_OF_CONDUCT.md", ".env.example", "docker-compose.yml", 
        "Dockerfile", "requirements.txt", ".gitignore"
    ]
    
    essential_dirs = [
        "src", "docs", "tests", "tools", "scripts", "config", 
        "examples", "validation", "requirements", "archive", 
        "data", "assets", ".git", ".github", ".vscode"
    ]
    
    print("Essential files:")
    for file_name in essential_files:
        if (project_root / file_name).exists():
            print(f"  ✓ {file_name}")
        else:
            print(f"  - {file_name} (missing)")
    
    print("Essential directories:")
    for dir_name in essential_dirs:
        if (project_root / dir_name).exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  - {dir_name}/ (missing)")

if __name__ == "__main__":
    main()
