#!/usr/bin/env python3
"""
Brain-Forge Root Directory Cleanup Script
Organize scattered files into proper subdirectories
"""

import os
import shutil
from pathlib import Path


class BrainForgeCleanup:
    """Organize Brain-Forge project files into proper directory structure"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.moves_made = []
        self.directories_created = []
        
    def create_directory_structure(self):
        """Create necessary subdirectories if they don't exist"""
        directories = {
            'docs/reports': 'Project reports and completion documentation',
            'docs/project': 'Project management and planning documents',
            'scripts/setup': 'Setup and installation scripts',
            'scripts/cleanup': 'Cleanup and maintenance scripts',
            'scripts/demo': 'Demonstration and validation scripts',
            'tools/validation': 'Validation and testing utilities',
            'tools/development': 'Development helper scripts',
            'archive/scripts': 'Archived script files',
            'archive/reports': 'Archived report files',
            'config/linting': 'Code quality and linting configurations',
            'config/docker': 'Docker and containerization configs',
        }
        
        for dir_path, description in directories.items():
            full_path = self.project_root / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                self.directories_created.append(str(dir_path))
                print(f"✓ Created directory: {dir_path} ({description})")
                
    def organize_files(self):
        """Move files to appropriate subdirectories"""
        file_moves = {
            # Reports and Documentation
            'COMPLETION_REPORT.md': 'docs/reports/',
            'FINAL_COMPLETION_STATUS.md': 'docs/reports/',
            'FINAL_COMPLETION_SUMMARY.md': 'docs/reports/',
            'FINAL_SUCCESS_REPORT.md': 'docs/reports/',
            'IMPLEMENTATION_COMPLETE.md': 'docs/reports/',
            'PROJECT_CLEANUP_SUMMARY.md': 'docs/reports/',
            'PROJECT_COMPLETION_REPORT.md': 'docs/reports/',
            'PROJECT_PROGRESS_TRACKER.md': 'docs/reports/',
            'PROJECT_REORGANIZATION_COMPLETE.md': 'docs/reports/',
            'PROJECT_STATUS.md': 'docs/reports/',
            
            # Project Documentation
            'CLEANUP_INSTRUCTIONS.md': 'docs/project/',
            'USAGE.md': 'docs/project/',
            
            # Setup Scripts
            'install_and_validate.sh': 'scripts/setup/',
            'restructure_project.sh': 'scripts/setup/',
            
            # Cleanup Scripts
            'cleanup_original_files.sh': 'scripts/cleanup/',
            'cleanup_scattered_files.py': 'scripts/cleanup/',
            'final_cleanup.py': 'scripts/cleanup/',
            
            # Demo and Validation Scripts
            'demonstrate_completion.sh': 'scripts/demo/',
            'quick_test.py': 'scripts/demo/',
            'quick_validation.py': 'scripts/demo/',
            
            # Validation Scripts
            'validate_brain_forge.py': 'tools/validation/',
            'validate_completion.py': 'tools/validation/',
            'validate_implementation.py': 'tools/validation/',
            'validate_infrastructure.py': 'tools/validation/',
            'run_validation.py': 'tools/validation/',
            
            # Development Scripts  
            'run_tests.py': 'tools/development/',
            'brain_forge_complete.py': 'tools/development/',
            
            # Test Files
            'test_basic_functionality.py': 'tests/',
            'test_core_imports.py': 'tests/',
            'test_imports.py': 'tests/',
            'test_project_completion.py': 'tests/',
            
            # Configuration Files
            '.flake8': 'config/linting/',
            '.mypy.ini': 'config/linting/',
            '.pre-commit-config.yaml': 'config/linting/',
            '.pylintrc': 'config/linting/',
            'pytest.ini': 'config/linting/',
            'pyproject.toml': 'config/linting/',
            
            # Cleanup Files (to archive)
            '.delete_brain_forge_complete.py': 'archive/scripts/',
        }
        
        for filename, target_dir in file_moves.items():
            source_path = self.project_root / filename
            target_path = self.project_root / target_dir / filename
            
            if source_path.exists():
                # Create target directory if it doesn't exist
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Move the file
                shutil.move(str(source_path), str(target_path))
                self.moves_made.append(f"{filename} → {target_dir}")
                print(f"✓ Moved: {filename} → {target_dir}")
            else:
                print(f"⚠ File not found: {filename}")
                
    def create_readme_files(self):
        """Create README files for new directories"""
        readme_content = {
            'docs/reports/README.md': """# Project Reports

This directory contains all project completion reports and status documentation.

## Files:
- Completion reports and final status documents
- Project progress tracking
- Implementation summaries
""",
            'docs/project/README.md': """# Project Documentation

This directory contains project management and planning documentation.

## Files:
- Cleanup instructions
- Usage guidelines
- Project planning documents
""",
            'scripts/README.md': """# Scripts Directory

This directory contains all project scripts organized by category.

## Subdirectories:
- `setup/` - Installation and setup scripts
- `cleanup/` - Cleanup and maintenance scripts  
- `demo/` - Demonstration and validation scripts
""",
            'tools/validation/README.md': """# Validation Tools

This directory contains validation and testing utilities.

## Files:
- Project validation scripts
- Infrastructure testing tools
- Completion verification utilities
""",
            'config/linting/README.md': """# Linting Configuration

This directory contains code quality and linting configuration files.

## Files:
- Python linting configurations (.flake8, .pylintrc, .mypy.ini)
- Pre-commit hooks configuration
- Testing configuration (pytest.ini)
- Project configuration (pyproject.toml)
""",
        }
        
        for readme_path, content in readme_content.items():
            full_path = self.project_root / readme_path
            if not full_path.exists():
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(content)
                print(f"✓ Created: {readme_path}")
                
    def cleanup_empty_directories(self):
        """Remove any empty directories that may have been left behind"""
        for root, dirs, files in os.walk(self.project_root, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    try:
                        dir_path.rmdir()
                        print(f"✓ Removed empty directory: {dir_path.relative_to(self.project_root)}")
                    except OSError:
                        pass  # Directory not empty or permission error
                        
    def generate_cleanup_summary(self):
        """Generate a summary of cleanup actions"""
        summary_path = self.project_root / 'docs' / 'project' / 'CLEANUP_SUMMARY.md'
        
        content = f"""# Root Directory Cleanup Summary

## Directories Created
{chr(10).join(f'- {d}' for d in self.directories_created)}

## Files Moved
{chr(10).join(f'- {m}' for m in self.moves_made)}

## Final Root Directory Structure
After cleanup, the root directory should contain only:
- Essential project files (README.md, LICENSE, CHANGELOG.md, etc.)
- Core configuration files (.env.example, docker-compose.yml, Dockerfile)
- Primary directories (src/, docs/, tests/, tools/, etc.)
- Version control files (.git/, .gitignore)

## Organized Structure
```
brain-forge/
├── src/                    # Source code
├── docs/                   # Documentation
│   ├── api/               # API documentation
│   ├── project/           # Project documentation
│   └── reports/           # Project reports
├── tests/                  # Test files
├── tools/                  # Development tools
│   ├── validation/        # Validation scripts
│   └── development/       # Development utilities
├── scripts/                # Project scripts
│   ├── setup/             # Setup scripts
│   ├── cleanup/           # Cleanup scripts
│   └── demo/              # Demo scripts
├── config/                 # Configuration files
│   ├── linting/           # Code quality configs
│   └── docker/            # Docker configurations
├── archive/                # Archived files
├── validation/             # Validation utilities
├── requirements/           # Dependencies
└── [core files]           # README, LICENSE, etc.
```

Date: {Path(__file__).stat().st_mtime}
"""
        
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            f.write(content)
        print(f"✓ Generated cleanup summary: {summary_path.relative_to(self.project_root)}")
        
    def run_cleanup(self):
        """Execute the complete cleanup process"""
        print("=== Brain-Forge Root Directory Cleanup ===\n")
        
        print("1. Creating directory structure...")
        self.create_directory_structure()
        print()
        
        print("2. Moving files to appropriate locations...")
        self.organize_files()
        print()
        
        print("3. Creating README files...")
        self.create_readme_files()
        print()
        
        print("4. Cleaning up empty directories...")
        self.cleanup_empty_directories()
        print()
        
        print("5. Generating cleanup summary...")
        self.generate_cleanup_summary()
        print()
        
        print("=== Cleanup Complete! ===")
        print(f"Directories created: {len(self.directories_created)}")
        print(f"Files moved: {len(self.moves_made)}")
        print("\nRoot directory is now clean and organized!")

def main():
    """Main execution function"""
    cleanup = BrainForgeCleanup()
    cleanup.run_cleanup()

if __name__ == "__main__":
    main()