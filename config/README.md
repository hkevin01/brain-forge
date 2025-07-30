# Configuration Directory

This directory contains all configuration files organized by category.

## Structure:
- `linting/` - Code quality and linting configurations

## Linting Configuration (`linting/`):
- `.flake8` - Python code style configuration
- `.mypy.ini` - Type checking configuration  
- `.pylintrc` - Python linting rules
- `.pre-commit-config.yaml` - Pre-commit hooks
- `pytest.ini` - Testing configuration
- `pyproject.toml` - Project and tool configuration

## Usage:
These configuration files are automatically used by development tools:
- Flake8 uses `.flake8` for code style checking
- MyPy uses `.mypy.ini` for type checking
- Pre-commit uses `.pre-commit-config.yaml` for git hooks
- Pytest uses `pytest.ini` for test configuration

The tools will automatically discover these configuration files in the project.
