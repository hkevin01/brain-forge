# Brain-Forge Linting Configuration

This document explains the linting configuration for the Brain-Forge project, designed to provide comprehensive code quality checking while suppressing common false positives that occur in scientific computing environments.

## Overview

The Brain-Forge project uses multiple linting tools configured to work together:

- **Black**: Code formatting (88 character line length)
- **isort**: Import organization 
- **Flake8**: Style guide enforcement with scientific computing adaptations
- **Pylint**: Comprehensive code analysis with complexity allowances
- **MyPy**: Type checking with flexible scientific library handling
- **Pre-commit**: Automated checking on commits

## Configuration Files

### `.flake8`
Primary style checking with common false positive suppression:
- **Line length**: Set to 88 characters (matches Black)
- **Ignored errors**: E203, E501, W503, W504, E402, F401, E722, C901, E741
- **Per-file ignores**: More lenient rules for tests, scripts, and scientific modules
- **Complexity limit**: Increased to 15 for scientific algorithms

### `.pylintrc` 
Comprehensive code analysis with scientific computing considerations:
- **Disabled checks**: Overly strict documentation, naming, and complexity rules
- **Increased limits**: More arguments, locals, branches, and statements allowed
- **Scientific naming**: Accepts common mathematical variable names (x, y, z, t, etc.)
- **Import handling**: Lenient treatment of dynamic imports and scientific libraries

### `.mypy.ini`
Type checking with flexibility for scientific libraries:
- **Import handling**: Ignores missing imports for scientific packages
- **Flexibility**: Disabled strict typing requirements that conflict with NumPy/SciPy usage
- **Per-module ignores**: Specific handling for scientific computing libraries

### `pyproject.toml`
Central configuration hub with tool-specific sections:
- **Black**: 88 character line length, Python 3.8+ target
- **isort**: Black-compatible profile
- **Coverage**: Comprehensive reporting with exclusions
- **Pytest**: Test discovery and coverage integration

### `.pre-commit-config.yaml`
Automated quality checking:
- **Black formatting**: Automatic code formatting
- **isort**: Import organization
- **Basic hooks**: Trailing whitespace, YAML/JSON validation
- **Flake8**: Style checking with our custom configuration
- **MyPy**: Type checking (limited to core modules)

### `.vscode/settings.json`
VS Code integration:
- **Linting tools**: Configured paths and arguments
- **Formatting**: Black integration with format-on-save
- **Problem reporting**: Suppressed common false positives
- **Testing**: Pytest integration with coverage

## Suppressed Common Issues

### Line Length (E501)
- **Why suppressed**: Scientific computing often requires longer expressions
- **Alternative**: Black handles line length at 88 characters
- **Per-file**: More lenient in scientific modules

### Import Issues (E402, F401, F403, F405)
- **E402**: Module level import not at top - sometimes needed for sys.path manipulation
- **F401**: Imported but unused - common in `__init__.py` files and scientific imports
- **F403/F405**: Star imports - sometimes needed for scientific libraries

### Exception Handling (E722)
- **Why suppressed**: Broad exception handling sometimes necessary for robust scientific code
- **Note**: Still enforces proper exception handling practices

### Complexity (C901)
- **Why suppressed**: Scientific algorithms can be inherently complex
- **Limit**: Increased to 15 (from default 10)
- **Alternative**: Focus on readability and testing

### Variable Names (E741)
- **Why suppressed**: Mathematical conventions use single letters (x, y, z, I, O, l)
- **Context**: Common in scientific computing and mathematical expressions

### Docstring Requirements
- **Pylint**: Disabled overly strict docstring requirements
- **Reason**: Too noisy during development, enforced at review stage

## Per-File Rules

### Test Files (`test_*.py`, `tests/*.py`)
- **More lenient**: Allows unused imports, longer lines, complexity
- **Reason**: Test files often need different patterns

### Scripts (`scripts/*.py`)
- **Import flexibility**: Allows top-level imports after sys.path manipulation
- **Line length**: More flexible for demo/utility scripts

### Scientific Modules (`src/specialized_tools.py`, processing, simulation)
- **Complexity**: Higher limits for scientific algorithms
- **Line length**: More flexible for mathematical expressions
- **Import handling**: Lenient for optional scientific dependencies

### Configuration Files
- **Line length**: More flexible for long configuration values
- **Import order**: Lenient for grouped configurations

## Tool Integration

### VS Code
- **Real-time linting**: Flake8 and MyPy integration
- **Format on save**: Black formatting automatically applied
- **Import organization**: isort runs on save
- **Problem panel**: Shows relevant issues, suppresses false positives

### Pre-commit Hooks
- **Automatic**: Runs on every commit
- **Fast feedback**: Catches issues before they reach CI
- **Configurable**: Can be bypassed with `--no-verify` if needed

### CI/CD Integration
- **GitHub Actions**: Can run full linting suite
- **Coverage reporting**: Integrated with pytest
- **Quality gates**: Configurable pass/fail criteria

## Usage Guidelines

### Running Linting Manually

```bash
# Format code with Black
black src/ tests/ scripts/

# Organize imports with isort  
isort src/ tests/ scripts/

# Check with Flake8
flake8 src/ tests/ scripts/

# Type check with MyPy (selective)
mypy src/core/ src/api/

# Comprehensive check with Pylint (optional)
pylint src/

# Run all pre-commit hooks
pre-commit run --all-files
```

### Installing Tools

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Configuring Your Editor

The provided `.vscode/settings.json` configures VS Code optimally. For other editors:

- **PyCharm**: Import the `.flake8` and `.pylintrc` configurations
- **Vim/Neovim**: Use ALE or similar with our configuration files
- **Emacs**: Configure flycheck to use our tools and settings

## Customization

### Adding New Ignored Rules

1. **Flake8**: Add to `extend-ignore` in `.flake8`
2. **Pylint**: Add to `disable` list in `.pylintrc`  
3. **MyPy**: Add module overrides in `.mypy.ini`

### Per-File Customization

Use `per-file-ignores` in `.flake8` or `# pylint: disable=` comments for specific files.

### Temporary Overrides

```python
# Disable specific warnings for a line
result = some_complex_expression()  # noqa: E501

# Disable pylint for a function
def complex_function():  # pylint: disable=too-many-branches
    pass

# Disable mypy for a module
import some_untyped_library  # type: ignore
```

## Best Practices

### What's Still Enforced
- **Security issues**: Hardcoded passwords, SQL injection risks
- **Logic errors**: Undefined variables, unreachable code
- **Import errors**: Actual missing modules (not scientific libraries)
- **Type consistency**: Basic type checking where applicable

### What's Relaxed
- **Style preferences**: Line length up to 88 characters
- **Scientific conventions**: Mathematical variable names, complex algorithms
- **Development patterns**: Broad exception handling, dynamic imports
- **Test flexibility**: More lenient rules for test files

### When to Override
- **Temporarily**: During development for quick iteration
- **Specifically**: For scientific algorithms that need complexity
- **Never**: For security or correctness issues

## Troubleshooting

### Common Issues

1. **"Module not found" for scientific libraries**
   - **Solution**: Already handled in MyPy configuration
   - **Check**: Verify `.mypy.ini` includes your library

2. **"Line too long" still appearing**
   - **Solution**: Check if file is in `per-file-ignores`
   - **Alternative**: Use Black formatting

3. **Pre-commit hooks failing**
   - **Solution**: Run `pre-commit run --all-files` to see specific issues
   - **Bypass**: Use `git commit --no-verify` if urgent

4. **VS Code not using configuration**
   - **Solution**: Reload window and check settings.json
   - **Verify**: Check Python interpreter and linting tool paths

### Getting Help

- **Documentation**: Check tool-specific docs for advanced configuration
- **Team standards**: Follow project conventions for consistency  
- **Override judiciously**: Use `# noqa` and `# pylint: disable` sparingly

## Summary

This linting configuration provides a balance between code quality enforcement and scientific computing practicality. It maintains important safety and correctness checks while suppressing the noise from common false positives in data science and neuroscience code.

The configuration is designed to:
- ✅ **Catch real issues**: Logic errors, security problems, type inconsistencies
- ✅ **Allow scientific patterns**: Mathematical naming, complex algorithms, flexible imports
- ✅ **Integrate smoothly**: Works with VS Code, pre-commit, and CI/CD
- ✅ **Stay maintainable**: Documented, configurable, and team-friendly
