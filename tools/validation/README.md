# Validation Tools

This directory contains validation and testing utilities for the Brain-Forge platform.

## Tools Available:
- `validate_brain_forge.py` - Core platform validation
- `validate_completion.py` - Project completion verification  
- `validate_implementation.py` - Implementation testing
- `validate_infrastructure.py` - Infrastructure validation
- `run_validation.py` - Comprehensive validation runner

## Usage:
```bash
# Run comprehensive validation
python tools/validation/run_validation.py

# Run specific validations
python tools/validation/validate_brain_forge.py
python tools/validation/validate_infrastructure.py
```

These tools help ensure the Brain-Forge platform is properly configured and functional.
