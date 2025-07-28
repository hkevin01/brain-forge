# Getting Started with Brain-Forge

## Installation

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements/base.txt`

## Basic Usage

```python
from src.integrated_system import IntegratedBrainSystem

# Initialize system
brain_system = IntegratedBrainSystem()

# Run brain transfer protocol
result = await brain_system.run_brain_transfer_protocol("subject_001")
```

## Documentation

See the `docs/` directory for detailed documentation.
