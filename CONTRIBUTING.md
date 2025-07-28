# Contributing to Brain-Forge

Thank you for your interest in contributing to Brain-Forge! This guide will help you get started with contributing to our neuroscience research platform.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Issue Reporting Guidelines](#issue-reporting-guidelines)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

Brain-Forge is committed to fostering an inclusive and welcoming community. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions.

## Getting Started

### Prerequisites

- Python 3.9+ 
- Git
- Basic understanding of neuroscience data processing
- Familiarity with MEG/EEG data analysis (helpful but not required)

### Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up development environment** (see [Development Setup](#development-setup))
4. **Create a branch** for your changes
5. **Make your changes** following our guidelines
6. **Test thoroughly** 
7. **Submit a pull request**

## How to Contribute

### Types of Contributions

We welcome many types of contributions:

- 🐛 **Bug reports and fixes**
- ✨ **New features and enhancements**
- 📚 **Documentation improvements**
- 🧪 **Tests and test improvements**
- ⚡ **Performance optimizations**
- 🔧 **Hardware driver support**
- 🎨 **UI/UX improvements**
- 🔒 **Security improvements**

### Contribution Areas

#### 🧠 Neuroscience Domain
- Signal processing algorithms
- Connectivity analysis methods
- Brain mapping techniques
- Real-time processing optimization
- Scientific validation and benchmarking

#### 💻 Software Engineering
- Core architecture improvements
- API design and implementation
- Performance optimization
- Testing infrastructure
- CI/CD pipeline enhancements

#### 🔌 Hardware Integration
- Device driver development
- Hardware abstraction layers
- Calibration procedures
- Real-time data acquisition
- Multi-device synchronization

#### 📖 Documentation & Education
- User guides and tutorials
- API documentation
- Scientific background explanations
- Example notebooks and scripts
- Video tutorials and demos

## Issue Reporting Guidelines

Brain-Forge uses a comprehensive issue template system. Please choose the appropriate template:

### 🐛 Bug Reports
Use the **Bug Report** template for:
- Software crashes or errors
- Incorrect analysis results
- Performance degradation
- Hardware communication issues
- Data corruption or loss

**Required Information:**
- Detailed reproduction steps
- Environment specifications
- Error messages and logs
- Minimal code example
- Data characteristics

### 🚀 Feature Requests
Use the **Feature Request** template for:
- New analysis methods
- Hardware support requests
- API enhancements
- Visualization improvements
- Performance optimizations

**Required Information:**
- Scientific justification
- Use cases and examples
- Implementation complexity estimate
- Success criteria
- Willingness to contribute

### ⚡ Hardware Issues
Use the **Hardware Issue** template for:
- Device connection problems
- Data quality issues
- Calibration problems
- Driver issues
- Safety concerns

**Required Information:**
- Hardware specifications
- Environment details
- Safety verification
- Troubleshooting steps attempted
- Data quality measurements

### 📚 Documentation Issues
Use the **Documentation Issue** template for:
- Missing documentation
- Unclear explanations
- Outdated information
- Formatting problems
- Translation needs

### ❓ Questions and Support
Use the **Question/Support** template for:
- Usage guidance
- Best practices
- Scientific method discussions
- Integration help
- General discussions

For more details, see [Issue Templates Guide](docs/ISSUE_TEMPLATES.md).

## Development Setup

### Environment Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/brain-forge.git
cd brain-forge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install Brain-Forge in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Hardware Setup (Optional)

For hardware-related development:

```bash
# Install hardware drivers (Linux)
sudo apt-get install libusb-1.0-0-dev

# Set up udev rules for hardware access
sudo cp hardware/udev_rules/*.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules

# Test hardware connectivity
python -m brain_forge.hardware.test_connections
```

### Verification

```bash
# Run tests to verify setup
pytest tests/

# Run style checks
flake8 brain_forge/
black --check brain_forge/

# Build documentation
cd docs/
make html
```

## Pull Request Process

### Before Submitting

1. **Create feature branch** from main
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make focused changes** - one feature/fix per PR

3. **Follow coding standards** (see below)

4. **Add/update tests** for your changes

5. **Update documentation** as needed

6. **Run full test suite**
   ```bash
   pytest tests/ --cov=brain_forge
   ```

7. **Check code style**
   ```bash
   pre-commit run --all-files
   ```

### PR Template

Use our comprehensive PR template that includes:
- Change type classification
- Brain-Forge component identification
- Testing coverage verification
- Documentation updates
- Performance impact assessment
- Breaking change notification

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by maintainers
3. **Scientific review** for algorithm changes
4. **Hardware testing** for hardware-related changes
5. **Documentation review** for user-facing changes

### Merging

- PRs require approval from at least one maintainer
- Complex changes may require multiple reviews
- Breaking changes require special approval
- Security-related changes undergo additional review

## Coding Standards

### Python Style

- **PEP 8** compliance (enforced by flake8)
- **Black** formatting (automated)
- **Type hints** for public APIs
- **Docstrings** for all public functions (Google style)

### Code Organization

```
brain_forge/
├── core/           # Core functionality
├── acquisition/    # Data acquisition
├── processing/     # Signal processing
├── analysis/       # Analysis methods
├── visualization/  # Plotting and displays
├── hardware/       # Hardware drivers
├── api/           # REST API and WebSocket
├── utils/         # Utilities
└── tests/         # Test files
```

### Naming Conventions

- **Classes:** PascalCase (`ConnectivityAnalyzer`)
- **Functions/methods:** snake_case (`compute_connectivity`)
- **Constants:** UPPER_SNAKE_CASE (`DEFAULT_SAMPLING_RATE`)
- **Files/modules:** snake_case (`connectivity_analysis.py`)

### Documentation Strings

```python
def compute_plv(data: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Compute Phase Locking Value (PLV) connectivity matrix.
    
    Args:
        data: EEG/MEG data with shape (n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        PLV connectivity matrix with shape (n_channels, n_channels)
        
    Raises:
        ValueError: If data has incorrect shape
        
    Example:
        >>> data = load_meg_data('subject01.fif')
        >>> plv_matrix = compute_plv(data, sampling_rate=1000)
        >>> print(plv_matrix.shape)  # (306, 306)
    """
```

### Error Handling

```python
# Use specific exceptions
raise ValueError(f"Invalid sampling rate: {sampling_rate}")

# Provide helpful error messages
if data.ndim != 2:
    raise ValueError(
        f"Data must be 2D (channels, samples), got {data.ndim}D"
    )

# Handle hardware errors gracefully
try:
    device.connect()
except HardwareError as e:
    logger.error(f"Failed to connect to device: {e}")
    raise DeviceConnectionError(f"Could not establish connection: {e}")
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── hardware/       # Hardware tests
├── performance/    # Performance benchmarks
├── fixtures/       # Test data and fixtures
└── conftest.py     # Pytest configuration
```

### Test Categories

#### Unit Tests
- Fast execution (<1s per test)
- No external dependencies
- Mock hardware interfaces
- Test individual functions/classes

#### Integration Tests
- Test component interactions
- Use real data files
- May use simulated hardware
- Longer execution time acceptable

#### Hardware Tests
- Require actual hardware
- Run in specialized test environment
- Tagged for selective execution
- Include safety checks

#### Performance Tests
- Benchmark critical algorithms
- Monitor for regressions
- Include memory usage testing
- Real-time constraint validation

### Test Examples

```python
import pytest
import numpy as np
from brain_forge.analysis.connectivity import compute_plv

class TestConnectivityAnalysis:
    """Tests for connectivity analysis functions."""
    
    def test_plv_basic_functionality(self):
        """Test basic PLV computation."""
        # Arrange
        n_channels, n_samples = 10, 1000
        data = np.random.randn(n_channels, n_samples)
        sampling_rate = 500.0
        
        # Act
        plv_matrix = compute_plv(data, sampling_rate)
        
        # Assert
        assert plv_matrix.shape == (n_channels, n_channels)
        assert np.allclose(np.diag(plv_matrix), 1.0)  # Self-connectivity = 1
        assert np.all(plv_matrix >= 0) and np.all(plv_matrix <= 1)
        
    def test_plv_invalid_input(self):
        """Test PLV with invalid input."""
        with pytest.raises(ValueError, match="Data must be 2D"):
            compute_plv(np.random.randn(10), sampling_rate=500)
            
    @pytest.mark.hardware
    def test_real_time_plv(self, meg_device):
        """Test real-time PLV computation with hardware."""
        # Test requires actual MEG device
        pass
```

### Test Data Management

```python
# Use fixtures for test data
@pytest.fixture
def sample_meg_data():
    """Sample MEG data for testing."""
    return load_test_data('sample_meg_10min.fif')

@pytest.fixture
def mock_hardware_device():
    """Mock hardware device for testing."""
    with patch('brain_forge.hardware.MEGDevice') as mock:
        yield mock.return_value
```

## Documentation

### Types of Documentation

#### API Documentation
- Auto-generated from docstrings
- Complete parameter descriptions
- Usage examples
- Return value specifications

#### User Guides
- Step-by-step tutorials
- Real-world examples
- Best practice recommendations
- Troubleshooting guides

#### Scientific Documentation
- Algorithm descriptions
- Method validation
- Performance benchmarks
- Literature references

#### Developer Documentation
- Architecture overview
- Contribution guidelines
- Testing instructions
- Release procedures

### Documentation Tools

- **Sphinx** for main documentation
- **NumPy/Google style** docstrings
- **Jupyter notebooks** for tutorials
- **ReadTheDocs** for hosting

### Writing Guidelines

1. **Clarity first** - prefer simple explanations
2. **Include examples** - show don't just tell
3. **Update with code** - keep docs synchronized
4. **Test examples** - ensure code samples work
5. **Scientific accuracy** - validate scientific content

## Community

### Communication Channels

- **GitHub Issues:** Bug reports and feature requests
- **GitHub Discussions:** General questions and ideas
- **Email:** security@brain-forge.org for security issues
- **Documentation:** Official guides and references

### Getting Help

1. **Search existing issues** and discussions
2. **Check documentation** and FAQ
3. **Ask in discussions** for general questions
4. **Create detailed issues** for bugs/features

### Recognition

Contributors are recognized through:
- **Contributors file** listing all contributors
- **Release notes** highlighting contributions
- **GitHub contributors graph** showing activity
- **Community highlights** for exceptional contributions

### Maintainer Responsibilities

Maintainers commit to:
- **Timely responses** to issues and PRs
- **Constructive feedback** during reviews
- **Inclusive community** fostering
- **Quality standards** maintenance
- **Scientific accuracy** verification

## Getting Started Checklist

- [ ] Read and understand the Code of Conduct
- [ ] Set up development environment
- [ ] Run tests to verify setup
- [ ] Choose an issue to work on (look for "good first issue" label)
- [ ] Create feature branch
- [ ] Make changes following guidelines
- [ ] Add tests for your changes
- [ ] Update documentation
- [ ] Submit pull request
- [ ] Respond to review feedback

## Questions?

- 💬 **General questions:** [GitHub Discussions](https://github.com/hkevin01/brain-forge/discussions)
- 📚 **Documentation:** [ReadTheDocs](https://brain-forge.readthedocs.io)
- 🐛 **Bug reports:** Use issue templates
- 📧 **Security issues:** security@brain-forge.org

---

**Thank you for contributing to Brain-Forge and advancing neuroscience research!** 🧠✨
