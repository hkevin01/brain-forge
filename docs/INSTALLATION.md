# Installation Guide

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Installation](#quick-installation)
- [Detailed Setup](#detailed-setup)
- [Hardware Configuration](#hardware-configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Operating System**:
- Linux (Ubuntu 20.04+, CentOS 8+) - Recommended
- macOS 10.15+ - Supported
- Windows 10+ - Limited support

**Hardware**:
- CPU: Intel i5-8th gen / AMD Ryzen 5 3600 or better
- RAM: 16 GB minimum, 32 GB recommended
- Storage: 100 GB available space (SSD recommended)
- GPU: OpenGL 4.1+ compatible (NVIDIA/AMD preferred for 3D visualization)
- Network: Gigabit Ethernet for real-time streaming

### Software Prerequisites

**Python Environment**:
```bash
# Check Python version (3.9+ required)
python3 --version
# or
python --version

# Install pip if not available
sudo apt-get install python3-pip  # Ubuntu/Debian
# or
brew install python3              # macOS
```

**Node.js (for React GUI)**:
```bash
# Install Node.js 16+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node --version
npm --version
```

**Git**:
```bash
# Install Git
sudo apt-get install git          # Ubuntu/Debian
brew install git                  # macOS

# Verify installation
git --version
```

## Quick Installation

### Option 1: Automated Setup Script

```bash
# Clone repository
git clone https://github.com/hkevin01/brain-forge.git
cd brain-forge

# Run automated setup
chmod +x setup.sh
./setup.sh

# Follow interactive prompts for configuration
```

### Option 2: Manual Installation

```bash
# 1. Clone repository
git clone https://github.com/hkevin01/brain-forge.git
cd brain-forge

# 2. Create Python virtual environment
python3 -m venv brain_forge_env
source brain_forge_env/bin/activate  # Linux/macOS
# or
brain_forge_env\Scripts\activate     # Windows

# 3. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install React GUI dependencies
cd demo-gui
npm install
cd ..

# 5. Configure system
cp config/default_config.yaml config/config.yaml
# Edit config/config.yaml as needed

# 6. Initialize database and storage
python scripts/initialize_system.py
```

## Detailed Setup

### Python Environment Setup

#### Using Conda (Recommended for Scientific Computing)

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

# Create conda environment
conda create -n brain-forge python=3.9
conda activate brain-forge

# Install scientific packages from conda-forge
conda install -c conda-forge numpy scipy matplotlib pandas
conda install -c conda-forge mne pyvista h5py

# Install remaining packages with pip
pip install -r requirements.txt
```

#### Using virtualenv

```bash
# Install virtualenv
pip install virtualenv

# Create virtual environment
virtualenv -p python3.9 brain_forge_env
source brain_forge_env/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Development Dependencies

```bash
# Install development tools (optional)
pip install -r requirements-dev.txt

# This includes:
# - pytest (testing framework)
# - black (code formatting)
# - flake8 (linting)
# - mypy (type checking)
# - sphinx (documentation)
```

### React GUI Setup

```bash
cd demo-gui

# Install dependencies
npm install

# Install additional development tools (optional)
npm install -D @types/node @types/react @types/react-dom
npm install -D eslint prettier @typescript-eslint/parser

# Build production version
npm run build

# Start development server
npm run dev
```

### System Configuration

#### Configuration Files

```bash
# Copy default configuration
cp config/default_config.yaml config/config.yaml

# Edit configuration file
nano config/config.yaml  # or use your preferred editor
```

**Example configuration (`config/config.yaml`)**:
```yaml
# Brain-Forge Configuration
system:
  name: "Brain-Forge Research Platform"
  version: "1.0.0"
  debug: false
  log_level: "INFO"

data_acquisition:
  sampling_rate: 1000  # Hz
  buffer_size: 10000   # samples
  devices:
    opm_helmet:
      enabled: true
      channels: 306
      port: "/dev/ttyUSB0"
    kernel_optical:
      enabled: true
      ip_address: "192.168.1.100"
      port: 8080
    accelerometer:
      enabled: true
      sensitivity: 2  # g

signal_processing:
  real_time:
    enabled: true
    latency_target: 100  # ms
  filters:
    high_pass: 0.1   # Hz
    low_pass: 100    # Hz
    notch: 60        # Hz (power line)

visualization:
  brain_model: "fsaverage"
  update_rate: 30    # Hz
  color_map: "RdYlBu_r"

network:
  web_port: 8501
  websocket_port: 8765
  api_port: 8000

storage:
  data_directory: "/data/brain_forge"
  compression: "gzip"
  backup_enabled: true
```

#### Directory Structure Setup

```bash
# Create data directories
sudo mkdir -p /data/brain_forge/{sessions,exports,backups}
sudo chown -R $USER:$USER /data/brain_forge

# Create log directory
mkdir -p logs

# Set permissions
chmod +x run*.sh
chmod +x scripts/*.py
```

## Hardware Configuration

### OPM Helmet Setup

```bash
# 1. Connect OPM helmet via USB 3.0
# 2. Install device drivers (if required)
sudo apt-get install libusb-1.0-0-dev

# 3. Configure device permissions
sudo tee /etc/udev/rules.d/99-opm-helmet.rules << EOF
SUBSYSTEM=="usb", ATTRS{idVendor}=="1234", ATTRS{idProduct}=="5678", MODE="0666"
EOF

# 4. Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# 5. Test device connection
python scripts/test_hardware.py --device omp_helmet
```

### Kernel Optical System Setup

```bash
# 1. Connect Kernel system via Ethernet
# 2. Configure network interface
sudo ip addr add 192.168.1.10/24 dev eth1
sudo ip link set eth1 up

# 3. Test connectivity
ping 192.168.1.100

# 4. Configure firewall (if needed)
sudo ufw allow from 192.168.1.0/24

# 5. Test device communication
python scripts/test_hardware.py --device kernel_optical
```

### Accelerometer Array Setup

```bash
# 1. Connect accelerometer arrays via USB
# 2. Install serial communication libraries
pip install pyserial

# 3. Identify device ports
ls /dev/ttyUSB*
ls /dev/ttyACM*

# 4. Test device communication
python scripts/test_hardware.py --device accelerometer
```

## Verification

### System Health Check

```bash
# Run comprehensive system test
python scripts/system_check.py

# Expected output:
# ✓ Python environment: OK
# ✓ Dependencies: OK
# ✓ Configuration: OK
# ✓ Hardware devices: OK
# ✓ Network connectivity: OK
# ✓ Storage access: OK
# ✓ GUI components: OK
```

### Component Testing

```bash
# Test individual components
python -m pytest tests/unit/           # Unit tests
python -m pytest tests/integration/    # Integration tests
python -m pytest tests/hardware/       # Hardware tests (requires devices)

# Test GUI components
cd demo-gui
npm test                               # React component tests

# Test Streamlit dashboard
streamlit run src/streamlit_app.py --server.headless true
```

### Performance Verification

```bash
# Run performance benchmarks
python scripts/benchmark_system.py

# Expected results:
# Data acquisition: <10ms latency
# Signal processing: <50ms latency
# Visualization: >30 FPS
# WebSocket latency: <20ms
```

## Troubleshooting

### Common Issues

#### Permission Errors
```bash
# Fix file permissions
chmod +x run*.sh
sudo chown -R $USER:$USER /data/brain_forge

# Fix device permissions
sudo usermod -a -G dialout $USER
# Log out and log back in
```

#### Python Dependencies
```bash
# Clear pip cache
pip cache purge

# Reinstall dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# For conda users
conda clean --all
conda env remove -n brain-forge
# Recreate environment
```

#### Node.js Issues
```bash
# Clear npm cache
npm cache clean --force

# Remove node_modules and reinstall
cd demo-gui
rm -rf node_modules package-lock.json
npm install
```

#### Hardware Connection Issues
```bash
# Check device connections
lsusb                    # USB devices
ip addr show             # Network interfaces
dmesg | tail             # Recent kernel messages

# Test device communication
python scripts/diagnose_hardware.py
```

#### Port Conflicts
```bash
# Check port usage
sudo netstat -tulpn | grep :8501  # Streamlit
sudo netstat -tulpn | grep :8765  # WebSocket
sudo netstat -tulpn | grep :3000  # React dev server

# Kill conflicting processes
sudo fuser -k 8501/tcp
```

### Getting Help

1. **Check Logs**:
   ```bash
   tail -f logs/brain_forge.log
   tail -f logs/error.log
   ```

2. **Enable Debug Mode**:
   ```yaml
   # In config/config.yaml
   system:
     debug: true
     log_level: "DEBUG"
   ```

3. **Run Diagnostic Script**:
   ```bash
   python scripts/diagnose_system.py > diagnostic_report.txt
   ```

4. **Community Support**:
   - GitHub Issues: https://github.com/hkevin01/brain-forge/issues
   - Discussions: https://github.com/hkevin01/brain-forge/discussions
   - Documentation: https://brain-forge.readthedocs.io

### Advanced Configuration

#### GPU Acceleration
```bash
# Install CUDA support (NVIDIA GPUs)
conda install cudatoolkit
pip install cupy-cuda11x

# Verify GPU availability
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

#### Cluster Deployment
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Build Brain-Forge container
docker build -t brain-forge:latest .

# Deploy with Docker Compose
docker-compose up -d
```

#### Custom Hardware Integration
```python
# Create custom device driver
# See docs/hardware_integration.md for details
from brain_forge.hardware import HardwareDevice

class CustomDevice(HardwareDevice):
    def connect(self):
        # Implement connection logic
        pass

    def acquire_data(self):
        # Implement data acquisition
        pass
```

---

For additional installation support, please refer to our [troubleshooting guide](TROUBLESHOOTING.md) or contact the development team.
