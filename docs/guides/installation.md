# Brain-Forge Installation Guide
## Complete Setup Instructions for Multi-Modal Brain-Computer Interface System

**Version**: 1.0  
**Date**: July 28, 2025  
**Compatibility**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+  

---

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04, macOS 10.15, or Windows 10
- **CPU**: Intel i7-8700K or AMD Ryzen 7 2700X (8 cores, 3.2GHz)
- **RAM**: 32GB DDR4
- **GPU**: NVIDIA GTX 1080 Ti or AMD RX 6800 XT (8GB VRAM)
- **Storage**: 1TB NVMe SSD
- **Network**: Gigabit Ethernet

### Recommended Requirements
- **OS**: Ubuntu 22.04 LTS or macOS 13.0+
- **CPU**: Intel i9-12900K or AMD Ryzen 9 5900X (12+ cores, 3.7GHz+)
- **RAM**: 64GB DDR4-3200 or DDR5
- **GPU**: NVIDIA RTX 4080 or better (16GB+ VRAM)
- **Storage**: 2TB NVMe SSD (PCIe 4.0)
- **Network**: 10 Gigabit Ethernet

### Hardware Requirements
- **OPM Helmet**: Compatible magnetometer array
- **Kernel Optical Helmet**: Time-domain NIRS system
- **Accelerometer Array**: 12-channel motion tracking system
- **Synchronized Clock**: GPS or atomic clock reference

---

## Pre-Installation Setup

### 1. Hardware Setup

#### OPM Helmet Configuration
```bash
# Check for OPM helmet connectivity
lsusb | grep "OPM"
dmesg | tail -20  # Check for hardware recognition

# Install OPM drivers (if required)
sudo apt update
sudo apt install linux-headers-$(uname -r)
sudo modprobe opm_driver

# Verify OPM detection
ls /dev/omp*
```

#### Kernel Optical Helmet Setup
```bash
# Check optical helmet connection
lsusb | grep -i "kernel\|optical"

# Install optical drivers
sudo apt install libusb-1.0-0-dev
sudo usermod -a -G dialout $USER  # Add user to dialout group

# Verify optical system
ls /dev/ttyACM* /dev/ttyUSB*
```

#### Accelerometer Array Setup
```bash
# Check accelerometer connections
dmesg | grep -i "accel\|imu"

# Install motion tracking drivers
sudo apt install libiio-utils
sudo systemctl enable iio-sensor-proxy

# Test accelerometer detection
iio_info | grep -i accel
```

### 2. System Preparation

#### Linux (Ubuntu/Debian)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    software-properties-common

# Install Python development tools
sudo apt install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    python3-setuptools \
    python3-wheel

# Install system libraries
sudo apt install -y \
    libhdf5-dev \
    libfftw3-dev \
    libblas-dev \
    liblapack-dev \
    libopenmpi-dev \
    libssl-dev \
    libffi-dev \
    libyaml-dev
```

#### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.10 cmake hdf5 fftw openmpi libyaml
brew install --cask anaconda  # Optional: Anaconda distribution
```

#### Windows
```powershell
# Install Chocolatey
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install dependencies
choco install python310 git cmake visualstudio2019buildtools
choco install anaconda3  # Recommended for Windows
```

---

## Installation Methods

### Method 1: PyPI Installation (Recommended)

```bash
# Create virtual environment
python3.10 -m venv brain_forge_env
source brain_forge_env/bin/activate  # Linux/macOS
# brain_forge_env\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Brain-Forge
pip install brain-forge[all]

# Verify installation
python -c "import brain_forge; print(brain_forge.__version__)"
```

### Method 2: Development Installation

```bash
# Clone repository
git clone https://github.com/brain-forge/brain-forge.git
cd brain-forge

# Create development environment
python3.10 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev,test,docs]"

# Run tests to verify installation
pytest tests/
```

### Method 3: Docker Installation

```bash
# Pull Docker image
docker pull brainforge/brain-forge:latest

# Run with hardware access
docker run -it --privileged \
    -v /dev:/dev \
    -v $(pwd)/data:/app/data \
    brainforge/brain-forge:latest

# Or build from source
git clone https://github.com/brain-forge/brain-forge.git
cd brain-forge
docker build -t brain-forge .
```

### Method 4: Conda Installation

```bash
# Add conda-forge channel
conda config --add channels conda-forge
conda config --add channels brain-forge

# Create environment
conda create -n brain_forge python=3.10
conda activate brain_forge

# Install Brain-Forge
conda install brain-forge

# Install additional packages
conda install jupyter lab plotly dash
```

---

## Detailed Component Installation

### 1. Core Scientific Libraries

```bash
# Neuroimaging libraries
pip install mne==1.5.0
pip install nilearn==0.10.1
pip install dipy==1.7.0

# Neural simulation libraries
pip install brian2==2.5.1
pip install nest-simulator==3.5
pip install tvb-library==2.6

# Real-time processing
pip install pylsl==1.16.0
pip install timeflux==0.10.0

# Machine learning
pip install torch==2.0.1
pip install tensorflow==2.13.0
pip install scikit-learn==1.3.0
pip install transformers==4.32.0

# Scientific computing
pip install numpy==1.24.3
pip install scipy==1.11.1
pip install pandas==2.0.3
pip install h5py==3.9.0

# Visualization
pip install matplotlib==3.7.2
pip install plotly==5.15.0
pip install mayavi==4.8.1
pip install dash==2.12.1
```

### 2. Hardware Interface Libraries

```bash
# USB and serial communication
pip install pyserial==3.5
pip install pyusb==1.2.1
pip install libusb1==3.0.0

# Low-level hardware access
pip install RPi.GPIO==0.7.1  # Raspberry Pi support
pip install gpiozero==1.6.2

# Data acquisition
pip install pydaq==2.1.0
pip install nidaqmx==0.6.5  # National Instruments
pip install pyvisa==1.13.0  # VISA instruments
```

### 3. GPU Acceleration Setup

#### NVIDIA CUDA
```bash
# Install CUDA toolkit (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Install cuDNN
sudo apt install libcudnn8 libcudnn8-dev

# Verify CUDA installation
nvcc --version
nvidia-smi
```

#### AMD ROCm (Alternative)
```bash
# Install ROCm (Ubuntu)
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dev rocm-libs

# Add user to render group
sudo usermod -a -G render $USER
```

---

## Configuration

### 1. Initial Configuration

```bash
# Create configuration directory
mkdir -p ~/.brain_forge/config
mkdir -p ~/.brain_forge/data
mkdir -p ~/.brain_forge/logs

# Generate default configuration
brain-forge config init

# Edit configuration
nano ~/.brain_forge/config/brain_forge.yaml
```

### 2. Hardware Configuration

```yaml
# ~/.brain_forge/config/brain_forge.yaml
hardware:
  omp_helmet:
    device_path: "/dev/omp0"
    channels: 128
    sampling_rate: 1000
    calibration_file: "~/.brain_forge/config/omp_calibration.json"
    
  optical_helmet:
    device_path: "/dev/ttyACM0"
    channels: 64
    wavelengths: [650, 750, 850]
    laser_power: 5.0  # mW
    
  accelerometer:
    device_paths: ["/dev/iio:device0", "/dev/iio:device1"]
    sensor_count: 12
    range_g: 16
    sampling_rate: 1000

processing:
  real_time_enabled: true
  gpu_acceleration: true
  compression_algorithm: "neural_lz"
  compression_quality: "high"

simulation:
  resolution: "high"
  dynamics: "real_time"
  parallel_processing: true
  save_intermediate: false

logging:
  level: "INFO"
  file: "~/.brain_forge/logs/brain_forge.log"
  max_size: "100MB"
  backup_count: 5
```

### 3. Environment Variables

```bash
# Add to ~/.bashrc or ~/.zshrc
export BRAIN_FORGE_CONFIG_DIR="$HOME/.brain_forge/config"
export BRAIN_FORGE_DATA_DIR="$HOME/.brain_forge/data"
export BRAIN_FORGE_LOG_DIR="$HOME/.brain_forge/logs"

# GPU settings
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

# MKL optimization
export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8

# Apply changes
source ~/.bashrc
```

---

## Hardware Calibration

### 1. OMP Helmet Calibration

```bash
# Run calibration wizard
brain-forge calibrate omp --interactive

# Manual calibration
brain-forge calibrate omp \
    --device /dev/omp0 \
    --channels 128 \
    --duration 300 \
    --output ~/.brain_forge/config/omp_calibration.json

# Verify calibration
brain-forge verify omp --calibration-file ~/.brain_forge/config/omp_calibration.json
```

### 2. Optical Helmet Calibration

```bash
# Calibrate optical sensors
brain-forge calibrate optical \
    --device /dev/ttyACM0 \
    --wavelengths 650,750,850 \
    --phantom-type "tissue_equivalent" \
    --output ~/.brain_forge/config/optical_calibration.json

# Test optical measurements
brain-forge test optical --duration 60
```

### 3. Accelerometer Calibration

```bash
# Calibrate motion sensors
brain-forge calibrate accelerometer \
    --devices /dev/iio:device0,/dev/iio:device1 \
    --gravity-calibration \
    --output ~/.brain_forge/config/accel_calibration.json

# Test motion tracking
brain-forge test accelerometer --duration 30
```

---

## Verification and Testing

### 1. System Verification

```bash
# Run comprehensive system check
brain-forge system check

# Check hardware connectivity
brain-forge hardware status

# Test all components
brain-forge test all --duration 60
```

### 2. Performance Benchmarks

```bash
# Run performance benchmarks
brain-forge benchmark --full

# GPU acceleration test
brain-forge benchmark gpu

# Real-time processing test
brain-forge benchmark realtime
```

### 3. Sample Data Collection

```python
# test_installation.py
import asyncio
from brain_forge.core.config import BrainForgeConfig
from brain_forge.acquisition.stream_manager import StreamManager

async def test_data_collection():
    """Test basic data collection functionality"""
    
    # Load configuration
    config = BrainForgeConfig()
    
    # Initialize stream manager
    stream_manager = StreamManager(config)
    
    try:
        # Initialize streams
        init_result = await stream_manager.initialize_all_streams()
        print(f"Stream initialization: {init_result}")
        
        # Start streaming
        await stream_manager.start_synchronized_streaming()
        
        # Collect test data
        print("Collecting test data for 10 seconds...")
        test_data = []
        
        for i in range(100):  # 10 seconds at 10Hz
            sample = await stream_manager.get_synchronized_sample()
            test_data.append(sample)
            await asyncio.sleep(0.1)
            
        print(f"Successfully collected {len(test_data)} samples")
        
        # Verify data quality
        for stream_name, data in test_data[-1].items():
            print(f"{stream_name}: shape={data[0].shape}, timestamp={data[1]}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        return False
        
    print("Installation test completed successfully!")
    return True

# Run test
if __name__ == "__main__":
    success = asyncio.run(test_data_collection())
    exit(0 if success else 1)
```

---

## Troubleshooting

### Common Issues

#### 1. Hardware Not Detected

```bash
# Check USB connections
lsusb -v | grep -A 5 -B 5 "OMP\|Kernel\|Accelerometer"

# Check permissions
sudo chmod 666 /dev/omp* /dev/ttyACM* /dev/iio:device*

# Restart udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

#### 2. Python Import Errors

```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Reinstall with verbose output
pip install --upgrade --force-reinstall brain-forge -v

# Check for conflicts
pip check
```

#### 3. GPU Acceleration Issues

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Update GPU drivers
sudo apt update && sudo apt install nvidia-driver-535

# Verify GPU memory
nvidia-smi
```

#### 4. Permission Issues

```bash
# Add user to necessary groups
sudo usermod -a -G dialout,plugdev,gpio $USER

# Set udev rules for hardware access
sudo nano /etc/udev/rules.d/99-brain-forge.rules

# Add rules:
SUBSYSTEM=="usb", ATTRS{idVendor}=="1234", ATTRS{idProduct}=="5678", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="iio", MODE="0666", GROUP="plugdev"

# Reload rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Logout and login again
```

### Performance Issues

#### 1. Slow Processing

```bash
# Check CPU usage
htop

# Monitor GPU usage
nvidia-smi -l 1

# Check memory usage
free -h

# Optimize settings
brain-forge config set processing.optimization_level "high"
brain-forge config set processing.parallel_threads 16
```

#### 2. Real-time Processing Delays

```bash
# Set real-time scheduling
sudo chrt -f 99 python brain_forge_app.py

# Increase system limits
echo "* soft realtime -1" | sudo tee -a /etc/security/limits.conf
echo "* hard realtime -1" | sudo tee -a /etc/security/limits.conf

# Disable CPU frequency scaling
sudo cpupower frequency-set --governor performance
```

---

## Maintenance

### Regular Updates

```bash
# Update Brain-Forge
pip install --upgrade brain-forge

# Update all dependencies
pip install --upgrade -r requirements.txt

# Check for security updates
pip audit

# Update hardware drivers
sudo apt update && sudo apt upgrade
```

### Backup Configuration

```bash
# Backup configuration
tar -czf brain_forge_backup_$(date +%Y%m%d).tar.gz ~/.brain_forge/

# Restore configuration
tar -xzf brain_forge_backup_20250728.tar.gz -C ~/
```

### Log Management

```bash
# Rotate logs
brain-forge logs rotate

# Clean old logs
brain-forge logs clean --older-than 30d

# Monitor logs
tail -f ~/.brain_forge/logs/brain_forge.log
```

---

## Advanced Configuration

### Multi-User Setup

```bash
# System-wide installation
sudo pip install brain-forge

# Create shared configuration
sudo mkdir -p /etc/brain_forge/
sudo cp ~/.brain_forge/config/brain_forge.yaml /etc/brain_forge/

# Set permissions
sudo chown -R brain_forge:brain_forge /etc/brain_forge/
sudo chmod -R 755 /etc/brain_forge/
```

### High-Performance Computing

```bash
# Install MPI support
sudo apt install openmpi-bin openmpi-common libopenmpi-dev

# Configure for cluster usage
brain-forge config set simulation.parallel_backend "mpi"
brain-forge config set simulation.nodes 8
brain-forge config set simulation.cores_per_node 32
```

### Container Deployment

```dockerfile
# Custom Dockerfile
FROM ubuntu:22.04
RUN apt update && apt install -y python3.10 python3-pip
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python", "-m", "brain_forge.server"]
```

---

## Support and Resources

### Documentation
- **Official Documentation**: https://docs.brain-forge.org
- **API Reference**: https://api.brain-forge.org
- **Tutorials**: https://tutorials.brain-forge.org

### Community
- **GitHub Issues**: https://github.com/brain-forge/brain-forge/issues
- **Discussion Forum**: https://community.brain-forge.org
- **Discord Channel**: https://discord.gg/brainforge

### Professional Support
- **Email**: support@brain-forge.org
- **Enterprise Support**: enterprise@brain-forge.org
- **Training Programs**: training@brain-forge.org

---

Installation is now complete! Proceed to the User Guide for detailed usage instructions and examples.
