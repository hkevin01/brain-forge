# Hardware Integration Stack

This document details the hardware integration capabilities and requirements for Brain-Forge's multi-modal brain scanning platform.

## 🔧 Device Interface Architecture

Brain-Forge supports three primary hardware categories for comprehensive brain monitoring:

### 1. OPM Helmet (Magnetoencephalography)
- **Technology**: Optically Pumped Magnetometers
- **Channels**: 306 sensors in helmet configuration
- **Sampling Rate**: 1000 Hz
- **Sensitivity**: <10 fT/√Hz
- **Interface**: USB/Serial communication via PySerial

### 2. Kernel Optical Helmet System
- **Flow Helmet**: Real-time brain activity pattern detection
- **Flux Helmet**: Neuron speed measurement capabilities
- **Technology**: Time-domain near-infrared spectroscopy
- **Channels**: 32 Flow + 64 Flux = 96 total optical channels
- **Interface**: Serial communication with custom protocols

### 3. Accelerometer Arrays (Motion Tracking)
- **Technology**: 3-axis MEMS accelerometers
- **Purpose**: Motion artifact detection and compensation
- **Sampling**: 1000 Hz synchronized with neural data
- **Interface**: I2C/Bluetooth LE communication

## 📡 Communication Protocols

### Serial Communication Stack
```python
# Core serial communication
pyserial>=3.5          # Primary serial interface
pyusb>=1.2.0           # USB device management
smbus2>=0.4.0          # I2C interface for sensors
```

### Wireless Communication
```python
# Bluetooth Low Energy
bleak>=0.13.0          # BLE device communication
pybluez>=0.23          # Classic Bluetooth support
```

### Real-time Streaming
```python
# Lab Streaming Layer
pylsl>=1.14.0          # Multi-device synchronization
timeflux>=0.6.0        # Real-time processing pipeline
asyncio                # Asynchronous device handling
```

## 🚀 GPU Acceleration Support

Brain-Forge leverages GPU acceleration for real-time neural signal processing:

### NVIDIA CUDA Stack
```python
# CUDA acceleration libraries
cupy>=9.0.0            # GPU-accelerated NumPy replacement
numba>=0.54.0          # JIT compilation with CUDA kernels
pycuda>=2021.1         # Low-level CUDA programming
```

**Docker GPU Support:**
```dockerfile
# NVIDIA Container Runtime
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install NVIDIA Container Toolkit
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
```

### AMD ROCm Stack
```bash
# ROCm platform for AMD GPUs
rocm-docs-core         # ROCm documentation
hip-dev                # HIP development environment
rocblas-dev            # GPU-accelerated BLAS
rocfft-dev             # GPU-accelerated FFT
```

**ROCm Installation:**
```bash
# Add ROCm repository
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

# Install ROCm
sudo apt update
sudo apt install rocm-dkms rocm-libs
```

## 🔄 Real-time Processing Pipeline

### Data Flow Architecture
```
OMP Helmet (306ch) ──┐
                     ├── LSL Synchronization ──► GPU Processing ──► Analysis ──► Visualization
Kernel Optical ──────┤
                     │
Accelerometers ──────┘
```

### Performance Requirements
- **Latency Target**: <100ms end-to-end processing
- **Throughput**: >1GB/s sustained data handling
- **Synchronization**: ±10μs timing accuracy across devices
- **Memory**: <16GB RAM for standard operation

## 🛠️ Hardware Interface Examples

### OMP Helmet Interface
```python
import serial
import pylsl
import numpy as np
from typing import Optional

class OMPHelmetInterface:
    """Interface for 306-channel OPM helmet system"""
    
    def __init__(self, port: str = '/dev/ttyUSB0', 
                 channels: int = 306, sample_rate: int = 1000):
        self.port = port
        self.channels = channels
        self.sample_rate = sample_rate
        self.serial_conn: Optional[serial.Serial] = None
        self.lsl_outlet: Optional[pylsl.StreamOutlet] = None
        
    def connect(self) -> bool:
        """Establish connection to OMP helmet"""
        try:
            # Initialize serial connection
            self.serial_conn = serial.Serial(
                self.port, 
                baudrate=115200,
                timeout=1.0
            )
            
            # Create LSL outlet for streaming
            info = pylsl.StreamInfo(
                name='OMP_MEG_Data',
                type='MEG',
                channel_count=self.channels,
                nominal_srate=self.sample_rate,
                channel_format=pylsl.cf_float32,
                source_id='omp_helmet_001'
            )
            
            self.lsl_outlet = pylsl.StreamOutlet(info)
            return True
            
        except Exception as e:
            print(f"Failed to connect to OMP helmet: {e}")
            return False
    
    def stream_data(self) -> None:
        """Start real-time data streaming"""
        if not self.serial_conn or not self.lsl_outlet:
            raise RuntimeError("Device not connected")
            
        while True:
            try:
                # Read binary data (4 bytes per float, 306 channels)
                raw_data = self.serial_conn.read(self.channels * 4)
                
                if len(raw_data) == self.channels * 4:
                    # Convert to numpy array
                    neural_data = np.frombuffer(raw_data, dtype=np.float32)
                    
                    # Stream via LSL
                    self.lsl_outlet.push_sample(neural_data.tolist())
                    
            except Exception as e:
                print(f"Streaming error: {e}")
                break
```

### Kernel Optical Interface
```python
import asyncio
import struct
from typing import Dict, List

class KernelOpticalInterface:
    """Interface for Kernel Flow/Flux optical helmet system"""
    
    def __init__(self):
        self.flow_channels = 32
        self.flux_channels = 64
        self.total_channels = self.flow_channels + self.flux_channels
        
    async def initialize_helmets(self) -> Dict[str, bool]:
        """Initialize both Flow and Flux helmets"""
        results = {}
        
        # Initialize Flow helmet
        try:
            self.flow_serial = serial.Serial('/dev/ttyUSB1', 230400)
            await self.send_command(self.flow_serial, b'\x01\x00\xFF')  # Start command
            results['flow'] = True
        except Exception as e:
            print(f"Flow helmet init failed: {e}")
            results['flow'] = False
            
        # Initialize Flux helmet
        try:
            self.flux_serial = serial.Serial('/dev/ttyUSB2', 230400)
            await self.send_command(self.flux_serial, b'\x02\x00\xFF')  # Start command
            results['flux'] = True
        except Exception as e:
            print(f"Flux helmet init failed: {e}")
            results['flux'] = False
            
        return results
    
    async def acquire_optical_data(self) -> Dict[str, np.ndarray]:
        """Acquire data from both optical systems"""
        tasks = [
            self.read_flow_data(),
            self.read_flux_data()
        ]
        
        flow_data, flux_data = await asyncio.gather(*tasks)
        
        return {
            'flow': flow_data,
            'flux': flux_data,
            'combined': np.concatenate([flow_data, flux_data])
        }
```

### GPU-Accelerated Processing
```python
import cupy as cp
from cupyx.scipy import signal as gpu_signal

class GPUSignalProcessor:
    """GPU-accelerated neural signal processing"""
    
    def __init__(self, channels: int = 306, sample_rate: int = 1000):
        self.channels = channels
        self.sample_rate = sample_rate
        
        # Pre-compute filter coefficients on GPU
        self.setup_filters()
        
    def setup_filters(self):
        """Initialize GPU-based filters"""
        # Bandpass filter for neural signals (1-100 Hz)
        self.bp_b, self.bp_a = gpu_signal.butter(
            4, [1, 100], btype='band', fs=self.sample_rate
        )
        
        # Notch filter for 60 Hz powerline noise
        self.notch_b, self.notch_a = gpu_signal.iirnotch(
            60, 30, fs=self.sample_rate
        )
    
    def process_realtime(self, neural_data: np.ndarray) -> np.ndarray:
        """Real-time GPU processing of neural signals"""
        # Move data to GPU
        gpu_data = cp.asarray(neural_data)
        
        # Apply bandpass filter
        filtered = gpu_signal.filtfilt(self.bp_b, self.bp_a, gpu_data, axis=1)
        
        # Apply notch filter
        cleaned = gpu_signal.filtfilt(self.notch_b, self.notch_a, filtered, axis=1)
        
        # Compute spectral features
        freqs, psd = gpu_signal.periodogram(cleaned, fs=self.sample_rate, axis=1)
        
        # Return processed data to CPU
        return {
            'filtered': cp.asnumpy(cleaned),
            'psd': cp.asnumpy(psd),
            'frequencies': cp.asnumpy(freqs)
        }
```

## 🐳 Docker GPU Configuration

### NVIDIA Docker Setup
```dockerfile
# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Brain-Forge with GPU support
COPY requirements/gpu.txt .
RUN pip3 install -r gpu.txt

# Install Brain-Forge
COPY . /app
WORKDIR /app
RUN pip3 install -e .

# Expose ports for API
EXPOSE 8000 8001

# Start with GPU support
CMD ["python3", "-c", "import cupy; print(f'GPU available: {cupy.cuda.is_available()}')"]
```

### Docker Compose with GPU
```yaml
version: '3.8'
services:
  brain-forge:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./data:/app/data
      - /dev:/dev  # Hardware access
    privileged: true  # Required for hardware access
    ports:
      - "8000:8000"
      - "8001:8001"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - NVIDIA_VISIBLE_DEVICES=all
```

## 🔧 Hardware Troubleshooting

### Common Issues and Solutions

**Issue**: Device not found
```bash
# Check USB devices
lsusb

# Check serial ports
ls -la /dev/tty*

# Add user to dialout group for serial access
sudo usermod -a -G dialout $USER
```

**Issue**: Permission denied for hardware access
```bash
# Set appropriate permissions
sudo chmod 666 /dev/ttyUSB*
sudo chmod 666 /dev/ttyACM*

# Create udev rules for persistent permissions
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="1234", ATTRS{idProduct}=="5678", GROUP="dialout", MODE="0666"' | sudo tee /etc/udev/rules.d/99-brain-forge.rules
```

**Issue**: GPU out of memory
```python
# Monitor GPU memory usage
import cupy as cp

def check_gpu_memory():
    mempool = cp.get_default_memory_pool()
    print(f"Used: {mempool.used_bytes()} bytes")
    print(f"Total: {mempool.total_bytes()} bytes")
    
# Clear GPU memory
mempool.free_all_blocks()
```

**Issue**: Real-time processing delays
```python
# Optimize for real-time performance
import os

# Set CPU affinity for critical processes
os.sched_setaffinity(0, {0, 1})  # Use first 2 CPU cores

# Set process priority
os.nice(-10)  # Higher priority

# Use memory locking to prevent swapping
import mlock
mlock.mlockall()
```

## 📊 Performance Monitoring

### Real-time Metrics
```python
import time
import psutil
from collections import deque

class PerformanceMonitor:
    """Monitor real-time processing performance"""
    
    def __init__(self, window_size: int = 100):
        self.latencies = deque(maxlen=window_size)
        self.throughputs = deque(maxlen=window_size)
        
    def measure_latency(self, start_time: float) -> float:
        """Measure processing latency"""
        latency = time.time() - start_time
        self.latencies.append(latency)
        return latency
    
    def measure_throughput(self, data_size: int, duration: float) -> float:
        """Measure data throughput"""
        throughput = data_size / duration
        self.throughputs.append(throughput)
        return throughput
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        return {
            'avg_latency_ms': np.mean(self.latencies) * 1000,
            'max_latency_ms': np.max(self.latencies) * 1000,
            'avg_throughput_mbps': np.mean(self.throughputs) / (1024*1024),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        }
```

## 🚀 Deployment Configurations

### Production Deployment
```yaml
# Kubernetes deployment with GPU support
apiVersion: apps/v1
kind: Deployment
metadata:
  name: brain-forge-processing
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: brain-forge
        image: brain-forge:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
          requests:
            memory: "16Gi"
            cpu: "4"
        volumeMounts:
        - name: device-access
          mountPath: /dev
      volumes:
      - name: device-access
        hostPath:
          path: /dev
```

### Development Environment
```bash
#!/bin/bash
# Development setup script

# Create development environment
python -m venv brain-forge-dev
source brain-forge-dev/bin/activate

# Install development dependencies
pip install -r requirements/dev.txt
pip install -r requirements/gpu.txt

# Set up pre-commit hooks
pre-commit install

# Configure hardware permissions
sudo usermod -a -G dialout $USER
sudo usermod -a -G plugdev $USER

echo "Development environment ready!"
echo "Please log out and back in for group changes to take effect."
```

---

This hardware integration stack provides the foundation for Brain-Forge's multi-modal brain scanning capabilities, enabling real-time processing of neural data with GPU acceleration and comprehensive device support.
