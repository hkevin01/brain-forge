# Troubleshooting Guide

## Table of Contents
- [Quick Diagnostics](#quick-diagnostics)
- [Common Issues](#common-issues)
- [Hardware Problems](#hardware-problems)
- [Software Issues](#software-issues)
- [Performance Problems](#performance-problems)
- [Network Connectivity](#network-connectivity)
- [GUI Issues](#gui-issues)
- [Advanced Debugging](#advanced-debugging)

## Quick Diagnostics

### System Health Check

Run the automated system diagnostic to identify issues:

```bash
# Comprehensive system check
python scripts/system_check.py

# Quick hardware check
python scripts/test_hardware.py --all

# Network connectivity test
python scripts/test_network.py

# Performance benchmark
python scripts/benchmark_system.py --quick
```

### Log Analysis

```bash
# View real-time system logs
tail -f logs/brain_forge.log

# Check error logs
grep -i error logs/brain_forge.log | tail -20

# View hardware logs
tail -f logs/hardware.log

# Check processing pipeline logs
grep -i "processing" logs/brain_forge.log
```

### Quick Status Check

```bash
# Check running processes
ps aux | grep brain_forge

# Check port usage
sudo netstat -tulpn | grep -E ':(8501|8765|3000|8000)'

# Check system resources
top -p $(pgrep -f brain_forge)

# Check disk space
df -h /data/brain_forge
```

## Common Issues

### 1. Application Won't Start

**Symptoms**: Brain-Forge fails to initialize or crashes on startup

**Diagnostic Steps**:
```bash
# Check Python environment
python --version
pip list | grep -E "(numpy|scipy|mne|pyvista)"

# Verify configuration
python -c "from brain_forge.config import Config; print(Config.validate('config/config.yaml'))"

# Test basic imports
python -c "import brain_forge; print('Import successful')"
```

**Common Solutions**:

```bash
# Option 1: Reinstall dependencies
pip uninstall brain-forge -y
pip install -r requirements.txt
pip install -e .

# Option 2: Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Option 3: Reset configuration
cp config/default_config.yaml config/config.yaml

# Option 4: Check file permissions
chmod -R 755 src/
chmod +x run*.sh
```

### 2. Import Errors

**Error**: `ModuleNotFoundError: No module named 'brain_forge'`

**Solutions**:
```bash
# Install in development mode
pip install -e .

# Add to Python path temporarily
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Check virtual environment
which python
pip show brain-forge
```

**Error**: `ImportError: cannot import name 'specific_module'`

**Solutions**:
```bash
# Check for circular imports
python -c "import ast; import sys; print(sys.path)"

# Reinstall specific dependencies
pip uninstall mne pyvista -y
pip install mne pyvista

# Clear import cache
python -c "import importlib; importlib.invalidate_caches()"
```

### 3. Configuration Issues

**Symptoms**: System starts but behaves unexpectedly

**Diagnostic**:
```bash
# Validate configuration syntax
python -c "
import yaml
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)
    print('Configuration is valid YAML')
"

# Check configuration values
python scripts/validate_config.py config/config.yaml
```

**Solutions**:
```bash
# Reset to defaults
cp config/default_config.yaml config/config.yaml

# Fix common configuration errors
sed -i 's/enabled: yes/enabled: true/g' config/config.yaml
sed -i 's/enabled: no/enabled: false/g' config/config.yaml

# Validate device paths
ls -la /dev/ttyUSB* /dev/ttyACM*
```

## Hardware Problems

### 1. Device Connection Failures

**Error**: `DeviceConnectionError: Failed to connect to OPM helmet`

**Diagnostic Steps**:
```bash
# Check USB connections
lsusb
dmesg | grep -i usb | tail -10

# Check device permissions
ls -la /dev/ttyUSB*
groups $USER

# Test serial communication
python -c "
import serial
ser = serial.Serial('/dev/ttyUSB0', 9600)
print('Serial connection successful')
ser.close()
"
```

**Solutions**:
```bash
# Fix USB permissions
sudo usermod -a -G dialout $USER
# Log out and log back in

# Create udev rules
sudo tee /etc/udev/rules.d/99-brain-forge.rules << EOF
SUBSYSTEM=="usb", ATTRS{idVendor}=="1234", ATTRS{idProduct}=="5678", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1234", ATTRS{idProduct}=="5678", MODE="0666"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger

# Reset USB devices
sudo modprobe -r usbserial
sudo modprobe usbserial
```

### 2. Data Acquisition Issues

**Symptoms**: No data received or corrupted data from devices

**Diagnostic**:
```bash
# Test device communication
python scripts/test_hardware.py --device omp_helmet --verbose

# Check data integrity
python -c "
from brain_forge.hardware import OMPHelmet
helmet = OMPHelmet()
helmet.connect()
data = helmet.acquire_data(1.0)
print(f'Data shape: {data.shape}')
print(f'Data range: {data.min():.3f} to {data.max():.3f}')
print(f'Data std: {data.std():.3f}')
helmet.disconnect()
"
```

**Solutions**:
```bash
# Recalibrate devices
python scripts/calibrate_hardware.py --device omp_helmet

# Check cable connections
# Ensure all cables are properly seated
# Try different USB ports/cables

# Update device firmware (if applicable)
python scripts/update_firmware.py --device omp_helmet

# Adjust acquisition parameters
# Edit config/config.yaml:
# data_acquisition:
#   sampling_rate: 500  # Reduce if having issues
#   buffer_size: 5000   # Reduce buffer size
```

### 3. Kernel Optical System Issues

**Error**: `NetworkError: Cannot connect to Kernel system at 192.168.1.100`

**Diagnostic**:
```bash
# Test network connectivity
ping 192.168.1.100

# Check network interface
ip addr show
route -n

# Test port connectivity
telnet 192.168.1.100 8080
```

**Solutions**:
```bash
# Configure network interface
sudo ip addr add 192.168.1.10/24 dev eth1
sudo ip link set eth1 up

# Configure firewall
sudo ufw allow from 192.168.1.0/24

# Check Kernel system status
curl http://192.168.1.100:8080/status

# Reset network configuration
sudo systemctl restart NetworkManager
```

## Software Issues

### 1. Processing Pipeline Errors

**Error**: `ProcessingError: Signal processing failed at step 'ica_removal'`

**Diagnostic**:
```bash
# Test individual processing steps
python -c "
from brain_forge.processing import SignalProcessor
import numpy as np

data = np.random.randn(306, 10000)
processor = SignalProcessor()

# Test each step individually
filtered = processor.apply_butterworth_filter(data)
print('Filtering: OK')

cleaned = processor.remove_artifacts(filtered)
print('Artifact removal: OK')
"
```

**Solutions**:
```bash
# Adjust processing parameters
# Edit config/config.yaml:
# signal_processing:
#   ica:
#     n_components: 15  # Reduce from 20
#     max_iter: 100     # Reduce iterations

# Skip problematic steps temporarily
python -c "
from brain_forge.processing import SignalProcessor
processor = SignalProcessor()
processor.disable_step('ica_removal')
"

# Check data quality
python scripts/analyze_data_quality.py session_id
```

### 2. Memory Issues

**Error**: `MemoryError: Unable to allocate array`

**Diagnostic**:
```bash
# Check available memory
free -h
cat /proc/meminfo | grep Available

# Check Python memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

**Solutions**:
```bash
# Reduce buffer sizes
# Edit config/config.yaml:
# data_acquisition:
#   buffer_size: 5000    # Reduce from 10000
# visualization:
#   max_history: 1000    # Reduce history

# Enable data compression
# Edit config/config.yaml:
# storage:
#   compression: "gzip"
#   compression_level: 6

# Process data in chunks
python -c "
from brain_forge.processing import ChunkedProcessor
processor = ChunkedProcessor(chunk_size=1000)
"

# Increase system swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 3. Threading/Concurrency Issues

**Symptoms**: Deadlocks, race conditions, or inconsistent behavior

**Diagnostic**:
```bash
# Check for deadlocks
python -c "
import threading
print('Active threads:', threading.active_count())
for thread in threading.enumerate():
    print(f'Thread: {thread.name}, Alive: {thread.is_alive()}')
"

# Enable debug logging
# Edit config/config.yaml:
# system:
#   debug: true
#   log_level: "DEBUG"
```

**Solutions**:
```bash
# Reduce thread count
# Edit config/config.yaml:
# processing:
#   max_workers: 2  # Reduce from 4

# Add proper synchronization
python -c "
import threading
import queue

# Use thread-safe queue for communication
data_queue = queue.Queue(maxsize=100)
"

# Restart with single-threaded mode
python scripts/run_single_threaded.py
```

## Performance Problems

### 1. High Processing Latency

**Symptoms**: Processing takes longer than expected (>100ms)

**Diagnostic**:
```bash
# Run performance profiling
python -m cProfile -o profile.stats scripts/benchmark_processing.py
python -c "
import pstats
stats = pstats.Stats('profile.stats')
stats.sort_stats('cumulative').print_stats(20)
"

# Check CPU usage
top -p $(pgrep -f brain_forge)
iostat -x 1 5
```

**Solutions**:
```bash
# Optimize processing parameters
# Edit config/config.yaml:
# signal_processing:
#   real_time:
#     chunk_size: 500     # Smaller chunks
#     overlap: 50         # Reduced overlap
#   filters:
#     order: 4            # Lower filter order

# Enable GPU acceleration (if available)
pip install cupy-cuda11x
# Edit config/config.yaml:
# processing:
#   use_gpu: true

# Optimize NumPy
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

### 2. Memory Leaks

**Symptoms**: Memory usage increases over time

**Diagnostic**:
```bash
# Monitor memory usage over time
python scripts/monitor_memory.py &
MONITOR_PID=$!

# Run your application
python scripts/run_long_test.py

# Stop monitoring
kill $MONITOR_PID

# Check for memory leaks
python -m memory_profiler scripts/test_memory_leak.py
```

**Solutions**:
```bash
# Force garbage collection
python -c "
import gc
gc.set_debug(gc.DEBUG_LEAK)
gc.collect()
"

# Use memory pools
python -c "
from brain_forge.utils import MemoryPool
pool = MemoryPool(max_size=1000)
"

# Limit cache sizes
# Edit config/config.yaml:
# data_management:
#   cache_size: 100     # Limit cache
#   auto_cleanup: true  # Enable auto cleanup
```

## Network Connectivity

### 1. WebSocket Connection Issues

**Error**: `WebSocket connection failed`

**Diagnostic**:
```bash
# Test WebSocket connectivity
curl -i -N -H "Connection: Upgrade" \
     -H "Upgrade: websocket" \
     -H "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
     -H "Sec-WebSocket-Version: 13" \
     http://localhost:8765/

# Check if WebSocket server is running
ps aux | grep websocket
netstat -tulpn | grep :8765
```

**Solutions**:
```bash
# Restart WebSocket server
./run_websocket_bridge.sh

# Check firewall settings
sudo ufw status
sudo ufw allow 8765

# Test with different port
# Edit config/config.yaml:
# network:
#   websocket_port: 8766

# Check for port conflicts
sudo lsof -i :8765
```

### 2. API Server Issues

**Error**: `API server not responding`

**Diagnostic**:
```bash
# Test API endpoints
curl http://localhost:8000/api/v1/status
curl -v http://localhost:8000/health

# Check server logs
tail -f logs/api_server.log
```

**Solutions**:
```bash
# Restart API server
python -m brain_forge.api.server

# Increase timeout settings
# Edit config/config.yaml:
# api:
#   timeout: 60
#   max_connections: 100

# Check database connectivity
python -c "
from brain_forge.database import DatabaseManager
db = DatabaseManager()
print('Database connection:', db.test_connection())
"
```

## GUI Issues

### 1. Streamlit Dashboard Problems

**Error**: Streamlit app won't load or crashes

**Diagnostic**:
```bash
# Test Streamlit directly
streamlit run src/streamlit_app.py --server.headless true

# Check Streamlit logs
tail -f ~/.streamlit/logs/streamlit.log

# Test in debug mode
streamlit run src/streamlit_app.py --logger.level debug
```

**Solutions**:
```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/

# Update Streamlit
pip install --upgrade streamlit

# Check browser compatibility
# Try different browser or incognito mode

# Reduce visualization complexity
# Edit streamlit_app.py:
# st.set_page_config(
#     page_title="Brain-Forge",
#     layout="centered"  # Change from "wide"
# )
```

### 2. React GUI Issues

**Error**: React development server fails to start

**Diagnostic**:
```bash
cd demo-gui

# Check Node.js version
node --version
npm --version

# Check for dependency issues
npm ls
npm audit

# Clear npm cache
npm cache clean --force
```

**Solutions**:
```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Fix permission issues
sudo chown -R $USER:$USER node_modules

# Use different port
npm start -- --port 3001

# Build production version
npm run build
```

### 3. 3D Visualization Issues

**Error**: 3D brain visualization not rendering

**Diagnostic**:
```bash
# Test PyVista directly
python -c "
import pyvista as pv
print('PyVista version:', pv.__version__)
print('VTK version:', pv.vtk_version_info)

# Test basic rendering
sphere = pv.Sphere()
sphere.plot(off_screen=True)
print('Basic rendering: OK')
"

# Check OpenGL support
glxinfo | grep -i opengl
```

**Solutions**:
```bash
# Install required OpenGL libraries
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# Use software rendering (if hardware acceleration fails)
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330

# Update graphics drivers
sudo ubuntu-drivers autoinstall

# Use headless rendering
python -c "
import pyvista as pv
pv.start_xvfb()  # For headless systems
"
```

## Advanced Debugging

### 1. Enable Debug Mode

```bash
# Set debug environment variables
export BRAIN_FORGE_DEBUG=1
export BRAIN_FORGE_LOG_LEVEL=DEBUG

# Enable verbose logging
# Edit config/config.yaml:
# system:
#   debug: true
#   log_level: "DEBUG"
#   verbose_errors: true
```

### 2. Profiling Tools

```bash
# CPU profiling
python -m cProfile -o profile.stats main.py
snakeviz profile.stats

# Memory profiling
pip install memory_profiler
python -m memory_profiler scripts/test_script.py

# Line-by-line profiling
pip install line_profiler
kernprof -l -v scripts/test_script.py
```

### 3. Debugging Specific Components

```python
# Debug signal processing
import logging
logging.getLogger('brain_forge.processing').setLevel(logging.DEBUG)

# Debug hardware communication
import logging
logging.getLogger('brain_forge.hardware').setLevel(logging.DEBUG)

# Debug visualization
import pyvista as pv
pv.set_plot_theme('document')  # Better for debugging
```

### 4. Creating Debug Reports

```bash
# Generate comprehensive debug report
python scripts/generate_debug_report.py > debug_report.txt

# Include system information
python -c "
import platform
import sys
import pkg_resources

print('Python version:', sys.version)
print('Platform:', platform.platform())
print('Architecture:', platform.architecture())

print('\nInstalled packages:')
for pkg in sorted([str(d) for d in pkg_resources.working_set]):
    print(pkg)
" >> debug_report.txt

# Include recent logs
echo -e '\n\n=== Recent Logs ===' >> debug_report.txt
tail -100 logs/brain_forge.log >> debug_report.txt
```

## Getting Help

### 1. Community Support

- **GitHub Issues**: https://github.com/hkevin01/brain-forge/issues
- **Discussions**: https://github.com/hkevin01/brain-forge/discussions
- **Documentation**: https://brain-forge.readthedocs.io

### 2. Professional Support

For commercial support and custom development:
- Email: support@brain-forge.org
- Documentation: https://brain-forge.org/support

### 3. Reporting Bugs

When reporting issues, please include:

1. **Environment Information**:
   ```bash
   python scripts/system_info.py
   ```

2. **Error Logs**:
   ```bash
   tail -50 logs/brain_forge.log
   ```

3. **Configuration**:
   ```bash
   cat config/config.yaml
   ```

4. **Steps to Reproduce**:
   - Exact commands or actions taken
   - Expected behavior
   - Actual behavior

5. **Debug Report**:
   ```bash
   python scripts/generate_debug_report.py
   ```

---

If you cannot resolve an issue using this guide, please don't hesitate to reach out to the community or support team for assistance.
