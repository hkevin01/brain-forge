# Brain-Forge API Reference
## Comprehensive Brain-Computer Interface System API

**Version**: 1.0  
**Date**: July 28, 2025  
**API Version**: v1.0.0  

---

## Overview

The Brain-Forge API provides comprehensive access to multi-modal brain scanning, real-time processing, and neural simulation capabilities. The API is organized into several main modules: acquisition, processing, simulation, and utilities.

---

## Core Modules

### Configuration Management

#### `brain_forge.core.config`

**BrainForgeConfig**
```python
class BrainForgeConfig:
    """Main configuration management class"""
    
    def __init__(self, config_file: str = 'config.yaml'):
        """Initialize configuration from file or defaults"""
        
    def load_from_file(self, config_file: str) -> None:
        """Load configuration from YAML file"""
        
    def save_to_file(self, config_file: str) -> None:
        """Save current configuration to YAML file"""
        
    def validate_config(self) -> bool:
        """Validate configuration parameters"""
        
    # Properties
    hardware: HardwareConfig
    processing: ProcessingConfig
    simulation: SimulationConfig
```

**HardwareConfig**
```python
@dataclass
class HardwareConfig:
    """Hardware configuration settings"""
    opm_channels: int = 128
    optical_channels: int = 64
    accelerometer_count: int = 12
    sampling_rate: int = 1000
    calibration_enabled: bool = True
    real_time_mode: bool = True
```

**Example Usage**
```python
from brain_forge.core.config import BrainForgeConfig

# Initialize with default settings
config = BrainForgeConfig()

# Load from custom file
config = BrainForgeConfig('my_config.yaml')

# Modify settings
config.hardware.opm_channels = 256
config.processing.compression_quality = 'ultra_high'

# Save configuration
config.save_to_file('updated_config.yaml')
```

---

## Data Acquisition Module

### Hardware Interfaces

#### `brain_forge.acquisition.omp_helmet`

**OPMHelmetInterface**
```python
class OPMHelmetInterface:
    """Interface for Optically Pumped Magnetometer helmet"""
    
    def __init__(self, channels: int = 128, sampling_rate: int = 1000):
        """Initialize OPM helmet interface"""
        
    def initialize_sensors(self) -> bool:
        """Initialize and calibrate OPM sensor array"""
        
    def start_streaming(self) -> None:
        """Begin real-time data streaming"""
        
    def stop_streaming(self) -> None:
        """Stop data streaming"""
        
    def get_sample(self) -> Tuple[np.ndarray, float]:
        """Get single sample with timestamp"""
        
    def stream_data(self, duration: float = None) -> Iterator[Tuple[np.ndarray, float]]:
        """Stream data for specified duration"""
        
    def calibrate_sensors(self) -> Dict[str, Any]:
        """Perform sensor calibration"""
        
    # Properties
    is_streaming: bool
    channel_count: int
    sampling_rate: int
    calibration_status: Dict[str, bool]
```

**Example Usage**
```python
from brain_forge.acquisition.omp_helmet import OPMHelmetInterface

# Initialize OPM helmet
omp = OPMHelmetInterface(channels=128, sampling_rate=1000)

# Calibrate sensors
calibration_result = omp.calibrate_sensors()
print(f"Calibration successful: {calibration_result['success']}")

# Start streaming
omp.start_streaming()

# Collect data for 10 seconds
for data, timestamp in omp.stream_data(duration=10.0):
    print(f"Received {data.shape} at {timestamp}")
    
# Stop streaming
omp.stop_streaming()
```

#### `brain_forge.acquisition.kernel_optical`

**KernelOpticalInterface**
```python
class KernelOpticalInterface:
    """Interface for Kernel optical helmet"""
    
    def __init__(self, wavelengths: List[int] = [650, 750, 850], channels: int = 64):
        """Initialize Kernel optical interface"""
        
    def start_optical_measurement(self) -> None:
        """Begin optical measurements"""
        
    def measure_hemodynamics(self) -> Dict[str, np.ndarray]:
        """Measure blood oxygenation and flow"""
        
    def get_blood_flow_data(self) -> Tuple[np.ndarray, float]:
        """Get real-time blood flow measurements"""
        
    def calibrate_optical_sensors(self) -> bool:
        """Calibrate optical measurement system"""
        
    # Properties
    wavelengths: List[int]
    channel_count: int
    measurement_active: bool
    laser_power: float  # mW
```

**Example Usage**
```python
from brain_forge.acquisition.kernel_optical import KernelOpticalInterface

# Initialize with custom wavelengths
optical = KernelOpticalInterface(wavelengths=[650, 850], channels=32)

# Calibrate optical sensors
if optical.calibrate_optical_sensors():
    print("Optical calibration successful")
    
# Start measurements
optical.start_optical_measurement()

# Get hemodynamic data
hemo_data = optical.measure_hemodynamics()
print(f"HbO2: {hemo_data['hbo2'].shape}")
print(f"Hb: {hemo_data['hb'].shape}")
```

#### `brain_forge.acquisition.accelerometer`

**AccelerometerInterface**
```python
class AccelerometerInterface:
    """Interface for accelerometer array"""
    
    def __init__(self, sensor_count: int = 12, range_g: int = 16):
        """Initialize accelerometer interface"""
        
    def start_motion_tracking(self) -> None:
        """Begin motion tracking"""
        
    def get_motion_data(self) -> Tuple[Dict[str, np.ndarray], float]:
        """Get motion data from all sensors"""
        
    def detect_motion_artifacts(self, neural_data: np.ndarray) -> np.ndarray:
        """Detect motion artifacts in neural data"""
        
    def compensate_motion_artifacts(self, neural_data: np.ndarray) -> np.ndarray:
        """Remove motion artifacts from neural data"""
        
    # Properties
    sensor_count: int
    range_g: int
    tracking_active: bool
```

### Stream Management

#### `brain_forge.acquisition.stream_manager`

**StreamManager**
```python
class StreamManager:
    """Manage multiple data streams with synchronization"""
    
    def __init__(self, config: BrainForgeConfig):
        """Initialize stream manager with configuration"""
        
    async def initialize_all_streams(self) -> Dict[str, bool]:
        """Initialize all hardware streams"""
        
    async def start_synchronized_streaming(self) -> None:
        """Start synchronized data collection"""
        
    async def get_synchronized_sample(self) -> Dict[str, Tuple[np.ndarray, float]]:
        """Get synchronized sample from all streams"""
        
    def add_stream(self, name: str, stream_interface) -> None:
        """Add new data stream"""
        
    def remove_stream(self, name: str) -> None:
        """Remove data stream"""
        
    # Properties
    active_streams: Dict[str, Any]
    synchronization_accuracy: float  # microseconds
    is_streaming: bool
```

**Example Usage**
```python
from brain_forge.acquisition.stream_manager import StreamManager
from brain_forge.core.config import BrainForgeConfig

# Initialize stream manager
config = BrainForgeConfig()
stream_manager = StreamManager(config)

# Initialize all streams
await stream_manager.initialize_all_streams()

# Start synchronized streaming
await stream_manager.start_synchronized_streaming()

# Collect synchronized data
for i in range(100):
    synchronized_data = await stream_manager.get_synchronized_sample()
    print(f"Sample {i}: {synchronized_data.keys()}")
```

---

## Data Processing Module

### Compression

#### `brain_forge.processing.compression`

**NeuralCompressor**
```python
class NeuralCompressor:
    """Advanced neural data compression"""
    
    def __init__(self, algorithm: str = 'neural_lz', quality: str = 'high'):
        """Initialize neural compressor"""
        
    def compress(self, neural_data: np.ndarray) -> Dict[str, Any]:
        """Compress multi-modal neural data"""
        
    def decompress(self, compressed_data: Dict[str, Any]) -> np.ndarray:
        """Decompress neural data"""
        
    def get_compression_ratio(self) -> float:
        """Get achieved compression ratio"""
        
    def set_quality_level(self, quality: str) -> None:
        """Set compression quality: 'low', 'medium', 'high', 'ultra_high'"""
        
    # Properties
    algorithm: str
    quality: str
    compression_ratio: float
```

**Example Usage**
```python
from brain_forge.processing.compression import NeuralCompressor

# Initialize compressor
compressor = NeuralCompressor(algorithm='neural_lz', quality='high')

# Compress data
compressed = compressor.compress(neural_data)
print(f"Compression ratio: {compressed['compression_ratio']:.2f}x")

# Decompress
decompressed = compressor.decompress(compressed)
print(f"Decompressed shape: {decompressed.shape}")
```

### Pattern Recognition

#### `brain_forge.processing.patterns`

**TransformerPatternDetector**
```python
class TransformerPatternDetector:
    """Neural pattern detection using transformer architecture"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize pattern detector"""
        
    def detect_patterns(self, brain_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Detect neural patterns in real-time data"""
        
    def train_model(self, training_data: np.ndarray, labels: np.ndarray) -> None:
        """Train pattern detection model"""
        
    def save_model(self, model_path: str) -> None:
        """Save trained model"""
        
    def load_model(self, model_path: str) -> None:
        """Load pre-trained model"""
        
    # Properties
    model_loaded: bool
    pattern_types: List[str]
```

### Artifact Removal

#### `brain_forge.processing.artifacts`

**MotionArtifactRemover**
```python
class MotionArtifactRemover:
    """Remove motion artifacts from neural data"""
    
    def __init__(self, motion_data: Optional[np.ndarray] = None):
        """Initialize artifact remover"""
        
    def remove_artifacts(self, neural_data: np.ndarray, motion_data: np.ndarray) -> np.ndarray:
        """Remove motion artifacts from neural data"""
        
    def detect_artifacts(self, neural_data: np.ndarray) -> np.ndarray:
        """Detect artifact locations"""
        
    def set_sensitivity(self, sensitivity: float) -> None:
        """Set artifact detection sensitivity (0.0 to 1.0)"""
        
    # Properties
    sensitivity: float
    artifacts_detected: int
```

---

## Brain Simulation Module

### Digital Brain Twin

#### `brain_forge.simulation.brain_twin`

**DigitalBrainTwin**
```python
class DigitalBrainTwin:
    """High-fidelity digital brain simulation"""
    
    def __init__(self, resolution: str = 'high', dynamics: str = 'real_time'):
        """Initialize digital brain twin"""
        
    def initialize_brain_structure(self) -> None:
        """Initialize anatomical brain structure"""
        
    def apply_brain_data(self, brain_data: np.ndarray) -> None:
        """Apply real brain data to simulation"""
        
    def run_simulation(self, duration: float = 1.0) -> Dict[str, np.ndarray]:
        """Run brain simulation"""
        
    def get_simulation_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        
    def save_simulation(self, filepath: str) -> None:
        """Save simulation state"""
        
    def load_simulation(self, filepath: str) -> None:
        """Load simulation state"""
        
    # Properties
    resolution: str
    dynamics: str
    simulation_active: bool
    brain_regions: List[str]
```

**Example Usage**
```python
from brain_forge.simulation.brain_twin import DigitalBrainTwin

# Create brain twin
brain_twin = DigitalBrainTwin(resolution='high', dynamics='real_time')

# Apply real brain data
brain_twin.apply_brain_data(processed_neural_data)

# Run simulation
results = brain_twin.run_simulation(duration=10.0)
print(f"Simulation completed: {results.keys()}")

# Save simulation
brain_twin.save_simulation('my_brain_simulation.pkl')
```

### Transfer Learning

#### `brain_forge.simulation.transfer`

**BrainTransferSystem**
```python
class BrainTransferSystem:
    """Transfer learning for brain pattern mapping"""
    
    def __init__(self, source_resolution: str = 'high', target_resolution: str = 'high'):
        """Initialize brain transfer system"""
        
    def extract_brain_features(self, brain_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract key features from brain data"""
        
    def map_to_simulation(self, source_features: Dict[str, np.ndarray], 
                         target_simulation: DigitalBrainTwin) -> DigitalBrainTwin:
        """Map source brain features to target simulation"""
        
    def validate_transfer(self, source_data: np.ndarray, 
                         transferred_simulation: DigitalBrainTwin) -> Dict[str, float]:
        """Validate transfer learning accuracy"""
        
    # Properties
    feature_extractor_trained: bool
    transfer_accuracy: float
```

---

## Utilities Module

### Visualization

#### `brain_forge.utils.visualization`

**BrainVisualizer**
```python
class BrainVisualizer:
    """Comprehensive brain data visualization"""
    
    def __init__(self, backend: str = 'plotly'):
        """Initialize visualizer"""
        
    def plot_brain_activity(self, brain_data: np.ndarray, 
                           brain_atlas: Optional[Any] = None) -> Any:
        """Plot real-time brain activity"""
        
    def plot_connectivity_matrix(self, connectivity: np.ndarray) -> Any:
        """Plot brain connectivity matrix"""
        
    def plot_simulation_results(self, simulation_data: Dict[str, np.ndarray]) -> Any:
        """Visualize simulation results"""
        
    def create_interactive_brain_plot(self, data: np.ndarray) -> Any:
        """Create interactive 3D brain visualization"""
        
    def export_visualization(self, plot: Any, filename: str, format: str = 'html') -> None:
        """Export visualization to file"""
```

### Performance Optimization

#### `brain_forge.utils.optimization`

**RealTimeOptimizer**
```python
class RealTimeOptimizer:
    """Optimize processing for real-time performance"""
    
    def __init__(self):
        """Initialize optimizer"""
        
    def optimize_processing_pipeline(self, data_stream: np.ndarray) -> np.ndarray:
        """Optimize processing for real-time performance"""
        
    def enable_gpu_acceleration(self) -> bool:
        """Enable GPU acceleration if available"""
        
    def benchmark_performance(self) -> Dict[str, float]:
        """Benchmark system performance"""
        
    # Properties
    gpu_available: bool
    processing_latency: float  # milliseconds
```

---

## Integration Examples

### Complete Brain Scanning Pipeline

```python
import asyncio
from brain_forge.core.config import BrainForgeConfig
from brain_forge.acquisition.stream_manager import StreamManager
from brain_forge.processing.compression import NeuralCompressor
from brain_forge.processing.patterns import TransformerPatternDetector
from brain_forge.simulation.brain_twin import DigitalBrainTwin
from brain_forge.simulation.transfer import BrainTransferSystem

async def complete_brain_scanning_pipeline():
    """Complete brain scanning and simulation pipeline"""
    
    # Initialize configuration
    config = BrainForgeConfig('brain_forge_config.yaml')
    
    # Set up data acquisition
    stream_manager = StreamManager(config)
    await stream_manager.initialize_all_streams()
    
    # Initialize processing components
    compressor = NeuralCompressor(algorithm='neural_lz', quality='high')
    pattern_detector = TransformerPatternDetector()
    
    # Initialize simulation components
    brain_twin = DigitalBrainTwin(resolution='high', dynamics='real_time')
    transfer_system = BrainTransferSystem()
    
    # Start data acquisition
    await stream_manager.start_synchronized_streaming()
    
    # Collect and process data
    collected_data = []
    for i in range(1000):  # 1000 samples
        # Get synchronized data
        sync_data = await stream_manager.get_synchronized_sample()
        
        # Combine multi-modal data
        combined_data = np.concatenate([
            sync_data['omp'][0],
            sync_data['optical'][0].flatten(),
            sync_data['motion'][0].flatten()
        ])
        
        collected_data.append(combined_data)
    
    # Convert to numpy array
    brain_data = np.array(collected_data)
    
    # Compress data
    compressed_data = compressor.compress(brain_data)
    print(f"Compression ratio: {compressed_data['compression_ratio']:.2f}x")
    
    # Detect patterns
    patterns = pattern_detector.detect_patterns(brain_data)
    print(f"Detected patterns: {patterns.keys()}")
    
    # Apply to brain simulation
    brain_twin.apply_brain_data(brain_data)
    simulation_results = brain_twin.run_simulation(duration=10.0)
    
    # Transfer learning
    brain_features = transfer_system.extract_brain_features(brain_data)
    transferred_twin = transfer_system.map_to_simulation(brain_features, brain_twin)
    
    # Validate transfer
    validation_results = transfer_system.validate_transfer(brain_data, transferred_twin)
    print(f"Transfer accuracy: {validation_results['overall_score']:.2f}")
    
    return {
        'raw_data': brain_data,
        'compressed_data': compressed_data,
        'patterns': patterns,
        'simulation_results': simulation_results,
        'validation': validation_results
    }

# Run the pipeline
if __name__ == "__main__":
    results = asyncio.run(complete_brain_scanning_pipeline())
    print("Brain scanning pipeline completed successfully!")
```

### Real-time Brain Monitoring

```python
from brain_forge.acquisition.omp_helmet import OPMHelmetInterface
from brain_forge.processing.artifacts import MotionArtifactRemover
from brain_forge.utils.visualization import BrainVisualizer

def real_time_brain_monitoring():
    """Real-time brain activity monitoring"""
    
    # Initialize components
    omp = OPMHelmetInterface(channels=128, sampling_rate=1000)
    artifact_remover = MotionArtifactRemover()
    visualizer = BrainVisualizer(backend='plotly')
    
    # Calibrate and start streaming
    omp.calibrate_sensors()
    omp.start_streaming()
    
    try:
        # Real-time monitoring loop
        for data, timestamp in omp.stream_data():
            # Remove artifacts
            clean_data = artifact_remover.remove_artifacts(data, motion_data)
            
            # Real-time visualization
            plot = visualizer.plot_brain_activity(clean_data)
            
            # Update display (pseudo-code)
            update_display(plot)
            
            # Process for 60 seconds
            if timestamp - start_time > 60.0:
                break
                
    finally:
        omp.stop_streaming()

# Run monitoring
real_time_brain_monitoring()
```

---

## Error Handling

### Exception Classes

```python
from brain_forge.core.exceptions import (
    BrainForgeError,
    HardwareError,
    ProcessingError,
    SimulationError,
    ConfigurationError
)

# Example error handling
try:
    omp = OPMHelmetInterface(channels=128)
    omp.initialize_sensors()
except HardwareError as e:
    print(f"Hardware initialization failed: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except BrainForgeError as e:
    print(f"General Brain-Forge error: {e}")
```

---

## Configuration Files

### Sample Configuration (config.yaml)

```yaml
# Brain-Forge Configuration File
hardware:
  omp_channels: 128
  optical_channels: 64
  accelerometer_count: 12
  sampling_rate: 1000
  calibration_enabled: true
  real_time_mode: true

processing:
  compression_algorithm: "neural_lz"
  compression_quality: "high"
  artifact_removal: true
  real_time_threshold: 0.001
  pattern_detection: true

simulation:
  resolution: "high"
  dynamics: "real_time"
  validation_enabled: true
  transfer_learning: true
  save_intermediate_results: false

logging:
  level: "INFO"
  file: "brain_forge.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

privacy:
  encryption_enabled: true
  local_processing_only: true
  data_anonymization: true
  consent_required: true
```

---

This API reference provides comprehensive documentation for all Brain-Forge modules and functions. For additional examples and tutorials, see the guides section of the documentation.
