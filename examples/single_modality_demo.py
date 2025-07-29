#!/usr/bin/env python3
"""
Brain-Forge Single Modality Demo - Kernel Flow2 Focus

This demo demonstrates focused development with a single modality approach,
as recommended for incremental development. Starting with Kernel Flow2
optical brain imaging for more achievable initial targets.

Key Features Demonstrated:
- Single modality focus (Kernel Flow2)
- Realistic performance targets (<500ms latency)
- Mock hardware interfaces for development
- Motor imagery BCI application
- Conservative compression ratios (1.5-3x)
"""

import asyncio
import sys
from pathlib import Path
from time import sleep, time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import signal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.config import BrainForgeConfig
from core.logger import get_logger

logger = get_logger(__name__)


class MockKernelFlow2Interface:
    """Mock interface for Kernel Flow2 helmet for development without hardware"""
    
    def __init__(self, n_channels: int = 32, sampling_rate: float = 100.0):
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.is_connected = False
        self.is_streaming = False
        
    def initialize(self) -> bool:
        """Initialize mock hardware connection"""
        try:
            logger.info("Initializing Mock Kernel Flow2 helmet...")
            sleep(0.5)  # Simulate initialization time
            self.is_connected = True
            logger.info(f"Mock Kernel Flow2 initialized: {self.n_channels} channels @ {self.sampling_rate} Hz")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Mock Kernel Flow2: {e}")
            return False
            
    def start_acquisition(self) -> None:
        """Start mock data acquisition"""
        if not self.is_connected:
            raise RuntimeError("Hardware not initialized")
        self.is_streaming = True
        logger.info("Mock Kernel Flow2 streaming started")
        
    def stop_acquisition(self) -> None:
        """Stop mock data acquisition"""
        self.is_streaming = False
        logger.info("Mock Kernel Flow2 streaming stopped")
        
    def get_data_stream(self) -> np.ndarray:
        """Generate realistic fNIRS signals for motor imagery"""
        if not self.is_streaming:
            return np.zeros((self.n_channels, 1))
            
        # Generate realistic hemodynamic response patterns
        t = np.linspace(0, 1.0/self.sampling_rate, 1)
        
        # Simulate motor cortex activation (channels 1-8)
        motor_channels = np.zeros((8, len(t)))
        for i in range(8):
            # Hemodynamic response function
            hrf = self._generate_hrf(t) * (0.5 + 0.3 * np.random.random())
            motor_channels[i] = hrf + 0.1 * np.random.randn(len(t))
            
        # Simulate other brain regions (channels 9-32)
        other_channels = 0.05 * np.random.randn(24, len(t))
        
        # Combine all channels
        data = np.vstack([motor_channels, other_channels])
        
        return data
        
    def _generate_hrf(self, t: np.ndarray) -> np.ndarray:
        """Generate hemodynamic response function"""
        # Simplified HRF: gamma function convolution
        a1, a2 = 6, 16
        b1, b2 = 1, 1
        
        hrf = (t**(a1-1) * np.exp(-t/b1) / (b1**a1 * np.math.gamma(a1)) - 
               0.35 * t**(a2-1) * np.exp(-t/b2) / (b2**a2 * np.math.gamma(a2)))
        
        return hrf


class RealisticSignalProcessor:
    """Signal processor with conservative, achievable targets"""
    
    def __init__(self, sampling_rate: float = 100.0):
        self.sampling_rate = sampling_rate
        self.target_latency = 0.5  # 500ms - realistic target
        self.compression_ratio = 2.0  # Conservative 2x compression
        
    def apply_bandpass_filter(self, data: np.ndarray, 
                            low_freq: float = 0.01, 
                            high_freq: float = 0.5) -> np.ndarray:
        """Apply bandpass filter optimized for fNIRS hemodynamic signals"""
        start_time = time()
        
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # 4th order Butterworth filter
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=1)
        
        processing_time = time() - start_time
        logger.info(f"Bandpass filtering completed in {processing_time*1000:.1f}ms")
        
        return filtered_data
        
    def detect_motor_imagery(self, data: np.ndarray) -> Dict[str, float]:
        """Detect motor imagery patterns in fNIRS data"""
        start_time = time()
        
        # Focus on motor cortex channels (1-8)
        motor_data = data[:8, :]
        
        # Calculate relative signal changes
        baseline = np.mean(motor_data[:, :int(self.sampling_rate)], axis=1)
        activation = np.mean(motor_data[:, -int(self.sampling_rate):], axis=1)
        
        relative_change = (activation - baseline) / baseline
        
        # Simple threshold-based detection
        motor_activation = np.mean(relative_change[relative_change > 0])
        confidence = min(1.0, motor_activation / 0.05)  # Normalized confidence
        
        processing_time = time() - start_time
        
        results = {
            'motor_activation': float(motor_activation),
            'confidence': float(confidence),
            'processing_time_ms': processing_time * 1000,
            'channels_active': int(np.sum(relative_change > 0.02))
        }
        
        logger.info(f"Motor imagery detection: {confidence:.2f} confidence in {processing_time*1000:.1f}ms")
        
        return results
        
    def compress_data(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """Conservative data compression with 1.5-3x ratio"""
        start_time = time()
        
        # Simple downsampling compression
        downsample_factor = 2
        compressed_data = data[:, ::downsample_factor]
        
        original_size = data.nbytes
        compressed_size = compressed_data.nbytes
        compression_ratio = original_size / compressed_size
        
        processing_time = time() - start_time
        logger.info(f"Data compressed {compression_ratio:.1f}x in {processing_time*1000:.1f}ms")
        
        return compressed_data, compression_ratio


class MotorImageryBCIDemo:
    """Focused BCI demo for motor imagery detection - specific clinical application"""
    
    def __init__(self):
        self.config = BrainForgeConfig()
        self.kernel_interface = MockKernelFlow2Interface()
        self.processor = RealisticSignalProcessor()
        self.is_running = False
        
        # Performance tracking
        self.latency_history = []
        self.accuracy_history = []
        
    async def initialize_system(self) -> bool:
        """Initialize the BCI system"""
        logger.info("=== Initializing Motor Imagery BCI System ===")
        
        # Initialize hardware (mock)
        if not self.kernel_interface.initialize():
            logger.error("Failed to initialize Kernel Flow2 interface")
            return False
            
        logger.info("✓ Single modality system initialized successfully")
        logger.info("✓ Realistic performance targets: <500ms latency, 2x compression")
        logger.info("✓ Focus application: Motor imagery detection")
        
        return True
        
    async def run_motor_imagery_session(self, duration: float = 30.0) -> Dict:
        """Run a motor imagery detection session"""
        logger.info(f"=== Starting Motor Imagery Session ({duration}s) ===")
        
        if not await self.initialize_system():
            return {}
            
        self.kernel_interface.start_acquisition()
        self.is_running = True
        
        session_data = {
            'detections': [],
            'latencies': [],
            'compressions': [],
            'total_duration': duration
        }
        
        start_time = time()
        
        try:
            while time() - start_time < duration and self.is_running:
                # Acquire data (realistic 1-second windows)
                data_window = []
                for _ in range(int(self.processor.sampling_rate)):
                    sample = self.kernel_interface.get_data_stream()
                    data_window.append(sample)
                    await asyncio.sleep(1.0 / self.processor.sampling_rate)
                
                data_window = np.concatenate(data_window, axis=1)
                
                # Process data with realistic latency
                filtered_data = self.processor.apply_bandpass_filter(data_window)
                detection_result = self.processor.detect_motor_imagery(filtered_data)
                compressed_data, compression_ratio = self.processor.compress_data(filtered_data)
                
                # Track performance
                session_data['detections'].append(detection_result)
                session_data['latencies'].append(detection_result['processing_time_ms'])
                session_data['compressions'].append(compression_ratio)
                
                # Real-time feedback
                if detection_result['confidence'] > 0.7:
                    logger.info(f"🧠 Motor imagery detected! Confidence: {detection_result['confidence']:.2f}")
                    
                # Check latency target
                if detection_result['processing_time_ms'] > 500:
                    logger.warning(f"⚠️  Latency exceeded target: {detection_result['processing_time_ms']:.1f}ms")
                    
        except KeyboardInterrupt:
            logger.info("Session interrupted by user")
        finally:
            self.kernel_interface.stop_acquisition()
            self.is_running = False
            
        logger.info("=== Motor Imagery Session Complete ===")
        return session_data
        
    def analyze_session_performance(self, session_data: Dict) -> None:
        """Analyze session performance with realistic metrics"""
        if not session_data or not session_data['detections']:
            logger.warning("No session data to analyze")
            return
            
        detections = session_data['detections']
        latencies = session_data['latencies']
        compressions = session_data['compressions']
        
        # Calculate performance metrics
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        avg_compression = np.mean(compressions)
        
        detection_rate = len([d for d in detections if d['confidence'] > 0.5]) / len(detections)
        high_confidence_rate = len([d for d in detections if d['confidence'] > 0.7]) / len(detections)
        
        # Performance analysis
        logger.info("=== Session Performance Analysis ===")
        logger.info(f"Average Processing Latency: {avg_latency:.1f}ms (Target: <500ms)")
        logger.info(f"Maximum Processing Latency: {max_latency:.1f}ms")
        logger.info(f"Average Compression Ratio: {avg_compression:.1f}x (Target: 1.5-3x)")
        logger.info(f"Motor Imagery Detection Rate: {detection_rate:.1%}")
        logger.info(f"High Confidence Detection Rate: {high_confidence_rate:.1%}")
        
        # Performance targets assessment
        latency_achieved = avg_latency < 500
        compression_achieved = 1.5 <= avg_compression <= 3.0
        detection_achieved = detection_rate > 0.6
        
        logger.info("=== Target Achievement ===")
        logger.info(f"✓ Latency Target (<500ms): {'ACHIEVED' if latency_achieved else 'MISSED'}")
        logger.info(f"✓ Compression Target (1.5-3x): {'ACHIEVED' if compression_achieved else 'MISSED'}")
        logger.info(f"✓ Detection Target (>60%): {'ACHIEVED' if detection_achieved else 'MISSED'}")
        
    def visualize_session_results(self, session_data: Dict) -> None:
        """Create visualizations for session analysis"""
        if not session_data or not session_data['detections']:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Brain-Forge Single Modality BCI - Motor Imagery Session Analysis', fontsize=14)
        
        detections = session_data['detections']
        latencies = session_data['latencies']
        compressions = session_data['compressions']
        confidences = [d['confidence'] for d in detections]
        
        # Latency over time
        axes[0, 0].plot(latencies, 'b-', alpha=0.7)
        axes[0, 0].axhline(y=500, color='r', linestyle='--', label='Target (500ms)')
        axes[0, 0].set_title('Processing Latency Over Time')
        axes[0, 0].set_ylabel('Latency (ms)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Confidence distribution
        axes[0, 1].hist(confidences, bins=20, alpha=0.7, color='green')
        axes[0, 1].axvline(x=0.7, color='r', linestyle='--', label='High Confidence')
        axes[0, 1].set_title('Detection Confidence Distribution')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Compression ratios
        axes[1, 0].plot(compressions, 'g-', alpha=0.7)
        axes[1, 0].axhline(y=1.5, color='r', linestyle='--', label='Min Target')
        axes[1, 0].axhline(y=3.0, color='r', linestyle='--', label='Max Target')
        axes[1, 0].set_title('Compression Ratio Over Time')
        axes[1, 0].set_ylabel('Compression Ratio')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance summary
        avg_latency = np.mean(latencies)
        avg_compression = np.mean(compressions)
        detection_rate = len([c for c in confidences if c > 0.5]) / len(confidences)
        
        summary_text = f"""Performance Summary:
        
        Average Latency: {avg_latency:.1f}ms
        Target: <500ms ({'✓' if avg_latency < 500 else '✗'})
        
        Average Compression: {avg_compression:.1f}x
        Target: 1.5-3x ({'✓' if 1.5 <= avg_compression <= 3.0 else '✗'})
        
        Detection Rate: {detection_rate:.1%}
        Target: >60% ({'✓' if detection_rate > 0.6 else '✗'})
        
        System Status: {'OPERATIONAL' if all([avg_latency < 500, 1.5 <= avg_compression <= 3.0, detection_rate > 0.6]) else 'OPTIMIZATION NEEDED'}
        """
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=10)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()


async def main():
    """Main demo function showcasing single modality approach"""
    logger.info("=== Brain-Forge Single Modality Demo ===")
    logger.info("Focus: Kernel Flow2 optical brain imaging")
    logger.info("Application: Motor imagery BCI")
    logger.info("Targets: <500ms latency, 1.5-3x compression, >60% detection")
    
    # Create BCI demo instance
    bci_demo = MotorImageryBCIDemo()
    
    try:
        # Run a 30-second motor imagery session
        logger.info("\n🚀 Starting demonstration...")
        session_data = await bci_demo.run_motor_imagery_session(duration=30.0)
        
        # Analyze performance
        bci_demo.analyze_session_performance(session_data)
        
        # Create visualizations
        bci_demo.visualize_session_results(session_data)
        
        logger.info("\n✓ Single modality demo completed successfully!")
        logger.info("Next steps: Validate with real hardware, then add second modality")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
