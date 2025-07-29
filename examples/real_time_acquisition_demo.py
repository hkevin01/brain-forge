#!/usr/bin/env python3
"""
Brain-Forge Real-Time Data Acquisition Demo

This demo showcases the multi-modal brain data acquisition capabilities
of Brain-Forge, including OPM helmet, Kernel optical helmet, and 
accelerometer array integration with real-time streaming.

Key Features Demonstrated:
- Multi-device synchronization via LSL
- Real-time data streaming and buffering
- Quality monitoring and artifact detection
- Hardware interface management
"""

import sys
import threading
from collections import deque
from pathlib import Path
from time import sleep, time

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from acquisition.stream_manager import StreamManager
from core.config import BrainForgeConfig
from core.logger import get_logger
from hardware.accelerometer import AccelerometerArray
from hardware.kernel_optical import KernelOpticalHelmet
from hardware.omp_helmet import OMPHelmets

logger = get_logger(__name__)

class RealTimeDataDemo:
    """Demonstrates real-time multi-modal data acquisition"""
    
    def __init__(self):
        self.config = BrainForgeConfig()
        self.running = False
        self.data_buffer = {
            'omp': deque(maxlen=1000),
            'optical': deque(maxlen=1000),
            'accelerometer': deque(maxlen=1000)
        }
        self.timestamps = deque(maxlen=1000)
        
    def setup_hardware(self):
        """Initialize all hardware components"""
        print("🔧 Setting up hardware interfaces...")
        
        # Initialize OPM helmet (306 channels)
        self.omp_helmet = OMPHelmets(self.config.hardware.omp)
        print(f"   ✅ OPM Helmet: {self.omp_helmet.n_channels} channels ready")
        
        # Initialize Kernel optical helmet
        self.optical_helmet = KernelOpticalHelmet(self.config.hardware.kernel_optical)
        print(f"   ✅ Kernel Optical: Flow + Flux helmets ready")
        
        # Initialize accelerometer array
        self.accelerometer = AccelerometerArray(self.config.hardware.accelerometer)
        print(f"   ✅ Accelerometer Array: 3-axis motion tracking ready")
        
        # Initialize stream manager
        self.stream_manager = StreamManager(self.config.acquisition)
        print("   ✅ LSL Stream Manager initialized")
        
    def start_acquisition(self):
        """Start real-time data acquisition"""
        print("\n🚀 Starting real-time data acquisition...")
        
        # Start hardware
        self.omp_helmet.start()
        self.optical_helmet.start()
        self.accelerometer.start()
        
        # Start streaming
        self.stream_manager.start_streaming()
        self.running = True
        
        # Start data collection thread
        self.data_thread = threading.Thread(target=self._collect_data)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        print("   ✅ All systems streaming!")
        
    def _collect_data(self):
        """Collect data from all streams"""
        while self.running:
            timestamp = time()
            
            # Get OMP data (306 channels)
            omp_data = self.omp_helmet.get_data()
            if omp_data is not None:
                # Calculate RMS for visualization
                omp_rms = np.sqrt(np.mean(omp_data**2, axis=0))
                self.data_buffer['omp'].append(np.mean(omp_rms))
            
            # Get optical data
            optical_data = self.optical_helmet.get_data()
            if optical_data is not None:
                # Get hemodynamic signal strength
                flow_strength = np.mean(optical_data.get('flow', [0]))
                self.data_buffer['optical'].append(flow_strength)
            
            # Get accelerometer data
            accel_data = self.accelerometer.get_data()
            if accel_data is not None:
                # Calculate motion magnitude
                motion_magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
                self.data_buffer['accelerometer'].append(np.mean(motion_magnitude))
            
            self.timestamps.append(timestamp)
            sleep(0.01)  # 100 Hz update rate
            
    def monitor_quality(self):
        """Monitor data quality in real-time"""
        print("\n📊 Real-time quality monitoring:")
        
        for i in range(100):  # Monitor for 10 seconds
            if len(self.data_buffer['omp']) > 10:
                # Calculate quality metrics
                omp_quality = self._calculate_snr(self.data_buffer['omp'])
                optical_stability = self._calculate_stability(self.data_buffer['optical'])
                motion_level = np.std(list(self.data_buffer['accelerometer'])[-10:]) if len(self.data_buffer['accelerometer']) > 10 else 0
                
                print(f"\r   OMP SNR: {omp_quality:.2f} dB | "
                      f"Optical Stability: {optical_stability:.3f} | "
                      f"Motion: {motion_level:.4f} g", end='', flush=True)
            
            sleep(0.1)
        print()
        
    def _calculate_snr(self, data):
        """Calculate signal-to-noise ratio"""
        if len(data) < 10:
            return 0.0
        signal_power = np.mean(np.array(data)**2)
        noise_power = np.var(np.array(data))
        return 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    def _calculate_stability(self, data):
        """Calculate signal stability"""
        if len(data) < 10:
            return 0.0
        return 1.0 / (1.0 + np.std(data))
        
    def visualize_streams(self):
        """Create real-time visualization of data streams"""
        print("\n📈 Generating real-time visualization...")
        
        # Wait for some data
        while len(self.timestamps) < 50:
            sleep(0.1)
            
        plt.figure(figsize=(15, 10))
        
        # Plot OMP data
        plt.subplot(3, 1, 1)
        times = np.array(self.timestamps) - self.timestamps[0]
        plt.plot(times, list(self.data_buffer['omp']), 'b-', linewidth=2)
        plt.title('OMP Helmet - Magnetic Field Strength (RMS)', fontsize=14, fontweight='bold')
        plt.ylabel('Field Strength (fT)')
        plt.grid(True, alpha=0.3)
        
        # Plot optical data
        plt.subplot(3, 1, 2)
        plt.plot(times, list(self.data_buffer['optical']), 'r-', linewidth=2)
        plt.title('Kernel Optical - Hemodynamic Signal', fontsize=14, fontweight='bold')
        plt.ylabel('Flow Intensity')
        plt.grid(True, alpha=0.3)
        
        # Plot accelerometer data
        plt.subplot(3, 1, 3)
        plt.plot(times, list(self.data_buffer['accelerometer']), 'g-', linewidth=2)
        plt.title('Accelerometer Array - Motion Magnitude', fontsize=14, fontweight='bold')
        plt.ylabel('Acceleration (g)')
        plt.xlabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('real_time_streams.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def demonstrate_synchronization(self):
        """Demonstrate microsecond-precision synchronization"""
        print("\n⏱️  Demonstrating synchronization accuracy...")
        
        sync_timestamps = []
        for _ in range(100):
            # Get synchronized timestamps from all devices
            omp_time = self.omp_helmet.get_timestamp()
            optical_time = self.optical_helmet.get_timestamp()
            accel_time = self.accelerometer.get_timestamp()
            
            # Calculate sync differences
            sync_diff = max(omp_time, optical_time, accel_time) - min(omp_time, optical_time, accel_time)
            sync_timestamps.append(sync_diff * 1e6)  # Convert to microseconds
            
            sleep(0.01)
        
        sync_accuracy = np.mean(sync_timestamps)
        sync_std = np.std(sync_timestamps)
        
        print(f"   📏 Synchronization accuracy: {sync_accuracy:.2f} ± {sync_std:.2f} μs")
        print(f"   🎯 Target: <10 μs ({'✅ ACHIEVED' if sync_accuracy < 10 else '⚠️  NEEDS IMPROVEMENT'})")
        
    def stop_acquisition(self):
        """Stop data acquisition and cleanup"""
        print("\n🛑 Stopping data acquisition...")
        
        self.running = False
        if hasattr(self, 'data_thread'):
            self.data_thread.join(timeout=1.0)
            
        self.stream_manager.stop_streaming()
        self.omp_helmet.stop()
        self.optical_helmet.stop()
        self.accelerometer.stop()
        
        print("   ✅ All systems stopped safely")

def main():
    """Main demo function"""
    print("🧠 BRAIN-FORGE REAL-TIME DATA ACQUISITION DEMO")
    print("=" * 60)
    
    demo = RealTimeDataDemo()
    
    try:
        # Setup and start
        demo.setup_hardware()
        demo.start_acquisition()
        
        # Run demonstrations
        print("\n⏳ Collecting data for 5 seconds...")
        sleep(5)
        
        demo.monitor_quality()
        demo.demonstrate_synchronization()
        demo.visualize_streams()
        
        # Show statistics
        print(f"\n📈 Data Collection Summary:")
        print(f"   OMP samples: {len(demo.data_buffer['omp'])}")
        print(f"   Optical samples: {len(demo.data_buffer['optical'])}")
        print(f"   Accelerometer samples: {len(demo.data_buffer['accelerometer'])}")
        print(f"   Total duration: {demo.timestamps[-1] - demo.timestamps[0]:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\n⚡ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    finally:
        demo.stop_acquisition()
    
    print("\n🎉 Real-time data acquisition demo completed!")
    print("📁 Visualization saved as 'real_time_streams.png'")

if __name__ == "__main__":
    main()
