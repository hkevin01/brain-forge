#!/usr/bin/env python3
"""
Hardware Calibration Script for Brain-Forge Platform

This script performs calibration and validation of hardware components
in the Brain-Forge brain-computer interface system, including OPM helmets,
Kernel optical helmets, and accelerometer arrays.

Usage:
    python scripts/calibrate_hardware.py --device=all
    python scripts/calibrate_hardware.py --device=omp --output=calibration.json
    python scripts/calibrate_hardware.py --device=kernel --quick
"""

import argparse
import json
import time
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Hardware simulation constants
DEVICE_CONFIGS = {
    'omp': {
        'name': 'OPM Helmet',
        'channels': 306,
        'sampling_rate': 1000,
        'noise_floor_ft': 5e-15,  # Tesla
        'bandwidth_hz': [0.1, 400],
        'calibration_duration_s': 30
    },
    'kernel': {
        'name': 'Kernel Optical Helmet',
        'channels': 52,
        'sampling_rate': 100,
        'noise_floor_od': 1e-6,  # Optical density
        'bandwidth_hz': [0.01, 50],
        'calibration_duration_s': 60
    },
    'accelerometer': {
        'name': 'Accelerometer Array',
        'channels': 3,
        'sampling_rate': 1000,
        'noise_floor_g': 1e-6,  # g-force units
        'bandwidth_hz': [0.1, 500],
        'calibration_duration_s': 10
    }
}


@dataclass
class CalibrationResult:
    """Data class for storing calibration results."""
    device_name: str
    channel_id: int
    offset: float
    gain: float
    noise_level: float
    snr_db: float
    frequency_response: Dict[str, float]
    success: bool = True
    error_message: Optional[str] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class DeviceCalibration:
    """Complete calibration data for a device."""
    device_type: str
    device_name: str
    serial_number: str
    calibration_date: str
    channel_results: List[CalibrationResult]
    overall_performance: Dict[str, float]
    success: bool = True
    notes: str = ""


class HardwareCalibrator:
    """Hardware calibration system for Brain-Forge devices."""
    
    def __init__(self, output_file: str = "calibration_results.json",
                 quick_mode: bool = False, simulation_mode: bool = True):
        self.output_file = output_file
        self.quick_mode = quick_mode
        self.simulation_mode = simulation_mode
        self.calibration_results: List[DeviceCalibration] = []
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create calibration data directory
        self.data_dir = project_root / "data" / "calibration_files"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def _generate_test_signal(self, duration_s: float, fs: int,
                              signal_type: str = 'noise') -> np.ndarray:
        """Generate test signals for calibration."""
        n_samples = int(duration_s * fs)
        t = np.linspace(0, duration_s, n_samples)
        
        if signal_type == 'noise':
            # White noise signal
            return np.random.randn(n_samples)
        elif signal_type == 'sine':
            # Multi-frequency sine wave test signal
            frequencies = [1, 10, 40, 100]  # Hz
            signal = np.zeros(n_samples)
            for freq in frequencies:
                if freq < fs / 2:  # Nyquist limit
                    signal += np.sin(2 * np.pi * freq * t) / len(frequencies)
            return signal
        elif signal_type == 'chirp':
            # Frequency sweep from 0.1 Hz to fs/2
            from scipy.signal import chirp
            return chirp(t, 0.1, duration_s, fs/2, method='logarithmic')
        else:
            return np.zeros(n_samples)
    
    def _analyze_signal_quality(self, signal: np.ndarray, fs: int,
                                expected_noise: float) -> Dict[str, float]:
        """Analyze signal quality metrics."""
        # Calculate noise floor
        noise_level = np.std(signal)
        
        # Calculate SNR (assuming signal contains both signal and noise)
        signal_power = np.mean(signal ** 2)
        noise_power = expected_noise ** 2
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = 60
        
        # Frequency domain analysis
        from scipy import signal as sp_signal
        frequencies, psd = sp_signal.welch(signal, fs, nperseg=1024)
        
        # Find peak frequency and power
        peak_idx = np.argmax(psd)
        peak_frequency = frequencies[peak_idx]
        peak_power = psd[peak_idx]
        
        # Calculate bandwidth at -3dB
        half_max = peak_power / 2
        indices = np.where(psd >= half_max)[0]
        if len(indices) > 1:
            bandwidth = frequencies[indices[-1]] - frequencies[indices[0]]
        else:
            bandwidth = fs / 2  # Full bandwidth if no clear peak
        
        return {
            'noise_level': noise_level,
            'snr_db': snr_db,
            'peak_frequency_hz': peak_frequency,
            'peak_power_db': 10 * np.log10(peak_power),
            'bandwidth_hz': bandwidth,
            'total_power_db': 10 * np.log10(np.mean(psd))
        }
    
    def calibrate_omp_helmet(self,
                             serial_number: str = "OMP-SIM-001"
                             ) -> DeviceCalibration:
        """Calibrate OPM helmet sensors."""
        self.logger.info("Starting OPM helmet calibration...")
        
        config = DEVICE_CONFIGS['omp']
        duration = config['calibration_duration_s']
        if self.quick_mode:
            duration = min(duration, 10)  # Quick mode: max 10 seconds
        
        channel_results = []
        
        for channel in range(config['channels']):
            if self.quick_mode and channel > 20:  # Quick mode: only first 20 channels
                break
                
            self.logger.info(f"Calibrating OMP channel {channel + 1}/{config['channels']}")
            
            try:
                # Generate test magnetic field signal
                test_signal = self._generate_test_signal(
                    duration, config['sampling_rate'], 'sine'
                )
                
                # Add realistic noise
                noise = np.random.randn(len(test_signal)) * config['noise_floor_ft']
                measured_signal = test_signal + noise
                
                # Analyze signal quality
                quality_metrics = self._analyze_signal_quality(
                    measured_signal, config['sampling_rate'], config['noise_floor_ft']
                )
                
                # Calculate calibration parameters
                # Offset calibration (DC component)
                offset = np.mean(measured_signal)
                
                # Gain calibration (response to known signal)
                expected_amplitude = 1.0  # Tesla
                measured_amplitude = np.std(test_signal)
                gain = expected_amplitude / measured_amplitude if measured_amplitude > 0 else 1.0
                
                # Frequency response (simplified)
                freq_response = {
                    'dc_response': 1.0,
                    'low_freq_3db': 0.1,  # Hz
                    'high_freq_3db': 400.0,  # Hz
                    'phase_delay_ms': 1.0
                }
                
                result = CalibrationResult(
                    device_name=config['name'],
                    channel_id=channel,
                    offset=offset,
                    gain=gain,
                    noise_level=quality_metrics['noise_level'],
                    snr_db=quality_metrics['snr_db'],
                    frequency_response=freq_response,
                    success=True
                )
                
                channel_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to calibrate OMP channel {channel}: {e}")
                error_result = CalibrationResult(
                    device_name=config['name'],
                    channel_id=channel,
                    offset=0.0,
                    gain=1.0,
                    noise_level=float('inf'),
                    snr_db=-60.0,
                    frequency_response={},
                    success=False,
                    error_message=str(e)
                )
                channel_results.append(error_result)
        
        # Calculate overall performance metrics
        successful_channels = [r for r in channel_results if r.success]
        if successful_channels:
            avg_snr = np.mean([r.snr_db for r in successful_channels])
            avg_noise = np.mean([r.noise_level for r in successful_channels])
            offset_std = np.std([r.offset for r in successful_channels])
            gain_std = np.std([r.gain for r in successful_channels])
        else:
            avg_snr = -60.0
            avg_noise = float('inf')
            offset_std = float('inf')
            gain_std = float('inf')
        
        overall_performance = {
            'average_snr_db': avg_snr,
            'average_noise_level': avg_noise,
            'offset_stability': offset_std,
            'gain_uniformity': gain_std,
            'successful_channels': len(successful_channels),
            'total_channels': len(channel_results),
            'success_rate': len(successful_channels) / len(channel_results) * 100
        }
        
        device_calibration = DeviceCalibration(
            device_type='omp',
            device_name=config['name'],
            serial_number=serial_number,
            calibration_date=datetime.now().isoformat(),
            channel_results=channel_results,
            overall_performance=overall_performance,
            success=len(successful_channels) > len(channel_results) * 0.8,  # 80% success
            notes=f"Calibrated {len(successful_channels)}/{len(channel_results)} channels successfully"
        )
        
        self.logger.info(f"OMP calibration complete: {len(successful_channels)}/{len(channel_results)} channels successful")
        return device_calibration
    
    def calibrate_kernel_optical(self, serial_number: str = "KERNEL-SIM-001"
                                ) -> DeviceCalibration:
        """Calibrate Kernel optical helmet sensors."""
        self.logger.info("Starting Kernel optical helmet calibration...")
        
        config = DEVICE_CONFIGS['kernel']
        duration = config['calibration_duration_s']
        if self.quick_mode:
            duration = min(duration, 15)  # Quick mode: max 15 seconds
        
        channel_results = []
        
        for channel in range(config['channels']):
            self.logger.info(f"Calibrating Kernel channel {channel + 1}/{config['channels']}")
            
            try:
                # Generate test optical signal (blood flow simulation)
                test_signal = self._generate_test_signal(
                    duration, config['sampling_rate'], 'sine'
                )
                
                # Add realistic optical noise
                noise = np.random.randn(len(test_signal)) * config['noise_floor_od']
                measured_signal = test_signal * 0.01 + noise  # Scale for optical density
                
                # Analyze signal quality
                quality_metrics = self._analyze_signal_quality(
                    measured_signal, config['sampling_rate'], config['noise_floor_od']
                )
                
                # Calculate calibration parameters
                offset = np.mean(measured_signal)
                
                # Gain calibration for optical signals
                expected_amplitude = 0.01  # Optical density units
                measured_amplitude = np.std(test_signal * 0.01)
                gain = expected_amplitude / measured_amplitude if measured_amplitude > 0 else 1.0
                
                # Frequency response for optical signals
                freq_response = {
                    'dc_response': 0.95,  # Slightly reduced DC response
                    'low_freq_3db': 0.01,  # Hz
                    'high_freq_3db': 50.0,  # Hz
                    'phase_delay_ms': 10.0  # Higher delay for optical
                }
                
                result = CalibrationResult(
                    device_name=config['name'],
                    channel_id=channel,
                    offset=offset,
                    gain=gain,
                    noise_level=quality_metrics['noise_level'],
                    snr_db=quality_metrics['snr_db'],
                    frequency_response=freq_response,
                    success=True
                )
                
                channel_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to calibrate Kernel channel {channel}: {e}")
                error_result = CalibrationResult(
                    device_name=config['name'],
                    channel_id=channel,
                    offset=0.0,
                    gain=1.0,
                    noise_level=float('inf'),
                    snr_db=-60.0,
                    frequency_response={},
                    success=False,
                    error_message=str(e)
                )
                channel_results.append(error_result)
        
        # Calculate overall performance
        successful_channels = [r for r in channel_results if r.success]
        if successful_channels:
            avg_snr = np.mean([r.snr_db for r in successful_channels])
            avg_noise = np.mean([r.noise_level for r in successful_channels])
            offset_std = np.std([r.offset for r in successful_channels])
            gain_std = np.std([r.gain for r in successful_channels])
        else:
            avg_snr = -60.0
            avg_noise = float('inf')
            offset_std = float('inf')
            gain_std = float('inf')
        
        overall_performance = {
            'average_snr_db': avg_snr,
            'average_noise_level': avg_noise,
            'offset_stability': offset_std,
            'gain_uniformity': gain_std,
            'successful_channels': len(successful_channels),
            'total_channels': len(channel_results),
            'success_rate': len(successful_channels) / len(channel_results) * 100
        }
        
        device_calibration = DeviceCalibration(
            device_type='kernel',
            device_name=config['name'],
            serial_number=serial_number,
            calibration_date=datetime.now().isoformat(),
            channel_results=channel_results,
            overall_performance=overall_performance,
            success=len(successful_channels) == len(channel_results),  # All channels must work
            notes=f"Optical calibration: {len(successful_channels)}/{len(channel_results)} channels"
        )
        
        self.logger.info(f"Kernel calibration complete: {len(successful_channels)}/{len(channel_results)} channels successful")
        return device_calibration
    
    def calibrate_accelerometer(self, serial_number: str = "ACCEL-SIM-001"
                               ) -> DeviceCalibration:
        """Calibrate accelerometer array."""
        self.logger.info("Starting accelerometer calibration...")
        
        config = DEVICE_CONFIGS['accelerometer']
        duration = config['calibration_duration_s']
        if self.quick_mode:
            duration = min(duration, 5)  # Quick mode: max 5 seconds
        
        channel_results = []
        axes = ['X', 'Y', 'Z']
        
        for channel in range(config['channels']):
            axis_name = axes[channel]
            self.logger.info(f"Calibrating accelerometer {axis_name}-axis")
            
            try:
                # Generate test acceleration signal
                test_signal = self._generate_test_signal(
                    duration, config['sampling_rate'], 'chirp'
                )
                
                # Add gravity offset for vertical axis
                gravity_offset = 9.81 if channel == 2 else 0.0  # Z-axis has gravity
                
                # Add realistic noise
                noise = np.random.randn(len(test_signal)) * config['noise_floor_g']
                measured_signal = test_signal + gravity_offset + noise
                
                # Analyze signal quality
                quality_metrics = self._analyze_signal_quality(
                    measured_signal, config['sampling_rate'], config['noise_floor_g']
                )
                
                # Calculate calibration parameters
                offset = np.mean(measured_signal) - gravity_offset  # Remove gravity component
                
                # Gain calibration (1g reference)
                expected_amplitude = 1.0  # g
                measured_amplitude = np.std(test_signal)
                gain = expected_amplitude / measured_amplitude if measured_amplitude > 0 else 1.0
                
                # Frequency response for accelerometer
                freq_response = {
                    'dc_response': 1.0,
                    'low_freq_3db': 0.1,  # Hz
                    'high_freq_3db': 500.0,  # Hz
                    'phase_delay_ms': 0.1  # Low delay for accelerometer
                }
                
                result = CalibrationResult(
                    device_name=f"{config['name']} ({axis_name}-axis)",
                    channel_id=channel,
                    offset=offset,
                    gain=gain,
                    noise_level=quality_metrics['noise_level'],
                    snr_db=quality_metrics['snr_db'],
                    frequency_response=freq_response,
                    success=True
                )
                
                channel_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to calibrate accelerometer {axis_name}-axis: {e}")
                error_result = CalibrationResult(
                    device_name=f"{config['name']} ({axis_name}-axis)",
                    channel_id=channel,
                    offset=0.0,
                    gain=1.0,
                    noise_level=float('inf'),
                    snr_db=-60.0,
                    frequency_response={},
                    success=False,
                    error_message=str(e)
                )
                channel_results.append(error_result)
        
        # Calculate overall performance
        successful_channels = [r for r in channel_results if r.success]
        if successful_channels:
            avg_snr = np.mean([r.snr_db for r in successful_channels])
            avg_noise = np.mean([r.noise_level for r in successful_channels])
            offset_std = np.std([r.offset for r in successful_channels])
            gain_std = np.std([r.gain for r in successful_channels])
        else:
            avg_snr = -60.0
            avg_noise = float('inf')
            offset_std = float('inf')
            gain_std = float('inf')
        
        overall_performance = {
            'average_snr_db': avg_snr,
            'average_noise_level': avg_noise,
            'offset_stability': offset_std,
            'gain_uniformity': gain_std,
            'successful_channels': len(successful_channels),
            'total_channels': len(channel_results),
            'success_rate': len(successful_channels) / len(channel_results) * 100
        }
        
        device_calibration = DeviceCalibration(
            device_type='accelerometer',
            device_name=config['name'],
            serial_number=serial_number,
            calibration_date=datetime.now().isoformat(),
            channel_results=channel_results,
            overall_performance=overall_performance,
            success=len(successful_channels) == len(channel_results),  # All axes must work
            notes=f"Accelerometer calibration: {len(successful_channels)}/{len(channel_results)} axes"
        )
        
        self.logger.info(f"Accelerometer calibration complete: {len(successful_channels)}/{len(channel_results)} axes successful")
        return device_calibration
    
    def run_full_calibration(self, devices: List[str] = None) -> Dict[str, Any]:
        """Run calibration for specified devices."""
        if devices is None:
            devices = ['omp', 'kernel', 'accelerometer']
        
        self.logger.info(f"Starting full hardware calibration for devices: {', '.join(devices)}")
        
        calibration_functions = {
            'omp': self.calibrate_omp_helmet,
            'kernel': self.calibrate_kernel_optical,
            'accelerometer': self.calibrate_accelerometer
        }
        
        for device in devices:
            if device in calibration_functions:
                try:
                    result = calibration_functions[device]()
                    self.calibration_results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to calibrate {device}: {e}")
            else:
                self.logger.warning(f"Unknown device type: {device}")
        
        return self.generate_calibration_report()
    
    def generate_calibration_report(self) -> Dict[str, Any]:
        """Generate comprehensive calibration report."""
        total_devices = len(self.calibration_results)
        successful_devices = len([r for r in self.calibration_results if r.success])
        
        total_channels = sum(len(r.channel_results) for r in self.calibration_results)
        successful_channels = sum(
            len([c for c in r.channel_results if c.success]) 
            for r in self.calibration_results
        )
        
        report = {
            "calibration_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_devices": total_devices,
                "successful_devices": successful_devices,
                "device_success_rate": (successful_devices / total_devices * 100) if total_devices > 0 else 0,
                "total_channels": total_channels,
                "successful_channels": successful_channels,
                "channel_success_rate": (successful_channels / total_channels * 100) if total_channels > 0 else 0,
                "quick_mode": self.quick_mode,
                "simulation_mode": self.simulation_mode
            },
            "device_results": [asdict(result) for result in self.calibration_results],
            "recommendations": self._generate_calibration_recommendations()
        }
        
        return report
    
    def _generate_calibration_recommendations(self) -> List[str]:
        """Generate calibration recommendations."""
        recommendations = []
        
        if not self.calibration_results:
            return ["No calibration results available. Run calibration first."]
        
        for device_result in self.calibration_results:
            if not device_result.success:
                recommendations.append(
                    f"Device {device_result.device_name} failed calibration. Check hardware connections."
                )
                continue
            
            performance = device_result.overall_performance
            
            if performance['success_rate'] < 90:
                recommendations.append(
                    f"{device_result.device_name}: {performance['success_rate']:.1f}% channel success rate. "
                    "Check failing channels for hardware issues."
                )
            
            if performance['average_snr_db'] < 20:
                recommendations.append(
                    f"{device_result.device_name}: Low SNR ({performance['average_snr_db']:.1f} dB). "
                    "Consider improving shielding or reducing environmental noise."
                )
            
            if performance.get('gain_uniformity', 0) > 0.1:
                recommendations.append(
                    f"{device_result.device_name}: High gain variation across channels. "
                    "Consider individual channel gain correction."
                )
        
        if len(recommendations) == 0:
            recommendations.append("All devices calibrated successfully. System ready for operation.")
        
        return recommendations
    
    def save_calibration_results(self, report: Dict[str, Any]) -> None:
        """Save calibration results to file."""
        # Save main report
        with open(self.output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save individual device calibration files
        for device_result in self.calibration_results:
            device_file = self.data_dir / f"{device_result.device_type}_calibration_{device_result.serial_number}.json"
            with open(device_file, 'w') as f:
                json.dump(asdict(device_result), f, indent=2, default=str)
        
        self.logger.info(f"Calibration results saved to: {self.output_file}")
        self.logger.info(f"Individual device files saved to: {self.data_dir}")


def main():
    """Main entry point for calibration script."""
    parser = argparse.ArgumentParser(description="Brain-Forge Hardware Calibration Suite")
    parser.add_argument("--device", default="all", 
                       choices=['all', 'omp', 'kernel', 'accelerometer'],
                       help="Device to calibrate")
    parser.add_argument("--output", default="calibration_results.json",
                       help="Output file for calibration results")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick calibration with reduced duration")
    parser.add_argument("--simulation", action="store_true", default=True,
                       help="Run in simulation mode (default)")
    
    args = parser.parse_args()
    
    # Determine devices to calibrate
    if args.device == 'all':
        devices = ['omp', 'kernel', 'accelerometer']
    else:
        devices = [args.device]
    
    calibrator = HardwareCalibrator(
        output_file=args.output,
        quick_mode=args.quick,
        simulation_mode=args.simulation
    )
    
    try:
        report = calibrator.run_full_calibration(devices)
        calibrator.save_calibration_results(report)
        
        # Print summary
        print("\n" + "="*60)
        print("BRAIN-FORGE HARDWARE CALIBRATION SUMMARY")
        print("="*60)
        summary = report['calibration_summary']
        print(f"Devices calibrated: {summary['successful_devices']}/{summary['total_devices']}")
        print(f"Device success rate: {summary['device_success_rate']:.1f}%")
        print(f"Channels calibrated: {summary['successful_channels']}/{summary['total_channels']}")
        print(f"Channel success rate: {summary['channel_success_rate']:.1f}%")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        print(f"\nResults saved to: {args.output}")
        
        return 0 if summary['device_success_rate'] == 100 else 1
        
    except Exception as e:
        print(f"Calibration failed with error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
