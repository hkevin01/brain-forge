"""
Brown Accelo-hat Accelerometer Array Driver
===========================================

This module implements the driver for the Brown University Accelo-hat
accelerometer array system for brain injury monitoring and force measurement.

The Accelo-hat is designed for:
- High-resolution acceleration measurement during brain injury events
- Multi-point acceleration mapping across the head
- Real-time impact detection and characterization
- Force vector analysis for injury assessment
- Microsecond-precision timing for multi-modal integration

Based on Brown University's brain injury research platform with:
- Dense accelerometer array configuration
- High-frequency sampling (>1kHz) capability
- Real-time impact detection algorithms
- Force magnitude and direction computation
- Integration with other BCI modalities

Author: Brain-Forge Development Team
Date: 2025-01-28
License: MIT
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import signal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize random number generator
rng = np.random.default_rng()


class ImpactSeverity(Enum):
    """Impact severity classification"""
    NO_IMPACT = "no_impact"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class AccelerometerType(Enum):
    """Accelerometer sensor types"""
    MEMS_HIGH_G = "mems_high_g"  # ±200g MEMS
    MEMS_LOW_G = "mems_low_g"    # ±16g MEMS
    PIEZOELECTRIC = "piezoelectric"  # High frequency response


@dataclass
class AccelerometerConfig:
    """Configuration for individual accelerometer"""
    sensor_id: str
    position: np.ndarray  # 3D position on helmet (mm)
    sensor_type: AccelerometerType
    range_g: float  # Maximum acceleration range (g)
    sensitivity: float  # mV/g
    bandwidth_hz: float  # Sensor bandwidth
    is_active: bool = True
    calibration_offset: Optional[np.ndarray] = None  # [x,y,z] offset in g
    calibration_scale: Optional[np.ndarray] = None   # [x,y,z] scale factors


@dataclass
class ImpactThresholds:
    """Thresholds for impact detection and classification"""
    mild_threshold_g: float = 15.0      # 15g for mild impact
    moderate_threshold_g: float = 50.0  # 50g for moderate impact
    severe_threshold_g: float = 100.0   # 100g for severe impact
    critical_threshold_g: float = 150.0 # 150g for critical impact

    duration_threshold_ms: float = 10.0  # Minimum impact duration
    rise_time_threshold_ms: float = 5.0  # Maximum rise time for impact


@dataclass
class AcceloHatConfig:
    """Complete Accelo-hat system configuration"""
    accelerometers: List[AccelerometerConfig]
    sample_rate: int = 2000  # Hz - high frequency for impact detection
    impact_thresholds: ImpactThresholds = None
    sync_precision: float = 1e-6  # 1 microsecond sync precision
    buffer_size: int = 10000  # Number of samples to buffer
    enable_realtime_analysis: bool = True
    enable_impact_detection: bool = True


class ImpactDetector:
    """
    Real-time impact detection and analysis system

    Detects and characterizes impact events from accelerometer data
    including magnitude, direction, duration, and severity classification.
    """

    def __init__(self, config: AcceloHatConfig):
        self.config = config
        self.thresholds = config.impact_thresholds or ImpactThresholds()

        # Impact detection state
        self.is_impact_detected = False
        self.impact_start_time = None
        self.impact_data_buffer = []
        self.background_noise_estimate = 0.5  # g RMS

        # Signal processing
        self.sample_rate = config.sample_rate
        self.filter_coeffs = self._design_impact_filter()

        # Impact history
        self.impact_events = []
        self.last_impact_time = 0

    def _design_impact_filter(self) -> Dict[str, np.ndarray]:
        """Design filters for impact detection"""
        nyquist = self.sample_rate / 2

        # High-pass filter to remove DC and low-frequency noise
        hp_cutoff = 1.0  # Hz
        hp_b, hp_a = signal.butter(4, hp_cutoff / nyquist, btype='high')

        # Low-pass filter to prevent aliasing
        lp_cutoff = 500.0  # Hz
        lp_b, lp_a = signal.butter(4, lp_cutoff / nyquist, btype='low')

        return {
            'highpass_b': hp_b,
            'highpass_a': hp_a,
            'lowpass_b': lp_b,
            'lowpass_a': lp_a
        }

    def process_acceleration_data(self, accel_data: np.ndarray,
                                timestamps: np.ndarray) -> Dict[str, Any]:
        """
        Process acceleration data for impact detection

        Args:
            accel_data: N_samples x N_sensors x 3 acceleration data (g)
            timestamps: N_samples timestamp array

        Returns:
            Dict with impact detection results
        """
        n_samples, n_sensors, _ = accel_data.shape

        # Compute magnitude for each sensor
        accel_magnitude = np.linalg.norm(accel_data, axis=2)  # N_samples x N_sensors

        # Apply filtering
        filtered_magnitude = np.zeros_like(accel_magnitude)
        for sensor_idx in range(n_sensors):
            # High-pass filter
            filtered_data = signal.filtfilt(
                self.filter_coeffs['highpass_b'],
                self.filter_coeffs['highpass_a'],
                accel_magnitude[:, sensor_idx]
            )
            # Low-pass filter
            filtered_magnitude[:, sensor_idx] = signal.filtfilt(
                self.filter_coeffs['lowpass_b'],
                self.filter_coeffs['lowpass_a'],
                filtered_data
            )

        # Detect impacts
        impact_results = self._detect_impacts(
            filtered_magnitude, timestamps, accel_data
        )

        # Compute summary statistics
        max_acceleration = np.max(filtered_magnitude)
        peak_sensor_idx = np.unravel_index(
            np.argmax(filtered_magnitude), filtered_magnitude.shape
        )

        return {
            'max_acceleration_g': float(max_acceleration),
            'peak_sensor_id': peak_sensor_idx[1],
            'peak_timestamp': timestamps[peak_sensor_idx[0]],
            'filtered_magnitude': filtered_magnitude,
            'impact_events': impact_results['events'],
            'is_impact_detected': impact_results['impact_detected'],
            'background_noise_g': float(self.background_noise_estimate)
        }

    def _detect_impacts(self, magnitude_data: np.ndarray,
                       timestamps: np.ndarray,
                       raw_accel: np.ndarray) -> Dict[str, Any]:
        """Detect impact events in acceleration data"""
        impact_events = []
        impact_detected = False

        # Simple threshold-based detection
        threshold_exceeded = magnitude_data > self.thresholds.mild_threshold_g

        if np.any(threshold_exceeded):
            # Find impact regions
            impact_regions = self._find_impact_regions(
                threshold_exceeded, timestamps, magnitude_data
            )

            for region in impact_regions:
                start_idx, end_idx = region['indices']
                impact_duration = region['duration_ms']

                # Skip short-duration events (likely noise)
                if impact_duration < self.thresholds.duration_threshold_ms:
                    continue

                # Extract impact characteristics
                impact_magnitude = np.max(magnitude_data[start_idx:end_idx])
                impact_time = timestamps[start_idx]

                # Classify severity
                severity = self._classify_impact_severity(impact_magnitude)

                # Compute impact vector (simplified)
                impact_vector = self._compute_impact_vector(
                    raw_accel[start_idx:end_idx], magnitude_data[start_idx:end_idx]
                )

                impact_event = {
                    'timestamp': impact_time,
                    'duration_ms': impact_duration,
                    'magnitude_g': float(impact_magnitude),
                    'severity': severity.value,
                    'impact_vector': impact_vector.tolist(),
                    'start_index': int(start_idx),
                    'end_index': int(end_idx)
                }

                impact_events.append(impact_event)
                impact_detected = True

                # Log significant impacts
                if severity != ImpactSeverity.MILD:
                    logger.warning(f"Impact detected: {severity.value} "
                                 f"({impact_magnitude:.1f}g) at {impact_time:.6f}s")

        # Update background noise estimate
        if not impact_detected:
            noise_estimate = np.std(magnitude_data)
            self.background_noise_estimate = (
                0.99 * self.background_noise_estimate + 0.01 * noise_estimate
            )

        return {
            'events': impact_events,
            'impact_detected': impact_detected,
            'total_events': len(impact_events)
        }

    def _find_impact_regions(self, threshold_mask: np.ndarray,
                           timestamps: np.ndarray,
                           magnitude_data: np.ndarray) -> List[Dict[str, Any]]:
        """Find continuous regions where threshold is exceeded"""
        regions = []

        # Find transitions
        diff_mask = np.diff(threshold_mask.astype(int), axis=0)

        for sensor_idx in range(threshold_mask.shape[1]):
            sensor_mask = threshold_mask[:, sensor_idx]
            sensor_diff = diff_mask[:, sensor_idx]

            # Find start and end indices
            starts = np.where(sensor_diff == 1)[0] + 1  # +1 because diff shifts indices
            ends = np.where(sensor_diff == -1)[0] + 1

            # Handle edge cases
            if sensor_mask[0]:  # Starts with True
                starts = np.concatenate([[0], starts])
            if sensor_mask[-1]:  # Ends with True
                ends = np.concatenate([ends, [len(sensor_mask)]])

            # Create regions
            for start_idx, end_idx in zip(starts, ends):
                if start_idx < end_idx:
                    duration_ms = (timestamps[end_idx-1] - timestamps[start_idx]) * 1000
                    regions.append({
                        'sensor_idx': sensor_idx,
                        'indices': (start_idx, end_idx),
                        'duration_ms': duration_ms,
                        'start_time': timestamps[start_idx],
                        'end_time': timestamps[end_idx-1]
                    })

        return regions

    def _classify_impact_severity(self, magnitude_g: float) -> ImpactSeverity:
        """Classify impact severity based on magnitude"""
        if magnitude_g >= self.thresholds.critical_threshold_g:
            return ImpactSeverity.CRITICAL
        elif magnitude_g >= self.thresholds.severe_threshold_g:
            return ImpactSeverity.SEVERE
        elif magnitude_g >= self.thresholds.moderate_threshold_g:
            return ImpactSeverity.MODERATE
        elif magnitude_g >= self.thresholds.mild_threshold_g:
            return ImpactSeverity.MILD
        else:
            return ImpactSeverity.NO_IMPACT

    def _compute_impact_vector(self, accel_segment: np.ndarray,
                              magnitude_segment: np.ndarray) -> np.ndarray:
        """Compute primary impact direction vector"""
        # Find peak impact point
        peak_idx = np.argmax(magnitude_segment)

        # Average acceleration vectors around peak
        window_size = min(10, len(accel_segment))
        start_idx = max(0, peak_idx - window_size // 2)
        end_idx = min(len(accel_segment), start_idx + window_size)

        # Compute average direction vector across all sensors
        avg_vector = np.mean(accel_segment[start_idx:end_idx], axis=(0, 1))

        # Normalize
        magnitude = np.linalg.norm(avg_vector)
        if magnitude > 0:
            return avg_vector / magnitude
        else:
            return np.array([0.0, 0.0, 0.0])


class AcceloHatDriver:
    """
    Main driver for Brown Accelo-hat accelerometer array system

    Manages:
    - High-frequency accelerometer data acquisition (>1kHz)
    - Real-time impact detection and analysis
    - Multi-point acceleration mapping
    - Force vector computation and analysis
    - Microsecond-precision synchronization
    """

    def __init__(self, config: AcceloHatConfig):
        self.config = config
        self.impact_detector = ImpactDetector(config) if config.enable_impact_detection else None

        # Data buffers
        self.acceleration_buffer = []
        self.timestamp_buffer = []
        self.impact_events = []

        # System state
        self.is_acquiring = False
        self.acquisition_start_time = None
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Calibration and processing
        self.calibration_data = self._load_calibration()

        # Statistics
        self.stats = {
            'samples_acquired': 0,
            'impacts_detected': 0,
            'max_acceleration_g': 0.0,
            'avg_acceleration_g': 0.0,
            'processing_latency_ms': 0.0,
            'last_impact_time': 0.0
        }

    def _load_calibration(self) -> Dict[str, np.ndarray]:
        """Load accelerometer calibration data"""
        # In practice, this would load from calibration files
        calibration = {}

        for accel in self.config.accelerometers:
            if accel.calibration_offset is None:
                offset = np.zeros(3)
            else:
                offset = np.array(accel.calibration_offset)

            if accel.calibration_scale is None:
                scale = np.ones(3)
            else:
                scale = np.array(accel.calibration_scale)

            calibration[accel.sensor_id] = {
                'offset': offset,
                'scale': scale
            }

        return calibration

    async def initialize(self):
        """Initialize Accelo-hat system"""
        logger.info("Initializing Brown Accelo-hat accelerometer array...")

        # Initialize hardware interfaces (simulated)
        await self._initialize_accelerometers()
        await self._initialize_timing_system()
        await self._perform_self_test()

        active_sensors = sum(1 for accel in self.config.accelerometers if accel.is_active)
        logger.info(f"Accelo-hat initialized with {active_sensors} active accelerometers")
        logger.info(f"Sample rate: {self.config.sample_rate} Hz")
        logger.info(f"Impact detection: {'enabled' if self.config.enable_impact_detection else 'disabled'}")

    async def _initialize_accelerometers(self):
        """Initialize accelerometer sensors"""
        logger.info("Configuring accelerometer sensors...")

        # Check sensor configurations
        high_g_count = sum(1 for accel in self.config.accelerometers
                          if accel.sensor_type == AccelerometerType.MEMS_HIGH_G and accel.is_active)
        low_g_count = sum(1 for accel in self.config.accelerometers
                         if accel.sensor_type == AccelerometerType.MEMS_LOW_G and accel.is_active)
        piezo_count = sum(1 for accel in self.config.accelerometers
                         if accel.sensor_type == AccelerometerType.PIEZOELECTRIC and accel.is_active)

        logger.info(f"Sensor distribution: {high_g_count} high-g MEMS, "
                   f"{low_g_count} low-g MEMS, {piezo_count} piezoelectric")

    async def _initialize_timing_system(self):
        """Initialize high-precision timing system"""
        logger.info("Initializing high-precision timing...")
        self.timing_reference = time.time()
        logger.info(f"Timing precision: {self.config.sync_precision*1e6:.1f} microseconds")

    async def _perform_self_test(self):
        """Perform system self-test"""
        logger.info("Performing system self-test...")

        # Test accelerometer connectivity (simulated)
        failed_sensors = []
        for accel in self.config.accelerometers:
            if accel.is_active:
                # Simulate random sensor failure (1% chance)
                if rng.random() < 0.01:
                    failed_sensors.append(accel.sensor_id)

        if failed_sensors:
            logger.warning(f"Self-test failed for sensors: {failed_sensors}")
        else:
            logger.info("Self-test passed for all sensors")

    def _get_precision_timestamp(self) -> float:
        """Get high-precision timestamp"""
        return time.time()

    def _apply_calibration(self, raw_data: np.ndarray, sensor_id: str) -> np.ndarray:
        """Apply calibration to raw accelerometer data"""
        if sensor_id not in self.calibration_data:
            return raw_data

        cal = self.calibration_data[sensor_id]
        return (raw_data - cal['offset']) * cal['scale']

    async def _acquire_acceleration_sample(self) -> Dict[str, Any]:
        """Acquire single acceleration measurement from all sensors"""
        timestamp = self._get_precision_timestamp()

        # Simulate realistic accelerometer data
        n_sensors = len([a for a in self.config.accelerometers if a.is_active])

        # Generate baseline acceleration (gravity + small vibrations)
        baseline_accel = np.zeros((n_sensors, 3))
        baseline_accel[:, 2] = 1.0  # 1g downward (gravity)

        # Add small random vibrations
        vibration_noise = rng.normal(0, 0.1, (n_sensors, 3))  # 0.1g RMS noise

        # Simulate occasional larger movements
        if rng.random() < 0.001:  # 0.1% chance of impact simulation
            impact_magnitude = rng.uniform(20, 100)  # 20-100g impact
            impact_direction = rng.normal(0, 1, 3)
            impact_direction /= np.linalg.norm(impact_direction)

            # Apply impact to random subset of sensors
            impact_sensors = rng.choice(n_sensors, size=max(1, n_sensors//3), replace=False)
            for sensor_idx in impact_sensors:
                baseline_accel[sensor_idx] += impact_magnitude * impact_direction

        # Combine baseline + noise
        raw_acceleration = baseline_accel + vibration_noise

        # Apply calibration
        calibrated_data = np.zeros_like(raw_acceleration)
        active_sensors = [a for a in self.config.accelerometers if a.is_active]

        for i, accel_config in enumerate(active_sensors):
            calibrated_data[i] = self._apply_calibration(
                raw_acceleration[i], accel_config.sensor_id
            )

        return {
            'timestamp': timestamp,
            'acceleration_g': calibrated_data,
            'sensor_ids': [a.sensor_id for a in active_sensors],
            'sample_count': n_sensors
        }

    async def start_acquisition(self, duration: Optional[float] = None):
        """Start accelerometer data acquisition"""
        if self.is_acquiring:
            logger.warning("Acquisition already in progress")
            return

        logger.info("Starting Accelo-hat data acquisition...")
        if duration:
            logger.info(f"Acquisition duration: {duration:.1f} seconds")

        self.is_acquiring = True
        self.acquisition_start_time = time.time()

        # Clear buffers
        self.acceleration_buffer = []
        self.timestamp_buffer = []
        self.impact_events = []

        # Start acquisition loop
        await self._acquisition_loop(duration)

    async def _acquisition_loop(self, duration: Optional[float] = None):
        """Main data acquisition loop"""
        sample_interval = 1.0 / self.config.sample_rate
        processing_buffer = []

        while self.is_acquiring:
            loop_start = time.time()

            # Check duration limit
            if duration and (loop_start - self.acquisition_start_time) >= duration:
                break

            # Acquire sample
            sample = await self._acquire_acceleration_sample()

            # Store in buffers
            self.acceleration_buffer.append(sample['acceleration_g'])
            self.timestamp_buffer.append(sample['timestamp'])
            processing_buffer.append(sample)

            self.stats['samples_acquired'] += 1

            # Update acceleration statistics
            max_accel = np.max(np.linalg.norm(sample['acceleration_g'], axis=1))
            avg_accel = np.mean(np.linalg.norm(sample['acceleration_g'], axis=1))

            self.stats['max_acceleration_g'] = max(self.stats['max_acceleration_g'], max_accel)
            self.stats['avg_acceleration_g'] = (
                0.99 * self.stats['avg_acceleration_g'] + 0.01 * avg_accel
            )

            # Process batch for impact detection
            if (self.config.enable_realtime_analysis and
                len(processing_buffer) >= 100):  # Process every 100 samples

                await self._process_batch(processing_buffer)
                processing_buffer = []

            # Maintain buffer size
            if len(self.acceleration_buffer) > self.config.buffer_size:
                self.acceleration_buffer.pop(0)
                self.timestamp_buffer.pop(0)

            # Maintain precise timing
            processing_time = time.time() - loop_start
            self.stats['processing_latency_ms'] = (
                0.9 * self.stats['processing_latency_ms'] +
                0.1 * processing_time * 1000
            )

            sleep_time = max(0, sample_interval - processing_time)
            await asyncio.sleep(sleep_time)

        # Process remaining buffer
        if processing_buffer:
            await self._process_batch(processing_buffer)

        self.is_acquiring = False
        logger.info("Accelo-hat acquisition completed")

    async def _process_batch(self, batch_data: List[Dict[str, Any]]):
        """Process batch of acceleration data for impacts"""
        if not self.impact_detector:
            return

        # Combine batch data
        accel_array = np.array([sample['acceleration_g'] for sample in batch_data])
        timestamp_array = np.array([sample['timestamp'] for sample in batch_data])

        # Process for impacts
        impact_results = self.impact_detector.process_acceleration_data(
            accel_array, timestamp_array
        )

        # Store detected impacts
        if impact_results['impact_events']:
            self.impact_events.extend(impact_results['impact_events'])
            self.stats['impacts_detected'] += len(impact_results['impact_events'])

            # Update last impact time
            latest_impact = max(
                impact_results['impact_events'],
                key=lambda x: x['timestamp']
            )
            self.stats['last_impact_time'] = latest_impact['timestamp']

    async def stop_acquisition(self):
        """Stop data acquisition"""
        logger.info("Stopping Accelo-hat acquisition...")
        self.is_acquiring = False

    def get_latest_acceleration(self) -> Optional[Dict[str, Any]]:
        """Get latest acceleration measurement"""
        if not self.acceleration_buffer:
            return None

        latest_accel = self.acceleration_buffer[-1]
        latest_timestamp = self.timestamp_buffer[-1]

        # Compute magnitude for each sensor
        magnitudes = np.linalg.norm(latest_accel, axis=1)

        return {
            'timestamp': latest_timestamp,
            'acceleration_g': latest_accel.tolist(),
            'magnitudes_g': magnitudes.tolist(),
            'max_magnitude_g': float(np.max(magnitudes)),
            'sensor_count': len(latest_accel)
        }

    def get_impact_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent impact events"""
        # Sort by timestamp and return most recent
        sorted_impacts = sorted(
            self.impact_events,
            key=lambda x: x['timestamp'],
            reverse=True
        )
        return sorted_impacts[:limit]

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        active_sensors = sum(1 for a in self.config.accelerometers if a.is_active)

        return {
            'is_acquiring': self.is_acquiring,
            'active_sensors': active_sensors,
            'sample_rate_hz': self.config.sample_rate,
            'samples_acquired': self.stats['samples_acquired'],
            'impacts_detected': self.stats['impacts_detected'],
            'max_acceleration_g': f"{self.stats['max_acceleration_g']:.2f}",
            'avg_acceleration_g': f"{self.stats['avg_acceleration_g']:.2f}",
            'processing_latency_ms': f"{self.stats['processing_latency_ms']:.2f}",
            'last_impact_time': self.stats['last_impact_time'],
            'buffer_size': len(self.acceleration_buffer),
            'impact_detection_enabled': self.config.enable_impact_detection
        }

    def export_data(self) -> Dict[str, Any]:
        """Export acquired data"""
        return {
            'config': {
                'sample_rate': self.config.sample_rate,
                'sensor_count': len(self.config.accelerometers),
                'active_sensors': [a.sensor_id for a in self.config.accelerometers if a.is_active]
            },
            'acceleration_data': [accel.tolist() for accel in self.acceleration_buffer],
            'timestamps': self.timestamp_buffer,
            'impact_events': self.impact_events,
            'stats': self.stats,
            'acquisition_duration': time.time() - self.acquisition_start_time if self.acquisition_start_time else 0
        }


# Example usage
async def main():
    """Example usage of Accelo-hat system"""

    # Create accelerometer configurations
    accelerometers = []

    # Create a realistic helmet accelerometer layout
    positions = [
        [0, 50, 80],     # Front center
        [-30, 40, 70],   # Front left
        [30, 40, 70],    # Front right
        [-50, 0, 60],    # Left side
        [50, 0, 60],     # Right side
        [-30, -40, 70],  # Rear left
        [30, -40, 70],   # Rear right
        [0, -50, 80],    # Rear center
    ]

    for i, pos in enumerate(positions):
        # Mix of high-g and low-g sensors
        sensor_type = (AccelerometerType.MEMS_HIGH_G if i % 2 == 0
                      else AccelerometerType.MEMS_LOW_G)
        range_g = 200.0 if sensor_type == AccelerometerType.MEMS_HIGH_G else 16.0

        accel = AccelerometerConfig(
            sensor_id=f"ACCEL_{i+1:02d}",
            position=np.array(pos),
            sensor_type=sensor_type,
            range_g=range_g,
            sensitivity=50.0,  # mV/g
            bandwidth_hz=1000.0
        )
        accelerometers.append(accel)

    # Create system configuration
    config = AcceloHatConfig(
        accelerometers=accelerometers,
        sample_rate=2000,  # 2 kHz
        impact_thresholds=ImpactThresholds(),
        enable_realtime_analysis=True,
        enable_impact_detection=True
    )

    # Initialize and run system
    accelo_hat = AcceloHatDriver(config)
    await accelo_hat.initialize()

    # Start acquisition
    logger.info("Starting 10-second acquisition with impact detection...")
    await accelo_hat.start_acquisition(duration=10.0)

    # Get latest acceleration
    latest_accel = accelo_hat.get_latest_acceleration()
    if latest_accel:
        logger.info(f"Latest acceleration: {latest_accel['max_magnitude_g']:.2f}g max")

    # Get impact history
    impacts = accelo_hat.get_impact_history()
    if impacts:
        logger.info(f"Detected {len(impacts)} impact events:")
        for impact in impacts[:3]:  # Show first 3
            logger.info(f"  {impact['severity']}: {impact['magnitude_g']:.1f}g "
                       f"at {impact['timestamp']:.3f}s")
    else:
        logger.info("No impacts detected during acquisition")

    # Print system status
    status = accelo_hat.get_system_status()
    logger.info("Final System Status:")
    for key, value in status.items():
        logger.info(f"  {key}: {value}")

    # Export data
    exported_data = accelo_hat.export_data()
    logger.info(f"Exported {len(exported_data['acceleration_data'])} acceleration samples")
    logger.info(f"Detected {len(exported_data['impact_events'])} impact events")


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(main())
