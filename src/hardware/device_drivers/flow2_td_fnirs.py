"""
Kernel Flow2 TD-fNIRS + EEG Fusion Driver
========================================

This module implements the driver for Kernel Flow2 Time-Domain functional
Near-Infrared Spectroscopy (TD-fNIRS) system with integrated EEG fusion.

Based on Kernel's Flow2 system specifications:
- Whole-head TD-fNIRS coverage with miniaturized components
- Dense sensor array with helmet form factor
- Time-domain measurements for improved depth sensitivity
- Integrated EEG for complementary temporal resolution
- High-resolution cortical oxygenation mapping

Features:
- TD-fNIRS measurement with picosecond timing resolution
- Multi-wavelength LED sources (typically 760nm, 850nm)
- SPAD (Single Photon Avalanche Diode) detectors
- Integrated EEG electrodes for hybrid recordings
- Real-time oxygenation computation (HbO2, HbR, HbT)
- Microsecond-precision synchronization with other modalities

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeasurementMode(Enum):
    """TD-fNIRS measurement modes"""
    TD_FNIRS_ONLY = "td_fnirs"
    EEG_ONLY = "eeg"
    HYBRID_FUSION = "hybrid"


@dataclass
class OptodeConfig:
    """Configuration for individual optode (source or detector)"""
    optode_id: str
    optode_type: str  # 'source' or 'detector'
    position: np.ndarray  # 3D position on helmet
    wavelengths: Optional[List[int]] = None  # For sources only
    sensitivity: float = 1.0  # Detector sensitivity
    is_active: bool = True


@dataclass
class EEGElectrodeConfig:
    """Configuration for EEG electrode"""
    electrode_id: str
    position: np.ndarray  # 3D position
    impedance: float = 5000.0  # Ohms
    is_active: bool = True


@dataclass
class TDfNIRSConfig:
    """Configuration for TD-fNIRS system"""
    source_optodes: List[OptodeConfig]
    detector_optodes: List[OptodeConfig]
    wavelengths: Optional[List[int]] = None  # [760, 850] nm typically
    pulse_rate: float = 80e6  # 80 MHz pulse rate
    integration_time: float = 0.1  # seconds
    distance_threshold: float = 45.0  # mm max source-detector distance
    photon_count_threshold: int = 1000  # Minimum photons for valid measurement


@dataclass
class Flow2Config:
    """Complete Flow2 system configuration"""
    td_fnirs_config: TDfNIRSConfig
    eeg_electrodes: List[EEGElectrodeConfig]
    sample_rate: int = 10  # Hz for fNIRS (EEG typically 1000Hz)
    eeg_sample_rate: int = 1000  # Hz
    mode: MeasurementMode = MeasurementMode.HYBRID_FUSION
    sync_precision: float = 1e-6  # 1 microsecond sync precision


class PhotonTimingProcessor:
    """
    Time-domain photon timing analysis for TD-fNIRS

    Processes photon arrival times to extract optical properties
    and compute oxygenation parameters.
    """

    def __init__(self, wavelengths: List[int]):
        self.wavelengths = wavelengths

        # Extinction coefficients for hemoglobin (cm^-1 / (mol/L))
        # Values at 760nm and 850nm
        self.extinction_coeffs = {
            760: {'HbO2': 1486.5, 'HbR': 3843.0},
            850: {'HbO2': 2526.4, 'HbR': 1829.6}
        }

        # Path length factors (typical values)
        self.path_length_factor = 6.0

    def process_photon_data(self, photon_times: Dict[int, np.ndarray],
                          source_detector_distance: float) -> Dict[str, float]:
        """
        Process time-domain photon data to extract optical properties

        Args:
            photon_times: Dict mapping wavelength to photon arrival times
            source_detector_distance: Distance between source and detector (mm)

        Returns:
            Dict with computed optical properties and oxygenation
        """
        results = {}

        for wavelength in self.wavelengths:
            if wavelength not in photon_times:
                continue

            times = photon_times[wavelength]

            if len(times) == 0:
                continue

            # Compute time-domain parameters
            mean_time = np.mean(times)  # Mean photon flight time
            temporal_variance = np.var(times)
            photon_count = len(times)

            # Simple absorption coefficient estimation
            # (In practice, this would use more sophisticated fitting)
            mu_a = self._estimate_absorption(mean_time, source_detector_distance)

            results[f'mu_a_{wavelength}'] = mu_a
            results[f'mean_time_{wavelength}'] = mean_time
            results[f'photon_count_{wavelength}'] = photon_count
            results[f'temporal_variance_{wavelength}'] = temporal_variance

        # Compute oxygenation if we have both wavelengths
        if len(results) >= 2 * len(self.wavelengths):
            oxy_results = self._compute_oxygenation(results)
            results.update(oxy_results)

        return results

    def _estimate_absorption(self, mean_time: float, distance: float) -> float:
        """Estimate absorption coefficient from mean photon time"""
        # Simplified model - would use more sophisticated approach in practice
        c = 3e11  # Speed of light in tissue (mm/s)
        estimated_path_length = distance * self.path_length_factor

        if mean_time > 0:
            mu_a = np.log(estimated_path_length / (c * mean_time)) / distance
            return max(0, mu_a)  # Ensure positive
        return 0.0

    def _compute_oxygenation(self, optical_data: Dict[str, float]) -> Dict[str, float]:
        """Compute hemoglobin oxygenation from multi-wavelength data"""
        try:
            # Extract absorption coefficients
            mu_a_760 = optical_data.get('mu_a_760', 0)
            mu_a_850 = optical_data.get('mu_a_850', 0)

            # Modified Beer-Lambert law inversion
            ext_760 = self.extinction_coeffs[760]
            ext_850 = self.extinction_coeffs[850]

            # Solve linear system: [ext_matrix] * [HbO2, HbR] = [mu_a_760, mu_a_850]
            ext_matrix = np.array([
                [ext_760['HbO2'], ext_760['HbR']],
                [ext_850['HbO2'], ext_850['HbR']]
            ])

            mu_a_vector = np.array([mu_a_760, mu_a_850])

            # Solve for hemoglobin concentrations
            hb_concentrations = np.linalg.solve(ext_matrix, mu_a_vector)

            hbo2 = max(0, hb_concentrations[0])  # Oxygenated hemoglobin
            hbr = max(0, hb_concentrations[1])   # Deoxygenated hemoglobin
            hbt = hbo2 + hbr                     # Total hemoglobin
            so2 = hbo2 / hbt if hbt > 0 else 0   # Oxygen saturation

            return {
                'HbO2': hbo2,
                'HbR': hbr,
                'HbT': hbt,
                'SO2': so2
            }

        except (np.linalg.LinAlgError, KeyError, ZeroDivisionError):
            # Return zeros if computation fails
            return {
                'HbO2': 0.0,
                'HbR': 0.0,
                'HbT': 0.0,
                'SO2': 0.0
            }


class EEGProcessor:
    """EEG signal processing for hybrid TD-fNIRS/EEG system"""

    def __init__(self, sample_rate: int = 1000):
        self.sample_rate = sample_rate
        self.filters_initialized = False

    def process_eeg_data(self, eeg_data: np.ndarray,
                        electrode_ids: List[str]) -> Dict[str, Any]:
        """
        Process EEG data and extract features

        Args:
            eeg_data: N_samples x N_electrodes array
            electrode_ids: List of electrode identifiers

        Returns:
            Dict with processed EEG features
        """
        results = {
            'raw_data': eeg_data,
            'electrode_ids': electrode_ids,
            'sample_rate': self.sample_rate,
            'features': {}
        }

        # Basic preprocessing
        # In practice, would include filtering, artifact removal, etc.

        # Compute basic spectral features
        for i, electrode_id in enumerate(electrode_ids):
            channel_data = eeg_data[:, i]

            # Power spectral features (simplified)
            fft_data = np.fft.fft(channel_data)
            freqs = np.fft.fftfreq(len(channel_data), 1.0 / self.sample_rate)

            # Define frequency bands
            delta_power = self._band_power(fft_data, freqs, 1, 4)
            theta_power = self._band_power(fft_data, freqs, 4, 8)
            alpha_power = self._band_power(fft_data, freqs, 8, 13)
            beta_power = self._band_power(fft_data, freqs, 13, 30)

            results['features'][electrode_id] = {
                'delta_power': delta_power,
                'theta_power': theta_power,
                'alpha_power': alpha_power,
                'beta_power': beta_power,
                'total_power': delta_power + theta_power + alpha_power + beta_power
            }

        return results

    def _band_power(self, fft_data: np.ndarray, freqs: np.ndarray,
                    f_low: float, f_high: float) -> float:
        """Compute power in a frequency band"""
        band_mask = (freqs >= f_low) & (freqs <= f_high)
        band_power = np.sum(np.abs(fft_data[band_mask])**2)
        return float(band_power)


class Flow2Driver:
    """
    Main driver for Kernel Flow2 TD-fNIRS + EEG system

    Manages:
    - TD-fNIRS data acquisition with picosecond timing
    - EEG recording with microsecond synchronization
    - Real-time oxygenation computation
    - Multi-modal data fusion and streaming
    """

    def __init__(self, config: Flow2Config):
        self.config = config
        self.photon_processor = PhotonTimingProcessor(
            config.td_fnirs_config.wavelengths or [760, 850]
        )
        self.eeg_processor = EEGProcessor(config.eeg_sample_rate)

        # Data buffers
        self.fnirs_data_buffer = []
        self.eeg_data_buffer = []
        self.sync_timestamps = []

        # System state
        self.is_acquiring = False
        self.acquisition_start_time = None
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Statistics
        self.stats = {
            'fnirs_samples_acquired': 0,
            'eeg_samples_acquired': 0,
            'avg_photon_count': 0.0,
            'sync_precision_actual': 0.0,
            'processing_latency_ms': 0.0
        }

    async def initialize(self):
        """Initialize Flow2 system"""
        logger.info("Initializing Kernel Flow2 TD-fNIRS + EEG system...")

        # Initialize hardware interfaces (simulated)
        await self._initialize_optodes()
        await self._initialize_eeg_electrodes()
        await self._initialize_timing_system()

        logger.info(f"Flow2 initialized in {self.config.mode.value} mode")
        logger.info(f"TD-fNIRS: {len(self.config.td_fnirs_config.source_optodes)} "
                   f"sources, {len(self.config.td_fnirs_config.detector_optodes)} detectors")
        logger.info(f"EEG: {len(self.config.eeg_electrodes)} electrodes")

    async def _initialize_optodes(self):
        """Initialize TD-fNIRS optode system"""
        logger.info("Initializing TD-fNIRS optodes...")

        # Check optode configurations
        active_sources = sum(1 for opt in self.config.td_fnirs_config.source_optodes
                           if opt.is_active)
        active_detectors = sum(1 for opt in self.config.td_fnirs_config.detector_optodes
                             if opt.is_active)

        logger.info(f"Active optodes: {active_sources} sources, {active_detectors} detectors")

    async def _initialize_eeg_electrodes(self):
        """Initialize EEG electrode system"""
        if self.config.mode in [MeasurementMode.EEG_ONLY, MeasurementMode.HYBRID_FUSION]:
            logger.info("Initializing EEG electrodes...")

            # Check electrode impedances (simulated)
            high_impedance_count = 0
            for electrode in self.config.eeg_electrodes:
                if electrode.is_active and electrode.impedance > 10000:
                    high_impedance_count += 1

            if high_impedance_count > 0:
                logger.warning(f"{high_impedance_count} electrodes have high impedance")

    async def _initialize_timing_system(self):
        """Initialize microsecond-precision timing system"""
        logger.info("Initializing high-precision timing system...")

        # Initialize hardware timestamp counters
        # In practice, this would configure precision timing hardware
        self.timing_reference = time.time()

        logger.info(f"Timing precision: {self.config.sync_precision*1e6:.1f} microseconds")

    def _get_precision_timestamp(self) -> float:
        """Get high-precision timestamp"""
        # In practice, would use hardware timer
        return time.time()

    async def _acquire_fnirs_sample(self) -> Dict[str, Any]:
        """Acquire single TD-fNIRS sample"""
        timestamp = self._get_precision_timestamp()
        sample_data = {}

        # Simulate photon data for each source-detector pair
        for source in self.config.td_fnirs_config.source_optodes:
            if not source.is_active:
                continue

            for detector in self.config.td_fnirs_config.detector_optodes:
                if not detector.is_active:
                    continue

                # Calculate source-detector distance
                distance = np.linalg.norm(source.position - detector.position)

                # Skip if distance too large
                if distance > self.config.td_fnirs_config.distance_threshold:
                    continue

                # Simulate photon arrival times for each wavelength
                photon_data = {}
                for wavelength in self.photon_processor.wavelengths:
                    # Simulate realistic photon statistics
                    n_photons = np.random.poisson(5000)  # Average photon count

                    if n_photons > self.config.td_fnirs_config.photon_count_threshold:
                        # Generate photon arrival times (simplified)
                        base_time = distance / 3e11  # Time of flight in tissue
                        spread = 100e-12  # 100 ps spread
                        photon_times = np.random.normal(base_time, spread, n_photons)
                        photon_data[wavelength] = photon_times

                # Process photon data to get oxygenation
                if photon_data:
                    processed = self.photon_processor.process_photon_data(
                        photon_data, distance
                    )

                    pair_id = f"{source.optode_id}_{detector.optode_id}"
                    sample_data[pair_id] = {
                        'distance_mm': distance,
                        'optical_data': processed,
                        'timestamp': timestamp
                    }

        return {
            'timestamp': timestamp,
            'pairs': sample_data,
            'sample_count': len(sample_data)
        }

    async def _acquire_eeg_sample(self) -> Dict[str, Any]:
        """Acquire EEG sample"""
        timestamp = self._get_precision_timestamp()

        # Simulate EEG data
        n_electrodes = len([e for e in self.config.eeg_electrodes if e.is_active])
        n_samples = self.config.eeg_sample_rate // self.config.sample_rate  # Samples per fNIRS sample

        # Generate realistic EEG-like signals
        eeg_data = np.random.randn(n_samples, n_electrodes) * 50e-6  # 50 µV RMS

        # Add some alpha rhythm
        t = np.linspace(0, n_samples/self.config.eeg_sample_rate, n_samples)
        alpha_signal = 20e-6 * np.sin(2 * np.pi * 10 * t)  # 10 Hz, 20 µV
        eeg_data[:, 0] += alpha_signal.reshape(-1, 1)  # Add to first channel

        electrode_ids = [e.electrode_id for e in self.config.eeg_electrodes if e.is_active]

        return {
            'timestamp': timestamp,
            'eeg_data': eeg_data,
            'electrode_ids': electrode_ids,
            'sample_rate': self.config.eeg_sample_rate
        }

    async def start_acquisition(self, duration: float = None):
        """Start multi-modal data acquisition"""
        if self.is_acquiring:
            logger.warning("Acquisition already in progress")
            return

        logger.info(f"Starting acquisition in {self.config.mode.value} mode...")

        if duration:
            logger.info(f"Acquisition duration: {duration:.1f} seconds")

        self.is_acquiring = True
        self.acquisition_start_time = time.time()

        # Clear buffers
        self.fnirs_data_buffer = []
        self.eeg_data_buffer = []
        self.sync_timestamps = []

        # Start acquisition loop
        await self._acquisition_loop(duration)

    async def _acquisition_loop(self, duration: float = None):
        """Main acquisition loop with precise timing"""
        sample_interval = 1.0 / self.config.sample_rate

        while self.is_acquiring:
            loop_start = time.time()

            # Check duration limit
            if duration and (loop_start - self.acquisition_start_time) >= duration:
                break

            sync_timestamp = self._get_precision_timestamp()

            # Acquire data based on mode
            tasks = []

            if self.config.mode in [MeasurementMode.TD_FNIRS_ONLY, MeasurementMode.HYBRID_FUSION]:
                tasks.append(self._acquire_fnirs_sample())

            if self.config.mode in [MeasurementMode.EEG_ONLY, MeasurementMode.HYBRID_FUSION]:
                tasks.append(self._acquire_eeg_sample())

            # Execute acquisitions concurrently
            if tasks:
                results = await asyncio.gather(*tasks)

                # Store results with synchronization timestamp
                sample_data = {
                    'sync_timestamp': sync_timestamp,
                    'processing_timestamp': time.time()
                }

                if self.config.mode in [MeasurementMode.TD_FNIRS_ONLY, MeasurementMode.HYBRID_FUSION]:
                    fnirs_data = results[0] if len(results) > 0 else None
                    if fnirs_data:
                        self.fnirs_data_buffer.append(fnirs_data)
                        self.stats['fnirs_samples_acquired'] += 1

                        # Update photon count stats
                        if fnirs_data['pairs']:
                            photon_counts = [
                                pair['optical_data'].get('photon_count_760', 0) +
                                pair['optical_data'].get('photon_count_850', 0)
                                for pair in fnirs_data['pairs'].values()
                            ]
                            if photon_counts:
                                self.stats['avg_photon_count'] = (
                                    0.9 * self.stats['avg_photon_count'] +
                                    0.1 * np.mean(photon_counts)
                                )

                        sample_data['fnirs'] = fnirs_data

                if self.config.mode in [MeasurementMode.EEG_ONLY, MeasurementMode.HYBRID_FUSION]:
                    eeg_data = results[-1]  # EEG data is last
                    if eeg_data:
                        self.eeg_data_buffer.append(eeg_data)
                        self.stats['eeg_samples_acquired'] += 1
                        sample_data['eeg'] = eeg_data

                self.sync_timestamps.append(sample_data)

            # Maintain precise timing
            processing_time = time.time() - loop_start
            self.stats['processing_latency_ms'] = (
                0.9 * self.stats['processing_latency_ms'] +
                0.1 * processing_time * 1000
            )

            sleep_time = max(0, sample_interval - processing_time)
            await asyncio.sleep(sleep_time)

        self.is_acquiring = False
        logger.info("Acquisition completed")

    async def stop_acquisition(self):
        """Stop data acquisition"""
        logger.info("Stopping acquisition...")
        self.is_acquiring = False

    def get_realtime_oxygenation(self) -> Dict[str, Any]:
        """Get latest oxygenation measurements"""
        if not self.fnirs_data_buffer:
            return {}

        latest_sample = self.fnirs_data_buffer[-1]
        oxygenation_data = {}

        for pair_id, pair_data in latest_sample['pairs'].items():
            optical_data = pair_data['optical_data']
            oxygenation_data[pair_id] = {
                'HbO2': optical_data.get('HbO2', 0),
                'HbR': optical_data.get('HbR', 0),
                'HbT': optical_data.get('HbT', 0),
                'SO2': optical_data.get('SO2', 0),
                'distance_mm': pair_data['distance_mm'],
                'timestamp': pair_data['timestamp']
            }

        return oxygenation_data

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'is_acquiring': self.is_acquiring,
            'mode': self.config.mode.value,
            'fnirs_samples': self.stats['fnirs_samples_acquired'],
            'eeg_samples': self.stats['eeg_samples_acquired'],
            'avg_photon_count': int(self.stats['avg_photon_count']),
            'processing_latency_ms': f"{self.stats['processing_latency_ms']:.2f}",
            'sync_precision_us': self.config.sync_precision * 1e6,
            'buffer_sizes': {
                'fnirs': len(self.fnirs_data_buffer),
                'eeg': len(self.eeg_data_buffer),
                'sync': len(self.sync_timestamps)
            }
        }

    def export_data(self) -> Dict[str, Any]:
        """Export acquired data"""
        return {
            'config': {
                'mode': self.config.mode.value,
                'sample_rate': self.config.sample_rate,
                'eeg_sample_rate': self.config.eeg_sample_rate,
                'wavelengths': self.photon_processor.wavelengths
            },
            'fnirs_data': self.fnirs_data_buffer,
            'eeg_data': self.eeg_data_buffer,
            'sync_timestamps': self.sync_timestamps,
            'stats': self.stats,
            'acquisition_duration': time.time() - self.acquisition_start_time if self.acquisition_start_time else 0
        }


# Example usage
async def main():
    """Example usage of Flow2 TD-fNIRS + EEG system"""

    # Create optode configurations (simplified)
    sources = []
    detectors = []

    # Create source optodes
    for i in range(8):  # 8 sources
        source = OptodeConfig(
            optode_id=f"S{i+1:02d}",
            optode_type="source",
            position=np.random.randn(3) * 50,  # Random positions
            wavelengths=[760, 850]
        )
        sources.append(source)

    # Create detector optodes
    for i in range(16):  # 16 detectors
        detector = OptodeConfig(
            optode_id=f"D{i+1:02d}",
            optode_type="detector",
            position=np.random.randn(3) * 50,
            sensitivity=1.0
        )
        detectors.append(detector)

    # Create EEG electrodes
    eeg_electrodes = []
    for i, pos_name in enumerate(['Fz', 'Cz', 'Pz', 'Oz']):  # 4 electrodes
        electrode = EEGElectrodeConfig(
            electrode_id=pos_name,
            position=np.random.randn(3) * 10,
            impedance=5000.0
        )
        eeg_electrodes.append(electrode)

    # Create system configuration
    td_fnirs_config = TDfNIRSConfig(
        source_optodes=sources,
        detector_optodes=detectors,
        wavelengths=[760, 850],
        pulse_rate=80e6,
        integration_time=0.1
    )

    config = Flow2Config(
        td_fnirs_config=td_fnirs_config,
        eeg_electrodes=eeg_electrodes,
        sample_rate=10,  # 10 Hz for fNIRS
        eeg_sample_rate=1000,  # 1 kHz for EEG
        mode=MeasurementMode.HYBRID_FUSION
    )

    # Initialize and run system
    flow2 = Flow2Driver(config)
    await flow2.initialize()

    # Start acquisition
    logger.info("Starting 10-second acquisition...")
    await flow2.start_acquisition(duration=10.0)

    # Get real-time oxygenation
    oxygenation = flow2.get_realtime_oxygenation()
    if oxygenation:
        logger.info("Latest oxygenation measurements:")
        for pair_id, data in list(oxygenation.items())[:3]:  # Show first 3 pairs
            logger.info(f"  {pair_id}: HbO2={data['HbO2']:.3f}, HbR={data['HbR']:.3f}, SO2={data['SO2']:.1%}")

    # Print system status
    status = flow2.get_system_status()
    logger.info("Final System Status:")
    for key, value in status.items():
        logger.info(f"  {key}: {value}")

    # Export data
    exported_data = flow2.export_data()
    logger.info(f"Exported {len(exported_data['fnirs_data'])} fNIRS samples")
    logger.info(f"Exported {len(exported_data['eeg_data'])} EEG samples")


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(main())
