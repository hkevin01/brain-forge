"""
Brain-Forge Multi-Modal BCI Integration System
=============================================

This module integrates all four hardware systems for Version 0.2.0:
- NIBIB OMP helmet with matrix coil compensation
- Kernel Flow2 TD-fNIRS + EEG fusion
- Brown Accelo-hat accelerometer array
- Microsecond-precision synchronization

Provides unified interface for multi-modal BCI data acquisition with
real-time processing and sub-millisecond synchronization.

Author: Brain-Forge Development Team
Date: 2025-01-28
License: MIT
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..device_drivers.accelo_hat import AcceloHatConfig, AcceloHatDriver
from ..device_drivers.flow2_td_fnirs import (Flow2Config, Flow2Driver,
                                             MeasurementMode)
# Import our hardware drivers
from ..device_drivers.omp_helmet import OMPHelmetConfig, OMPHelmetDriver
from ..synchronization.multimodal_sync import (ModalityConfig, ModalityType,
                                               MultiModalSynchronizer)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize random number generator
rng = np.random.default_rng(42)


@dataclass
class BrainForgeConfig:
    """Complete Brain-Forge system configuration"""
    omp_config: OMPHelmetConfig
    flow2_config: Flow2Config
    accelo_config: AcceloHatConfig
    sync_rate: float = 100.0  # Hz
    output_directory: str = "./brain_forge_data"
    enable_realtime_display: bool = True
    processing_latency_target_ms: float = 100.0  # <100ms target


class BrainForgeSystem:
    """
    Main Brain-Forge multi-modal BCI system

    Coordinates all hardware systems with microsecond synchronization
    and real-time processing for brain-computer interface applications.
    """

    def __init__(self, config: BrainForgeConfig):
        self.config = config

        # Initialize hardware drivers
        self.omp_driver = None
        self.flow2_driver = None
        self.accelo_driver = None
        self.synchronizer = None

        # System state
        self.is_running = False
        self.session_start_time = None
        self.output_dir = Path(config.output_directory)

        # Real-time processing
        self.processing_stats = {
            'total_samples': 0,
            'processing_latency_ms': 0.0,
            'sync_quality_percent': 0.0,
            'data_throughput_mbps': 0.0
        }

        # Data fusion results
        self.latest_fusion_result = None

        logger.info("Brain-Forge system initialized")

    async def initialize_hardware(self):
        """Initialize all hardware systems"""
        logger.info("Initializing Brain-Forge hardware systems...")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OMP helmet
        logger.info("Initializing NIBIB OMP helmet...")
        self.omp_driver = OMPHelmetDriver(self.config.omp_config)
        await self.omp_driver.initialize()

        # Initialize Flow2 TD-fNIRS + EEG
        logger.info("Initializing Kernel Flow2 TD-fNIRS + EEG...")
        self.flow2_driver = Flow2Driver(self.config.flow2_config)
        await self.flow2_driver.initialize()

        # Initialize Accelo-hat
        logger.info("Initializing Brown Accelo-hat...")
        self.accelo_driver = AcceloHatDriver(self.config.accelo_config)
        await self.accelo_driver.initialize()

        # Initialize synchronization system
        logger.info("Initializing microsecond synchronization...")
        modality_configs = [
            ModalityConfig(
                modality_id="omp_helmet",
                modality_type=ModalityType.OMP_MAGNETOMETRY,
                sample_rate=self.config.omp_config.sample_rate,
                priority=3,
                sync_tolerance_us=25.0
            ),
            ModalityConfig(
                modality_id="flow2_fnirs",
                modality_type=ModalityType.TD_FNIRS,
                sample_rate=self.config.flow2_config.sample_rate,
                priority=2,
                sync_tolerance_us=50.0
            ),
            ModalityConfig(
                modality_id="flow2_eeg",
                modality_type=ModalityType.EEG,
                sample_rate=self.config.flow2_config.eeg_sample_rate,
                priority=3,
                sync_tolerance_us=10.0
            ),
            ModalityConfig(
                modality_id="accelo_hat",
                modality_type=ModalityType.ACCELEROMETRY,
                sample_rate=self.config.accelo_config.sample_rate,
                priority=1,
                sync_tolerance_us=5.0
            )
        ]

        self.synchronizer = MultiModalSynchronizer(modality_configs, self.config.sync_rate)

        # Register synchronization callback
        self.synchronizer.register_sync_callback(self._process_synchronized_sample)

        logger.info("All hardware systems initialized successfully")

    def _process_synchronized_sample(self, sync_sample):
        """Process synchronized multi-modal sample"""
        processing_start = time.time()

        # Extract data from each modality
        omp_data = sync_sample.modality_data.get('omp_helmet')
        fnirs_data = sync_sample.modality_data.get('flow2_fnirs')
        eeg_data = sync_sample.modality_data.get('flow2_eeg')
        accel_data = sync_sample.modality_data.get('accelo_hat')

        # Perform multi-modal data fusion
        fusion_result = self._fuse_multimodal_data(
            omp_data, fnirs_data, eeg_data, accel_data, sync_sample.master_timestamp
        )

        # Update processing statistics
        processing_time = (time.time() - processing_start) * 1000  # ms
        self.processing_stats['processing_latency_ms'] = (
            0.9 * self.processing_stats['processing_latency_ms'] +
            0.1 * processing_time
        )
        self.processing_stats['total_samples'] += 1

        # Store latest result
        self.latest_fusion_result = fusion_result

        # Log significant events
        if fusion_result.get('significant_event'):
            logger.info(f"Significant event detected: {fusion_result['event_type']}")

    def _fuse_multimodal_data(self, omp_data, fnirs_data, eeg_data, accel_data, timestamp):
        """Fuse data from all modalities"""
        fusion_result = {
            'timestamp': timestamp,
            'modalities_present': [],
            'brain_state': {},
            'movement_state': {},
            'significant_event': False,
            'event_type': None,
            'confidence': 0.0
        }

        # Process OMP magnetometry data
        if omp_data is not None:
            fusion_result['modalities_present'].append('omp')
            # Extract magnetic field features
            field_magnitude = np.mean(np.linalg.norm(omp_data['magnetic_field'], axis=1))
            fusion_result['brain_state']['magnetic_field_nt'] = float(field_magnitude * 1e9)  # nT

            # Check for magnetic artifacts
            if field_magnitude > 1e-12:  # 1 pT threshold
                fusion_result['significant_event'] = True
                fusion_result['event_type'] = 'magnetic_artifact'

        # Process TD-fNIRS data
        if fnirs_data is not None:
            fusion_result['modalities_present'].append('fnirs')
            # Extract oxygenation features
            if 'HbO2' in fnirs_data:
                fusion_result['brain_state']['hbo2_change'] = fnirs_data['HbO2']
                fusion_result['brain_state']['hbr_change'] = fnirs_data.get('HbR', 0)

                # Detect significant oxygenation changes
                if abs(fnirs_data['HbO2']) > 0.5:  # μM threshold
                    fusion_result['significant_event'] = True
                    fusion_result['event_type'] = 'oxygenation_change'

        # Process EEG data
        if eeg_data is not None:
            fusion_result['modalities_present'].append('eeg')
            # Extract spectral features (simplified)
            if hasattr(eeg_data, 'shape') and len(eeg_data.shape) > 0:
                eeg_power = np.mean(eeg_data**2)
                fusion_result['brain_state']['eeg_power_uv2'] = float(eeg_power * 1e12)  # μV²

                # Detect high-power events (artifacts or seizures)
                if eeg_power > 100e-12:  # 100 μV² threshold
                    fusion_result['significant_event'] = True
                    fusion_result['event_type'] = 'high_eeg_activity'

        # Process accelerometry data
        if accel_data is not None:
            fusion_result['modalities_present'].append('accelerometry')
            # Extract movement features
            if hasattr(accel_data, 'shape') and len(accel_data.shape) > 1:
                accel_magnitude = np.linalg.norm(accel_data, axis=1)
                max_accel = np.max(accel_magnitude)
                fusion_result['movement_state']['max_acceleration_g'] = float(max_accel)

                # Detect impacts
                if max_accel > 10.0:  # 10g threshold
                    fusion_result['significant_event'] = True
                    fusion_result['event_type'] = 'impact_detected'

        # Compute overall confidence based on number of modalities
        n_modalities = len(fusion_result['modalities_present'])
        fusion_result['confidence'] = min(1.0, n_modalities / 4.0)

        return fusion_result

    async def start_acquisition(self, duration: Optional[float] = None):
        """Start multi-modal data acquisition"""
        if self.is_running:
            logger.warning("Acquisition already in progress")
            return

        logger.info("Starting Brain-Forge multi-modal acquisition...")
        if duration:
            logger.info(f"Acquisition duration: {duration:.1f} seconds")

        self.is_running = True
        self.session_start_time = time.time()

        # Start all systems concurrently
        tasks = []

        # Start synchronization
        sync_task = asyncio.create_task(self.synchronizer.start_synchronization())
        tasks.append(sync_task)

        # Start hardware data acquisition with data forwarding
        omp_task = asyncio.create_task(self._run_omp_acquisition(duration))
        flow2_task = asyncio.create_task(self._run_flow2_acquisition(duration))
        accelo_task = asyncio.create_task(self._run_accelo_acquisition(duration))

        tasks.extend([omp_task, flow2_task, accelo_task])

        # Wait for completion or duration
        try:
            if duration:
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=duration + 5.0)
            else:
                await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            logger.info("Acquisition completed by timeout")
        except Exception as e:
            logger.error(f"Acquisition error: {e}")
        finally:
            await self.stop_acquisition()

    async def _run_omp_acquisition(self, duration: Optional[float]):
        """Run OMP acquisition with data forwarding"""
        # Custom acquisition loop to forward data to synchronizer
        sample_interval = 1.0 / self.config.omp_config.sample_rate
        start_time = time.time()

        while self.is_running:
            if duration and (time.time() - start_time) >= duration:
                break

            # Simulate getting sample from OMP driver
            # In real implementation, this would be from actual driver
            sample_data = {
                'magnetic_field': rng.normal(0, 1e-15, (48, 3)),  # 48 sensors, 3 axes
                'compensation_active': True,
                'field_quality': 'good'
            }

            # Forward to synchronizer
            self.synchronizer.add_sample('omp_helmet', sample_data)

            await asyncio.sleep(sample_interval)

    async def _run_flow2_acquisition(self, duration: Optional[float]):
        """Run Flow2 acquisition with data forwarding"""
        fnirs_interval = 1.0 / self.config.flow2_config.sample_rate
        eeg_interval = 1.0 / self.config.flow2_config.eeg_sample_rate
        start_time = time.time()

        fnirs_counter = 0
        eeg_counter = 0

        while self.is_running:
            if duration and (time.time() - start_time) >= duration:
                break

            current_time = time.time()

            # fNIRS data (lower rate)
            if (current_time - start_time) >= fnirs_counter * fnirs_interval:
                fnirs_data = {
                    'HbO2': rng.normal(0, 0.1),
                    'HbR': rng.normal(0, 0.05),
                    'HbT': 0.0,
                    'SO2': 0.75
                }
                self.synchronizer.add_sample('flow2_fnirs', fnirs_data)
                fnirs_counter += 1

            # EEG data (higher rate)
            if (current_time - start_time) >= eeg_counter * eeg_interval:
                eeg_data = rng.normal(0, 50e-6, 64)  # 64 channels
                self.synchronizer.add_sample('flow2_eeg', eeg_data)
                eeg_counter += 1

            await asyncio.sleep(min(fnirs_interval, eeg_interval) / 10)  # Fast polling

    async def _run_accelo_acquisition(self, duration: Optional[float]):
        """Run Accelo-hat acquisition with data forwarding"""
        sample_interval = 1.0 / self.config.accelo_config.sample_rate
        start_time = time.time()

        while self.is_running:
            if duration and (time.time() - start_time) >= duration:
                break

            # Simulate accelerometer data
            accel_data = rng.normal(0, 0.1, (8, 3))  # 8 sensors, 3 axes
            accel_data[:, 2] += 1.0  # Add gravity

            # Forward to synchronizer
            self.synchronizer.add_sample('accelo_hat', accel_data)

            await asyncio.sleep(sample_interval)

    async def stop_acquisition(self):
        """Stop all data acquisition"""
        logger.info("Stopping Brain-Forge acquisition...")
        self.is_running = False

        # Stop synchronization
        if self.synchronizer:
            await self.synchronizer.stop_synchronization()

        # Stop individual drivers
        if self.omp_driver:
            await self.omp_driver.stop_acquisition()
        if self.flow2_driver:
            await self.flow2_driver.stop_acquisition()
        if self.accelo_driver:
            await self.accelo_driver.stop_acquisition()

        logger.info("All systems stopped")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'system': {
                'is_running': self.is_running,
                'session_duration': time.time() - (self.session_start_time or time.time()),
                'processing_latency_ms': f"{self.processing_stats['processing_latency_ms']:.2f}",
                'total_samples': self.processing_stats['total_samples'],
                'target_latency_ms': self.config.processing_latency_target_ms
            }
        }

        # Get individual system status
        if self.omp_driver:
            status['omp_helmet'] = self.omp_driver.get_system_status()
        if self.flow2_driver:
            status['flow2'] = self.flow2_driver.get_system_status()
        if self.accelo_driver:
            status['accelo_hat'] = self.accelo_driver.get_system_status()
        if self.synchronizer:
            status['synchronization'] = self.synchronizer.get_sync_statistics()

        return status

    def get_latest_brain_state(self) -> Optional[Dict[str, Any]]:
        """Get latest fused brain state"""
        if self.latest_fusion_result:
            return {
                'timestamp': self.latest_fusion_result['timestamp'],
                'brain_state': self.latest_fusion_result['brain_state'],
                'movement_state': self.latest_fusion_result['movement_state'],
                'modalities_active': self.latest_fusion_result['modalities_present'],
                'confidence': self.latest_fusion_result['confidence'],
                'significant_event': self.latest_fusion_result['significant_event']
            }
        return None

    async def save_session_data(self, filename: Optional[str] = None):
        """Save complete session data"""
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"brain_forge_session_{timestamp}.json"

        filepath = self.output_dir / filename

        # Collect data from all systems
        session_data = {
            'metadata': {
                'timestamp': time.time(),
                'session_duration': time.time() - (self.session_start_time or time.time()),
                'brain_forge_version': '0.2.0',
                'config': {
                    'sync_rate': self.config.sync_rate,
                    'target_latency_ms': self.config.processing_latency_target_ms
                }
            },
            'system_status': self.get_system_status(),
            'processing_stats': self.processing_stats
        }

        # Add synchronized data
        if self.synchronizer:
            sync_data = self.synchronizer.export_synchronized_data()
            session_data['synchronized_data'] = sync_data

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)

        logger.info(f"Session data saved to {filepath}")
        return str(filepath)


# Example usage and testing
async def main():
    """Example Brain-Forge system usage"""

    # Create minimal configurations for testing
    # (In practice, these would be loaded from config files)

    # OMP helmet config (simplified)
    omp_config = OMPHelmetConfig(
        sensor_positions=np.random.randn(48, 3) * 0.1,  # 48 sensors
        sample_rate=1200,
        enable_compensation=True
    )

    # Flow2 config (simplified)
    flow2_config = Flow2Config(
        td_fnirs_config=None,  # Would be properly configured
        eeg_electrodes=[],     # Would be properly configured
        sample_rate=10,
        eeg_sample_rate=1000,
        mode=MeasurementMode.HYBRID_FUSION
    )

    # Accelo-hat config (simplified)
    accelo_config = AcceloHatConfig(
        accelerometers=[],  # Would be properly configured
        sample_rate=2000,
        enable_impact_detection=True
    )

    # Brain-Forge system config
    system_config = BrainForgeConfig(
        omp_config=omp_config,
        flow2_config=flow2_config,
        accelo_config=accelo_config,
        sync_rate=100.0,
        processing_latency_target_ms=100.0
    )

    # Initialize system
    brain_forge = BrainForgeSystem(system_config)
    await brain_forge.initialize_hardware()

    # Run acquisition
    logger.info("Starting 15-second multi-modal acquisition...")
    await brain_forge.start_acquisition(duration=15.0)

    # Get final status
    status = brain_forge.get_system_status()
    logger.info("Final System Status:")
    logger.info(f"  Processing latency: {status['system']['processing_latency_ms']} ms")
    logger.info(f"  Total samples: {status['system']['total_samples']}")

    # Get latest brain state
    brain_state = brain_forge.get_latest_brain_state()
    if brain_state:
        logger.info("Latest Brain State:")
        logger.info(f"  Active modalities: {brain_state['modalities_active']}")
        logger.info(f"  Confidence: {brain_state['confidence']:.2f}")
        logger.info(f"  Significant event: {brain_state['significant_event']}")

    # Save session data
    save_path = await brain_forge.save_session_data()
    logger.info(f"Session data saved to: {save_path}")

    logger.info("Brain-Forge Version 0.2.0 Multi-Modal Integration Complete!")


if __name__ == "__main__":
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
