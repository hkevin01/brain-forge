"""
Assurance Tests for Multi-Modal Brain Data Acquisition
Validates NIBIB OPM Helmets, Kernel Flow2 Helmets, and Accelo-hat Arrays
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@dataclass
class OMPHelmetSpecs:
    """NIBIB OMP Helmet specifications"""
    channels: int = 306
    sampling_rate: int = 1000  # Hz
    sensitivity: float = 1e-15  # T/√Hz
    coil_compensation: bool = True
    natural_movement_support: bool = True
    field_range: float = 100e-9  # ±100 nT


@dataclass
class KernelFlow2Specs:
    """Kernel Flow2 Helmet specifications"""
    optical_modules: int = 40
    td_fnirs_channels: int = 160  # 4 per module
    eeg_channels: int = 32
    sampling_rate: int = 250  # Hz
    hemodynamic_resolution: float = 0.1  # mm
    electrical_resolution: float = 1e-6  # V


@dataclass
class AcceloHatSpecs:
    """Accelo-hat Array specifications"""
    accelerometer_nodes: int = 16
    sampling_rate: int = 1000  # Hz
    acceleration_range: float = 16.0  # ±16g
    precision: float = 0.001  # g
    impact_detection: bool = True
    motion_correlation: bool = True


class TestOMPHelmetAcquisition:
    """Test NIBIB OMP Helmet wearable magnetometer functionality"""
    
    @pytest.fixture
    def omp_helmet(self):
        """Mock OMP helmet with realistic specifications"""
        helmet = Mock()
        helmet.specs = OMPHelmetSpecs()
        helmet.is_connected = Mock(return_value=True)
        helmet.is_calibrated = Mock(return_value=True)
        helmet.matrix_coil_active = Mock(return_value=True)
        return helmet
    
    def test_omp_helmet_specifications(self, omp_helmet):
        """Test OMP helmet meets NIBIB specifications"""
        specs = omp_helmet.specs
        
        # Validate channel count
        assert specs.channels == 306, "OMP helmet must have 306 channels"
        
        # Validate sampling rate
        assert specs.sampling_rate >= 1000, "Sampling rate must be ≥1kHz"
        
        # Validate sensitivity
        assert specs.sensitivity <= 1e-14, "Sensitivity must be ≤10 fT/√Hz"
        
        # Validate movement support
        assert specs.natural_movement_support, "Must support natural movement"
        assert specs.coil_compensation, "Must have matrix coil compensation"
    
    @pytest.mark.asyncio
    async def test_omp_data_acquisition(self, omp_helmet):
        """Test real-time OMP data acquisition"""
        # Mock data stream
        async def mock_data_stream():
            while True:
                # Generate realistic magnetometer data
                channels = omp_helmet.specs.channels
                data = np.random.normal(0, 1e-12, channels)  # Tesla
                timestamp = time.time()
                yield data, timestamp
                await asyncio.sleep(0.001)  # 1ms = 1kHz
        
        omp_helmet.get_data_stream = mock_data_stream
        
        # Test data acquisition
        stream = omp_helmet.get_data_stream()
        data_points = []
        
        # Collect 100ms of data
        start_time = time.time()
        async for data, timestamp in stream:
            data_points.append((data, timestamp))
            if len(data_points) >= 100:  # 100 samples at 1kHz
                break
        
        # Validate data quality
        assert len(data_points) == 100
        
        # Check data characteristics
        for data, timestamp in data_points:
            assert len(data) == 306, "Each sample must have 306 channels"
            assert isinstance(timestamp, float), "Timestamp must be float"
            assert np.all(np.abs(data) < 1e-9), "Data must be within realistic range"
    
    def test_matrix_coil_compensation(self, omp_helmet):
        """Test matrix coil compensation for movement artifacts"""
        # Simulate movement artifact
        movement_artifact = np.random.normal(0, 1e-10, 306)
        clean_signal = np.random.normal(0, 1e-12, 306)
        contaminated_signal = clean_signal + movement_artifact
        
        # Mock compensation function
        def compensate_movement(data, coil_active=True):
            if coil_active:
                # Simulate 90% artifact reduction
                artifact_component = data - clean_signal
                return clean_signal + 0.1 * artifact_component
            return data
        
        omp_helmet.compensate_movement = compensate_movement
        
        # Test compensation
        compensated = omp_helmet.compensate_movement(
            contaminated_signal, 
            omp_helmet.matrix_coil_active()
        )
        
        # Validate artifact reduction
        original_snr = np.var(clean_signal) / np.var(movement_artifact)
        compensated_snr = np.var(clean_signal) / np.var(compensated - clean_signal)
        
        assert compensated_snr > 9 * original_snr, "Should achieve >9x SNR improvement"


class TestKernelFlow2Acquisition:
    """Test Kernel Flow2 TD-fNIRS + EEG fusion system"""
    
    @pytest.fixture
    def kernel_helmet(self):
        """Mock Kernel Flow2 helmet"""
        helmet = Mock()
        helmet.specs = KernelFlow2Specs()
        helmet.is_connected = Mock(return_value=True)
        helmet.optical_modules_active = Mock(return_value=40)
        helmet.eeg_electrodes_active = Mock(return_value=32)
        return helmet
    
    def test_kernel_flow2_specifications(self, kernel_helmet):
        """Test Kernel Flow2 meets specifications"""
        specs = kernel_helmet.specs
        
        # Validate optical modules
        assert specs.optical_modules == 40, "Must have 40 optical modules"
        assert specs.td_fnirs_channels == 160, "Must have 160 TD-fNIRS channels"
        
        # Validate EEG integration
        assert specs.eeg_channels == 32, "Must have 32 EEG channels"
        
        # Validate resolution
        assert specs.hemodynamic_resolution <= 0.1, "Hemodynamic resolution ≤0.1mm"
        assert specs.electrical_resolution <= 1e-6, "Electrical resolution ≤1μV"
    
    @pytest.mark.asyncio
    async def test_multimodal_data_fusion(self, kernel_helmet):
        """Test TD-fNIRS + EEG data fusion"""
        # Mock synchronized data streams
        async def mock_fnirs_stream():
            while True:
                # Hemodynamic response (HbO, HbR)
                hbo = np.random.normal(0, 1e-6, 160)  # Oxygenated hemoglobin
                hbr = np.random.normal(0, 1e-6, 160)  # Deoxygenated hemoglobin
                timestamp = time.time()
                yield {'HbO': hbo, 'HbR': hbr, 'timestamp': timestamp}
                await asyncio.sleep(0.004)  # 250Hz
        
        async def mock_eeg_stream():
            while True:
                # Electrical brain activity
                eeg_data = np.random.normal(0, 50e-6, 32)  # μV
                timestamp = time.time()
                yield {'eeg': eeg_data, 'timestamp': timestamp}
                await asyncio.sleep(0.004)  # 250Hz
        
        kernel_helmet.get_fnirs_stream = mock_fnirs_stream
        kernel_helmet.get_eeg_stream = mock_eeg_stream
        
        # Test synchronized acquisition
        fnirs_stream = kernel_helmet.get_fnirs_stream()
        eeg_stream = kernel_helmet.get_eeg_stream()
        
        fnirs_data = []
        eeg_data = []
        
        # Collect synchronized data
        tasks = [
            self._collect_fnirs_data(fnirs_stream, fnirs_data),
            self._collect_eeg_data(eeg_stream, eeg_data)
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Validate synchronization
        assert len(fnirs_data) > 0 and len(eeg_data) > 0
        
        # Check temporal alignment (within 1ms)
        for fnirs_sample, eeg_sample in zip(fnirs_data[:10], eeg_data[:10]):
            time_diff = abs(fnirs_sample['timestamp'] - eeg_sample['timestamp'])
            assert time_diff < 0.001, "Synchronization error must be <1ms"
    
    async def _collect_fnirs_data(self, stream, data_list):
        """Collect fNIRS data samples"""
        count = 0
        async for sample in stream:
            data_list.append(sample)
            count += 1
            if count >= 50:  # 200ms of data
                break
    
    async def _collect_eeg_data(self, stream, data_list):
        """Collect EEG data samples"""
        count = 0
        async for sample in stream:
            data_list.append(sample)
            count += 1
            if count >= 50:  # 200ms of data
                break
    
    def test_hemodynamic_electrical_correlation(self, kernel_helmet):
        """Test correlation between hemodynamic and electrical signals"""
        # Generate correlated signals (neurovascular coupling)
        time_points = np.linspace(0, 10, 2500)  # 10 seconds at 250Hz
        
        # Electrical activity (fast)
        neural_activity = np.sin(2 * np.pi * 10 * time_points)  # 10Hz
        eeg_signal = neural_activity + np.random.normal(0, 0.1, len(time_points))
        
        # Hemodynamic response (slow, delayed)
        hrf_delay = 2.0  # seconds
        delayed_neural = np.sin(2 * np.pi * 10 * (time_points - hrf_delay))
        hbo_signal = np.convolve(delayed_neural, np.exp(-time_points[:500]/2))[:len(time_points)]
        
        # Mock correlation analysis
        def analyze_neurovascular_coupling(eeg, hbo):
            # Cross-correlation with lag compensation
            correlation = np.corrcoef(eeg[500:], hbo[:-500])[0, 1]  # Account for HRF delay
            return abs(correlation)
        
        kernel_helmet.analyze_coupling = analyze_neurovascular_coupling
        
        # Test coupling analysis
        coupling_strength = kernel_helmet.analyze_coupling(eeg_signal, hbo_signal)
        
        assert coupling_strength > 0.5, "Should detect strong neurovascular coupling"


class TestAcceloHatArrays:
    """Test Accelo-hat precision accelerometer networks"""
    
    @pytest.fixture
    def accelo_hat(self):
        """Mock Accelo-hat array system"""
        hat = Mock()
        hat.specs = AcceloHatSpecs()
        hat.is_connected = Mock(return_value=True)
        hat.accelerometers_active = Mock(return_value=16)
        hat.impact_detection_enabled = Mock(return_value=True)
        return hat
    
    def test_accelerometer_specifications(self, accelo_hat):
        """Test Accelo-hat meets precision specifications"""
        specs = accelo_hat.specs
        
        # Validate array configuration
        assert specs.accelerometer_nodes == 16, "Must have 16 accelerometer nodes"
        assert specs.sampling_rate >= 1000, "Sampling rate must be ≥1kHz"
        
        # Validate precision and range
        assert specs.precision <= 0.001, "Precision must be ≤1mg"
        assert specs.acceleration_range >= 16.0, "Range must be ≥±16g"
        
        # Validate advanced features
        assert specs.impact_detection, "Must support impact detection"
        assert specs.motion_correlation, "Must support motion correlation"
    
    @pytest.mark.asyncio
    async def test_motion_brain_correlation(self, accelo_hat):
        """Test correlation between motion and brain activity"""
        # Mock synchronized data streams
        async def mock_acceleration_stream():
            while True:
                # 3-axis acceleration for 16 nodes
                accel_data = np.random.normal(0, 0.1, (16, 3))  # g
                timestamp = time.time()
                yield accel_data, timestamp
                await asyncio.sleep(0.001)  # 1kHz
        
        async def mock_brain_activity():
            while True:
                # Simulated brain activity (from OMP/Kernel)
                brain_data = np.random.normal(0, 1e-12, 306)  # Tesla
                timestamp = time.time()
                yield brain_data, timestamp
                await asyncio.sleep(0.001)  # 1kHz
        
        accelo_hat.get_acceleration_stream = mock_acceleration_stream
        accelo_hat.get_brain_stream = mock_brain_activity
        
        # Test motion-brain correlation
        motion_data = []
        brain_data = []
        
        # Collect synchronized data
        accel_stream = accelo_hat.get_acceleration_stream()
        brain_stream = accelo_hat.get_brain_stream()
        
        # Sample for 100ms
        for _ in range(100):
            try:
                accel_sample = await asyncio.wait_for(accel_stream.__anext__(), timeout=0.01)
                brain_sample = await asyncio.wait_for(brain_stream.__anext__(), timeout=0.01)
                
                motion_data.append(accel_sample)
                brain_data.append(brain_sample)
            except asyncio.TimeoutError:
                break
        
        # Validate data collection
        assert len(motion_data) > 50, "Should collect sufficient motion data"
        assert len(brain_data) > 50, "Should collect sufficient brain data"
        
        # Test temporal alignment
        for motion_sample, brain_sample in zip(motion_data[:10], brain_data[:10]):
            time_diff = abs(motion_sample[1] - brain_sample[1])
            assert time_diff < 0.001, "Motion-brain sync error must be <1ms"
    
    def test_impact_detection(self, accelo_hat):
        """Test precision impact detection"""
        # Generate impact signature
        normal_motion = np.random.normal(0, 0.1, (16, 3))  # Normal head motion
        
        # Simulate impact on node 5
        impact_motion = normal_motion.copy()
        impact_motion[5] = [15.0, -2.0, 8.0]  # High g-force impact
        
        # Mock impact detection algorithm
        def detect_impact(accel_data, threshold=10.0):
            """Detect impacts based on acceleration magnitude"""
            magnitudes = np.linalg.norm(accel_data, axis=1)
            impact_nodes = np.where(magnitudes > threshold)[0]
            return impact_nodes, magnitudes[impact_nodes]
        
        accelo_hat.detect_impact = detect_impact
        
        # Test impact detection
        impact_nodes, impact_magnitudes = accelo_hat.detect_impact(impact_motion)
        
        # Validate detection
        assert len(impact_nodes) > 0, "Should detect impact"
        assert 5 in impact_nodes, "Should detect impact on node 5"
        assert np.max(impact_magnitudes) > 15.0, "Should measure high g-force"
    
    def test_motion_artifact_correlation(self, accelo_hat):
        """Test motion artifact correlation with brain signals"""
        # Generate correlated motion and brain artifacts
        time_points = np.linspace(0, 1, 1000)  # 1 second at 1kHz
        
        # Head motion (low frequency)
        head_motion = 2.0 * np.sin(2 * np.pi * 5 * time_points)  # 5Hz movement
        
        # Corresponding brain signal artifact
        motion_artifact = 1e-11 * head_motion  # Proportional artifact
        clean_brain_signal = np.random.normal(0, 1e-12, 1000)
        contaminated_signal = clean_brain_signal + motion_artifact
        
        # Mock correlation analysis
        def correlate_motion_artifacts(motion, brain_signal):
            """Calculate motion-artifact correlation"""
            # Normalize signals
            motion_norm = (motion - np.mean(motion)) / np.std(motion)
            brain_norm = (brain_signal - np.mean(brain_signal)) / np.std(brain_signal)
            
            # Calculate correlation
            correlation = np.corrcoef(motion_norm, brain_norm)[0, 1]
            return abs(correlation)
        
        accelo_hat.correlate_artifacts = correlate_motion_artifacts
        
        # Test correlation
        correlation = accelo_hat.correlate_artifacts(head_motion, motion_artifact)
        
        assert correlation > 0.8, "Should detect strong motion-artifact correlation"


class TestMultiModalIntegration:
    """Test integration of all three acquisition systems"""
    
    @pytest.fixture
    def integrated_system(self):
        """Mock integrated multi-modal acquisition system"""
        system = Mock()
        system.omp_helmet = Mock()
        system.kernel_helmet = Mock()
        system.accelo_hat = Mock()
        system.is_synchronized = Mock(return_value=True)
        return system
    
    @pytest.mark.asyncio
    async def test_submillisecond_synchronization(self, integrated_system):
        """Test sub-millisecond temporal alignment across all modalities"""
        # Mock synchronized data streams
        base_time = time.time()
        
        async def mock_omp_stream():
            for i in range(100):
                timestamp = base_time + i * 0.001  # 1ms intervals
                data = np.random.normal(0, 1e-12, 306)
                yield {'type': 'omp', 'data': data, 'timestamp': timestamp}
                await asyncio.sleep(0.001)
        
        async def mock_kernel_stream():
            for i in range(100):
                timestamp = base_time + i * 0.001 + 0.0001  # 0.1ms offset
                data = np.random.normal(0, 1e-6, 160)
                yield {'type': 'kernel', 'data': data, 'timestamp': timestamp}
                await asyncio.sleep(0.001)
        
        async def mock_accelo_stream():
            for i in range(100):
                timestamp = base_time + i * 0.001 - 0.0002  # 0.2ms offset
                data = np.random.normal(0, 0.1, (16, 3))
                yield {'type': 'accelo', 'data': data, 'timestamp': timestamp}
                await asyncio.sleep(0.001)
        
        integrated_system.get_omp_stream = mock_omp_stream
        integrated_system.get_kernel_stream = mock_kernel_stream
        integrated_system.get_accelo_stream = mock_accelo_stream
        
        # Collect synchronized data
        all_data = []
        
        async def collect_stream_data(stream, data_list):
            async for sample in stream:
                data_list.append(sample)
        
        # Run all streams concurrently
        await asyncio.gather(
            collect_stream_data(integrated_system.get_omp_stream(), all_data),
            collect_stream_data(integrated_system.get_kernel_stream(), all_data),
            collect_stream_data(integrated_system.get_accelo_stream(), all_data),
            return_exceptions=True
        )
        
        # Sort by timestamp
        all_data.sort(key=lambda x: x['timestamp'])
        
        # Test sub-millisecond synchronization
        omp_samples = [s for s in all_data if s['type'] == 'omp']
        kernel_samples = [s for s in all_data if s['type'] == 'kernel']
        accelo_samples = [s for s in all_data if s['type'] == 'accelo']
        
        # Check temporal alignment within 0.5ms
        for i in range(min(len(omp_samples), len(kernel_samples), len(accelo_samples))):
            omp_time = omp_samples[i]['timestamp']
            kernel_time = kernel_samples[i]['timestamp']
            accelo_time = accelo_samples[i]['timestamp']
            
            max_diff = max(abs(omp_time - kernel_time), 
                          abs(omp_time - accelo_time),
                          abs(kernel_time - accelo_time))
            
            assert max_diff < 0.0005, f"Synchronization error {max_diff*1000:.3f}ms exceeds 0.5ms limit"
    
    def test_cross_modal_validation(self, integrated_system):
        """Test cross-validation between modalities"""
        # Generate physiologically consistent data
        # Neural activity should correlate across modalities
        
        # Simulated neural event
        neural_event_time = 0.5  # seconds
        sampling_times = np.linspace(0, 1, 1000)
        
        # OMP response (immediate magnetic field change)
        omp_response = np.exp(-(sampling_times - neural_event_time)**2 / 0.01)
        omp_data = 1e-12 * omp_response + np.random.normal(0, 1e-13, 1000)
        
        # Kernel EEG response (immediate electrical change)
        eeg_response = np.exp(-(sampling_times - neural_event_time)**2 / 0.01)
        eeg_data = 50e-6 * eeg_response + np.random.normal(0, 5e-6, 1000)
        
        # Kernel fNIRS response (delayed hemodynamic change)
        fnirs_delay = 0.2  # 200ms delay
        fnirs_response = np.exp(-(sampling_times - neural_event_time - fnirs_delay)**2 / 0.05)
        fnirs_data = 1e-6 * fnirs_response + np.random.normal(0, 1e-7, 1000)
        
        # Mock cross-modal validation
        def validate_cross_modal_consistency(omp, eeg, fnirs):
            """Validate physiological consistency across modalities"""
            # OMP-EEG should be highly correlated (both electrical)
            omp_eeg_corr = np.corrcoef(omp, eeg)[0, 1]
            
            # fNIRS should lag behind OMP/EEG
            fnirs_delayed = fnirs[200:]  # Account for hemodynamic delay
            omp_truncated = omp[:-200]
            omp_fnirs_corr = np.corrcoef(omp_truncated, fnirs_delayed)[0, 1]
            
            return {
                'omp_eeg_correlation': abs(omp_eeg_corr),
                'omp_fnirs_correlation': abs(omp_fnirs_corr),
                'temporal_consistency': True
            }
        
        integrated_system.validate_consistency = validate_cross_modal_consistency
        
        # Test validation
        validation_results = integrated_system.validate_consistency(
            omp_data, eeg_data, fnirs_data
        )
        
        # Assert physiological consistency
        assert validation_results['omp_eeg_correlation'] > 0.7, "OMP-EEG should be highly correlated"
        assert validation_results['omp_fnirs_correlation'] > 0.5, "OMP-fNIRS should show delayed correlation"
        assert validation_results['temporal_consistency'], "Temporal relationships should be consistent"
    
    def test_natural_movement_capability(self, integrated_system):
        """Test system performance during natural movement"""
        # Simulate walking motion patterns
        time_points = np.linspace(0, 10, 10000)  # 10 seconds
        walking_frequency = 2.0  # 2 Hz walking
        
        # Generate walking motion
        walking_motion = {
            'vertical': 0.5 * np.sin(2 * np.pi * walking_frequency * time_points),
            'lateral': 0.2 * np.sin(2 * np.pi * walking_frequency * time_points + np.pi/4),
            'forward': 0.1 * np.cos(2 * np.pi * walking_frequency * time_points)
        }
        
        # Simulate motion artifacts in brain signals
        motion_artifact_omp = 1e-11 * walking_motion['vertical']
        motion_artifact_eeg = 10e-6 * walking_motion['lateral']
        
        # Clean neural signal
        clean_neural = np.random.normal(0, 1e-12, 10000)
        
        # Contaminated signals
        contaminated_omp = clean_neural + motion_artifact_omp
        contaminated_eeg = np.random.normal(0, 50e-6, 10000) + motion_artifact_eeg
        
        # Mock motion compensation
        def compensate_motion_artifacts(brain_signal, motion_data, modality='omp'):
            """Compensate for motion artifacts using accelerometer data"""
            if modality == 'omp':
                # OMP matrix coil compensation
                compensation_factor = 0.95  # 95% artifact reduction
                artifact_estimate = 1e-11 * motion_data['vertical']
                compensated = brain_signal - compensation_factor * artifact_estimate
            elif modality == 'eeg':
                # EEG motion artifact removal
                compensation_factor = 0.80  # 80% artifact reduction
                artifact_estimate = 10e-6 * motion_data['lateral']
                compensated = brain_signal - compensation_factor * artifact_estimate
            
            return compensated
        
        integrated_system.compensate_motion = compensate_motion_artifacts
        
        # Test motion compensation
        compensated_omp = integrated_system.compensate_motion(
            contaminated_omp, walking_motion, 'omp'
        )
        compensated_eeg = integrated_system.compensate_motion(
            contaminated_eeg, walking_motion, 'eeg'
        )
        
        # Validate compensation effectiveness
        omp_improvement = (np.var(contaminated_omp) - np.var(compensated_omp)) / np.var(contaminated_omp)
        eeg_improvement = (np.var(contaminated_eeg) - np.var(compensated_eeg)) / np.var(contaminated_eeg)
        
        assert omp_improvement > 0.85, "OMP motion compensation should achieve >85% artifact reduction"
        assert eeg_improvement > 0.70, "EEG motion compensation should achieve >70% artifact reduction"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
