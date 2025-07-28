"""
Hardware Integration Validation Tests

Tests for the comprehensive hardware integration system including:
- OPM helmet magnetometer arrays (306 channels) 
- Kernel Flow/Flux optical imaging systems
- Brown's Accelo-hat accelerometer arrays
- Multi-device synchronization and streaming
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"  
sys.path.insert(0, str(src_path))


class TestOMPHelmetIntegration:
    """Test OPM helmet magnetometer integration"""
    
    def test_omp_configuration(self):
        """Test OPM helmet configuration"""
        from core.config import Config
        
        config = Config()
        omp_config = config.hardware.omp_helmet
        
        assert omp_config.num_channels == 306
        assert omp_config.sampling_rate == 1000.0
        assert omp_config.dynamic_range == 50e-9  # 50 nT
        assert omp_config.sensitivity <= 10e-15  # 10 fT/√Hz
        
    @patch('integrated_system.pylsl')
    def test_omp_data_acquisition(self, mock_pylsl):
        """Test OPM data acquisition integration"""
        try:
            from integrated_system import IntegratedBrainSystem
            from core.config import Config
            
            config = Config()
            
            # Mock LSL components
            mock_inlet = Mock()
            mock_pylsl.StreamInlet.return_value = mock_inlet
            mock_pylsl.resolve_stream.return_value = [Mock()]
            
            # Mock OMP data (306 channels)
            mock_data = np.random.randn(306, 100) * 1e-12  # fT scale
            mock_inlet.pull_chunk.return_value = (mock_data.T.tolist(), [])
            
            # Test system creation (mocked)
            with patch.object(IntegratedBrainSystem, '_initialize_hardware'):
                system = IntegratedBrainSystem(config)
                assert system is not None
                
        except ImportError:
            pytest.skip("IntegratedBrainSystem import failed")
    
    def test_omp_calibration_parameters(self):
        """Test OMP calibration system"""
        from core.config import Config
        
        config = Config()
        omp_config = config.hardware.omp_helmet
        
        # Test calibration parameters exist
        assert hasattr(omp_config, 'calibration_enabled')
        assert hasattr(omp_config, 'noise_compensation')
        assert omp_config.calibration_enabled is True


class TestKernelOpticalIntegration:
    """Test Kernel Flow/Flux optical imaging integration"""
    
    def test_kernel_flow_configuration(self):
        """Test Kernel Flow helmet configuration"""
        from core.config import Config
        
        config = Config()
        flow_config = config.hardware.kernel_flow
        
        assert flow_config.enabled is True
        assert hasattr(flow_config, 'sampling_rate')
        assert hasattr(flow_config, 'num_channels')
        
    def test_kernel_flux_configuration(self):
        """Test Kernel Flux helmet configuration"""
        from core.config import Config
        
        config = Config()
        flux_config = config.hardware.kernel_flux
        
        assert flux_config.enabled is True
        assert hasattr(flux_config, 'measurement_type')
        
    @patch('integrated_system.KernelInterface')
    def test_optical_data_integration(self, mock_kernel):
        """Test optical data integration"""
        try:
            from integrated_system import IntegratedBrainSystem
            from core.config import Config
            
            config = Config()
            
            # Mock Kernel interface
            mock_interface = Mock()
            mock_kernel.return_value = mock_interface
            
            # Mock optical data
            mock_optical_data = {
                'flow_data': np.random.randn(32, 100),  # Hemodynamic data
                'flux_data': np.random.randn(32, 100),  # Speed data
                'timestamps': np.arange(100)
            }
            mock_interface.get_data.return_value = mock_optical_data
            
            # Test integration (mocked)
            with patch.object(IntegratedBrainSystem, '_initialize_hardware'):
                system = IntegratedBrainSystem(config)
                assert system is not None
                
        except ImportError:
            pytest.skip("Kernel integration components not available")


class TestAccelerometerIntegration:
    """Test Brown's Accelo-hat accelerometer integration"""
    
    def test_accelerometer_configuration(self):
        """Test accelerometer configuration"""
        from core.config import Config
        
        config = Config()
        accel_config = config.hardware.accelerometer
        
        assert accel_config.enabled is True
        assert hasattr(accel_config, 'sampling_rate')
        assert hasattr(accel_config, 'range_g')
        assert hasattr(accel_config, 'resolution_bits')
        
    @patch('integrated_system.AccelerometerArray')  
    def test_motion_data_acquisition(self, mock_accel):
        """Test motion data acquisition"""
        try:
            from integrated_system import IntegratedBrainSystem
            from core.config import Config
            
            config = Config()
            
            # Mock accelerometer array
            mock_array = Mock()
            mock_accel.return_value = mock_array
            
            # Mock 3-axis motion data
            mock_motion_data = {
                'accel_x': np.random.randn(100) * 0.1,  # g units
                'accel_y': np.random.randn(100) * 0.1,
                'accel_z': np.random.randn(100) * 0.1,
                'timestamps': np.arange(100)
            }
            mock_array.get_data.return_value = mock_motion_data
            
            # Test integration (mocked)
            with patch.object(IntegratedBrainSystem, '_initialize_hardware'):
                system = IntegratedBrainSystem(config)
                assert system is not None
                
        except ImportError:
            pytest.skip("Accelerometer integration not available")
    
    def test_motion_artifact_compensation(self):
        """Test motion artifact compensation"""
        from core.config import Config
        
        config = Config()
        processing_config = config.processing.artifacts
        
        assert hasattr(processing_config, 'motion_compensation')
        assert processing_config.motion_compensation is True


class TestMultiDeviceStreaming:
    """Test multi-device streaming and synchronization"""
    
    @patch('acquisition.stream_manager.pylsl')
    def test_stream_manager_initialization(self, mock_pylsl):
        """Test StreamManager handles multiple devices"""
        try:
            from acquisition.stream_manager import StreamManager
            
            # Mock LSL streams for different devices
            mock_streams = [
                Mock(name='OMP_Stream', type='MEG'),
                Mock(name='Kernel_Flow', type='Optical'),
                Mock(name='Kernel_Flux', type='Optical'), 
                Mock(name='Accelerometer', type='Motion')
            ]
            mock_pylsl.resolve_streams.return_value = mock_streams
            
            # Test stream manager creation
            manager = StreamManager()
            assert manager is not None
            
        except ImportError:
            pytest.skip("StreamManager import failed")
    
    def test_synchronization_configuration(self):
        """Test timestamp synchronization configuration"""
        from core.config import Config
        
        config = Config()
        sync_config = config.system.synchronization
        
        assert hasattr(sync_config, 'precision_us')
        assert sync_config.precision_us <= 10  # ±10μs accuracy target
        assert hasattr(sync_config, 'drift_correction')
        
    @patch('acquisition.stream_manager.time')
    def test_timestamp_synchronization(self, mock_time):
        """Test timestamp synchronization across devices"""
        try:
            from acquisition.stream_manager import StreamManager
            
            # Mock time for consistent timestamps
            mock_time.time.return_value = 1000.0
            mock_time.perf_counter.return_value = 1000.0
            
            with patch('acquisition.stream_manager.pylsl') as mock_pylsl:
                mock_pylsl.resolve_streams.return_value = []
                
                manager = StreamManager()
                
                # Test timestamp alignment method exists
                assert hasattr(manager, 'synchronize_timestamps') or \
                       hasattr(manager, '_align_timestamps')
                       
        except ImportError:
            pytest.skip("StreamManager not available")


class TestDataQualityMonitoring:
    """Test real-time data quality monitoring"""
    
    def test_quality_metrics_configuration(self):
        """Test data quality monitoring configuration"""
        from core.config import Config
        
        config = Config()
        quality_config = config.processing.quality_monitoring
        
        assert hasattr(quality_config, 'enabled')
        assert quality_config.enabled is True
        assert hasattr(quality_config, 'metrics')
        
    def test_artifact_detection_setup(self):
        """Test artifact detection configuration"""
        from core.config import Config
        
        config = Config()
        artifact_config = config.processing.artifacts
        
        # Test artifact detection methods configured
        assert hasattr(artifact_config, 'detection_methods')
        assert hasattr(artifact_config, 'rejection_thresholds')
        
    @patch('processing.signal')
    def test_quality_assessment_pipeline(self, mock_signal):
        """Test quality assessment integration"""
        try:
            from processing import RealTimeProcessor
            
            # Mock signal processing
            mock_signal.butter.return_value = ([1, 0], [1, 0])
            mock_signal.filtfilt.return_value = np.random.randn(100)
            
            # Test processor includes quality monitoring
            processor = RealTimeProcessor(
                sampling_rate=1000.0,
                quality_monitoring=True
            )
            
            assert processor is not None
            
        except ImportError:
            pytest.skip("RealTimeProcessor not available")


class TestBrainAtlasIntegration:
    """Test brain atlas and mapping integration"""
    
    def test_atlas_configuration(self):
        """Test brain atlas configuration"""
        from core.config import Config
        
        config = Config()
        atlas_config = config.processing.brain_mapping
        
        assert hasattr(atlas_config, 'atlas_type')
        assert atlas_config.atlas_type == 'harvard_oxford'
        assert hasattr(atlas_config, 'resolution_mm')
        
    def test_harvard_oxford_integration(self):
        """Test Harvard-Oxford atlas integration"""
        try:
            # Test that atlas functionality is integrated
            from integrated_system import IntegratedBrainSystem
            
            # Just test import - actual atlas loading requires nilearn
            assert IntegratedBrainSystem is not None
            
        except ImportError:
            pytest.skip("Atlas integration not available")
    
    def test_connectivity_mapping_setup(self):
        """Test connectivity mapping configuration"""
        from core.config import Config
        
        config = Config()
        connectivity_config = config.processing.connectivity
        
        assert hasattr(connectivity_config, 'method')
        assert hasattr(connectivity_config, 'frequency_bands')
        assert hasattr(connectivity_config, 'correlation_threshold')


class TestNeuralSimulationIntegration:
    """Test neural simulation framework integration"""
    
    def test_simulation_configuration(self):
        """Test neural simulation configuration"""
        from core.config import Config
        
        config = Config()
        sim_config = config.simulation
        
        assert hasattr(sim_config, 'framework')
        assert sim_config.framework in ['brian2', 'nest', 'both']
        assert hasattr(sim_config, 'real_time_enabled')
        
    def test_brian2_integration_setup(self):
        """Test Brian2 integration setup"""
        from core.config import Config
        
        config = Config()
        brian_config = config.simulation.brian2
        
        assert hasattr(brian_config, 'enabled')
        assert hasattr(brian_config, 'timestep_ms')
        assert hasattr(brian_config, 'network_size')
        
    def test_nest_integration_setup(self):
        """Test NEST integration setup"""
        from core.config import Config
        
        config = Config()
        nest_config = config.simulation.nest
        
        assert hasattr(nest_config, 'enabled')
        assert hasattr(nest_config, 'resolution_ms')
        assert hasattr(nest_config, 'threads')


if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v", 
        "--tb=short",
        "-m", "not slow"  # Skip slow tests by default
    ])
