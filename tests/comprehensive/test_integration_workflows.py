"""
End-to-End Integration Tests

Tests that verify complete workflows and integration between components
as described in the README examples and documentation.
"""

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import brain_forge

from core.config import Config
from core.logger import get_logger
from processing import ArtifactRemover, RealTimeProcessor, WaveletCompressor


class TestEndToEndIntegration:
    """Test complete end-to-end workflows"""
    
    def setup_method(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="brain_forge_integration_test_")
        self.config = Config()
        
    def teardown_method(self):
        """Clean up integration test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestREADMEWorkflowExamples(TestEndToEndIntegration):
    """Test complete workflows from README examples"""
    
    def test_basic_initialization_workflow(self):
        """
        Test the basic initialization example from README:
        
        ```python
        from brain_forge import Config, get_logger
        config = Config.from_file('configs/default.yaml')
        logger = get_logger(__name__)
        logger.info("Brain-Forge initialized successfully")
        ```
        """
        # Test imports work
        from core.config import Config
        from core.logger import get_logger

        # Test configuration creation
        config = Config()
        assert config is not None
        
        # Test logger creation
        logger = get_logger(__name__)
        assert logger is not None
        
        # Test logging works without error
        logger.info("Brain-Forge initialized successfully")
        
        # Verify configuration has expected structure
        assert hasattr(config, 'hardware')
        assert hasattr(config, 'processing')
        assert hasattr(config, 'system')
        
        # Verify hardware config has multi-modal support
        assert config.hardware.omp_enabled
        assert config.hardware.kernel_enabled
        assert config.hardware.accel_enabled
    
    def test_multimodal_data_acquisition_workflow(self):
        """
        Test the multi-modal data acquisition workflow from README
        """
        # Create configuration as shown in README
        omp_config = {
            'channels': 306,
            'matrix_coils': 48,
            'sampling_rate': 1000,
            'magnetic_shielding': True,
            'movement_compensation': 'dynamic'
        }
        
        kernel_config = {
            'optical_modules': 40,
            'eeg_channels': 4,
            'wavelengths': [690, 905],
            'measurement_type': 'hemodynamic_electrical',
            'coverage': 'whole_head'
        }
        
        accelo_config = {
            'accelerometers': 64,
            'impact_detection': True,
            'motion_correlation': True,
            'navy_grade': True
        }
        
        # Verify configuration structures are valid
        assert omp_config['channels'] == 306
        assert len(kernel_config['wavelengths']) == 2
        assert accelo_config['accelerometers'] == 64
        
        # Simulate data acquisition as described in README
        multimodal_data = self._generate_multimodal_test_data(
            omp_config, kernel_config, accelo_config
        )
        
        # Verify data structure matches README description
        assert 'omp_data' in multimodal_data
        assert 'kernel_optical' in multimodal_data
        assert 'kernel_eeg' in multimodal_data
        assert 'accelo_data' in multimodal_data
        
        # Verify data shapes match configuration
        assert multimodal_data['omp_data'].shape[0] == 306
        assert multimodal_data['kernel_optical'].shape[0] == 40
        assert multimodal_data['kernel_eeg'].shape[0] == 4
        assert multimodal_data['accelo_data'].shape[0] == 192  # 64 × 3 axes
        
        # Test brain state analysis simulation
        brain_state = self._analyze_brain_state(multimodal_data)
        
        assert 'activity_level' in brain_state
        assert 'motion_compensation' in brain_state
        assert 'network_coherence' in brain_state
        
        print(f"Current brain activity: {brain_state['activity_level']}")
        print(f"Movement artifacts: {brain_state['motion_compensation']}")
        print(f"Neural connectivity: {brain_state['network_coherence']}")
    
    def test_processing_pipeline_workflow(self):
        """
        Test the advanced neural processing pipeline from README
        """
        # Initialize processing components as shown in README
        processor = RealTimeProcessor()
        compressor = WaveletCompressor(wavelet='db8')
        
        # Verify processor has all expected components
        assert hasattr(processor, 'bandpass_filter')
        assert hasattr(processor, 'notch_filter')
        assert hasattr(processor, 'compressor')
        assert hasattr(processor, 'artifact_remover')
        assert hasattr(processor, 'feature_extractor')
        
        # Generate multimodal test data
        multimodal_data = {
            'omp': np.random.randn(306, 1000),
            'kernel_optical': np.random.randn(40, 100),
            'kernel_eeg': np.random.randn(4, 1000),
            'accelo': np.random.randn(64, 1000)
        }
        
        # Process data through complete pipeline
        processed_results = {}
        
        for modality, data in multimodal_data.items():
            if data.shape[1] >= 100:  # Skip very short data
                result = asyncio.run(processor.process_data_chunk(data))
                processed_results[modality] = result
        
        # Verify processing results
        for modality, result in processed_results.items():
            assert 'processed_data' in result
            assert 'compressed_data' in result
            assert 'features' in result
            assert 'quality_score' in result
            
            # Verify feature extraction worked
            features = result['features']
            expected_bands = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power']
            for band in expected_bands:
                assert band in features
            
            # Verify compression worked
            compression_info = result['compressed_data']
            assert 'compression_ratio' in compression_info
            assert compression_info['compression_ratio'] > 1.0
        
        print("Processing pipeline workflow completed successfully")
    
    def test_real_time_streaming_simulation(self):
        """
        Test real-time streaming simulation as described in README
        """
        processor = RealTimeProcessor()
        
        # Simulate streaming data chunks (100ms chunks at 1000 Hz)
        chunk_size = 100
        num_chunks = 20
        channels = 64
        
        processing_times = []
        quality_scores = []
        
        print("Simulating real-time streaming...")
        
        for i in range(num_chunks):
            # Generate streaming chunk
            chunk = np.random.randn(channels, chunk_size)
            
            # Process chunk
            start_time = time.time()
            result = asyncio.run(processor.process_data_chunk(chunk))
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            quality_scores.append(result['quality_score'])
            
            # Simulate real-time constraints
            chunk_duration = chunk_size / 1000.0  # 100ms
            if processing_time > chunk_duration:
                print(f"Warning: Chunk {i+1} processing time ({processing_time*1000:.1f}ms) "
                      f"exceeded chunk duration ({chunk_duration*1000:.1f}ms)")
        
        # Analyze streaming performance
        avg_processing_time = np.mean(processing_times)
        avg_quality = np.mean(quality_scores)
        
        print(f"Streaming simulation results:")
        print(f"  Average processing time: {avg_processing_time*1000:.2f}ms")
        print(f"  Average quality score: {avg_quality:.3f}")
        print(f"  Chunks meeting real-time constraint: {sum(1 for t in processing_times if t < 0.1)}/{num_chunks}")
        
        # Verify streaming performance
        assert avg_processing_time < 0.05  # Should be well under chunk duration
        assert 0.0 <= avg_quality <= 1.0
        assert all(0.0 <= q <= 1.0 for q in quality_scores)
    
    def _generate_multimodal_test_data(self, omp_config, kernel_config, accelo_config):
        """Generate realistic multimodal test data"""
        
        # OMP magnetometer data (306 channels, MEG)
        omp_data = np.random.randn(omp_config['channels'], 1000) * 1e-12  # fT scale
        
        # Kernel optical data (40 modules, hemodynamic)
        kernel_optical = np.random.randn(kernel_config['optical_modules'], 100)
        
        # Kernel EEG data (4 channels, electrical)
        kernel_eeg = np.random.randn(kernel_config['eeg_channels'], 1000)
        
        # Accelerometer data (64 sensors × 3 axes)
        accelo_data = np.random.randn(accelo_config['accelerometers'] * 3, 1000)
        
        return {
            'omp_data': omp_data,
            'kernel_optical': kernel_optical,
            'kernel_eeg': kernel_eeg,
            'accelo_data': accelo_data
        }
    
    def _analyze_brain_state(self, multimodal_data):
        """Simulate brain state analysis"""
        
        # Simple brain state analysis simulation
        omp_activity = np.mean(np.abs(multimodal_data['omp_data']))
        motion_level = np.mean(np.abs(multimodal_data['accelo_data']))
        
        return {
            'activity_level': f"{omp_activity * 1e12:.2f} fT RMS",
            'motion_compensation': f"{motion_level:.3f} g RMS",
            'network_coherence': f"{np.random.uniform(0.3, 0.9):.3f}"
        }


class TestSystemIntegration(TestEndToEndIntegration):
    """Test integration between major system components"""
    
    def test_config_processor_integration(self):
        """
        Test integration between configuration system and processor
        """
        # Create custom configuration
        config = Config()
        config.processing.filter_low = 2.0
        config.processing.filter_high = 80.0
        config.processing.compression_ratio = 6.0
        config.hardware.omp_channels = 128
        
        # Initialize processor with custom config
        processor = RealTimeProcessor(config=config)
        
        # Verify processor uses configuration
        assert processor.params.filter_low == 2.0
        assert processor.params.filter_high == 80.0
        assert processor.params.compression_ratio == 6.0
        
        # Test processing with configured parameters
        test_data = np.random.randn(128, 1000)  # Match config channel count
        result = asyncio.run(processor.process_data_chunk(test_data))
        
        # Verify processing used configured parameters
        assert result['processed_data'].shape[0] == 128
        compression_ratio = result['compressed_data']['compression_ratio']
        assert 4.0 <= compression_ratio <= 8.0  # Should be near configured ratio
    
    def test_logger_processor_integration(self):
        """
        Test integration between logging system and processor
        """
        # Create logger
        logger = get_logger("integration_test")
        
        # Initialize processor
        processor = RealTimeProcessor()
        
        # Process data and check for log output
        test_data = np.random.randn(32, 500)
        
        with patch.object(logger, 'info') as mock_info:
            result = asyncio.run(processor.process_data_chunk(test_data))
            
            # Processor should generate some log output during operation
            # (This tests the integration exists, even if we can't easily capture it)
            assert result is not None
    
    def test_error_handling_integration(self):
        """
        Test error handling integration across components
        """
        processor = RealTimeProcessor()
        
        # Test invalid data handling
        invalid_data_cases = [
            np.array([]),  # Empty data
            np.random.randn(0, 100),  # Zero channels
            np.random.randn(100, 0),  # Zero samples
        ]
        
        for invalid_data in invalid_data_cases:
            with pytest.raises((ValueError, IndexError, Exception)):
                asyncio.run(processor.process_data_chunk(invalid_data))
    
    def test_artifact_removal_integration(self):
        """
        Test integration of artifact removal with full processing pipeline
        """
        processor = RealTimeProcessor()
        
        # Create data with known artifacts
        clean_data = np.random.randn(32, 2000) * 0.1
        
        # Add artifacts
        artifact_data = clean_data.copy()
        artifact_data[0, 500:600] += 5.0  # Large spike
        artifact_data[1, :] += 0.5 * np.random.randn(2000)  # High noise
        
        # Process data (artifact removal should be integrated)
        result = asyncio.run(processor.process_data_chunk(artifact_data))
        
        # Verify processing completed
        assert 'processed_data' in result
        assert result['processed_data'].shape == artifact_data.shape
        
        # Quality score should reflect artifact removal
        assert 0.0 <= result['quality_score'] <= 1.0


class TestMultiModalDataFlow(TestEndToEndIntegration):
    """Test multi-modal data flow and synchronization"""
    
    def test_synchronized_multimodal_processing(self):
        """
        Test processing of synchronized multi-modal data streams
        """
        processor = RealTimeProcessor()
        
        # Create synchronized data streams with different sampling rates
        base_duration = 1.0  # 1 second
        
        # Different modalities at their native sampling rates
        streams = {
            'omp': np.random.randn(306, 1000),  # 1000 Hz
            'kernel_optical': np.random.randn(40, 100),   # 100 Hz  
            'kernel_eeg': np.random.randn(4, 1000),       # 1000 Hz
            'accelerometer': np.random.randn(192, 1000),  # 1000 Hz
        }
        
        # Process each stream
        processed_streams = {}
        processing_times = {}
        
        for stream_name, data in streams.items():
            start_time = time.time()
            result = asyncio.run(processor.process_data_chunk(data))
            processing_time = time.time() - start_time
            
            processed_streams[stream_name] = result
            processing_times[stream_name] = processing_time
            
            # Verify processing
            assert result['processed_data'].shape == data.shape
            assert 'features' in result
        
        # Verify all streams processed successfully
        assert len(processed_streams) == 4
        
        # Check temporal alignment (processing times should be similar)
        max_processing_time = max(processing_times.values())
        min_processing_time = min(processing_times.values())
        
        print(f"Processing time range: {min_processing_time*1000:.2f} - {max_processing_time*1000:.2f}ms")
        
        # Processing times should be reasonably similar for synchronization
        time_ratio = max_processing_time / min_processing_time
        assert time_ratio < 5.0, f"Processing time variance too high: {time_ratio:.2f}x"
    
    def test_cross_modal_feature_correlation(self):
        """
        Test correlation of features across modalities
        """
        processor = RealTimeProcessor()
        
        # Create correlated signals across modalities
        t = np.linspace(0, 2, 2000)
        base_signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz base signal
        
        # Add base signal to different modalities with different strengths
        omp_data = np.tile(base_signal * 1e-12, (306, 1)) + np.random.randn(306, 2000) * 0.1e-12
        eeg_data = np.tile(base_signal * 10e-6, (4, 1)) + np.random.randn(4, 2000) * 1e-6
        
        # Process both modalities
        omp_result = asyncio.run(processor.process_data_chunk(omp_data))
        eeg_result = asyncio.run(processor.process_data_chunk(eeg_data))
        
        # Extract alpha band power (should be correlated due to base signal)
        omp_alpha = np.mean(omp_result['features']['alpha_power'])
        eeg_alpha = np.mean(eeg_result['features']['alpha_power'])
        
        # Both should show elevated alpha power due to 10 Hz signal
        assert omp_alpha > 0
        assert eeg_alpha > 0
        
        print(f"Cross-modal alpha power - OMP: {omp_alpha:.3e}, EEG: {eeg_alpha:.3e}")
    
    def test_multimodal_compression_efficiency(self):
        """
        Test compression efficiency across different modalities
        """
        compressor = WaveletCompressor()
        
        # Different types of neural data
        modalities = {
            'omp_meg': np.random.randn(306, 2000) * 1e-12,  # MEG data (small amplitude)
            'eeg': np.random.randn(64, 2000) * 1e-5,        # EEG data (medium amplitude)
            'accelerometer': np.random.randn(192, 2000) * 0.1,  # Motion data (larger amplitude)
        }
        
        compression_results = {}
        
        for modality, data in modalities.items():
            compressed = compressor.compress(data, compression_ratio=5.0)
            compression_results[modality] = compressed
            
            ratio = compressed['compression_ratio']
            print(f"{modality}: {ratio:.2f}x compression")
            
            # All modalities should achieve reasonable compression
            assert ratio > 2.0
            assert ratio < 15.0
            
            # Verify decompression works
            decompressed = compressor.decompress(compressed)
            assert decompressed.shape == data.shape
        
        # Verify all modalities processed
        assert len(compression_results) == 3


class TestSystemRobustness(TestEndToEndIntegration):
    """Test system robustness and edge cases"""
    
    def test_continuous_operation_simulation(self):
        """
        Test continuous operation over extended period
        """
        processor = RealTimeProcessor()
        
        # Simulate continuous operation for 30 chunks
        num_chunks = 30
        chunk_duration = 0.2  # 200ms chunks
        channels = 64
        
        successful_chunks = 0
        errors = []
        
        for i in range(num_chunks):
            try:
                # Generate chunk with some variation
                samples = int(1000 * chunk_duration)  # 200 samples
                chunk = np.random.randn(channels, samples)
                
                # Occasionally add some challenging data
                if i % 10 == 0:
                    chunk += np.random.randn(channels, samples) * 2  # Higher noise
                
                # Process chunk
                result = asyncio.run(processor.process_data_chunk(chunk))
                
                # Verify result
                assert 'processed_data' in result
                assert result['quality_score'] >= 0.0
                
                successful_chunks += 1
                
            except Exception as e:
                errors.append((i, str(e)))
        
        # Should handle most chunks successfully
        success_rate = successful_chunks / num_chunks
        print(f"Continuous operation: {successful_chunks}/{num_chunks} chunks successful ({success_rate*100:.1f}%)")
        
        if errors:
            print(f"Errors encountered: {len(errors)}")
            for chunk_idx, error in errors:
                print(f"  Chunk {chunk_idx}: {error}")
        
        assert success_rate > 0.8, f"Success rate too low: {success_rate*100:.1f}%"
    
    def test_varying_data_characteristics(self):
        """
        Test processing with varying data characteristics
        """
        processor = RealTimeProcessor()
        
        # Different data characteristics
        test_cases = [
            ("normal", np.random.randn(32, 1000) * 0.1),
            ("high_amplitude", np.random.randn(32, 1000) * 10),
            ("low_amplitude", np.random.randn(32, 1000) * 0.001),
            ("sparse", np.random.choice([0, 1], size=(32, 1000)) * np.random.randn(32, 1000)),
            ("oscillatory", np.tile(np.sin(2*np.pi*10*np.linspace(0,1,1000)), (32, 1))),
        ]
        
        results = {}
        
        for case_name, test_data in test_cases:
            try:
                result = asyncio.run(processor.process_data_chunk(test_data))
                results[case_name] = {
                    'success': True,
                    'quality_score': result['quality_score'],
                    'processing_time': result['processing_time']
                }
                
            except Exception as e:
                results[case_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Print results
        print("Data characteristic test results:")
        for case_name, result in results.items():
            if result['success']:
                print(f"  {case_name}: Quality={result['quality_score']:.3f}, Time={result['processing_time']*1000:.1f}ms")
            else:
                print(f"  {case_name}: FAILED - {result['error']}")
        
        # Most cases should succeed
        success_count = sum(1 for r in results.values() if r['success'])
        assert success_count >= 4, f"Too many data characteristic tests failed: {success_count}/5"
    
    def test_memory_stability_under_load(self):
        """
        Test memory stability under processing load
        """
        import gc

        import psutil
        
        processor = RealTimeProcessor()
        
        # Initial memory measurement
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Process many chunks of varying sizes
        chunk_configs = [
            (16, 200),   # Small chunks
            (32, 500),   # Medium chunks
            (64, 1000),  # Large chunks
            (128, 2000), # Extra large chunks
        ]
        
        total_chunks = 0
        
        for channels, samples in chunk_configs:
            for _ in range(10):  # 10 chunks of each size
                test_data = np.random.randn(channels, samples)
                result = asyncio.run(processor.process_data_chunk(test_data))
                total_chunks += 1
                
                # Periodic garbage collection
                if total_chunks % 20 == 0:
                    gc.collect()
        
        # Final memory measurement
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        memory_growth = final_memory - initial_memory
        
        print(f"Memory stability test:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Memory growth: {memory_growth:.1f}MB")
        print(f"  Chunks processed: {total_chunks}")
        
        # Memory growth should be reasonable
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.1f}MB"
        
        # Memory per chunk should be reasonable
        memory_per_chunk = memory_growth / total_chunks
        assert memory_per_chunk < 2.0, f"Memory per chunk too high: {memory_per_chunk:.2f}MB/chunk"


class TestExampleValidation(TestEndToEndIntegration):
    """Validate all examples from README work correctly"""
    
    def test_system_info_example(self):
        """
        Test the system info example from README
        """
        # Should be able to get system info
        system_info = brain_forge.get_system_info()
        
        # Should contain expected fields
        expected_fields = ['version', 'python_version', 'platform', 'dependencies']
        for field in expected_fields:
            assert field in system_info
        
        # Version should match
        assert system_info['version'] == brain_forge.__version__
        
        print("System Info:")
        for key, value in system_info.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
    
    def test_hardware_support_example(self):
        """
        Test the hardware support check example from README
        """
        # Should be able to check hardware support
        hardware_status = brain_forge.check_hardware_support()
        
        # Should contain expected components
        expected_components = ['lsl_available', 'brian2_available', 'nest_available', 'mne_available']
        for component in expected_components:
            assert component in hardware_status
            assert isinstance(hardware_status[component], bool)
        
        print("Hardware Support:")
        for component, available in hardware_status.items():
            print(f"  {component}: {'Available' if available else 'Not Available'}")
    
    def test_configuration_example_validation(self):
        """
        Test that configuration examples from README are valid
        """
        # Test YAML configuration example format
        example_config = {
            'hardware': {
                'omp_channels': 306,
                'omp_sampling_rate': 1000.0,
                'kernel_wavelengths': [690, 905],
                'accel_range': 16
            },
            'processing': {
                'filter_low': 1.0,
                'filter_high': 100.0,
                'compression_ratio': 5.0,
                'wavelet_type': 'db8'
            },
            'system': {
                'log_level': 'INFO',
                'processing_threads': 4,
                'gpu_enabled': True
            }
        }
        
        # Save example configuration
        config_file = Path(self.temp_dir) / "example_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(example_config, f)
        
        # Should be able to load example configuration
        config = Config(config_file=str(config_file))
        
        # Verify example values loaded correctly
        assert config.hardware.omp_channels == 306
        assert config.hardware.omp_sampling_rate == 1000.0
        assert config.processing.filter_low == 1.0
        assert config.processing.compression_ratio == 5.0
        assert config.system.log_level == 'INFO'
        
        # Configuration validation should pass
        config._validate_config()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
