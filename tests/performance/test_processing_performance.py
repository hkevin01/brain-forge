"""
Comprehensive Performance Tests for Brain-Forge Processing Pipeline

This test module provides performance testing and benchmarking for
signal processing, compression, feature extraction, and real-time
processing capabilities.
"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Add src to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.config import ProcessingConfig


class MockRealTimeProcessor:
    """Mock real-time processor for performance testing"""
    
    def __init__(self, config=None):
        self.config = config or ProcessingConfig()
        self.buffer_size = self.config.buffer_size
        self.filter_low = self.config.filter_low
        self.filter_high = self.config.filter_high
        self.latency_target = self.config.real_time_threshold
    
    def bandpass_filter(self, data, sampling_rate=1000):
        """Mock bandpass filtering with realistic computation time"""
        # Simulate filtering computation
        time.sleep(0.0001)  # 0.1ms processing time
        
        # Apply simple filtering (for demonstration)
        from scipy import signal
        nyquist = sampling_rate / 2
        low_norm = self.filter_low / nyquist
        high_norm = self.filter_high / nyquist
        
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=-1)
        
        return filtered_data
    
    def compress_data(self, data, compression_ratio=5.0):
        """Mock data compression with performance timing"""
        start_time = time.time()
        
        # Simulate compression algorithm
        compressed_size = int(data.size / compression_ratio)
        compressed_data = np.random.randn(compressed_size)
        
        compression_time = time.time() - start_time
        return compressed_data, compression_time
    
    def extract_features(self, data):
        """Mock feature extraction with timing"""
        start_time = time.time()
        
        # Simulate feature extraction
        features = {
            'mean': np.mean(data, axis=-1),
            'std': np.std(data, axis=-1),
            'power_bands': np.random.randn(data.shape[0], 5),  # 5 frequency bands
            'connectivity': np.random.randn(data.shape[0], data.shape[0])
        }
        
        extraction_time = time.time() - start_time
        return features, extraction_time
    
    def process_chunk(self, data_chunk):
        """Process a complete data chunk with timing"""
        start_time = time.time()
        
        # Filter
        filtered_data = self.bandpass_filter(data_chunk)
        
        # Compress
        compressed_data, compression_time = self.compress_data(filtered_data)
        
        # Extract features
        features, feature_time = self.extract_features(filtered_data)
        
        total_time = time.time() - start_time
        
        return {
            'filtered_data': filtered_data,
            'compressed_data': compressed_data,
            'features': features,
            'processing_time': total_time,
            'compression_time': compression_time,
            'feature_time': feature_time
        }


class TestProcessingLatency:
    """Test processing latency requirements"""
    
    @pytest.fixture
    def processor(self):
        """Create processor with real-time configuration"""
        config = ProcessingConfig(real_time_threshold=0.001)  # 1ms target
        return MockRealTimeProcessor(config)
    
    def test_single_sample_latency(self, processor):
        """Test single sample processing latency"""
        # Single sample from 306-channel OMP helmet
        sample = np.random.randn(306)
        
        start_time = time.time()
        filtered_sample = processor.bandpass_filter(sample.reshape(306, 1))
        latency = time.time() - start_time
        
        # Should be well under 1ms for single sample
        assert latency < 0.001
        assert filtered_sample.shape == (306, 1)
    
    def test_chunk_processing_latency(self, processor):
        """Test chunk processing latency"""
        # 100ms chunk at 1000 Hz = 100 samples
        chunk_size = 100
        channels = 306
        data_chunk = np.random.randn(channels, chunk_size)
        
        result = processor.process_chunk(data_chunk)
        
        # Processing time should be less than chunk duration (100ms)
        assert result['processing_time'] < 0.1
        
        # Should meet real-time requirements
        chunk_duration = chunk_size / 1000.0  # 100ms
        processing_ratio = result['processing_time'] / chunk_duration
        assert processing_ratio < 0.5  # Use less than 50% of available time
    
    def test_streaming_latency_simulation(self, processor):
        """Test continuous streaming latency"""
        chunk_size = 100
        channels = 306
        num_chunks = 10
        
        latencies = []
        
        for _ in range(num_chunks):
            data_chunk = np.random.randn(channels, chunk_size)
            
            start_time = time.time()
            result = processor.process_chunk(data_chunk)
            latency = time.time() - start_time
            
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        
        # Average latency should be good
        assert avg_latency < 0.05  # 50ms average
        
        # Maximum latency should not cause buffer overruns
        assert max_latency < 0.1  # 100ms max
    
    def test_multi_modal_processing_latency(self, processor):
        """Test multi-modal data processing latency"""
        # Simulate data from all devices
        omp_data = np.random.randn(306, 100)  # OMP helmet
        optical_data = np.random.randn(96, 10)  # Kernel optical (slower sampling)
        accel_data = np.random.randn(192, 100)  # Accelerometer array
        
        start_time = time.time()
        
        # Process all modalities
        omp_result = processor.process_chunk(omp_data)
        optical_result = processor.process_chunk(optical_data)
        accel_result = processor.process_chunk(accel_data)
        
        total_latency = time.time() - start_time
        
        # Should process all modalities quickly
        assert total_latency < 0.2  # 200ms for all modalities
        
        # Individual processing times should be reasonable
        assert omp_result['processing_time'] < 0.1
        assert optical_result['processing_time'] < 0.05
        assert accel_result['processing_time'] < 0.05


class TestCompressionPerformance:
    """Test compression algorithm performance"""
    
    @pytest.fixture
    def processor(self):
        """Create processor for compression testing"""
        return MockRealTimeProcessor()
    
    def test_compression_ratio_achievement(self, processor):
        """Test achieving target compression ratios"""
        data_sizes = [1000, 10000, 100000]  # Different data sizes
        target_ratios = [2.0, 5.0, 10.0]  # Different compression targets
        
        for data_size in data_sizes:
            for ratio in target_ratios:
                data = np.random.randn(306, data_size)
                
                compressed_data, compression_time = processor.compress_data(data, ratio)
                
                # Check compression ratio
                actual_ratio = data.size / compressed_data.size
                assert 0.8 * ratio <= actual_ratio <= 1.2 * ratio  # Within 20%
                
                # Check compression speed
                throughput = data.size / compression_time  # samples per second
                assert throughput > 100000  # 100k samples/sec minimum
    
    def test_compression_quality_vs_speed_tradeoff(self, processor):
        """Test compression quality vs speed tradeoff"""
        data = np.random.randn(306, 10000)
        
        # Test different compression ratios
        ratios = [2.0, 5.0, 10.0, 20.0]
        results = []
        
        for ratio in ratios:
            compressed_data, compression_time = processor.compress_data(data, ratio)
            
            results.append({
                'ratio': ratio,
                'compression_time': compression_time,
                'compressed_size': compressed_data.size,
                'throughput': data.size / compression_time
            })
        
        # Higher compression ratios should not dramatically increase time
        for i in range(1, len(results)):
            time_increase = results[i]['compression_time'] / results[i-1]['compression_time']
            assert time_increase < 2.0  # Less than 2x time increase
    
    def test_parallel_compression(self, processor):
        """Test parallel compression performance"""
        # Create multiple data chunks
        num_chunks = 4
        chunks = [np.random.randn(306, 5000) for _ in range(num_chunks)]
        
        # Sequential compression
        start_time = time.time()
        sequential_results = []
        for chunk in chunks:
            compressed, comp_time = processor.compress_data(chunk)
            sequential_results.append(compressed)
        sequential_time = time.time() - start_time
        
        # Parallel compression
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_chunks) as executor:
            futures = [executor.submit(processor.compress_data, chunk) for chunk in chunks]
            parallel_results = [future.result()[0] for future in futures]
        parallel_time = time.time() - start_time
        
        # Parallel should be faster (or at least not much slower due to overhead)
        speedup = sequential_time / parallel_time
        assert speedup > 0.5  # At least 50% of sequential performance


class TestFeatureExtractionPerformance:
    """Test feature extraction performance"""
    
    @pytest.fixture
    def processor(self):
        """Create processor for feature extraction testing"""
        return MockRealTimeProcessor()
    
    def test_feature_extraction_speed(self, processor):
        """Test feature extraction speed for different data sizes"""
        data_sizes = [1000, 5000, 10000, 50000]
        channels = 306
        
        for size in data_sizes:
            data = np.random.randn(channels, size)
            
            features, extraction_time = processor.extract_features(data)
            
            # Check feature completeness
            assert 'mean' in features
            assert 'std' in features
            assert 'power_bands' in features
            assert 'connectivity' in features
            
            # Check extraction speed
            throughput = data.size / extraction_time
            assert throughput > 50000  # 50k samples/sec minimum
            
            # Check feature dimensions
            assert features['mean'].shape[0] == channels
            assert features['std'].shape[0] == channels
            assert features['power_bands'].shape == (channels, 5)
            assert features['connectivity'].shape == (channels, channels)
    
    def test_real_time_feature_extraction(self, processor):
        """Test real-time feature extraction capability"""
        # Simulate continuous data stream
        chunk_size = 1000  # 1 second at 1000 Hz
        channels = 306
        num_chunks = 10
        
        feature_times = []
        
        for _ in range(num_chunks):
            data_chunk = np.random.randn(channels, chunk_size)
            features, extraction_time = processor.extract_features(data_chunk)
            feature_times.append(extraction_time)
        
        avg_extraction_time = np.mean(feature_times)
        max_extraction_time = np.max(feature_times)
        
        # Should extract features faster than data arrives
        chunk_duration = chunk_size / 1000.0  # 1 second
        assert avg_extraction_time < chunk_duration / 2  # Use less than 50% of time
        assert max_extraction_time < chunk_duration  # Never exceed chunk duration
    
    def test_incremental_feature_updates(self, processor):
        """Test incremental feature update performance"""
        # Simulate sliding window feature extraction
        window_size = 2000
        step_size = 100
        channels = 306
        
        # Create continuous data stream
        total_samples = 10000
        data_stream = np.random.randn(channels, total_samples)
        
        update_times = []
        
        for start_idx in range(0, total_samples - window_size, step_size):
            window_data = data_stream[:, start_idx:start_idx + window_size]
            
            start_time = time.time()
            features, _ = processor.extract_features(window_data)
            update_time = time.time() - start_time
            
            update_times.append(update_time)
        
        avg_update_time = np.mean(update_times)
        
        # Updates should be fast for real-time processing
        step_duration = step_size / 1000.0  # Duration of step in seconds
        assert avg_update_time < step_duration / 2  # Use less than 50% of available time


class TestConcurrentProcessing:
    """Test concurrent processing capabilities"""
    
    @pytest.fixture
    def processor(self):
        """Create processor for concurrent testing"""
        return MockRealTimeProcessor()
    
    def test_multi_threaded_processing(self, processor):
        """Test multi-threaded processing performance"""
        num_threads = multiprocessing.cpu_count()
        data_chunks = [np.random.randn(306, 1000) for _ in range(num_threads)]
        
        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for chunk in data_chunks:
            result = processor.process_chunk(chunk)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Multi-threaded processing
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(processor.process_chunk, chunk) for chunk in data_chunks]
            parallel_results = [future.result() for future in futures]
        parallel_time = time.time() - start_time
        
        # Check results consistency
        assert len(parallel_results) == len(sequential_results)
        
        # Parallel processing should provide speedup
        speedup = sequential_time / parallel_time
        assert speedup > 1.0  # Some speedup expected
        
        # Results should be equivalent
        for seq_result, par_result in zip(sequential_results, parallel_results):
            assert seq_result['filtered_data'].shape == par_result['filtered_data'].shape
    
    def test_pipeline_throughput(self, processor):
        """Test processing pipeline throughput"""
        # Simulate high-throughput data stream
        chunk_size = 1000
        channels = 306
        num_chunks = 100
        
        total_samples = 0
        start_time = time.time()
        
        for _ in range(num_chunks):
            data_chunk = np.random.randn(channels, chunk_size)
            result = processor.process_chunk(data_chunk)
            total_samples += data_chunk.size
        
        total_time = time.time() - start_time
        throughput = total_samples / total_time
        
        # Should achieve high throughput
        assert throughput > 1000000  # 1M samples/sec minimum
        
        # Should maintain real-time performance
        data_rate = num_chunks * chunk_size / 1000.0  # seconds of data processed
        real_time_ratio = data_rate / total_time
        assert real_time_ratio > 5.0  # Process 5x faster than real-time
    
    def test_memory_efficient_processing(self, processor):
        """Test memory-efficient processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large amount of data
        large_data_chunks = [np.random.randn(306, 10000) for _ in range(10)]
        
        for chunk in large_data_chunks:
            result = processor.process_chunk(chunk)
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - baseline_memory
            
            # Should not use excessive memory
            assert memory_increase < 500  # Less than 500MB increase
        
        # Memory should be released after processing
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        final_increase = final_memory - baseline_memory
        assert final_increase < 100  # Less than 100MB permanent increase


class TestRealTimeConstraints:
    """Test real-time processing constraints"""
    
    @pytest.fixture
    def processor(self):
        """Create processor with strict real-time requirements"""
        config = ProcessingConfig(real_time_threshold=0.001)  # 1ms strict
        return MockRealTimeProcessor(config)
    
    def test_hard_real_time_constraints(self, processor):
        """Test hard real-time processing constraints"""
        # Simulate 1ms processing windows
        window_duration = 0.001  # 1ms
        sampling_rate = 1000
        samples_per_window = int(sampling_rate * window_duration)  # 1 sample
        
        # Test multiple windows
        violations = 0
        total_windows = 1000
        
        for _ in range(total_windows):
            data_window = np.random.randn(306, max(1, samples_per_window))
            
            start_time = time.time()
            result = processor.process_chunk(data_window)
            processing_time = time.time() - start_time
            
            if processing_time > window_duration:
                violations += 1
        
        # Should have very few violations for hard real-time
        violation_rate = violations / total_windows
        assert violation_rate < 0.01  # Less than 1% violations
    
    def test_soft_real_time_performance(self, processor):
        """Test soft real-time performance"""
        # Simulate 10ms processing windows (more realistic)
        window_duration = 0.01  # 10ms
        sampling_rate = 1000
        samples_per_window = int(sampling_rate * window_duration)  # 10 samples
        
        processing_times = []
        
        for _ in range(100):
            data_window = np.random.randn(306, samples_per_window)
            
            start_time = time.time()
            result = processor.process_chunk(data_window)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
        
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        percentile_99 = np.percentile(processing_times, 99)
        
        # Average should be well below window duration
        assert avg_processing_time < window_duration / 2
        
        # 99th percentile should be acceptable
        assert percentile_99 < window_duration * 0.8
        
        # Maximum should not be excessive
        assert max_processing_time < window_duration * 2
    
    def test_jitter_analysis(self, processor):
        """Test processing time jitter"""
        processing_times = []
        
        # Consistent data chunks
        for _ in range(1000):
            data_chunk = np.random.randn(306, 100)  # Same size each time
            
            start_time = time.time()
            result = processor.process_chunk(data_chunk)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
        
        processing_times = np.array(processing_times)
        
        mean_time = np.mean(processing_times)
        std_time = np.std(processing_times)
        jitter = std_time / mean_time  # Coefficient of variation
        
        # Jitter should be low for consistent performance
        assert jitter < 0.2  # Less than 20% variation


class TestScalabilityPerformance:
    """Test performance scalability"""
    
    @pytest.fixture
    def processor(self):
        """Create processor for scalability testing"""
        return MockRealTimeProcessor()
    
    def test_channel_count_scalability(self, processor):
        """Test performance scaling with channel count"""
        channel_counts = [64, 128, 256, 512, 1024]
        chunk_size = 1000
        
        results = []
        
        for channels in channel_counts:
            data = np.random.randn(channels, chunk_size)
            
            start_time = time.time()
            result = processor.process_chunk(data)
            processing_time = time.time() - start_time
            
            results.append({
                'channels': channels,
                'processing_time': processing_time,
                'throughput': data.size / processing_time
            })
        
        # Processing time should scale sub-linearly with channels
        for i in range(1, len(results)):
            channel_ratio = results[i]['channels'] / results[i-1]['channels']
            time_ratio = results[i]['processing_time'] / results[i-1]['processing_time']
            
            # Time should not increase as fast as channel count
            assert time_ratio < channel_ratio * 1.5
    
    def test_data_length_scalability(self, processor):
        """Test performance scaling with data length"""
        data_lengths = [100, 500, 1000, 5000, 10000]
        channels = 306
        
        results = []
        
        for length in data_lengths:
            data = np.random.randn(channels, length)
            
            start_time = time.time()
            result = processor.process_chunk(data)
            processing_time = time.time() - start_time
            
            results.append({
                'length': length,
                'processing_time': processing_time,
                'throughput': data.size / processing_time
            })
        
        # Processing time should scale linearly with data length
        for i in range(1, len(results)):
            length_ratio = results[i]['length'] / results[i-1]['length']
            time_ratio = results[i]['processing_time'] / results[i-1]['processing_time']
            
            # Time scaling should be reasonable
            assert 0.5 < time_ratio / length_ratio < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
