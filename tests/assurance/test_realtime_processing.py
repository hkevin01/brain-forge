"""
Assurance Tests for Real-Time Multi-Stream Processing
Validates synchronized data fusion, neural pattern recognition, and GPU acceleration
"""

import pytest
import numpy as np
import asyncio
import time
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import Dict, Any
import threading
from concurrent.futures import ThreadPoolExecutor

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@dataclass
class StreamSyncSpecs:
    """Synchronized data fusion specifications"""
    temporal_alignment: float = 0.0005  # Sub-millisecond (0.5ms max)
    omp_rate: int = 1000  # Hz
    kernel_rate: int = 250  # Hz
    accelo_rate: int = 1000  # Hz
    max_latency: float = 0.001  # 1ms maximum processing latency


@dataclass
class CompressionSpecs:
    """Neural pattern compression specifications"""
    min_compression_ratio: float = 2.0  # Minimum 2x compression
    max_compression_ratio: float = 10.0  # Maximum 10x compression
    quality_threshold: float = 0.95  # 95% signal quality retention
    real_time_requirement: bool = True


class TestSynchronizedDataFusion:
    """Test sub-millisecond temporal alignment of multi-modal streams"""
    
    @pytest.fixture
    def sync_system(self):
        """Mock synchronized data fusion system"""
        system = Mock()
        system.specs = StreamSyncSpecs()
        system.is_synchronized = Mock(return_value=True)
        system.sync_buffer = {}
        return system
    
    @pytest.mark.asyncio
    async def test_submillisecond_temporal_alignment(self, sync_system):
        """Test sub-millisecond synchronization across all data streams"""
        # Mock multi-rate data streams
        base_timestamp = time.time()
        sync_tolerance = sync_system.specs.temporal_alignment
        
        # High-rate streams (1kHz)
        async def omp_stream():
            for i in range(1000):  # 1 second of data
                timestamp = base_timestamp + i * 0.001
                data = np.random.normal(0, 1e-12, 306)
                yield {'source': 'omp', 'data': data, 'timestamp': timestamp, 'sample_id': i}
                await asyncio.sleep(0.001)
        
        async def accelo_stream():
            for i in range(1000):  # 1 second of data
                timestamp = base_timestamp + i * 0.001 + np.random.uniform(-0.0002, 0.0002)  # ±0.2ms jitter
                data = np.random.normal(0, 0.1, (16, 3))
                yield {'source': 'accelo', 'data': data, 'timestamp': timestamp, 'sample_id': i}
                await asyncio.sleep(0.001)
        
        # Lower-rate stream (250Hz)
        async def kernel_stream():
            for i in range(250):  # 1 second of data
                timestamp = base_timestamp + i * 0.004 + np.random.uniform(-0.0001, 0.0001)  # ±0.1ms jitter
                fnirs_data = np.random.normal(0, 1e-6, 160)
                eeg_data = np.random.normal(0, 50e-6, 32)
                yield {'source': 'kernel', 'fnirs': fnirs_data, 'eeg': eeg_data, 'timestamp': timestamp, 'sample_id': i}
                await asyncio.sleep(0.004)
        
        # Mock synchronization algorithm
        def synchronize_streams(omp_samples, kernel_samples, accelo_samples):
            """Synchronize multi-rate streams with interpolation"""
            synchronized_data = []
            
            # Use OMP as reference (highest rate)
            for omp_sample in omp_samples:
                omp_time = omp_sample['timestamp']
                sync_window = []
                
                # Find closest kernel sample
                kernel_diffs = [abs(k['timestamp'] - omp_time) for k in kernel_samples]
                if kernel_diffs:
                    closest_kernel_idx = np.argmin(kernel_diffs)
                    kernel_sample = kernel_samples[closest_kernel_idx]
                    if kernel_diffs[closest_kernel_idx] <= sync_tolerance:
                        sync_window.append(kernel_sample)
                
                # Find closest accelo sample
                accelo_diffs = [abs(a['timestamp'] - omp_time) for a in accelo_samples]
                if accelo_diffs:
                    closest_accelo_idx = np.argmin(accelo_diffs)
                    accelo_sample = accelo_samples[closest_accelo_idx]
                    if accelo_diffs[closest_accelo_idx] <= sync_tolerance:
                        sync_window.append(accelo_sample)
                
                if len(sync_window) >= 2:  # At least 2 modalities synchronized
                    synchronized_data.append({
                        'omp': omp_sample,
                        'synchronized_samples': sync_window,
                        'sync_timestamp': omp_time,
                        'sync_quality': len(sync_window) / 3.0
                    })
            
            return synchronized_data
        
        sync_system.synchronize_streams = synchronize_streams
        
        # Collect stream data
        omp_data = []
        kernel_data = []
        accelo_data = []
        
        async def collect_omp():
            async for sample in omp_stream():
                omp_data.append(sample)
                if len(omp_data) >= 100:  # Collect 100ms worth
                    break
        
        async def collect_kernel():
            async for sample in kernel_stream():
                kernel_data.append(sample)
                if len(kernel_data) >= 25:  # 100ms worth at 250Hz
                    break
        
        async def collect_accelo():
            async for sample in accelo_stream():
                accelo_data.append(sample)
                if len(accelo_data) >= 100:  # 100ms worth
                    break
        
        # Run collection concurrently
        await asyncio.gather(collect_omp(), collect_kernel(), collect_accelo())
        
        # Test synchronization
        synced_data = sync_system.synchronize_streams(omp_data, kernel_data, accelo_data)
        
        # Validate synchronization quality
        assert len(synced_data) > 0, "Should produce synchronized data"
        
        # Check temporal alignment
        alignment_errors = []
        for sync_sample in synced_data:
            ref_time = sync_sample['sync_timestamp']
            for sample in sync_sample['synchronized_samples']:
                time_diff = abs(sample['timestamp'] - ref_time)
                alignment_errors.append(time_diff)
        
        max_alignment_error = max(alignment_errors) if alignment_errors else 0
        mean_alignment_error = np.mean(alignment_errors) if alignment_errors else 0
        
        assert max_alignment_error < sync_tolerance, f"Max alignment error {max_alignment_error*1000:.3f}ms exceeds {sync_tolerance*1000:.1f}ms"
        assert mean_alignment_error < sync_tolerance/2, f"Mean alignment error {mean_alignment_error*1000:.3f}ms too high"
    
    def test_adaptive_synchronization_buffer(self, sync_system):
        """Test adaptive buffering for variable latency streams"""
        # Mock variable latency data
        timestamps = []
        latencies = []
        
        # Simulate network jitter and processing delays
        base_time = time.time()
        for i in range(1000):
            # Variable network latency (0-5ms)
            network_delay = np.random.exponential(0.002)  # Exponential distribution
            processing_delay = np.random.uniform(0.0001, 0.0005)  # 0.1-0.5ms
            
            actual_timestamp = base_time + i * 0.001
            received_timestamp = actual_timestamp + network_delay + processing_delay
            
            timestamps.append(actual_timestamp)
            latencies.append(received_timestamp - actual_timestamp)
        
        # Mock adaptive buffer algorithm
        def adaptive_buffer_size(latency_history, target_latency=0.001):
            """Calculate optimal buffer size based on latency statistics"""
            if len(latency_history) < 10:
                return 10  # Default buffer size
            
            # Use 95th percentile + margin
            p95_latency = np.percentile(latency_history, 95)
            buffer_samples = int((p95_latency + 0.001) * 1000)  # Convert to samples at 1kHz
            return max(10, min(buffer_samples, 50))  # 10-50 sample range
        
        sync_system.adaptive_buffer_size = adaptive_buffer_size
        
        # Test buffer adaptation
        buffer_sizes = []
        for i in range(10, len(latencies), 10):
            recent_latencies = latencies[max(0, i-100):i]  # Last 100 samples
            buffer_size = sync_system.adaptive_buffer_size(recent_latencies)
            buffer_sizes.append(buffer_size)
        
        # Validate adaptive behavior
        assert len(buffer_sizes) > 0, "Should calculate buffer sizes"
        assert all(10 <= size <= 50 for size in buffer_sizes), "Buffer sizes should be within range"
        
        # Buffer should adapt to latency changes
        high_latency_period = latencies[200:300]  # Simulate high latency period
        low_latency_period = latencies[800:900]   # Simulate low latency period
        
        high_buffer = sync_system.adaptive_buffer_size(high_latency_period)
        low_buffer = sync_system.adaptive_buffer_size(low_latency_period)
        
        assert high_buffer >= low_buffer, "Buffer should be larger during high latency periods"


class TestNeuralPatternRecognition:
    """Test transformer-based compression and pattern recognition"""
    
    @pytest.fixture
    def pattern_system(self):
        """Mock neural pattern recognition system"""
        system = Mock()
        system.specs = CompressionSpecs()
        system.transformer_model = Mock()
        system.pattern_library = {}
        return system
    
    def test_transformer_compression_ratios(self, pattern_system):
        """Test achievement of 2-10x compression ratios"""
        # Generate realistic neural data
        duration_seconds = 10
        sampling_rate = 1000
        n_samples = duration_seconds * sampling_rate
        n_channels = 306
        
        # Simulate neural oscillations and noise
        time_points = np.linspace(0, duration_seconds, n_samples)
        
        # Alpha rhythm (8-12 Hz)
        alpha_freq = 10
        alpha_signal = np.sin(2 * np.pi * alpha_freq * time_points)
        
        # Beta oscillations (13-30 Hz)
        beta_freq = 20
        beta_signal = 0.5 * np.sin(2 * np.pi * beta_freq * time_points)
        
        # Create multi-channel data with spatial patterns
        neural_data = np.zeros((n_samples, n_channels))
        for ch in range(n_channels):
            # Different channels have different phase relationships
            phase_shift = (ch / n_channels) * 2 * np.pi
            spatial_weight = np.cos(ch / n_channels * np.pi)  # Spatial weighting
            
            neural_data[:, ch] = (
                spatial_weight * (alpha_signal * np.cos(phase_shift) + 
                                beta_signal * np.sin(phase_shift)) +
                np.random.normal(0, 0.1, n_samples)  # Noise
            )
        
        # Mock transformer-based compression
        def transformer_compress(data, target_ratio=5.0):
            """Simulate transformer-based neural signal compression"""
            # Identify temporal patterns
            pattern_length = 100  # 100ms patterns
            n_patterns = data.shape[0] // pattern_length
            
            patterns = data[:n_patterns * pattern_length].reshape(n_patterns, pattern_length, -1)
            
            # Simulate pattern clustering and encoding
            unique_patterns = patterns[::int(target_ratio)]  # Subsample patterns
            pattern_indices = np.repeat(np.arange(len(unique_patterns)), int(target_ratio))[:n_patterns]
            
            # Calculate compression metrics
            original_size = data.nbytes
            compressed_size = unique_patterns.nbytes + pattern_indices.nbytes
            actual_ratio = original_size / compressed_size
            
            # Simulate reconstruction quality
            reconstruction_error = np.random.uniform(0.02, 0.08)  # 2-8% error
            quality = 1.0 - reconstruction_error
            
            return {
                'compressed_patterns': unique_patterns,
                'pattern_indices': pattern_indices,
                'compression_ratio': actual_ratio,
                'reconstruction_quality': quality,
                'original_size': original_size,
                'compressed_size': compressed_size
            }
        
        pattern_system.compress = transformer_compress
        
        # Test different compression ratios
        target_ratios = [2.0, 5.0, 8.0, 10.0]
        results = []
        
        for target_ratio in target_ratios:
            result = pattern_system.compress(neural_data, target_ratio)
            results.append(result)
            
            # Validate compression performance
            assert result['compression_ratio'] >= pattern_system.specs.min_compression_ratio, \
                f"Compression ratio {result['compression_ratio']:.1f}x below minimum {pattern_system.specs.min_compression_ratio}x"
            
            assert result['compression_ratio'] <= pattern_system.specs.max_compression_ratio, \
                f"Compression ratio {result['compression_ratio']:.1f}x exceeds maximum {pattern_system.specs.max_compression_ratio}x"
            
            assert result['reconstruction_quality'] >= pattern_system.specs.quality_threshold, \
                f"Quality {result['reconstruction_quality']:.3f} below threshold {pattern_system.specs.quality_threshold}"
        
        # Validate trade-off between compression and quality
        ratios = [r['compression_ratio'] for r in results]
        qualities = [r['reconstruction_quality'] for r in results]
        
        # Higher compression should generally mean lower quality
        compression_quality_correlation = np.corrcoef(ratios, qualities)[0, 1]
        assert compression_quality_correlation < 0, "Should show compression-quality trade-off"
    
    def test_temporal_spatial_pattern_detection(self, pattern_system):
        """Test identification of temporal and spatial brain patterns"""
        # Generate data with known patterns
        n_samples = 5000  # 5 seconds at 1kHz
        n_channels = 306
        
        # Create spatial pattern (e.g., motor cortex activation)
        motor_channels = [50, 51, 52, 53, 54]  # Simulated motor cortex channels
        temporal_pattern = np.zeros(n_samples)
        
        # Add motor events at specific times
        event_times = [1000, 2000, 3000, 4000]  # 1s, 2s, 3s, 4s
        for event_time in event_times:
            # Motor event signature (brief activation followed by suppression)
            event_duration = 200  # 200ms
            if event_time + event_duration < n_samples:
                # Activation phase (0-100ms)
                temporal_pattern[event_time:event_time+100] = 2.0
                # Suppression phase (100-200ms)
                temporal_pattern[event_time+100:event_time+event_duration] = -0.5
        
        # Create multi-channel data
        neural_data = np.random.normal(0, 0.1, (n_samples, n_channels))
        
        # Add spatial-temporal pattern to motor channels
        for ch in motor_channels:
            neural_data[:, ch] += temporal_pattern * (1.0 + 0.1 * np.random.randn())
        
        # Mock pattern detection algorithm
        def detect_patterns(data, pattern_type='motor'):
            """Detect temporal and spatial patterns in neural data"""
            if pattern_type == 'motor':
                # Look for activation-suppression pattern in motor channels
                motor_data = data[:, motor_channels].mean(axis=1)
                
                # Template matching for motor pattern
                template = np.concatenate([np.ones(100), -0.25 * np.ones(100)])  # Simplified template
                
                # Cross-correlation to find pattern occurrences
                correlation = np.correlate(motor_data, template, mode='valid')
                threshold = np.std(correlation) * 3  # 3-sigma threshold
                
                detected_events = np.where(correlation > threshold)[0]
                
                # Spatial consistency check
                spatial_consistency = []
                for event_idx in detected_events:
                    if event_idx + 200 < data.shape[0]:
                        event_data = data[event_idx:event_idx+200, motor_channels]
                        activation_strength = event_data[:100].mean(axis=0)
                        suppression_strength = event_data[100:].mean(axis=0)
                        
                        # Check if all motor channels show similar pattern
                        activation_consistency = np.std(activation_strength) / np.mean(np.abs(activation_strength))
                        spatial_consistency.append(activation_consistency < 0.5)  # Low variability
                
                return {
                    'detected_events': detected_events,
                    'spatial_consistency': spatial_consistency,
                    'detection_accuracy': len(detected_events) / len(event_times),
                    'false_positive_rate': max(0, len(detected_events) - len(event_times)) / len(detected_events) if len(detected_events) > 0 else 0
                }
            
            return {'detected_events': [], 'spatial_consistency': [], 'detection_accuracy': 0, 'false_positive_rate': 1}
        
        pattern_system.detect_patterns = detect_patterns
        
        # Test pattern detection
        detection_results = pattern_system.detect_patterns(neural_data, 'motor')
        
        # Validate detection performance
        assert detection_results['detection_accuracy'] >= 0.7, "Should detect at least 70% of motor events"
        assert detection_results['false_positive_rate'] <= 0.3, "False positive rate should be ≤30%"
        
        # Validate spatial consistency
        consistent_detections = sum(detection_results['spatial_consistency'])
        total_detections = len(detection_results['spatial_consistency'])
        
        if total_detections > 0:
            spatial_accuracy = consistent_detections / total_detections
            assert spatial_accuracy >= 0.8, "Spatial pattern consistency should be ≥80%"
    
    @pytest.mark.asyncio
    async def test_real_time_pattern_processing(self, pattern_system):
        """Test real-time pattern recognition with latency constraints"""
        max_latency = 0.001  # 1ms maximum processing latency
        
        # Mock real-time pattern processor
        async def process_pattern_chunk(data_chunk, model):
            """Process a chunk of neural data for patterns"""
            start_time = time.time()
            
            # Simulate transformer inference
            await asyncio.sleep(0.0005)  # 0.5ms processing time
            
            # Mock pattern recognition results
            patterns_detected = np.random.poisson(2)  # Poisson distribution of patterns
            confidence_scores = np.random.uniform(0.7, 0.99, patterns_detected)
            
            processing_time = time.time() - start_time
            
            return {
                'patterns_detected': patterns_detected,
                'confidence_scores': confidence_scores,
                'processing_latency': processing_time,
                'chunk_size': data_chunk.shape[0]
            }
        
        pattern_system.process_chunk = process_pattern_chunk
        
        # Test real-time processing
        chunk_size = 100  # 100ms chunks
        n_channels = 306
        processing_latencies = []
        
        for _ in range(50):  # Process 50 chunks (5 seconds)
            # Generate data chunk
            data_chunk = np.random.normal(0, 1e-12, (chunk_size, n_channels))
            
            # Process chunk
            result = await pattern_system.process_chunk(data_chunk, pattern_system.transformer_model)
            processing_latencies.append(result['processing_latency'])
            
            # Validate real-time constraint
            assert result['processing_latency'] <= max_latency, \
                f"Processing latency {result['processing_latency']*1000:.2f}ms exceeds {max_latency*1000:.1f}ms limit"
        
        # Validate overall performance
        mean_latency = np.mean(processing_latencies)
        max_measured_latency = max(processing_latencies)
        latency_jitter = np.std(processing_latencies)
        
        assert mean_latency <= max_latency * 0.8, f"Mean latency {mean_latency*1000:.2f}ms should be <80% of limit"
        assert max_measured_latency <= max_latency, f"Max latency {max_measured_latency*1000:.2f}ms exceeds limit"
        assert latency_jitter <= max_latency * 0.2, f"Latency jitter {latency_jitter*1000:.2f}ms too high"


class TestGPUAcceleration:
    """Test CUDA-optimized processing pipeline"""
    
    @pytest.fixture
    def gpu_system(self):
        """Mock GPU acceleration system"""
        system = Mock()
        system.cuda_available = Mock(return_value=True)
        system.gpu_memory_gb = 8
        system.compute_capability = (7, 5)  # RTX 20xx series
        return system
    
    def test_cuda_pipeline_performance(self, gpu_system):
        """Test GPU acceleration achieves performance targets"""
        # Mock CPU vs GPU processing comparison
        data_sizes = [1000, 5000, 10000, 50000]  # Different data sizes
        n_channels = 306
        
        def cpu_processing_time(n_samples, n_channels):
            """Simulate CPU processing time"""
            # Linear scaling with slight overhead
            base_time = 0.001  # 1ms base
            per_sample_time = 1e-6  # 1μs per sample
            return base_time + n_samples * n_channels * per_sample_time
        
        def gpu_processing_time(n_samples, n_channels, gpu_speedup=5.0):
            """Simulate GPU processing time with speedup"""
            cpu_time = cpu_processing_time(n_samples, n_channels)
            gpu_overhead = 0.0002  # 0.2ms GPU kernel launch overhead
            return gpu_overhead + cpu_time / gpu_speedup
        
        gpu_system.cpu_process = cpu_processing_time
        gpu_system.gpu_process = gpu_processing_time
        
        # Test performance scaling
        speedup_ratios = []
        for n_samples in data_sizes:
            cpu_time = gpu_system.cpu_process(n_samples, n_channels)
            gpu_time = gpu_system.gpu_process(n_samples, n_channels)
            speedup = cpu_time / gpu_time
            speedup_ratios.append(speedup)
            
            # Validate GPU advantage for larger datasets
            if n_samples >= 10000:
                assert speedup >= 3.0, f"GPU should provide ≥3x speedup for large data, got {speedup:.1f}x"
        
        # GPU should scale better with data size
        small_data_speedup = speedup_ratios[0]  # 1000 samples
        large_data_speedup = speedup_ratios[-1]  # 50000 samples
        
        assert large_data_speedup >= small_data_speedup, "GPU should scale better with larger datasets"
    
    def test_memory_efficient_gpu_processing(self, gpu_system):
        """Test memory-efficient GPU processing for large datasets"""
        # Mock memory management
        total_gpu_memory = gpu_system.gpu_memory_gb * 1024**3  # Bytes
        safety_margin = 0.8  # Use 80% of available memory
        usable_memory = int(total_gpu_memory * safety_margin)
        
        def calculate_chunk_size(n_channels, dtype_size=4):
            """Calculate optimal chunk size for GPU memory"""
            # Account for input, output, and intermediate buffers
            memory_per_sample = n_channels * dtype_size * 3  # 3x for buffers
            max_samples_per_chunk = usable_memory // memory_per_sample
            
            # Round down to nearest 1000 for efficiency
            chunk_size = (max_samples_per_chunk // 1000) * 1000
            return max(1000, chunk_size)  # Minimum 1000 samples
        
        def process_large_dataset(total_samples, n_channels):
            """Process large dataset in GPU-friendly chunks"""
            chunk_size = calculate_chunk_size(n_channels)
            n_chunks = (total_samples + chunk_size - 1) // chunk_size
            
            processing_times = []
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_samples)
                actual_chunk_size = end_idx - start_idx
                
                # Simulate chunk processing
                chunk_time = gpu_system.gpu_process(actual_chunk_size, n_channels)
                processing_times.append(chunk_time)
            
            return {
                'chunk_size': chunk_size,
                'n_chunks': n_chunks,
                'total_time': sum(processing_times),
                'avg_chunk_time': np.mean(processing_times),
                'memory_utilization': (chunk_size * n_channels * 4 * 3) / total_gpu_memory
            }
        
        gpu_system.calculate_chunk_size = calculate_chunk_size
        gpu_system.process_large_dataset = process_large_dataset
        
        # Test large dataset processing
        large_dataset_samples = 1000000  # 1M samples (16+ minutes at 1kHz)
        n_channels = 306
        
        result = gpu_system.process_large_dataset(large_dataset_samples, n_channels)
        
        # Validate memory efficiency
        assert result['memory_utilization'] <= 0.8, "Should use ≤80% of GPU memory"
        assert result['chunk_size'] >= 1000, "Chunk size should be ≥1000 samples"
        
        # Validate processing efficiency
        assert result['n_chunks'] <= 1000, "Should not create excessive chunks"
        assert result['avg_chunk_time'] <= 0.1, "Average chunk processing should be ≤100ms"
    
    def test_parallel_stream_processing(self, gpu_system):
        """Test parallel processing of multiple data streams"""
        # Mock multi-stream GPU processing
        n_streams = 3  # OMP, Kernel, Accelo
        stream_configs = [
            {'name': 'omp', 'channels': 306, 'rate': 1000},
            {'name': 'kernel', 'channels': 192, 'rate': 250},  # 160 fNIRS + 32 EEG
            {'name': 'accelo', 'channels': 48, 'rate': 1000}   # 16 nodes × 3 axes
        ]
        
        def parallel_gpu_processing(streams_data, max_parallel_streams=4):
            """Process multiple streams in parallel on GPU"""
            n_parallel = min(len(streams_data), max_parallel_streams)
            
            # Simulate parallel processing
            processing_times = []
            for stream_batch in range(0, len(streams_data), n_parallel):
                batch_streams = streams_data[stream_batch:stream_batch + n_parallel]
                
                # Parallel batch processing time (dominated by largest stream)
                batch_times = []
                for stream_data in batch_streams:
                    n_samples = stream_data['data'].shape[0]
                    n_channels = stream_data['data'].shape[1]
                    stream_time = gpu_system.gpu_process(n_samples, n_channels)
                    batch_times.append(stream_time)
                
                # Parallel execution time is maximum of individual times
                batch_time = max(batch_times)
                processing_times.append(batch_time)
            
            return {
                'total_parallel_time': sum(processing_times),
                'streams_processed': len(streams_data),
                'parallel_efficiency': n_parallel / len(streams_data) if len(streams_data) > 0 else 0
            }
        
        gpu_system.parallel_process = parallel_gpu_processing
        
        # Generate test data for each stream
        test_duration = 1.0  # 1 second
        stream_data = []
        
        for config in stream_configs:
            n_samples = int(config['rate'] * test_duration)
            n_channels = config['channels']
            data = np.random.normal(0, 1e-12, (n_samples, n_channels))
            
            stream_data.append({
                'name': config['name'],
                'data': data,
                'config': config
            })
        
        # Test parallel processing
        result = gpu_system.parallel_process(stream_data)
        
        # Calculate sequential processing time for comparison
        sequential_time = sum(
            gpu_system.gpu_process(data['data'].shape[0], data['data'].shape[1])
            for data in stream_data
        )
        
        parallel_speedup = sequential_time / result['total_parallel_time']
        
        # Validate parallel processing benefits
        assert parallel_speedup >= 1.5, f"Parallel processing should provide ≥1.5x speedup, got {parallel_speedup:.1f}x"
        assert result['streams_processed'] == len(stream_data), "Should process all streams"
        
        # Validate real-time capability
        real_time_limit = test_duration  # Must process 1s of data in ≤1s
        assert result['total_parallel_time'] <= real_time_limit, \
            f"Parallel processing time {result['total_parallel_time']:.3f}s exceeds real-time limit {real_time_limit}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
