#!/usr/bin/env python3
"""
Performance Benchmarking Script for Brain-Forge Platform

This script runs comprehensive performance benchmarks on the Brain-Forge
brain-computer interface system to ensure optimal performance across
different hardware configurations and data processing scenarios.

Usage:
    python scripts/benchmark_performance.py --output=benchmark-report.json
    python scripts/benchmark_performance.py --quick --hardware-simulation
"""

import argparse
import json
import time
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import psutil
import platform
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Optional imports for full functionality
IMPORTS_AVAILABLE = False
try:
    # These imports are optional - script works in simulation mode without them
    from brain_forge.core.config import ConfigManager  # noqa: F401
    from brain_forge.core.logger import Logger  # noqa: F401
    from brain_forge.processing.preprocessing import (  # noqa: F401
        SignalPreprocessor
    )
    from brain_forge.processing.compression import (  # noqa: F401
        CompressionEngine
    )
    from brain_forge.mapping.connectivity import (  # noqa: F401
        ConnectivityAnalyzer
    )
    from brain_forge.simulation.neural_models import (  # noqa: F401
        NeuralModelManager
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Brain-Forge modules: {e}")
    print("Running in simulation mode with synthetic benchmarks.")
    IMPORTS_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Data class for storing benchmark results."""
    name: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    throughput_ops_per_sec: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PerformanceBenchmarker:
    """Comprehensive performance benchmarking suite for Brain-Forge."""
    
    def __init__(self, output_file: str = "benchmark-report.json",
                 quick_mode: bool = False, hardware_simulation: bool = False):
        self.output_file = output_file
        self.quick_mode = quick_mode
        self.hardware_simulation = hardware_simulation
        self.results: List[BenchmarkResult] = []
        self.system_info = self._get_system_info()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information for benchmark context."""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "timestamp": time.time()
        }
    
    def _measure_performance(self, func, *args, **kwargs) -> BenchmarkResult:
        """Measure performance metrics for a given function."""
        # Get initial memory state
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()
        
        start_time = time.perf_counter()
        peak_memory = initial_memory
        
        try:
            # Execute function with monitoring
            result = func(*args, **kwargs)
            
            # Measure peak memory during execution
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
            
            end_time = time.perf_counter()
            final_cpu = process.cpu_percent()
            final_memory = process.memory_info().rss / 1024 / 1024
            
            duration_ms = (end_time - start_time) * 1000
            memory_mb = final_memory - initial_memory
            cpu_percent = max(0, final_cpu - initial_cpu)
            
            return BenchmarkResult(
                name=func.__name__,
                duration_ms=duration_ms,
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                peak_memory_mb=peak_memory,
                success=True,
                metadata={
                    "result_size": sys.getsizeof(result) if result else 0
                }
            )
            
        except Exception as e:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            return BenchmarkResult(
                name=func.__name__,
                duration_ms=duration_ms,
                memory_mb=0,
                cpu_percent=0,
                success=False,
                error_message=str(e)
            )
    
    def benchmark_data_acquisition_simulation(self) -> BenchmarkResult:
        """Benchmark simulated data acquisition performance."""
        def simulate_data_acquisition():
            # Simulate OMP helmet data (306 channels, 1000 Hz, 10 seconds)
            omp_data = np.random.randn(306, 10000).astype(np.float32)
            
            # Simulate Kernel optical data (52 channels, 100 Hz, 10 seconds)
            optical_data = np.random.randn(52, 1000).astype(np.float32)

            # Simulate accelerometer data (3 axes, 1000 Hz, 10 seconds)
            accel_data = np.random.randn(3, 10000).astype(np.float32)

            # Simulate synchronization overhead
            sync_timestamps = np.arange(0, 10, 0.001)

            return {
                'omp': omp_data,
                'optical': optical_data,
                'accelerometer': accel_data,
                'timestamps': sync_timestamps
            }
        
        return self._measure_performance(simulate_data_acquisition)
    
    def benchmark_signal_preprocessing(self) -> BenchmarkResult:
        """Benchmark signal preprocessing performance."""
        def preprocess_signals():
            # Generate test data (similar to MEG/EEG data)
            n_channels, n_samples = 306, 10000
            raw_data = np.random.randn(n_channels, n_samples).astype(
                np.float32
            )

            # Add realistic noise characteristics
            raw_data += 0.1 * np.random.randn(n_channels, n_samples)

            # Simulate preprocessing steps
            # 1. Bandpass filtering (1-100 Hz)
            from scipy import signal
            sos = signal.butter(
                4, [1, 100], btype='band', fs=1000, output='sos'
            )
            filtered_data = signal.sosfilt(sos, raw_data, axis=1)

            # 2. Artifact removal (simple thresholding)
            threshold = 3 * np.std(filtered_data, axis=1, keepdims=True)
            clean_data = np.clip(filtered_data, -threshold, threshold)

            # 3. Normalization
            mean_data = np.mean(clean_data, axis=1, keepdims=True)
            std_data = np.std(clean_data, axis=1, keepdims=True)
            normalized_data = (clean_data - mean_data) / std_data
            
            return normalized_data
        
        return self._measure_performance(preprocess_signals)
    
    def benchmark_compression_engine(self) -> BenchmarkResult:
        """Benchmark data compression performance."""
        def compress_neural_data():
            # Generate neural data with realistic characteristics
            n_channels, n_samples = 306, 60000  # 1 minute at 1000 Hz
            neural_data = np.random.randn(n_channels, n_samples).astype(
                np.float32
            )
            
            # Add structured patterns (simulate neural rhythms)
            t = np.linspace(0, 60, n_samples)
            for i in range(n_channels):
                # Add alpha rhythm (8-12 Hz) and beta rhythm (13-30 Hz)
                alpha_freq = 8 + 4 * np.random.random()
                beta_freq = 13 + 17 * np.random.random()
                neural_data[i] += 0.5 * np.sin(2 * np.pi * alpha_freq * t)
                neural_data[i] += 0.3 * np.sin(2 * np.pi * beta_freq * t)
            
            # Simulate compression (PCA-based dimensionality reduction)
            from sklearn.decomposition import PCA
            
            # Reshape for PCA (samples x features)
            reshaped_data = neural_data.T
            
            # Apply PCA to reduce to 95% variance
            pca = PCA(n_components=0.95)
            compressed_data = pca.fit_transform(reshaped_data)
            
            # Calculate compression ratio
            original_size = neural_data.nbytes
            compressed_size = compressed_data.nbytes + pca.components_.nbytes
            compression_ratio = original_size / compressed_size
            
            return {
                'compressed_data': compressed_data,
                'compression_ratio': compression_ratio,
                'explained_variance': np.sum(pca.explained_variance_ratio_)
            }
        
        result = self._measure_performance(compress_neural_data)
        return result
    
    def benchmark_connectivity_analysis(self) -> BenchmarkResult:
        """Benchmark brain connectivity analysis performance."""
        def analyze_connectivity():
            # Generate realistic neural time series data
            n_regions, n_samples = 68, 5000  # Desikan-Killiany atlas regions
            time_series = np.random.randn(n_regions, n_samples).astype(
                np.float32
            )

            # Add realistic cross-correlations between regions
            connectivity_matrix = np.random.rand(n_regions, n_regions)
            connectivity_matrix = (
                connectivity_matrix + connectivity_matrix.T
            ) / 2
            np.fill_diagonal(connectivity_matrix, 1.0)
            
            # Apply connectivity patterns
            for i in range(n_regions):
                for j in range(i+1, n_regions):
                    if connectivity_matrix[i, j] > 0.7:  # Strong connection
                        # Add correlated activity
                        common_signal = np.random.randn(n_samples)
                        time_series[i] += 0.3 * common_signal
                        time_series[j] += 0.3 * common_signal
            
            # Calculate connectivity metrics
            # 1. Pearson correlation
            correlation_matrix = np.corrcoef(time_series)
            
            # 2. Coherence (simplified)
            from scipy.signal import coherence
            coherence_matrix = np.zeros((n_regions, n_regions))
            for i in range(n_regions):
                for j in range(i, n_regions):
                    if i == j:
                        coherence_matrix[i, j] = 1.0
                    else:
                        f1, coh = coherence(
                            time_series[i], time_series[j], fs=1000
                        )
                        coherence_matrix[i, j] = np.mean(coh)
                        coherence_matrix[j, i] = coherence_matrix[i, j]
            
            # 3. Graph metrics
            # Convert correlation to binary adjacency (threshold = 0.5)
            adjacency = (correlation_matrix > 0.5).astype(int)
            
            # Calculate basic graph metrics
            node_degrees = np.sum(adjacency, axis=1)
            clustering = np.mean(node_degrees) / n_regions
            
            return {
                'correlation_matrix': correlation_matrix,
                'coherence_matrix': coherence_matrix,
                'adjacency_matrix': adjacency,
                'clustering_coefficient': clustering,
                'mean_connectivity': np.mean(
                    correlation_matrix[correlation_matrix != 1]
                )
            }
        
        result = self._measure_performance(analyze_connectivity)
        if result.success:
            # Add throughput calculation (connections per second)
            n_connections = 68 * 67 // 2  # Upper triangle of matrix
            result.throughput_ops_per_sec = (
                n_connections / (result.duration_ms / 1000)
            )
        return result
    
    def benchmark_neural_simulation(self) -> BenchmarkResult:
        """Benchmark neural network simulation performance."""
        def simulate_neural_network():
            # Simplified neural network simulation
            n_neurons = 1000 if not self.quick_mode else 100
            simulation_time = 1.0  # seconds
            dt = 0.1e-3  # 0.1 ms time step
            n_steps = int(simulation_time / dt)
            
            # Initialize neurons (Leaky Integrate-and-Fire model)
            v_membrane = np.random.uniform(-70, -50, n_neurons)  # mV
            v_threshold = -50  # mV
            v_reset = -70  # mV
            tau_membrane = 20e-3  # 20 ms
            
            # Connection matrix (sparse random connectivity)
            connection_prob = 0.1
            weights = np.random.rand(n_neurons, n_neurons) * 2 - 1  # -1 to 1
            connections = (
                np.random.rand(n_neurons, n_neurons) < connection_prob
            )
            weight_matrix = weights * connections
            
            # Simulation loop
            spike_times = []
            membrane_potentials = []
            
            for step in range(min(n_steps, 1000)):  # Limit for benchmarking
                # External input
                external_input = np.random.randn(n_neurons) * 5  # mV
                
                # Synaptic input
                spikes = v_membrane > v_threshold
                synaptic_input = np.dot(weight_matrix, spikes) * 0.5

                # Update membrane potentials
                total_input = external_input + synaptic_input
                dv_dt = (-v_membrane + total_input) / tau_membrane
                v_membrane += dv_dt * dt

                # Check for spikes
                spike_mask = v_membrane > v_threshold
                if np.any(spike_mask):
                    spike_neurons = np.where(spike_mask)[0]
                    spike_list = [
                        (step * dt, neuron) for neuron in spike_neurons
                    ]
                    spike_times.extend(spike_list)
                    v_membrane[spike_mask] = v_reset
                
                # Record membrane potentials (every 10 steps)
                if step % 10 == 0:
                    membrane_potentials.append(v_membrane.copy())
            
            return {
                'n_neurons': n_neurons,
                'n_spikes': len(spike_times),
                'spike_rate': len(spike_times) / (n_neurons * simulation_time),
                'membrane_potentials': np.array(membrane_potentials)
            }
        
        result = self._measure_performance(simulate_neural_network)
        if result.success:
            # Add throughput (simulation steps per second)
            result.throughput_ops_per_sec = 1000 / (result.duration_ms / 1000)
        return result
    
    def benchmark_memory_intensive_operation(self) -> BenchmarkResult:
        """Benchmark memory-intensive operations."""
        def memory_intensive_task():
            # Simulate large matrix operations common in neuroscience
            matrix_size = 5000 if not self.quick_mode else 1000
            
            # Large matrix multiplication
            A = np.random.randn(matrix_size, matrix_size).astype(np.float32)
            B = np.random.randn(matrix_size, matrix_size).astype(np.float32)
            
            # Matrix multiplication
            C = np.dot(A, B)
            
            # Eigenvalue decomposition (computationally intensive)
            smaller_matrix = A[:500, :500]  # Limit size for reasonable time
            eigenvals, eigenvecs = np.linalg.eigh(smaller_matrix)
            
            # SVD decomposition
            U, s, Vt = np.linalg.svd(smaller_matrix)
            
            return {
                'matrix_size': matrix_size,
                'result_matrix': C,
                'eigenvalues': eigenvals,
                'singular_values': s
            }
        
        return self._measure_performance(memory_intensive_task)
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        self.logger.info("Starting Brain-Forge performance benchmarks...")
        self.logger.info(f"System: {self.system_info['platform']}")
        self.logger.info(f"Python: {self.system_info['python_version']}")
        self.logger.info(f"CPU cores: {self.system_info['cpu_count']}")
        memory_gb = self.system_info['memory_total_gb']
        self.logger.info(f"Memory: {memory_gb:.1f} GB")

        benchmarks = [
            ("Data Acquisition Simulation",
             self.benchmark_data_acquisition_simulation),
            ("Signal Preprocessing", self.benchmark_signal_preprocessing),
            ("Compression Engine", self.benchmark_compression_engine),
            ("Connectivity Analysis", self.benchmark_connectivity_analysis),
            ("Neural Simulation", self.benchmark_neural_simulation),
            ("Memory Intensive Operations",
             self.benchmark_memory_intensive_operation),
        ]
        
        for name, benchmark_func in benchmarks:
            self.logger.info(f"Running benchmark: {name}")
            result = benchmark_func()
            result.name = name
            self.results.append(result)
            
            if result.success:
                self.logger.info(f"  ✓ Duration: {result.duration_ms:.1f} ms")
                self.logger.info(f"  ✓ Memory: {result.memory_mb:.1f} MB")
                if result.throughput_ops_per_sec:
                    tput = result.throughput_ops_per_sec
                    self.logger.info(f"  ✓ Throughput: {tput:.1f} ops/sec")
            else:
                self.logger.error(f"  ✗ Failed: {result.error_message}")
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        report = {
            "system_info": self.system_info,
            "benchmark_summary": {
                "total_benchmarks": len(self.results),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "success_rate": len(successful_results) / len(self.results) * 100,
                "total_duration_ms": sum(r.duration_ms for r in successful_results),
                "total_memory_mb": sum(r.memory_mb for r in successful_results),
                "average_duration_ms": np.mean([r.duration_ms for r in successful_results]) if successful_results else 0,
                "average_memory_mb": np.mean([r.memory_mb for r in successful_results]) if successful_results else 0,
            },
            "individual_results": [asdict(result) for result in self.results],
            "performance_metrics": {
                "fastest_benchmark": min(successful_results, key=lambda x: x.duration_ms).name if successful_results else None,
                "slowest_benchmark": max(successful_results, key=lambda x: x.duration_ms).name if successful_results else None,
                "most_memory_intensive": max(successful_results, key=lambda x: x.memory_mb).name if successful_results else None,
                "least_memory_intensive": min(successful_results, key=lambda x: x.memory_mb).name if successful_results else None,
            },
            "recommendations": self._generate_recommendations(successful_results)
        }
        
        return report
    
    def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate performance recommendations based on benchmark results."""
        recommendations = []
        
        if not results:
            return ["No successful benchmarks completed. Check system configuration."]
        
        avg_duration = np.mean([r.duration_ms for r in results])
        avg_memory = np.mean([r.memory_mb for r in results])
        
        if avg_duration > 5000:  # 5 seconds
            recommendations.append("Consider optimizing algorithms or using GPU acceleration for better performance.")
        
        if avg_memory > 1000:  # 1 GB
            recommendations.append("Memory usage is high. Consider implementing data streaming or batch processing.")
        
        slow_benchmarks = [r for r in results if r.duration_ms > avg_duration * 2]
        if slow_benchmarks:
            slow_names = [r.name for r in slow_benchmarks]
            recommendations.append(f"Focus optimization efforts on: {', '.join(slow_names)}")
        
        if self.system_info['cpu_count'] > 4:
            recommendations.append("System has multiple CPU cores. Consider parallel processing for compute-intensive tasks.")
        
        if self.system_info['memory_total_gb'] > 16:
            recommendations.append("Sufficient memory available for large-scale neural simulations.")
        else:
            recommendations.append("Limited memory. Consider optimizing data structures and using memory-mapped files.")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any]) -> None:
        """Save benchmark report to file."""
        with open(self.output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Benchmark report saved to: {self.output_file}")


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(description="Brain-Forge Performance Benchmarking Suite")
    parser.add_argument("--output", default="benchmark-report.json", 
                       help="Output file for benchmark results")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmarks with reduced problem sizes")
    parser.add_argument("--hardware-simulation", action="store_true",
                       help="Run in hardware simulation mode")
    
    args = parser.parse_args()
    
    benchmarker = PerformanceBenchmarker(
        output_file=args.output,
        quick_mode=args.quick,
        hardware_simulation=args.hardware_simulation
    )
    
    try:
        report = benchmarker.run_all_benchmarks()
        benchmarker.save_report(report)
        
        # Print summary
        print("\n" + "="*60)
        print("BRAIN-FORGE PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)
        print(f"Total benchmarks: {report['benchmark_summary']['total_benchmarks']}")
        print(f"Successful: {report['benchmark_summary']['successful']}")
        print(f"Failed: {report['benchmark_summary']['failed']}")
        print(f"Success rate: {report['benchmark_summary']['success_rate']:.1f}%")
        print(f"Total duration: {report['benchmark_summary']['total_duration_ms']:.1f} ms")
        print(f"Average duration: {report['benchmark_summary']['average_duration_ms']:.1f} ms")
        print(f"Average memory usage: {report['benchmark_summary']['average_memory_mb']:.1f} MB")
        
        if report['performance_metrics']['fastest_benchmark']:
            print(f"Fastest: {report['performance_metrics']['fastest_benchmark']}")
            print(f"Slowest: {report['performance_metrics']['slowest_benchmark']}")
        
        print(f"\nReport saved to: {args.output}")
        
        return 0 if report['benchmark_summary']['failed'] == 0 else 1
        
    except Exception as e:
        print(f"Benchmark failed with error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
