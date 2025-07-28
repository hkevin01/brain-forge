"""
Real-time Processing Pipeline for Brain-Forge

This module implements advanced signal processing capabilities for multi-modal
brain data, including filtering, compression, feature extraction, and artifact removal.
"""

import numpy as np
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time
import logging
from concurrent.futures import ThreadPoolExecutor

from ..core.config import Config
from ..core.logger import get_logger
from ..core.exceptions import BrainForgeError

# Signal processing libraries
from scipy import signal, fft
from scipy.stats import zscore
import pywt  # PyWavelets for wavelet transforms
from sklearn.decomposition import PCA, ICA, FastICA
from sklearn.preprocessing import StandardScaler

logger = get_logger(__name__)


@dataclass
class ProcessingParameters:
    """Configuration parameters for signal processing"""
    sampling_rate: float = 1000.0
    filter_low: float = 1.0
    filter_high: float = 100.0
    notch_freq: float = 60.0
    artifact_threshold: float = 3.0
    compression_ratio: float = 5.0
    wavelet_type: str = 'db8'
    ica_components: int = 20


class RealTimeFilter:
    """Real-time digital filter implementation"""
    
    def __init__(self, filter_type: str, frequencies: Tuple[float, ...], 
                 sampling_rate: float, order: int = 4):
        self.filter_type = filter_type
        self.frequencies = frequencies
        self.sampling_rate = sampling_rate
        self.order = order
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Design filter coefficients
        self._design_filter()
        
        # Initialize filter states for real-time processing
        self.zi = None
        
    def _design_filter(self):
        """Design digital filter coefficients"""
        try:
            nyquist = self.sampling_rate / 2
            
            if self.filter_type == 'bandpass':
                low, high = self.frequencies
                self.sos = signal.butter(
                    self.order, 
                    [low / nyquist, high / nyquist], 
                    btype='band', 
                    output='sos'
                )
                
            elif self.filter_type == 'lowpass':
                cutoff = self.frequencies[0]
                self.sos = signal.butter(
                    self.order, 
                    cutoff / nyquist, 
                    btype='low', 
                    output='sos'
                )
                
            elif self.filter_type == 'highpass':
                cutoff = self.frequencies[0]
                self.sos = signal.butter(
                    self.order, 
                    cutoff / nyquist, 
                    btype='high', 
                    output='sos'
                )
                
            elif self.filter_type == 'notch':
                freq = self.frequencies[0]
                Q = 30  # Quality factor
                self.sos = signal.iirnotch(freq / nyquist, Q, output='sos')
                
            self.logger.info(f"{self.filter_type} filter designed: {self.frequencies} Hz")
            
        except Exception as e:
            self.logger.error(f"Filter design failed: {e}")
            raise BrainForgeError(f"Failed to design {self.filter_type} filter: {e}")
    
    def apply_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply filter to data chunk"""
        try:
            if self.zi is None:
                # Initialize filter states
                self.zi = signal.sosfilt_zi(self.sos)
                if data.ndim > 1:
                    self.zi = np.tile(self.zi[:, np.newaxis, :], 
                                     (1, data.shape[0], 1))
            
            # Apply filter
            if data.ndim == 1:
                filtered_data, self.zi = signal.sosfilt(
                    self.sos, data, zi=self.zi
                )
            else:
                filtered_data, self.zi = signal.sosfilt(
                    self.sos, data, zi=self.zi, axis=1
                )
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Filter application failed: {e}")
            raise BrainForgeError(f"Failed to apply filter: {e}")


class WaveletCompressor:
    """Wavelet-based neural signal compression"""
    
    def __init__(self, wavelet: str = 'db8', levels: int = 6, 
                 threshold_mode: str = 'soft'):
        self.wavelet = wavelet
        self.levels = levels
        self.threshold_mode = threshold_mode
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
    def compress(self, data: np.ndarray, compression_ratio: float = 5.0) -> Dict[str, Any]:
        """Compress neural data using wavelet transform"""
        try:
            compressed_data = {}
            
            if data.ndim == 1:
                # Single channel compression
                compressed_data = self._compress_channel(data, compression_ratio)
            else:
                # Multi-channel compression
                compressed_channels = []
                for i in range(data.shape[0]):
                    channel_compressed = self._compress_channel(
                        data[i], compression_ratio
                    )
                    compressed_channels.append(channel_compressed)
                
                compressed_data = {
                    'channels': compressed_channels,
                    'n_channels': data.shape[0],
                    'original_shape': data.shape
                }
            
            # Calculate actual compression ratio
            original_size = data.nbytes
            compressed_size = self._estimate_compressed_size(compressed_data)
            actual_ratio = original_size / compressed_size
            
            compressed_data['compression_ratio'] = actual_ratio
            compressed_data['original_size'] = original_size
            compressed_data['compressed_size'] = compressed_size
            
            self.logger.info(f"Compression complete: {actual_ratio:.2f}x ratio")
            return compressed_data
            
        except Exception as e:
            self.logger.error(f"Compression failed: {e}")
            raise BrainForgeError(f"Wavelet compression failed: {e}")
    
    def _compress_channel(self, channel_data: np.ndarray, 
                         compression_ratio: float) -> Dict[str, Any]:
        """Compress single channel using wavelets"""
        # Wavelet decomposition
        coeffs = pywt.wavedec(channel_data, self.wavelet, level=self.levels)
        
        # Calculate threshold for desired compression ratio
        all_coeffs = np.concatenate([coeffs[0]] + coeffs[1:])
        threshold = np.percentile(np.abs(all_coeffs), 
                                 (1 - 1/compression_ratio) * 100)
        
        # Apply thresholding
        coeffs_thresh = list(coeffs)
        coeffs_thresh[0] = coeffs[0]  # Keep approximation coefficients
        
        for i in range(1, len(coeffs)):
            coeffs_thresh[i] = pywt.threshold(
                coeffs[i], threshold, mode=self.threshold_mode
            )
        
        return {
            'coefficients': coeffs_thresh,
            'wavelet': self.wavelet,
            'levels': self.levels,
            'threshold': threshold,
            'original_length': len(channel_data)
        }
    
    def decompress(self, compressed_data: Dict[str, Any]) -> np.ndarray:
        """Decompress wavelet-compressed data"""
        try:
            if 'channels' in compressed_data:
                # Multi-channel decompression
                channels = []
                for channel_data in compressed_data['channels']:
                    reconstructed = pywt.waverec(
                        channel_data['coefficients'], 
                        channel_data['wavelet']
                    )
                    channels.append(reconstructed)
                
                return np.array(channels)
            else:
                # Single channel decompression
                return pywt.waverec(
                    compressed_data['coefficients'], 
                    compressed_data['wavelet']
                )
                
        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            raise BrainForgeError(f"Wavelet decompression failed: {e}")
    
    def _estimate_compressed_size(self, compressed_data: Dict[str, Any]) -> int:
        """Estimate compressed data size in bytes"""
        # Simplified estimation based on non-zero coefficients
        total_coeffs = 0
        
        if 'channels' in compressed_data:
            for channel in compressed_data['channels']:
                for coeff_level in channel['coefficients']:
                    total_coeffs += np.count_nonzero(coeff_level)
        else:
            for coeff_level in compressed_data['coefficients']:
                total_coeffs += np.count_nonzero(coeff_level)
        
        return total_coeffs * 8  # 8 bytes per float64


class ArtifactRemover:
    """Advanced artifact removal using ICA and statistical methods"""
    
    def __init__(self, method: str = 'fastica', n_components: int = 20):
        self.method = method
        self.n_components = n_components
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize ICA
        if method == 'fastica':
            self.ica = FastICA(n_components=n_components, random_state=42)
        else:
            self.ica = None
            
        self.is_fitted = False
        self.artifact_components = []
    
    def fit_ica(self, data: np.ndarray) -> None:
        """Fit ICA to data for artifact identification"""
        try:
            if data.shape[0] < self.n_components:
                self.logger.warning(f"Reducing ICA components from {self.n_components} to {data.shape[0]}")
                self.n_components = data.shape[0]
                self.ica.n_components = self.n_components
            
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data.T).T
            
            # Fit ICA
            self.ica.fit(data_scaled.T)
            self.is_fitted = True
            
            self.logger.info(f"ICA fitted with {self.n_components} components")
            
        except Exception as e:
            self.logger.error(f"ICA fitting failed: {e}")
            raise BrainForgeError(f"Failed to fit ICA: {e}")
    
    def identify_artifacts(self, data: np.ndarray, 
                          threshold: float = 3.0) -> List[int]:
        """Identify artifact components automatically"""
        try:
            if not self.is_fitted:
                self.fit_ica(data)
            
            # Transform data to ICA space
            data_scaled = StandardScaler().fit_transform(data.T).T
            sources = self.ica.transform(data_scaled.T).T
            
            artifact_components = []
            
            for i, source in enumerate(sources):
                # Check for artifacts using multiple criteria
                
                # 1. High amplitude variance (muscle artifacts)
                amplitude_z = np.abs(zscore(np.abs(source)))
                high_amplitude_ratio = np.mean(amplitude_z > threshold)
                
                # 2. High frequency content (muscle/EMG artifacts)
                freqs, psd = signal.welch(source, fs=1000)
                high_freq_power = np.sum(psd[freqs > 30]) / np.sum(psd)
                
                # 3. Skewness (eye blink artifacts)
                skewness = abs(signal.find_peaks(np.abs(source), 
                                               height=threshold * np.std(source))[0].size)
                
                # Classify as artifact if multiple criteria are met
                artifact_score = (
                    (high_amplitude_ratio > 0.05) * 1 +
                    (high_freq_power > 0.3) * 1 +
                    (skewness > 5) * 1
                )
                
                if artifact_score >= 2:
                    artifact_components.append(i)
            
            self.artifact_components = artifact_components
            self.logger.info(f"Identified {len(artifact_components)} artifact components")
            
            return artifact_components
            
        except Exception as e:
            self.logger.error(f"Artifact identification failed: {e}")
            raise BrainForgeError(f"Failed to identify artifacts: {e}")
    
    def remove_artifacts(self, data: np.ndarray, 
                        component_indices: Optional[List[int]] = None) -> np.ndarray:
        """Remove specified artifact components"""
        try:
            if not self.is_fitted:
                raise BrainForgeError("ICA must be fitted before removing artifacts")
            
            if component_indices is None:
                component_indices = self.artifact_components
            
            # Transform to ICA space
            data_scaled = StandardScaler().fit_transform(data.T).T
            sources = self.ica.transform(data_scaled.T).T
            
            # Zero out artifact components
            sources_clean = sources.copy()
            for idx in component_indices:
                sources_clean[idx] = 0
            
            # Transform back to sensor space
            data_clean = self.ica.inverse_transform(sources_clean.T).T
            
            # Rescale back to original scale
            scaler = StandardScaler().fit(data.T)
            data_clean_scaled = scaler.inverse_transform(
                StandardScaler().fit_transform(data_clean.T)
            ).T
            
            self.logger.info(f"Removed {len(component_indices)} artifact components")
            
            return data_clean_scaled
            
        except Exception as e:
            self.logger.error(f"Artifact removal failed: {e}")
            raise BrainForgeError(f"Failed to remove artifacts: {e}")


class FeatureExtractor:
    """Extract comprehensive features from neural signals"""
    
    def __init__(self, sampling_rate: float = 1000.0):
        self.sampling_rate = sampling_rate
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Define frequency bands
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
    
    def extract_spectral_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract spectral power features"""
        try:
            features = {}
            
            # Compute power spectral density
            freqs, psd = signal.welch(data, fs=self.sampling_rate, axis=-1)
            
            # Extract power in each frequency band
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.mean(psd[..., band_mask], axis=-1)
                features[f'{band_name}_power'] = band_power
            
            # Total power
            features['total_power'] = np.mean(psd, axis=-1)
            
            # Relative power
            for band_name in self.frequency_bands.keys():
                features[f'{band_name}_relative'] = (
                    features[f'{band_name}_power'] / features['total_power']
                )
            
            # Peak frequency
            peak_indices = np.argmax(psd, axis=-1)
            features['peak_frequency'] = freqs[peak_indices]
            
            # Spectral centroid
            features['spectral_centroid'] = np.sum(
                freqs * psd, axis=-1
            ) / np.sum(psd, axis=-1)
            
            self.logger.info("Spectral features extracted")
            return features
            
        except Exception as e:
            self.logger.error(f"Spectral feature extraction failed: {e}")
            raise BrainForgeError(f"Failed to extract spectral features: {e}")
    
    def extract_temporal_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract temporal domain features"""
        try:
            features = {}
            
            # Basic statistics
            features['mean'] = np.mean(data, axis=-1)
            features['std'] = np.std(data, axis=-1)
            features['variance'] = np.var(data, axis=-1)
            features['skewness'] = signal.skew(data, axis=-1)
            features['kurtosis'] = signal.kurtosis(data, axis=-1)
            
            # Signal complexity measures
            features['zero_crossings'] = self._count_zero_crossings(data)
            features['line_length'] = self._compute_line_length(data)
            features['hjorth_activity'] = np.var(data, axis=-1)
            
            mobility, complexity = self._compute_hjorth_parameters(data)
            features['hjorth_mobility'] = mobility
            features['hjorth_complexity'] = complexity
            
            self.logger.info("Temporal features extracted")
            return features
            
        except Exception as e:
            self.logger.error(f"Temporal feature extraction failed: {e}")
            raise BrainForgeError(f"Failed to extract temporal features: {e}")
    
    def _count_zero_crossings(self, data: np.ndarray) -> np.ndarray:
        """Count zero crossings in signal"""
        zero_crossings = np.sum(np.diff(np.signbit(data), axis=-1), axis=-1)
        return zero_crossings
    
    def _compute_line_length(self, data: np.ndarray) -> np.ndarray:
        """Compute line length (sum of absolute differences)"""
        line_length = np.sum(np.abs(np.diff(data, axis=-1)), axis=-1)
        return line_length
    
    def _compute_hjorth_parameters(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Hjorth mobility and complexity parameters"""
        # First derivative
        diff1 = np.diff(data, axis=-1)
        
        # Second derivative
        diff2 = np.diff(diff1, axis=-1)
        
        # Variance of signal and derivatives
        var_data = np.var(data, axis=-1)
        var_diff1 = np.var(diff1, axis=-1)
        var_diff2 = np.var(diff2, axis=-1)
        
        # Mobility = sqrt(var(diff1) / var(data))
        mobility = np.sqrt(var_diff1 / var_data)
        
        # Complexity = Mobility(diff1) / Mobility(data)
        mobility_diff1 = np.sqrt(var_diff2 / var_diff1)
        complexity = mobility_diff1 / mobility
        
        return mobility, complexity


class RealTimeProcessor:
    """Main real-time processing pipeline coordinator"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize processing parameters
        self.params = ProcessingParameters()
        
        # Initialize processing components
        self._initialize_components()
        
        # Processing statistics
        self.processing_times = []
        self.data_quality_scores = []
        
    def _initialize_components(self):
        """Initialize all processing components"""
        try:
            # Filters
            self.bandpass_filter = RealTimeFilter(
                'bandpass', 
                (self.params.filter_low, self.params.filter_high),
                self.params.sampling_rate
            )
            
            self.notch_filter = RealTimeFilter(
                'notch',
                (self.params.notch_freq,),
                self.params.sampling_rate
            )
            
            # Compression
            self.compressor = WaveletCompressor(
                wavelet=self.params.wavelet_type
            )
            
            # Artifact removal
            self.artifact_remover = ArtifactRemover(
                n_components=self.params.ica_components
            )
            
            # Feature extraction
            self.feature_extractor = FeatureExtractor(
                sampling_rate=self.params.sampling_rate
            )
            
            self.logger.info("Real-time processor initialized")
            
        except Exception as e:
            self.logger.error(f"Processor initialization failed: {e}")
            raise BrainForgeError(f"Failed to initialize processor: {e}")
    
    async def process_data_chunk(self, data_chunk: np.ndarray) -> Dict[str, Any]:
        """Process a chunk of incoming data"""
        start_time = time.time()
        
        try:
            results = {
                'processed_data': None,
                'compressed_data': None,
                'features': {},
                'quality_score': 0.0,
                'processing_time': 0.0
            }
            
            # Step 1: Apply filters
            filtered_data = self.bandpass_filter.apply_filter(data_chunk)
            filtered_data = self.notch_filter.apply_filter(filtered_data)
            
            # Step 2: Artifact removal (if enabled)
            if hasattr(self, 'artifact_remover') and self.artifact_remover.is_fitted:
                clean_data = self.artifact_remover.remove_artifacts(filtered_data)
            else:
                clean_data = filtered_data
            
            # Step 3: Feature extraction
            spectral_features = self.feature_extractor.extract_spectral_features(clean_data)
            temporal_features = self.feature_extractor.extract_temporal_features(clean_data)
            
            results['features'].update(spectral_features)
            results['features'].update(temporal_features)
            
            # Step 4: Compression
            compressed = self.compressor.compress(
                clean_data, 
                compression_ratio=self.params.compression_ratio
            )
            
            # Step 5: Quality assessment
            quality_score = self._assess_data_quality(clean_data, results['features'])
            
            # Compile results
            results.update({
                'processed_data': clean_data,
                'compressed_data': compressed,
                'quality_score': quality_score,
                'processing_time': time.time() - start_time
            })
            
            # Update statistics
            self.processing_times.append(results['processing_time'])
            self.data_quality_scores.append(quality_score)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Data chunk processing failed: {e}")
            raise BrainForgeError(f"Failed to process data chunk: {e}")
    
    def _assess_data_quality(self, data: np.ndarray, features: Dict[str, np.ndarray]) -> float:
        """Assess overall data quality"""
        try:
            quality_factors = []
            
            # 1. Signal-to-noise ratio estimate
            signal_power = np.mean(features.get('total_power', 1.0))
            noise_estimate = np.mean(features.get('std', 1.0))
            snr = signal_power / (noise_estimate + 1e-10)
            quality_factors.append(min(snr / 10, 1.0))  # Normalize to [0, 1]
            
            # 2. Artifact level (based on high-frequency content)
            gamma_ratio = np.mean(features.get('gamma_relative', 0.1))
            artifact_score = 1.0 - min(gamma_ratio * 5, 1.0)  # Less gamma = better
            quality_factors.append(artifact_score)
            
            # 3. Signal stability (low variance in statistical measures)
            stability_score = 1.0 / (1.0 + np.mean(features.get('std', 1.0)))
            quality_factors.append(stability_score)
            
            # Overall quality score
            quality_score = np.mean(quality_factors)
            
            return float(quality_score)
            
        except Exception as e:
            self.logger.warning(f"Quality assessment failed: {e}")
            return 0.5  # Default medium quality
    
    def get_processing_stats(self) -> Dict[str, float]:
        """Get processing performance statistics"""
        if not self.processing_times:
            return {}
        
        return {
            'mean_processing_time': np.mean(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'mean_quality_score': np.mean(self.data_quality_scores),
            'min_quality_score': np.min(self.data_quality_scores),
            'chunks_processed': len(self.processing_times)
        }


# Example usage and testing
if __name__ == "__main__":
    async def test_processor():
        """Test the real-time processor"""
        try:
            # Initialize processor
            processor = RealTimeProcessor()
            
            # Simulate data chunks
            for i in range(10):
                # Generate test data (64 channels, 1 second at 1000 Hz)
                test_data = np.random.randn(64, 1000) + np.sin(
                    2 * np.pi * 10 * np.linspace(0, 1, 1000)
                )  # 10 Hz sine wave + noise
                
                # Process chunk
                results = await processor.process_data_chunk(test_data)
                
                print(f"Chunk {i+1}: Quality={results['quality_score']:.3f}, "
                      f"Time={results['processing_time']:.3f}s")
            
            # Print statistics
            stats = processor.get_processing_stats()
            print("\nProcessing Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value:.4f}")
                
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Run test
    asyncio.run(test_processor())
