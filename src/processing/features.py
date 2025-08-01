"""
Feature Extraction for Brain-Forge

This module implements comprehensive feature extraction techniques for
neural signal analysis including spectral, temporal, and connectivity features.
"""

from typing import Dict, Tuple

import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew

from ..core.exceptions import FeatureExtractionError
from ..core.logger import get_logger

logger = get_logger(__name__)


class SpectralFeatureExtractor:
    """Extract frequency domain features from neural signals"""

    def __init__(self, sampling_rate: float = 1000.0):
        self.sampling_rate = sampling_rate

        # Define standard frequency bands
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }

        logger.info(f"Initialized spectral extractor at {sampling_rate} Hz")

    def extract_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract comprehensive spectral features"""
        if data.size == 0:
            raise FeatureExtractionError("Cannot extract features from empty data")

        try:
            features = {}

            # Compute power spectral density
            freqs, psd = signal.welch(
                data, fs=self.sampling_rate, axis=-1,
                nperseg=min(256, data.shape[-1]//4)
            )

            # Band power features
            band_features = self._extract_band_power(freqs, psd)
            features.update(band_features)

            # Spectral shape features
            shape_features = self._extract_spectral_shape(freqs, psd)
            features.update(shape_features)

            # Connectivity features
            if data.ndim > 1 and data.shape[0] > 1:
                conn_features = self._extract_coherence_features(data)
                features.update(conn_features)

            logger.info(f"Extracted {len(features)} spectral features")
            return features

        except Exception as e:
            logger.error(f"Spectral feature extraction failed: {e}")
            raise FeatureExtractionError(f"Spectral extraction failed: {e}")

    def _extract_band_power(self, freqs: np.ndarray,
                           psd: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract power in each frequency band"""
        features = {}

        # Extract power in each frequency band
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                band_power = np.mean(psd[..., band_mask], axis=-1)
                features[f'{band_name}_power'] = band_power

        # Total power
        features['total_power'] = np.mean(psd, axis=-1)

        # Relative power
        for band_name in self.frequency_bands.keys():
            if f'{band_name}_power' in features:
                features[f'{band_name}_relative'] = (
                    features[f'{band_name}_power'] /
                    (features['total_power'] + 1e-10)
                )

        return features

    def _extract_spectral_shape(self, freqs: np.ndarray,
                               psd: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract spectral shape features"""
        features = {}

        try:
            # Peak frequency
            peak_indices = np.argmax(psd, axis=-1)
            features['peak_frequency'] = freqs[peak_indices]

            # Spectral centroid
            features['spectral_centroid'] = (
                np.sum(freqs * psd, axis=-1) /
                (np.sum(psd, axis=-1) + 1e-10)
            )

            # Spectral spread
            centroid = features['spectral_centroid']
            if psd.ndim > 1:
                centroid = centroid[..., np.newaxis]
            spread = np.sum(((freqs - centroid) ** 2) * psd, axis=-1)
            features['spectral_spread'] = np.sqrt(
                spread / (np.sum(psd, axis=-1) + 1e-10)
            )

            # Spectral rolloff (95% of energy)
            cumsum_psd = np.cumsum(psd, axis=-1)
            total_energy = cumsum_psd[..., -1:]
            rolloff_threshold = 0.95 * total_energy
            rolloff_indices = np.argmax(
                cumsum_psd >= rolloff_threshold, axis=-1
            )
            features['spectral_rolloff'] = freqs[rolloff_indices]

        except Exception as e:
            logger.warning(f"Some spectral shape features failed: {e}")

        return features

    def _extract_coherence_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract coherence-based connectivity features"""
        features = {}

        try:
            n_channels = data.shape[0]
            if n_channels < 2:
                return features

            coherence_matrix = np.zeros((n_channels, n_channels))

            # Calculate coherence between all channel pairs
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    freqs, coh = signal.coherence(
                        data[i], data[j], fs=self.sampling_rate
                    )
                    mean_coherence = np.mean(coh)
                    coherence_matrix[i, j] = mean_coherence
                    coherence_matrix[j, i] = mean_coherence

            # Summary statistics
            features['mean_coherence'] = np.mean(coherence_matrix)
            features['max_coherence'] = np.max(coherence_matrix)
            features['coherence_std'] = np.std(coherence_matrix)

        except Exception as e:
            logger.warning(f"Coherence feature extraction failed: {e}")

        return features


class TemporalFeatureExtractor:
    """Extract time domain features from neural signals"""

    def __init__(self):
        logger.info("Initialized temporal feature extractor")

    def extract_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract comprehensive temporal features"""
        if data.size == 0:
            raise FeatureExtractionError("Cannot extract features from empty data")

        try:
            features = {}

            # Basic statistical features
            stat_features = self._extract_statistical_features(data)
            features.update(stat_features)

            # Signal complexity features
            complexity_features = self._extract_complexity_features(data)
            features.update(complexity_features)

            # Morphological features
            morph_features = self._extract_morphological_features(data)
            features.update(morph_features)

            logger.info(f"Extracted {len(features)} temporal features")
            return features

        except Exception as e:
            logger.error(f"Temporal feature extraction failed: {e}")
            raise FeatureExtractionError(f"Temporal extraction failed: {e}")

    def _extract_statistical_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract basic statistical features"""
        features = {}

        try:
            features['mean'] = np.mean(data, axis=-1)
            features['std'] = np.std(data, axis=-1)
            features['variance'] = np.var(data, axis=-1)
            features['skewness'] = skew(data, axis=-1, nan_policy='omit')
            features['kurtosis'] = kurtosis(data, axis=-1, nan_policy='omit')
            features['min'] = np.min(data, axis=-1)
            features['max'] = np.max(data, axis=-1)
            features['range'] = features['max'] - features['min']
            features['rms'] = np.sqrt(np.mean(data**2, axis=-1))

        except Exception as e:
            logger.warning(f"Statistical feature extraction failed: {e}")

        return features

    def _extract_complexity_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract signal complexity features"""
        features = {}

        try:
            # Zero crossings
            features['zero_crossings'] = self._count_zero_crossings(data)

            # Line length
            features['line_length'] = self._compute_line_length(data)

            # Hjorth parameters
            hjorth_features = self._compute_hjorth_parameters(data)
            features.update(hjorth_features)

            # Signal energy
            features['energy'] = np.sum(data**2, axis=-1)

        except Exception as e:
            logger.warning(f"Complexity feature extraction failed: {e}")

        return features

    def _extract_morphological_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract signal morphology features"""
        features = {}

        try:
            # Peak detection
            if data.ndim == 1:
                peaks, _ = signal.find_peaks(data)
                features['peak_count'] = len(peaks)
            else:
                peak_counts = []
                for channel in data:
                    peaks, _ = signal.find_peaks(channel)
                    peak_counts.append(len(peaks))
                features['peak_count'] = np.array(peak_counts)

            # Slope features
            diff_data = np.diff(data, axis=-1)
            features['mean_slope'] = np.mean(diff_data, axis=-1)
            features['max_slope'] = np.max(np.abs(diff_data), axis=-1)

        except Exception as e:
            logger.warning(f"Morphological feature extraction failed: {e}")

        return features

    def _count_zero_crossings(self, data: np.ndarray) -> np.ndarray:
        """Count zero crossings in signal"""
        try:
            zero_crossings = np.sum(np.diff(np.signbit(data), axis=-1), axis=-1)
            return zero_crossings
        except:
            return np.zeros(data.shape[:-1])

    def _compute_line_length(self, data: np.ndarray) -> np.ndarray:
        """Compute line length (sum of absolute differences)"""
        try:
            line_length = np.sum(np.abs(np.diff(data, axis=-1)), axis=-1)
            return line_length
        except:
            return np.zeros(data.shape[:-1])

    def _compute_hjorth_parameters(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute Hjorth mobility and complexity parameters"""
        features = {}

        try:
            # First derivative
            diff1 = np.diff(data, axis=-1)

            # Second derivative
            diff2 = np.diff(diff1, axis=-1)

            # Variance of signal and derivatives
            var_data = np.var(data, axis=-1)
            var_diff1 = np.var(diff1, axis=-1)
            var_diff2 = np.var(diff2, axis=-1)

            # Mobility = sqrt(var(diff1) / var(data))
            mobility = np.sqrt(var_diff1 / (var_data + 1e-10))

            # Complexity = Mobility(diff1) / Mobility(data)
            mobility_diff1 = np.sqrt(var_diff2 / (var_diff1 + 1e-10))
            complexity = mobility_diff1 / (mobility + 1e-10)

            features['hjorth_activity'] = var_data
            features['hjorth_mobility'] = mobility
            features['hjorth_complexity'] = complexity

        except Exception as e:
            logger.warning(f"Hjorth parameter computation failed: {e}")
            features['hjorth_activity'] = np.zeros(data.shape[:-1])
            features['hjorth_mobility'] = np.zeros(data.shape[:-1])
            features['hjorth_complexity'] = np.zeros(data.shape[:-1])

        return features


class ComprehensiveFeatureExtractor:
    """Combined spectral and temporal feature extraction"""

    def __init__(self, sampling_rate: float = 1000.0):
        self.spectral_extractor = SpectralFeatureExtractor(sampling_rate)
        self.temporal_extractor = TemporalFeatureExtractor()

    def extract_all_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract both spectral and temporal features"""
        try:
            # Extract spectral features
            spectral_features = self.spectral_extractor.extract_features(data)

            # Extract temporal features
            temporal_features = self.temporal_extractor.extract_features(data)

            # Combine features
            all_features = {**spectral_features, **temporal_features}

            logger.info(f"Extracted {len(all_features)} total features")
            return all_features

        except Exception as e:
            logger.error(f"Comprehensive feature extraction failed: {e}")
            raise FeatureExtractionError(f"Feature extraction failed: {e}")


def create_feature_extractor(feature_type: str = 'comprehensive',
                           sampling_rate: float = 1000.0) -> object:
    """Factory function for feature extractor creation"""
    if feature_type == 'spectral':
        return SpectralFeatureExtractor(sampling_rate)
    elif feature_type == 'temporal':
        return TemporalFeatureExtractor()
    elif feature_type == 'comprehensive':
        return ComprehensiveFeatureExtractor(sampling_rate)
    else:
        raise FeatureExtractionError(f"Unknown feature type: {feature_type}")
