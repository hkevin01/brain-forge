"""
Artifact Removal for Brain-Forge

This module implements advanced artifact removal techniques using ICA
and statistical methods for cleaning neural signal data.
"""

from typing import Dict, List, Optional

import numpy as np
from scipy import signal
from scipy.stats import zscore
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

from ..core.exceptions import ArtifactRemovalError
from ..core.logger import get_logger

logger = get_logger(__name__)


class ICArtifactRemover:
    """ICA-based artifact removal system"""

    def __init__(self, method: str = 'fastica', n_components: int = 20,
                 max_iter: int = 200):
        self.method = method
        self.n_components = n_components
        self.max_iter = max_iter
        self.is_fitted = False
        self.artifact_components = []
        self.scaler = StandardScaler()

        try:
            # Initialize ICA
            if method == 'fastica':
                self.ica = FastICA(
                    n_components=n_components,
                    max_iter=max_iter,
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported ICA method: {method}")

            logger.info(f"Initialized ICA artifact remover with {method}")

        except Exception as e:
            raise ArtifactRemovalError(f"Failed to initialize ICA: {e}")

    def fit(self, data: np.ndarray) -> None:
        """Fit ICA to data for artifact identification"""
        if data.size == 0:
            raise ArtifactRemovalError("Cannot fit ICA on empty data")

        try:
            # Adjust components if necessary
            if data.shape[0] < self.n_components:
                logger.warning(f"Reducing ICA components from "
                             f"{self.n_components} to {data.shape[0]}")
                self.n_components = data.shape[0]
                self.ica.n_components = self.n_components

            # Standardize data
            data_scaled = self.scaler.fit_transform(data.T).T

            # Fit ICA
            self.ica.fit(data_scaled.T)
            self.is_fitted = True

            logger.info(f"ICA fitted with {self.n_components} components")

        except Exception as e:
            logger.error(f"ICA fitting failed: {e}")
            raise ArtifactRemovalError(f"Failed to fit ICA: {e}")

    def identify_artifacts(self, data: np.ndarray,
                          threshold: float = 3.0) -> List[int]:
        """Automatically identify artifact components"""
        try:
            if not self.is_fitted:
                self.fit(data)

            # Transform data to ICA space
            data_scaled = self.scaler.transform(data.T).T
            sources = self.ica.transform(data_scaled.T).T

            artifact_components = []

            for i, source in enumerate(sources):
                artifact_score = self._calculate_artifact_score(
                    source, threshold
                )

                # Component is artifact if score >= 2
                if artifact_score >= 2:
                    artifact_components.append(i)

            self.artifact_components = artifact_components
            logger.info(f"Identified {len(artifact_components)} "
                       f"artifact components")

            return artifact_components

        except Exception as e:
            logger.error(f"Artifact identification failed: {e}")
            raise ArtifactRemovalError(f"Failed to identify artifacts: {e}")

    def _calculate_artifact_score(self, source: np.ndarray,
                                 threshold: float) -> int:
        """Calculate artifact likelihood score for a component"""
        score = 0

        try:
            # 1. High amplitude variance (muscle artifacts)
            amplitude_z = np.abs(zscore(np.abs(source)))
            high_amplitude_ratio = np.mean(amplitude_z > threshold)
            if high_amplitude_ratio > 0.05:
                score += 1

            # 2. High frequency content (muscle/EMG artifacts)
            try:
                freqs, psd = signal.welch(source, fs=1000, nperseg=min(256, len(source)//4))
                if len(psd) > 0:
                    high_freq_power = (np.sum(psd[freqs > 30]) /
                                     (np.sum(psd) + 1e-10))
                    if high_freq_power > 0.3:
                        score += 1
            except:
                pass  # Skip if frequency analysis fails

            # 3. Spike detection (eye blink artifacts)
            try:
                peaks, _ = signal.find_peaks(
                    np.abs(source),
                    height=threshold * np.std(source)
                )
                if len(peaks) > 5:
                    score += 1
            except:
                pass  # Skip if peak detection fails

            return score

        except Exception as e:
            logger.warning(f"Artifact score calculation failed: {e}")
            return 0

    def remove_artifacts(self, data: np.ndarray,
                        component_indices: Optional[List[int]] = None) -> np.ndarray:
        """Remove specified artifact components"""
        try:
            if not self.is_fitted:
                raise ArtifactRemovalError("ICA must be fitted before "
                                         "removing artifacts")

            if component_indices is None:
                component_indices = self.artifact_components

            if not component_indices:
                logger.info("No artifact components to remove")
                return data.copy()

            # Transform to ICA space
            data_scaled = self.scaler.transform(data.T).T
            sources = self.ica.transform(data_scaled.T).T

            # Zero out artifact components
            sources_clean = sources.copy()
            for idx in component_indices:
                if 0 <= idx < len(sources_clean):
                    sources_clean[idx] = 0

            # Transform back to sensor space
            data_clean_scaled = self.ica.inverse_transform(sources_clean.T).T

            # Scale back to original range
            data_clean = self.scaler.inverse_transform(data_clean_scaled.T).T

            logger.info(f"Removed {len(component_indices)} "
                       f"artifact components")

            return data_clean

        except Exception as e:
            logger.error(f"Artifact removal failed: {e}")
            raise ArtifactRemovalError(f"Failed to remove artifacts: {e}")

    def get_component_info(self) -> Dict:
        """Get information about ICA components"""
        if not self.is_fitted:
            return {'error': 'ICA not fitted'}

        return {
            'n_components': self.n_components,
            'artifact_components': self.artifact_components,
            'clean_components': [i for i in range(self.n_components)
                               if i not in self.artifact_components],
            'is_fitted': self.is_fitted
        }


class StatisticalArtifactRemover:
    """Statistical methods for artifact detection and removal"""

    def __init__(self, threshold_std: float = 3.0):
        self.threshold_std = threshold_std

    def remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """Remove statistical outliers using z-score thresholding"""
        try:
            if data.size == 0:
                return data

            # Calculate z-scores
            z_scores = np.abs(zscore(data, axis=1, nan_policy='omit'))

            # Replace outliers with interpolated values
            data_clean = data.copy()

            for ch in range(data.shape[0]):
                outlier_mask = z_scores[ch] > self.threshold_std
                if np.any(outlier_mask):
                    # Linear interpolation for outliers
                    valid_indices = np.where(~outlier_mask)[0]
                    outlier_indices = np.where(outlier_mask)[0]

                    if len(valid_indices) > 1:
                        interpolated = np.interp(
                            outlier_indices,
                            valid_indices,
                            data_clean[ch, valid_indices]
                        )
                        data_clean[ch, outlier_indices] = interpolated

            logger.info(f"Removed outliers with threshold {self.threshold_std}")
            return data_clean

        except Exception as e:
            logger.error(f"Outlier removal failed: {e}")
            raise ArtifactRemovalError(f"Failed to remove outliers: {e}")

    def detect_bad_channels(self, data: np.ndarray) -> List[int]:
        """Detect bad channels using statistical criteria"""
        try:
            bad_channels = []

            for ch in range(data.shape[0]):
                channel_data = data[ch]

                # Criteria for bad channels
                is_flat = np.std(channel_data) < 1e-6
                is_noisy = np.std(channel_data) > 10 * np.median([
                    np.std(data[i]) for i in range(data.shape[0])
                ])
                has_nans = np.any(np.isnan(channel_data))
                has_infs = np.any(np.isinf(channel_data))

                if is_flat or is_noisy or has_nans or has_infs:
                    bad_channels.append(ch)

            logger.info(f"Detected {len(bad_channels)} bad channels")
            return bad_channels

        except Exception as e:
            logger.error(f"Bad channel detection failed: {e}")
            return []


class HybridArtifactRemover:
    """Combined ICA and statistical artifact removal"""

    def __init__(self, ica_components: int = 20, stat_threshold: float = 3.0):
        self.ica_remover = ICArtifactRemover(n_components=ica_components)
        self.stat_remover = StatisticalArtifactRemover(
            threshold_std=stat_threshold
        )

    def clean_data(self, data: np.ndarray) -> Dict:
        """Apply comprehensive artifact cleaning"""
        try:
            result = {
                'original_data': data.copy(),
                'processing_steps': []
            }

            # Step 1: Remove statistical outliers
            data_clean = self.stat_remover.remove_outliers(data)
            result['processing_steps'].append('statistical_outlier_removal')

            # Step 2: Detect bad channels
            bad_channels = self.stat_remover.detect_bad_channels(data_clean)
            result['bad_channels'] = bad_channels

            # Step 3: ICA artifact removal
            if data_clean.shape[0] >= 4:  # Need minimum channels for ICA
                artifact_components = self.ica_remover.identify_artifacts(
                    data_clean
                )
                data_clean = self.ica_remover.remove_artifacts(
                    data_clean, artifact_components
                )
                result['processing_steps'].append('ica_artifact_removal')
                result['artifact_components'] = artifact_components

            result['cleaned_data'] = data_clean
            result['processing_steps'].append('cleaning_complete')

            logger.info(f"Hybrid cleaning complete with "
                       f"{len(result['processing_steps'])} steps")

            return result

        except Exception as e:
            logger.error(f"Hybrid artifact removal failed: {e}")
            raise ArtifactRemovalError(f"Hybrid cleaning failed: {e}")


def create_artifact_remover(method: str = 'hybrid',
                           **kwargs) -> object:
    """Factory function for artifact remover creation"""
    if method == 'ica':
        return ICArtifactRemover(**kwargs)
    elif method == 'statistical':
        return StatisticalArtifactRemover(**kwargs)
    elif method == 'hybrid':
        return HybridArtifactRemover(**kwargs)
    else:
        raise ArtifactRemovalError(f"Unknown artifact removal method: {method}")
