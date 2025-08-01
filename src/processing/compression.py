"""
Wavelet-based Data Compression for Brain-Forge

This module implements efficient wavelet compression algorithms for neural
signal data to reduce storage and transmission requirements.
"""

from typing import Any, Dict, List

import numpy as np
import pywt

from ..core.exceptions import CompressionError
from ..core.logger import get_logger

logger = get_logger(__name__)


class WaveletCompressor:
    """Wavelet-based neural signal compression"""

    def __init__(self, wavelet: str = 'db8', levels: int = 6,
                 threshold_mode: str = 'soft'):
        self.wavelet = wavelet
        self.levels = levels
        self.threshold_mode = threshold_mode

        try:
            # Validate wavelet selection
            if wavelet not in pywt.wavelist():
                raise ValueError(f"Invalid wavelet: {wavelet}")

            logger.info(f"Initialized wavelet compressor with {wavelet}, "
                       f"{levels} levels")
        except Exception as e:
            raise CompressionError(f"Failed to initialize compressor: {e}")

    def compress(self, data: np.ndarray,
                 compression_ratio: float = 5.0) -> Dict[str, Any]:
        """Compress neural data using wavelet transform"""
        if data.size == 0:
            raise CompressionError("Cannot compress empty data")

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
            actual_ratio = original_size / compressed_size if compressed_size > 0 else 0

            compressed_data['compression_ratio'] = actual_ratio
            compressed_data['original_size'] = original_size
            compressed_data['compressed_size'] = compressed_size

            logger.info(f"Compression complete: {actual_ratio:.2f}x ratio")
            return compressed_data

        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise CompressionError(f"Wavelet compression failed: {e}")

    def _compress_channel(self, channel_data: np.ndarray,
                         compression_ratio: float) -> Dict[str, Any]:
        """Compress single channel using wavelets"""
        try:
            # Wavelet decomposition
            coeffs = pywt.wavedec(channel_data, self.wavelet,
                                 level=self.levels)

            # Calculate threshold for desired compression ratio
            all_coeffs = np.concatenate([coeffs[0]] + coeffs[1:])
            threshold = np.percentile(
                np.abs(all_coeffs),
                (1 - 1/compression_ratio) * 100
            )

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

        except Exception as e:
            raise CompressionError(f"Channel compression failed: {e}")

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
            logger.error(f"Decompression failed: {e}")
            raise CompressionError(f"Wavelet decompression failed: {e}")

    def _estimate_compressed_size(self, compressed_data: Dict[str, Any]) -> int:
        """Estimate compressed data size in bytes"""
        total_coeffs = 0

        try:
            if 'channels' in compressed_data:
                for channel in compressed_data['channels']:
                    for coeff_level in channel['coefficients']:
                        total_coeffs += np.count_nonzero(coeff_level)
            else:
                for coeff_level in compressed_data['coefficients']:
                    total_coeffs += np.count_nonzero(coeff_level)

            return total_coeffs * 8  # 8 bytes per float64

        except Exception as e:
            logger.error(f"Size estimation failed: {e}")
            return 0

    def get_compression_metrics(self, original_data: np.ndarray,
                              compressed_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate detailed compression metrics"""
        try:
            original_size = original_data.nbytes
            compressed_size = compressed_data.get('compressed_size', 0)

            if compressed_size == 0:
                return {'error': 'Invalid compressed data'}

            compression_ratio = original_size / compressed_size
            space_savings = (1 - compressed_size / original_size) * 100

            # Calculate reconstruction quality (if possible)
            try:
                reconstructed = self.decompress(compressed_data)
                mse = np.mean((original_data - reconstructed) ** 2)
                snr = 10 * np.log10(np.var(original_data) / mse) if mse > 0 else float('inf')
            except:
                mse = float('nan')
                snr = float('nan')

            return {
                'compression_ratio': compression_ratio,
                'space_savings_percent': space_savings,
                'original_size_bytes': original_size,
                'compressed_size_bytes': compressed_size,
                'reconstruction_mse': mse,
                'reconstruction_snr_db': snr
            }

        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return {'error': str(e)}


class CompressionManager:
    """Manages multiple compression algorithms and strategies"""

    def __init__(self):
        self.compressors = {
            'wavelet': WaveletCompressor(),
            'wavelet_coif': WaveletCompressor(wavelet='coif4'),
            'wavelet_bior': WaveletCompressor(wavelet='bior4.4')
        }

    def compress_adaptive(self, data: np.ndarray,
                         target_ratio: float = 5.0) -> Dict[str, Any]:
        """Adaptively choose best compression method"""
        best_result = None
        best_ratio = 0
        best_method = None

        try:
            for method, compressor in self.compressors.items():
                try:
                    result = compressor.compress(data, target_ratio)
                    actual_ratio = result.get('compression_ratio', 0)

                    if actual_ratio > best_ratio:
                        best_ratio = actual_ratio
                        best_result = result
                        best_method = method

                except Exception as e:
                    logger.warning(f"Compression method {method} failed: {e}")
                    continue

            if best_result is None:
                raise CompressionError("All compression methods failed")

            best_result['compression_method'] = best_method
            logger.info(f"Best compression: {best_method} "
                       f"({best_ratio:.2f}x ratio)")

            return best_result

        except Exception as e:
            logger.error(f"Adaptive compression failed: {e}")
            raise CompressionError(f"Adaptive compression failed: {e}")

    def decompress_adaptive(self, compressed_data: Dict[str, Any]) -> np.ndarray:
        """Decompress using the original compression method"""
        try:
            method = compressed_data.get('compression_method', 'wavelet')

            if method not in self.compressors:
                raise CompressionError(f"Unknown compression method: {method}")

            compressor = self.compressors[method]
            return compressor.decompress(compressed_data)

        except Exception as e:
            logger.error(f"Adaptive decompression failed: {e}")
            raise CompressionError(f"Adaptive decompression failed: {e}")


def create_compressor(compression_type: str = 'wavelet',
                     **kwargs) -> WaveletCompressor:
    """Factory function to create compression instances"""
    if compression_type == 'wavelet':
        return WaveletCompressor(**kwargs)
    else:
        raise CompressionError(f"Unknown compression type: {compression_type}")
