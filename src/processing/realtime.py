"""
Real-time Processing Pipeline for Brain-Forge

This module orchestrates the complete signal processing pipeline including
filtering, compression, artifact removal, and feature extraction.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import numpy as np

from ..core.config import Config
from ..core.exceptions import ProcessingError
from ..core.logger import get_logger
from .artifacts import HybridArtifactRemover, create_artifact_remover
from .compression import CompressionManager, WaveletCompressor
from .features import ComprehensiveFeatureExtractor, create_feature_extractor
from .filters import FilterBank, RealTimeFilter, create_standard_filter_bank

logger = get_logger(__name__)


class RealTimeProcessor:
    """Orchestrates real-time neural signal processing pipeline"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.processing_stats = {
            'chunks_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'quality_scores': []
        }

        try:
            self._initialize_components()
            logger.info("Real-time processor initialized successfully")
        except Exception as e:
            raise ProcessingError(f"Failed to initialize processor: {e}")

    def _initialize_components(self):
        """Initialize all processing components"""
        try:
            # Initialize filter bank
            sampling_rate = self.config.processing.sampling_rate
            self.filter_bank = create_standard_filter_bank(sampling_rate)

            # Initialize main bandpass filter
            self.main_filter = RealTimeFilter(
                'bandpass',
                (self.config.processing.filter_low,
                 self.config.processing.filter_high),
                sampling_rate
            )

            # Initialize compressor
            self.compressor = WaveletCompressor(
                wavelet=self.config.processing.wavelet_type
            )
            self.compression_manager = CompressionManager()

            # Initialize artifact remover
            self.artifact_remover = create_artifact_remover(
                method='hybrid',
                ica_components=self.config.processing.ica_components
            )

            # Initialize feature extractor
            self.feature_extractor = create_feature_extractor(
                feature_type='comprehensive',
                sampling_rate=sampling_rate
            )

            # Thread pool for parallel processing
            self.executor = ThreadPoolExecutor(max_workers=4)

            logger.info("All processing components initialized")

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise ProcessingError(f"Component initialization failed: {e}")

    async def process_data_chunk(self, data_chunk: np.ndarray) -> Dict[str, Any]:
        """Process a chunk of real-time data"""
        start_time = time.time()

        try:
            if data_chunk.size == 0:
                raise ProcessingError("Cannot process empty data chunk")

            result = {
                'timestamp': start_time,
                'original_data': data_chunk.copy(),
                'processing_pipeline': []
            }

            # Step 1: Apply main bandpass filter
            filtered_data = self.main_filter.apply_filter(data_chunk)
            result['filtered_data'] = filtered_data
            result['processing_pipeline'].append('bandpass_filtering')

            # Step 2: Artifact removal (if enough channels)
            if filtered_data.shape[0] >= 4:
                try:
                    artifact_result = self.artifact_remover.clean_data(
                        filtered_data
                    )
                    clean_data = artifact_result['cleaned_data']
                    result['cleaned_data'] = clean_data
                    result['bad_channels'] = artifact_result.get('bad_channels', [])
                    result['processing_pipeline'].append('artifact_removal')
                except Exception as e:
                    logger.warning(f"Artifact removal failed: {e}")
                    clean_data = filtered_data
                    result['cleaned_data'] = clean_data
            else:
                clean_data = filtered_data
                result['cleaned_data'] = clean_data

            # Step 3: Feature extraction
            try:
                features = self.feature_extractor.extract_all_features(clean_data)
                result['features'] = features
                result['processing_pipeline'].append('feature_extraction')
            except Exception as e:
                logger.warning(f"Feature extraction failed: {e}")
                result['features'] = {}

            # Step 4: Frequency band analysis
            try:
                band_data = self.filter_bank.apply_filters(clean_data)
                result['frequency_bands'] = band_data
                result['processing_pipeline'].append('frequency_analysis')
            except Exception as e:
                logger.warning(f"Frequency analysis failed: {e}")
                result['frequency_bands'] = {}

            # Step 5: Data quality assessment
            quality_score = self._assess_data_quality(clean_data, result.get('features', {}))
            result['quality_score'] = quality_score
            result['processing_pipeline'].append('quality_assessment')

            # Step 6: Compression (optional)
            if self.config.processing.enable_compression:
                try:
                    compressed = self.compression_manager.compress_adaptive(
                        clean_data,
                        target_ratio=self.config.processing.compression_ratio
                    )
                    result['compressed_data'] = compressed
                    result['processing_pipeline'].append('compression')
                except Exception as e:
                    logger.warning(f"Compression failed: {e}")

            # Update processing statistics
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time, quality_score)

            result['processing_time'] = processing_time
            result['processing_pipeline'].append('complete')

            logger.debug(f"Data chunk processed in {processing_time:.3f}s")
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Data processing failed after {processing_time:.3f}s: {e}")
            raise ProcessingError(f"Data chunk processing failed: {e}")

    def _assess_data_quality(self, data: np.ndarray,
                           features: Dict[str, np.ndarray]) -> float:
        """Assess overall data quality score (0-1)"""
        try:
            quality_factors = []

            # Factor 1: Signal amplitude (not too low, not too high)
            rms = np.sqrt(np.mean(data**2))
            amplitude_score = 1.0 - min(1.0, abs(np.log10(rms) - np.log10(50e-6)) / 2)
            quality_factors.append(amplitude_score)

            # Factor 2: Frequency content
            if 'alpha_power' in features and 'total_power' in features:
                alpha_ratio = np.mean(features['alpha_power'] /
                                    (features['total_power'] + 1e-10))
                alpha_score = min(1.0, alpha_ratio * 5)  # Scale alpha presence
                quality_factors.append(alpha_score)

            # Factor 3: Signal stability (low variance in statistics)
            if data.shape[-1] > 100:  # Need enough samples
                windowed_means = [
                    np.mean(data[..., i:i+50])
                    for i in range(0, data.shape[-1]-50, 25)
                ]
                stability_score = 1.0 - min(1.0, np.std(windowed_means) /
                                          (np.mean(np.abs(windowed_means)) + 1e-10))
                quality_factors.append(stability_score)

            # Factor 4: No excessive artifacts
            if 'line_length' in features:
                line_length = np.mean(features['line_length'])
                artifact_score = 1.0 / (1.0 + line_length / 1000.0)  # Sigmoid
                quality_factors.append(artifact_score)

            # Overall quality score
            overall_quality = np.mean(quality_factors) if quality_factors else 0.5

            return float(np.clip(overall_quality, 0.0, 1.0))

        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5

    def _update_processing_stats(self, processing_time: float, quality_score: float):
        """Update processing performance statistics"""
        try:
            self.processing_stats['chunks_processed'] += 1
            self.processing_stats['total_processing_time'] += processing_time
            self.processing_stats['average_processing_time'] = (
                self.processing_stats['total_processing_time'] /
                self.processing_stats['chunks_processed']
            )
            self.processing_stats['quality_scores'].append(quality_score)

            # Keep only last 100 quality scores
            if len(self.processing_stats['quality_scores']) > 100:
                self.processing_stats['quality_scores'] = (
                    self.processing_stats['quality_scores'][-100:]
                )

        except Exception as e:
            logger.warning(f"Statistics update failed: {e}")

    def get_processing_stats(self) -> Dict[str, float]:
        """Get current processing performance statistics"""
        try:
            stats = self.processing_stats.copy()

            if stats['quality_scores']:
                stats['average_quality'] = np.mean(stats['quality_scores'])
                stats['quality_std'] = np.std(stats['quality_scores'])
            else:
                stats['average_quality'] = 0.0
                stats['quality_std'] = 0.0

            return stats

        except Exception as e:
            logger.error(f"Statistics retrieval failed: {e}")
            return {'error': str(e)}

    def reset_filters(self):
        """Reset all filter states"""
        try:
            self.main_filter.reset()
            self.filter_bank.reset_all()
            logger.info("All filters reset")
        except Exception as e:
            logger.error(f"Filter reset failed: {e}")
            raise ProcessingError(f"Filter reset failed: {e}")

    def shutdown(self):
        """Clean shutdown of processing components"""
        try:
            self.executor.shutdown(wait=True)
            logger.info("Processor shutdown complete")
        except Exception as e:
            logger.error(f"Processor shutdown failed: {e}")


async def test_processor():
    """Test the real-time processor with synthetic data"""
    try:
        # Create processor
        processor = RealTimeProcessor()

        # Generate test data
        sampling_rate = 1000
        duration = 1.0  # seconds
        n_channels = 8
        n_samples = int(sampling_rate * duration)

        # Simulate neural data
        t = np.linspace(0, duration, n_samples)
        data = np.zeros((n_channels, n_samples))

        for ch in range(n_channels):
            # Alpha rhythm (10 Hz)
            data[ch] += 2e-5 * np.sin(2 * np.pi * 10 * t)
            # Beta activity (20 Hz)
            data[ch] += 1e-5 * np.sin(2 * np.pi * 20 * t)
            # Noise
            data[ch] += 0.5e-5 * np.random.randn(n_samples)

        # Process data
        result = await processor.process_data_chunk(data)

        logger.info("Processor test completed successfully")
        logger.info(f"Processing pipeline: {result['processing_pipeline']}")
        logger.info(f"Quality score: {result['quality_score']:.3f}")
        logger.info(f"Processing time: {result['processing_time']:.3f}s")

        # Print statistics
        stats = processor.get_processing_stats()
        logger.info(f"Processing stats: {stats}")

        processor.shutdown()

        return True

    except Exception as e:
        logger.error(f"Processor test failed: {e}")
        return False


if __name__ == "__main__":
    # Run test
    asyncio.run(test_processor())
