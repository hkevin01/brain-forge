"""
Real-time Digital Filtering for Brain-Forge

This module implements various digital filters for neural signal processing,
including bandpass, lowpass, highpass, and notch filters.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import signal

from ..core.exceptions import FilteringError
from ..core.logger import get_logger

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
        self.history = None
        self.zi = None

        try:
            self._design_filter()
            logger.info(f"Initialized {filter_type} filter with "
                       f"frequencies {frequencies}")
        except Exception as e:
            raise FilteringError(f"Failed to initialize filter: {e}")

    def _design_filter(self):
        """Design the digital filter based on specifications"""
        try:
            nyquist = self.sampling_rate / 2

            if self.filter_type == 'bandpass':
                if len(self.frequencies) != 2:
                    raise ValueError("Bandpass filter requires exactly "
                                   "2 frequencies")
                low, high = self.frequencies
                self.sos = signal.butter(
                    self.order, [low/nyquist, high/nyquist],
                    btype='band', output='sos')
            elif self.filter_type == 'lowpass':
                if len(self.frequencies) != 1:
                    raise ValueError("Lowpass filter requires exactly "
                                   "1 frequency")
                cutoff = self.frequencies[0]
                self.sos = signal.butter(
                    self.order, cutoff/nyquist,
                    btype='low', output='sos')
            elif self.filter_type == 'highpass':
                if len(self.frequencies) != 1:
                    raise ValueError("Highpass filter requires exactly "
                                   "1 frequency")
                cutoff = self.frequencies[0]
                self.sos = signal.butter(
                    self.order, cutoff/nyquist,
                    btype='high', output='sos')
            elif self.filter_type == 'notch':
                if len(self.frequencies) != 1:
                    raise ValueError("Notch filter requires exactly "
                                   "1 frequency")
                freq = self.frequencies[0]
                Q = 30.0  # Quality factor
                self.sos = signal.iirnotch(freq, Q, self.sampling_rate)
                self.sos = np.array([self.sos])  # Convert to SOS format
            else:
                raise ValueError(f"Unsupported filter type: "
                               f"{self.filter_type}")

            # Initialize filter state
            self.zi = signal.sosfilt_zi(self.sos)

        except Exception as e:
            logger.error(f"Filter design failed: {e}")
            raise FilteringError(f"Filter design failed: {e}")

    def apply_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply the filter to input data with state preservation"""
        if data.size == 0:
            return data

        try:
            # Handle multi-channel data
            if data.ndim == 1:
                # Single channel
                if self.zi is None:
                    # First call - initialize state
                    filtered_data, self.zi = signal.sosfilt(self.sos, data, zi=self.zi * data[0])
                else:
                    # Subsequent calls - use previous state
                    filtered_data, self.zi = signal.sosfilt(self.sos, data, zi=self.zi)
            else:
                # Multi-channel data
                n_channels = data.shape[0]
                if self.zi is None or self.zi.shape[1] != n_channels:
                    # Initialize state for all channels
                    self.zi = np.tile(signal.sosfilt_zi(self.sos)[:, np.newaxis], (1, n_channels))

                filtered_data = np.zeros_like(data)
                for ch in range(n_channels):
                    filtered_data[ch, :], self.zi[:, ch] = signal.sosfilt(
                        self.sos, data[ch, :], zi=self.zi[:, ch]
                    )

            return filtered_data

        except Exception as e:
            logger.error(f"Filter application failed: {e}")
            raise FilteringError(f"Filter application failed: {e}")

    def reset(self):
        """Reset filter state"""
        try:
            self.zi = signal.sosfilt_zi(self.sos)
            logger.debug("Filter state reset")
        except Exception as e:
            logger.error(f"Filter reset failed: {e}")
            raise FilteringError(f"Filter reset failed: {e}")

    def get_frequency_response(self, frequencies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get filter frequency response"""
        try:
            w, h = signal.sosfreqz(self.sos, worN=frequencies, fs=self.sampling_rate)
            return w, h
        except Exception as e:
            logger.error(f"Frequency response calculation failed: {e}")
            raise FilteringError(f"Frequency response calculation failed: {e}")


class FilterBank:
    """Bank of multiple filters for different frequency bands"""

    def __init__(self, filter_specs: List[Dict[str, Any]], sampling_rate: float):
        self.sampling_rate = sampling_rate
        self.filters = {}

        try:
            for spec in filter_specs:
                name = spec['name']
                filter_type = spec['type']
                frequencies = spec['frequencies']
                order = spec.get('order', 4)

                self.filters[name] = RealTimeFilter(
                    filter_type, frequencies, sampling_rate, order
                )

            logger.info(f"Initialized filter bank with {len(self.filters)} filters")

        except Exception as e:
            raise FilteringError(f"Failed to initialize filter bank: {e}")

    def apply_filters(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply all filters to input data"""
        results = {}

        try:
            for name, filt in self.filters.items():
                results[name] = filt.apply_filter(data)

            return results

        except Exception as e:
            logger.error(f"Filter bank application failed: {e}")
            raise FilteringError(f"Filter bank application failed: {e}")

    def reset_all(self):
        """Reset all filters in the bank"""
        try:
            for filt in self.filters.values():
                filt.reset()
            logger.debug("All filters in bank reset")
        except Exception as e:
            logger.error(f"Filter bank reset failed: {e}")
            raise FilteringError(f"Filter bank reset failed: {e}")


def create_standard_filter_bank(sampling_rate: float) -> FilterBank:
    """Create a standard set of filters for neural signal processing"""

    filter_specs = [
        {
            'name': 'delta',
            'type': 'bandpass',
            'frequencies': (0.5, 4.0),
            'order': 4
        },
        {
            'name': 'theta',
            'type': 'bandpass',
            'frequencies': (4.0, 8.0),
            'order': 4
        },
        {
            'name': 'alpha',
            'type': 'bandpass',
            'frequencies': (8.0, 13.0),
            'order': 4
        },
        {
            'name': 'beta',
            'type': 'bandpass',
            'frequencies': (13.0, 30.0),
            'order': 4
        },
        {
            'name': 'gamma',
            'type': 'bandpass',
            'frequencies': (30.0, 100.0),
            'order': 4
        },
        {
            'name': 'notch_60hz',
            'type': 'notch',
            'frequencies': (60.0,),
            'order': 4
        },
        {
            'name': 'lowpass_anti_alias',
            'type': 'lowpass',
            'frequencies': (sampling_rate / 2.5,),
            'order': 6
        }
    ]

    return FilterBank(filter_specs, sampling_rate)
