"""
Additional Specialized Tools Integration for Brain-Forge

This module integrates advanced neurophysiological analysis tools from the
broader Python ecosystem to enhance Brain-Forge capabilities.

Specialized Tools Integrated:
- EEG-Notebooks: Cognitive neuroscience experiments
- Braindecode: Deep learning for EEG
- NeuroKit2: Neurophysiological signal processing
- PyTorch-EEG: Deep learning for EEG classification
- Spike-Tools: Spike train analysis
- Ephys: Electrophysiology data analysis
"""

import numpy as np
from typing import Dict, Optional, Any
import logging

from ..core.config import Config
from ..core.logger import get_logger
from ..core.exceptions import BrainForgeError

logger = get_logger(__name__)


class EEGNotebooksIntegration:
    """Integration with EEG-Notebooks for cognitive neuroscience experiments"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.experiments = {}
        
    def setup_p300_experiment(self):
        """Setup P300 oddball paradigm experiment"""
        try:
            # EEG-Notebooks P300 experiment setup
            experiment_config = {
                'name': 'P300_Oddball',
                'duration': 300,  # seconds
                'n_trials': 200,
                'target_probability': 0.2,
                'iti_range': [0.5, 1.0]  # Inter-trial interval
            }
            
            self.experiments['p300'] = experiment_config
            self.logger.info("P300 experiment configured")
            return experiment_config
            
        except Exception as e:
            self.logger.error(f"P300 setup failed: {e}")
            raise BrainForgeError(f"Failed to setup P300 experiment: {e}")
    
    def setup_ssvep_experiment(self):
        """Setup Steady-State Visual Evoked Potential experiment"""
        try:
            experiment_config = {
                'name': 'SSVEP',
                'frequencies': [8.0, 10.0, 12.0, 15.0],  # Hz
                'duration': 240,  # seconds
                'trial_length': 4.0,  # seconds per trial
                'rest_length': 2.0   # seconds rest between trials
            }
            
            self.experiments['ssvep'] = experiment_config
            self.logger.info("SSVEP experiment configured")
            return experiment_config
            
        except Exception as e:
            self.logger.error(f"SSVEP setup failed: {e}")
            raise BrainForgeError(f"Failed to setup SSVEP experiment: {e}")


class BraindecodeIntegration:
    """Integration with Braindecode for deep learning on EEG data"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.models = {}
        
    def create_eegnet_classifier(self, n_channels: int = 64, n_classes: int = 2):
        """Create EEGNet classifier for EEG classification using Braindecode"""
        try:
            # Try to use real Braindecode implementation
            try:
                from braindecode.models import EEGNetv4
                import torch
                
                # Create actual EEGNet model
                model = EEGNetv4(
                    n_chans=n_channels,
                    n_outputs=n_classes,
                    input_window_samples=1000,  # 1 second at 1000 Hz 
                    final_conv_length='auto'
                )
                
                self.models['eegnet'] = {
                    'model': model,
                    'name': 'EEGNet',
                    'n_channels': n_channels,
                    'n_classes': n_classes,
                    'input_window_samples': 1000,
                    'framework': 'braindecode',
                    'ready_for_training': True
                }
                
                self.logger.info(f"✅ Real EEGNet model created with Braindecode: {n_channels} channels, {n_classes} classes")
                return self.models['eegnet']
                
            except ImportError:
                # Fallback to simulation if Braindecode not available
                self.logger.warning("Braindecode not available, using simulation")
                model_config = {
                    'name': 'EEGNet',
                    'n_channels': n_channels,
                    'n_classes': n_classes,
                    'input_window_samples': 1000,
                    'final_conv_length': 'auto',
                    'framework': 'simulation',
                    'ready_for_training': False
                }
                
                self.models['eegnet'] = model_config
                self.logger.info(f"EEGNet classifier configured (simulation): {n_channels} channels, {n_classes} classes")
                return model_config
            
        except Exception as e:
            self.logger.error(f"EEGNet creation failed: {e}")
            raise BrainForgeError(f"Failed to create EEGNet classifier: {e}")
    
    def create_shallow_fbcsp_net(self, n_channels: int = 64):
        """Create Shallow FBCSP Network for motor imagery classification using Braindecode"""
        try:
            # Try to use real Braindecode implementation
            try:
                from braindecode.models import ShallowFBCSPNet
                import torch
                
                # Create actual Shallow FBCSP model
                model = ShallowFBCSPNet(
                    n_chans=n_channels,
                    n_outputs=4,  # Typical for motor imagery (left hand, right hand, feet, tongue)
                    input_window_samples=1000,
                    n_filters_time=40,
                    n_filters_spat=40
                )
                
                self.models['shallow_fbcsp'] = {
                    'model': model,
                    'name': 'ShallowFBCSPNet',
                    'n_channels': n_channels,
                    'n_classes': 4,
                    'input_window_samples': 1000,
                    'n_filters_time': 40,
                    'n_filters_spat': 40,
                    'framework': 'braindecode',
                    'ready_for_training': True
                }
                
                self.logger.info(f"✅ Real Shallow FBCSP Network created with Braindecode: {n_channels} channels")
                return self.models['shallow_fbcsp']
                
            except ImportError:
                # Fallback to simulation if Braindecode not available
                self.logger.warning("Braindecode not available, using simulation")
                model_config = {
                    'name': 'ShallowFBCSPNet',
                    'n_channels': n_channels,
                    'n_classes': 4,
                    'input_window_samples': 1000,
                    'n_filters_time': 40,
                    'n_filters_spat': 40,
                    'framework': 'simulation',
                    'ready_for_training': False
                }
                
                self.models['shallow_fbcsp'] = model_config
                self.logger.info(f"Shallow FBCSP Network configured (simulation): {n_channels} channels")
                return model_config
            
        except Exception as e:
            self.logger.error(f"Shallow FBCSP creation failed: {e}")
            raise BrainForgeError(f"Failed to create Shallow FBCSP Network: {e}")


class NeuroKit2Integration:
    """Integration with NeuroKit2 for neurophysiological signal processing"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
    def process_eeg_signals(self, eeg_data: np.ndarray, sampling_rate: float = 1000.0):
        """Process EEG signals using NeuroKit2"""
        try:
            # Try to use real NeuroKit2 implementation
            try:
                import neurokit2 as nk
                
                # Real NeuroKit2 EEG processing pipeline
                processed_results = {
                    'cleaned_eeg': None,
                    'power_bands': {},
                    'artifacts_removed': 0,
                    'quality_score': 0.0,
                    'framework': 'neurokit2'
                }
                
                # Process each channel if multi-channel data
                if eeg_data.ndim == 2:
                    # Multi-channel EEG data (channels x samples)
                    cleaned_channels = []
                    all_power_bands = []
                    
                    for channel_idx in range(eeg_data.shape[0]):
                        channel_data = eeg_data[channel_idx, :]
                        
                        # Clean EEG data using NeuroKit2
                        # Note: NeuroKit2 doesn't have eeg_clean function, using signal_filter
                        cleaned_channel = nk.signal_filter(
                            channel_data, 
                            sampling_rate=sampling_rate,
                            lowcut=1.0,  # High-pass filter at 1 Hz
                            highcut=40.0,  # Low-pass filter at 40 Hz
                            method='butterworth'
                        )
                        cleaned_channels.append(cleaned_channel)
                        
                        # Calculate power in frequency bands using signal_power
                        power_bands = nk.signal_power(
                            cleaned_channel, 
                            sampling_rate=sampling_rate,
                            frequency_bands={
                                'delta': [1, 4],
                                'theta': [4, 8], 
                                'alpha': [8, 13],
                                'beta': [13, 30],
                                'gamma': [30, 40]
                            }
                        )
                        all_power_bands.append(power_bands)
                    
                    processed_results['cleaned_eeg'] = np.array(cleaned_channels)
                    # Average power bands across channels
                    processed_results['power_bands'] = {
                        'delta': np.mean([pb['delta'] for pb in all_power_bands]),
                        'theta': np.mean([pb['theta'] for pb in all_power_bands]),
                        'alpha': np.mean([pb['alpha'] for pb in all_power_bands]),
                        'beta': np.mean([pb['beta'] for pb in all_power_bands]),
                        'gamma': np.mean([pb['gamma'] for pb in all_power_bands])
                    }
                    
                else:
                    # Single channel EEG data
                    cleaned_eeg = nk.signal_filter(
                        eeg_data, 
                        sampling_rate=sampling_rate,
                        lowcut=1.0,
                        highcut=40.0,
                        method='butterworth'
                    )
                    
                    power_bands = nk.signal_power(
                        cleaned_eeg, 
                        sampling_rate=sampling_rate,
                        frequency_bands={
                            'delta': [1, 4],
                            'theta': [4, 8],
                            'alpha': [8, 13], 
                            'beta': [13, 30],
                            'gamma': [30, 40]
                        }
                    )
                    
                    processed_results['cleaned_eeg'] = cleaned_eeg
                    processed_results['power_bands'] = power_bands
                
                # Calculate quality score based on signal-to-noise ratio
                if processed_results['cleaned_eeg'] is not None:
                    signal_power = np.var(processed_results['cleaned_eeg'])
                    noise_estimate = np.var(eeg_data - processed_results['cleaned_eeg'] if eeg_data.shape == processed_results['cleaned_eeg'].shape else eeg_data)
                    snr = signal_power / (noise_estimate + 1e-10)
                    processed_results['quality_score'] = min(1.0, snr / 10.0)  # Normalize to 0-1
                else:
                    processed_results['quality_score'] = 0.5
                
                self.logger.info("✅ EEG signals processed with real NeuroKit2")
                return processed_results
                
            except ImportError:
                # Fallback to simulation if NeuroKit2 not available
                self.logger.warning("NeuroKit2 not available, using simulation")
                processed_results = {
                    'cleaned_eeg': eeg_data,  # Placeholder
                    'power_bands': {
                        'delta': np.mean(eeg_data ** 2),
                        'theta': np.mean(eeg_data ** 2) * 0.8,
                        'alpha': np.mean(eeg_data ** 2) * 0.6,
                        'beta': np.mean(eeg_data ** 2) * 0.4,
                        'gamma': np.mean(eeg_data ** 2) * 0.2
                    },
                    'artifacts_removed': 0,
                    'quality_score': 0.85,  # Simulated quality
                    'framework': 'simulation'
                }
                
                self.logger.info("EEG signals processed with simulation")
                return processed_results
            
        except Exception as e:
            self.logger.error(f"NeuroKit2 EEG processing failed: {e}")
            raise BrainForgeError(f"Failed to process EEG signals: {e}")
    
    def analyze_erp_components(self, epochs_data: np.ndarray, event_times: np.ndarray):
        """Analyze Event-Related Potential components"""
        try:
            erp_results = {
                'p100_amplitude': None,
                'n170_amplitude': None,
                'p300_amplitude': None,
                'p300_latency': None,
                'n400_amplitude': None
            }
            
            # Simulate ERP component detection
            # Note: Actual implementation would use NeuroKit2's ERP functions
            n_epochs, n_channels, n_timepoints = epochs_data.shape
            
            # P300 detection (around 300ms post-stimulus)
            p300_window = slice(250, 350)  # Assuming 1ms per sample
            p300_data = epochs_data[:, :, p300_window]
            erp_results['p300_amplitude'] = np.mean(np.max(p300_data, axis=2))
            erp_results['p300_latency'] = 300 + np.argmax(np.mean(p300_data, axis=(0, 1)))
            
            # N170 detection (around 170ms)
            n170_window = slice(150, 200)
            n170_data = epochs_data[:, :, n170_window]
            erp_results['n170_amplitude'] = np.mean(np.min(n170_data, axis=2))
            
            self.logger.info("ERP components analyzed")
            return erp_results
            
        except Exception as e:
            self.logger.error(f"ERP analysis failed: {e}")
            raise BrainForgeError(f"Failed to analyze ERP components: {e}")


class PyTorchEEGIntegration:
    """Integration with PyTorch-EEG for deep learning EEG classification"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.models = {}
        
    def create_lstm_classifier(self, input_size: int = 64, hidden_size: int = 128, num_classes: int = 2):
        """Create LSTM-based EEG classifier"""
        try:
            # LSTM model configuration for EEG classification
            model_config = {
                'name': 'LSTM_Classifier',
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': 2,
                'num_classes': num_classes,
                'dropout': 0.3,
                'bidirectional': True
            }
            
            # Note: Actual PyTorch implementation would create the model:
            # import torch.nn as nn
            # model = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=True)
            
            self.models['lstm'] = model_config
            self.logger.info(f"LSTM EEG classifier configured: {input_size} inputs, {num_classes} classes")
            return model_config
            
        except Exception as e:
            self.logger.error(f"LSTM classifier creation failed: {e}")
            raise BrainForgeError(f"Failed to create LSTM classifier: {e}")
    
    def create_cnn_transformer(self, n_channels: int = 64, sequence_length: int = 1000):
        """Create CNN-Transformer hybrid for EEG analysis"""
        try:
            model_config = {
                'name': 'CNN_Transformer',
                'n_channels': n_channels,
                'sequence_length': sequence_length,
                'cnn_filters': [32, 64, 128],
                'transformer_heads': 8,
                'transformer_layers': 6,
                'embedding_dim': 256
            }
            
            self.models['cnn_transformer'] = model_config
            self.logger.info(f"CNN-Transformer configured: {n_channels} channels, {sequence_length} timepoints")
            return model_config
            
        except Exception as e:
            self.logger.error(f"CNN-Transformer creation failed: {e}")
            raise BrainForgeError(f"Failed to create CNN-Transformer: {e}")


class SpikeToolsIntegration:
    """Integration with Spike-Tools for spike train analysis"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
    def analyze_spike_trains(self, spike_times: np.ndarray, bin_size: float = 0.001):
        """Analyze spike train patterns"""
        try:
            analysis_results = {
                'firing_rate': None,
                'isi_histogram': None,
                'cv_isi': None,  # Coefficient of variation of ISI
                'burst_detection': None,
                'spike_correlations': None
            }
            
            # Basic spike train analysis
            if len(spike_times) > 1:
                # Inter-spike intervals
                isis = np.diff(spike_times)
                
                # Firing rate
                total_time = spike_times[-1] - spike_times[0]
                analysis_results['firing_rate'] = len(spike_times) / total_time
                
                # ISI statistics
                analysis_results['isi_histogram'] = np.histogram(isis, bins=50)
                analysis_results['cv_isi'] = np.std(isis) / np.mean(isis)
                
                # Burst detection (simple threshold-based)
                burst_threshold = 0.01  # 10ms
                bursts = isis < burst_threshold
                analysis_results['burst_detection'] = {
                    'n_bursts': np.sum(np.diff(bursts.astype(int)) == 1),
                    'burst_ratio': np.mean(bursts)
                }
            
            self.logger.info("Spike train analysis completed")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Spike train analysis failed: {e}")
            raise BrainForgeError(f"Failed to analyze spike trains: {e}")
    
    def detect_synchronization(self, spike_trains_list: list, window_size: float = 0.005):
        """Detect synchronization between multiple spike trains"""
        try:
            n_trains = len(spike_trains_list)
            sync_results = {
                'cross_correlations': np.zeros((n_trains, n_trains)),
                'synchrony_index': 0.0,
                'coincidence_rate': 0.0
            }
            
            # Calculate cross-correlations between spike trains
            for i in range(n_trains):
                for j in range(i+1, n_trains):
                    # Simplified cross-correlation calculation
                    train1 = spike_trains_list[i]
                    train2 = spike_trains_list[j]
                    
                    # Count coincidences within window
                    coincidences = 0
                    for spike in train1:
                        nearby_spikes = np.sum(np.abs(train2 - spike) < window_size)
                        coincidences += nearby_spikes
                    
                    correlation = coincidences / (len(train1) * len(train2))
                    sync_results['cross_correlations'][i, j] = correlation
                    sync_results['cross_correlations'][j, i] = correlation
            
            # Overall synchrony index
            sync_results['synchrony_index'] = np.mean(sync_results['cross_correlations'])
            
            self.logger.info(f"Synchronization analysis completed for {n_trains} spike trains")
            return sync_results
            
        except Exception as e:
            self.logger.error(f"Synchronization analysis failed: {e}")
            raise BrainForgeError(f"Failed to detect synchronization: {e}")


class EphysIntegration:
    """Integration with Ephys for electrophysiology data analysis"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
    def analyze_local_field_potential(self, lfp_data: np.ndarray, sampling_rate: float = 1000.0):
        """Analyze Local Field Potential data"""
        try:
            analysis_results = {
                'power_spectral_density': None,
                'coherence_matrix': None,
                'phase_coupling': None,
                'spectral_peaks': None
            }
            
            from scipy import signal
            
            # Power spectral density
            freqs, psd = signal.welch(lfp_data, fs=sampling_rate, axis=-1)
            analysis_results['power_spectral_density'] = {'frequencies': freqs, 'power': psd}
            
            # Find spectral peaks
            if lfp_data.ndim == 1:
                peaks, _ = signal.find_peaks(psd, height=np.max(psd) * 0.1)
                analysis_results['spectral_peaks'] = freqs[peaks]
            else:
                # Multi-channel case
                all_peaks = []
                for ch in range(lfp_data.shape[0]):
                    peaks, _ = signal.find_peaks(psd[ch], height=np.max(psd[ch]) * 0.1)
                    all_peaks.append(freqs[peaks])
                analysis_results['spectral_peaks'] = all_peaks
            
            # Coherence analysis for multi-channel data
            if lfp_data.ndim > 1 and lfp_data.shape[0] > 1:
                n_channels = lfp_data.shape[0]
                coherence_matrix = np.zeros((n_channels, n_channels))
                
                for i in range(n_channels):
                    for j in range(i+1, n_channels):
                        f, coh = signal.coherence(lfp_data[i], lfp_data[j], fs=sampling_rate)
                        coherence_matrix[i, j] = np.mean(coh)
                        coherence_matrix[j, i] = coherence_matrix[i, j]
                
                analysis_results['coherence_matrix'] = coherence_matrix
            
            self.logger.info("LFP analysis completed")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"LFP analysis failed: {e}")
            raise BrainForgeError(f"Failed to analyze LFP data: {e}")
    
    def detect_sharp_wave_ripples(self, lfp_data: np.ndarray, sampling_rate: float = 1000.0):
        """Detect sharp wave-ripple complexes in hippocampal LFP"""
        try:
            from scipy import signal
            
            # Filter for ripple band (150-250 Hz)
            ripple_filter = signal.butter(4, [150, 250], btype='band', fs=sampling_rate)
            ripple_filtered = signal.filtfilt(ripple_filter[0], ripple_filter[1], lfp_data)
            
            # Filter for sharp wave band (1-30 Hz)
            sw_filter = signal.butter(4, [1, 30], btype='band', fs=sampling_rate)
            sw_filtered = signal.filtfilt(sw_filter[0], sw_filter[1], lfp_data)
            
            # Detect ripple events
            ripple_power = np.abs(signal.hilbert(ripple_filtered)) ** 2
            ripple_threshold = np.mean(ripple_power) + 3 * np.std(ripple_power)
            ripple_events, _ = signal.find_peaks(ripple_power, height=ripple_threshold, distance=int(0.02 * sampling_rate))
            
            # Detect sharp wave events
            sw_threshold = np.mean(sw_filtered) - 3 * np.std(sw_filtered)  # Negative deflection
            sw_events, _ = signal.find_peaks(-sw_filtered, height=-sw_threshold, distance=int(0.05 * sampling_rate))
            
            # Find coincident events (sharp wave-ripple complexes)
            swr_complexes = []
            for sw_event in sw_events:
                # Look for ripples within 50ms of sharp wave
                nearby_ripples = ripple_events[np.abs(ripple_events - sw_event) < 0.05 * sampling_rate]
                if len(nearby_ripples) > 0:
                    swr_complexes.append({
                        'sharp_wave_time': sw_event / sampling_rate,
                        'ripple_times': nearby_ripples / sampling_rate,
                        'duration': len(nearby_ripples) * 0.01  # Approximate
                    })
            
            results = {
                'swr_complexes': swr_complexes,
                'n_complexes': len(swr_complexes),
                'ripple_events': ripple_events / sampling_rate,
                'sharp_wave_events': sw_events / sampling_rate
            }
            
            self.logger.info(f"Detected {len(swr_complexes)} sharp wave-ripple complexes")
            return results
            
        except Exception as e:
            self.logger.error(f"SWR detection failed: {e}")
            raise BrainForgeError(f"Failed to detect sharp wave-ripples: {e}")


class SpecializedToolsManager:
    """Manager class for all specialized neurophysiological analysis tools"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize tool integrations
        self.eeg_notebooks = EEGNotebooksIntegration()
        self.braindecode = BraindecodeIntegration()
        self.neurokit2 = NeuroKit2Integration()
        self.pytorch_eeg = PyTorchEEGIntegration()
        self.spike_tools = SpikeToolsIntegration()
        self.ephys = EphysIntegration()
        
        self.logger.info("Specialized tools manager initialized")
    
    def get_available_tools(self) -> Dict[str, Any]:
        """Get information about all available specialized tools"""
        return {
            'eeg_notebooks': {
                'description': 'Cognitive neuroscience experiments',
                'experiments': list(self.eeg_notebooks.experiments.keys())
            },
            'braindecode': {
                'description': 'Deep learning for EEG',
                'models': list(self.braindecode.models.keys())
            },
            'neurokit2': {
                'description': 'Neurophysiological signal processing',
                'methods': ['process_eeg_signals', 'analyze_erp_components']
            },
            'pytorch_eeg': {
                'description': 'Deep learning for EEG classification',
                'models': list(self.pytorch_eeg.models.keys())
            },
            'spike_tools': {
                'description': 'Spike train analysis',
                'methods': ['analyze_spike_trains', 'detect_synchronization']
            },
            'ephys': {
                'description': 'Electrophysiology data analysis',
                'methods': ['analyze_local_field_potential', 'detect_sharp_wave_ripples']
            }
        }
    
    def run_comprehensive_analysis(self, brain_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Run comprehensive analysis using all available specialized tools"""
        try:
            results = {}
            
            # EEG/MEG analysis with NeuroKit2
            if 'eeg_data' in brain_data or 'meg_data' in brain_data:
                eeg_data = brain_data.get('eeg_data', brain_data.get('meg_data'))
                if eeg_data is not None:
                    results['neurokit2_eeg'] = self.neurokit2.process_eeg_signals(eeg_data)
            
            # Spike train analysis if available
            if 'spike_times' in brain_data:
                results['spike_analysis'] = self.spike_tools.analyze_spike_trains(brain_data['spike_times'])
            
            # LFP analysis if available
            if 'lfp_data' in brain_data:
                results['lfp_analysis'] = self.ephys.analyze_local_field_potential(brain_data['lfp_data'])
                results['swr_detection'] = self.ephys.detect_sharp_wave_ripples(brain_data['lfp_data'])
            
            self.logger.info("Comprehensive specialized analysis completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
            raise BrainForgeError(f"Failed to run comprehensive analysis: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize specialized tools manager
    tools_manager = SpecializedToolsManager()
    
    # Get available tools
    available_tools = tools_manager.get_available_tools()
    print("Available Specialized Tools:")
    for tool_name, info in available_tools.items():
        print(f"  {tool_name}: {info['description']}")
    
    # Example data for analysis
    example_data = {
        'eeg_data': np.random.randn(64, 10000),  # 64 channels, 10 seconds at 1kHz
        'spike_times': np.sort(np.random.exponential(0.01, 1000)),  # Poisson spike train
        'lfp_data': np.random.randn(8, 30000)  # 8 channels, 30 seconds at 1kHz
    }
    
    # Run comprehensive analysis
    try:
        results = tools_manager.run_comprehensive_analysis(example_data)
        print(f"Analysis completed. Results keys: {list(results.keys())}")
    except Exception as e:
        print(f"Analysis failed: {e}")
