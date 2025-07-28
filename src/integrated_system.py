"""
Integrated Brain System for Brain-Forge Platform

This module implements a comprehensive brain-computer interface system that combines
multiple cutting-edge technologies for brain scanning, mapping, and simulation.

Key Components:
- Multi-modal data acquisition (OMP, Kernel optical, accelerometer)
- Real-time processing and compression
- Brain mapping and connectivity analysis
- Neural simulation and digital twin creation
- Pattern transfer and learning protocols
"""

import numpy as np
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

# Core Brain-Forge imports
from ..core.config import Config
from ..core.logger import get_logger
from ..core.exceptions import BrainForgeError

# Neuroimaging and analysis
import mne
from nilearn import plotting, datasets
from nilearn.connectome import ConnectivityMeasure
from sklearn.decomposition import PCA
from scipy import signal

# Real-time streaming
from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet

# Neural simulation
from brian2 import *
import nest

# Deep learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pyvista as pv

logger = get_logger(__name__)


@dataclass
class BrainData:
    """Container for multi-modal brain data"""
    meg_data: np.ndarray
    optical_data: np.ndarray
    accel_data: np.ndarray
    timestamps: np.ndarray
    sampling_rates: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class NeuralFeatures:
    """Container for extracted neural features"""
    power_spectrum: np.ndarray
    dominant_frequencies: np.ndarray
    firing_rates: np.ndarray
    synchronization: np.ndarray
    spatial_components: np.ndarray
    connectivity_matrix: np.ndarray


class IntegratedBrainSystem:
    """
    Comprehensive brain-computer interface system integrating multiple
    neuroimaging technologies for real-time brain scanning and simulation.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the integrated brain system"""
        self.config = config or Config()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Hardware interfaces
        self.data_streams = {}
        self.hardware_status = {
            'omp_helmet': False,
            'kernel_optical': False,
            'accelerometer': False
        }
        
        # Processing components
        self.brain_atlas = None
        self.functional_networks = None
        self.simulation_network = None
        self.compression_module = None
        
        # Data storage
        self.current_session_data = None
        self.connectivity_matrix = None
        
        # Performance metrics
        self.processing_latency = []
        self.compression_ratios = []
        
        # Initialize system components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            self.logger.info("Initializing Brain-Forge integrated system...")
            
            # Setup brain atlases
            self._setup_brain_atlas()
            
            # Initialize compression module
            self._initialize_compression()
            
            self.logger.info("Brain-Forge system initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            raise BrainForgeError(f"System initialization failed: {e}")
    
    def _setup_brain_atlas(self):
        """Load and configure brain atlas for mapping"""
        try:
            self.logger.info("Loading brain atlases...")
            
            # Harvard-Oxford atlas for structural mapping
            self.brain_atlas = datasets.fetch_atlas_harvard_oxford(
                'cort-maxprob-thr25-2mm'
            )
            
            # Functional networks atlas
            self.functional_networks = datasets.fetch_atlas_yeo_2011()
            
            self.logger.info("Brain atlases loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load brain atlases: {e}")
            raise BrainForgeError(f"Atlas loading failed: {e}")
    
    def _initialize_compression(self):
        """Initialize neural data compression module"""
        try:
            from ..processing.compression import NeuralCompressor
            
            self.compression_module = NeuralCompressor(
                compression_ratio=self.config.processing.compression.compression_ratio,
                algorithm=self.config.processing.compression.algorithm
            )
            
            self.logger.info("Compression module initialized")
            
        except ImportError:
            self.logger.warning("Compression module not available, using basic compression")
            self.compression_module = None
    
    async def initialize_hardware_streams(self) -> Dict[str, bool]:
        """
        Initialize real-time data streams from all hardware devices
        
        Returns:
            Dict mapping device names to connection status
        """
        self.logger.info("Initializing hardware streams...")
        
        # OMP Helmet (MEG-like data)
        try:
            meg_streams = resolve_stream('type', 'MEG')
            if meg_streams:
                self.data_streams['meg'] = StreamInlet(meg_streams[0])
                self.hardware_status['omp_helmet'] = True
                self.logger.info("OMP Helmet connected (306 channels)")
            else:
                self.logger.warning("OMP Helmet not found")
        except Exception as e:
            self.logger.error(f"Failed to connect OMP Helmet: {e}")
        
        # Kernel Optical Helmet
        try:
            optical_streams = resolve_stream('type', 'NIRS')
            if optical_streams:
                self.data_streams['optical'] = StreamInlet(optical_streams[0])
                self.hardware_status['kernel_optical'] = True
                self.logger.info("Kernel Optical Helmet connected")
            else:
                self.logger.warning("Kernel Optical Helmet not found")
        except Exception as e:
            self.logger.error(f"Failed to connect Kernel Optical: {e}")
        
        # Accelerometer data
        try:
            accel_streams = resolve_stream('type', 'Accelerometer')
            if accel_streams:
                self.data_streams['accel'] = StreamInlet(accel_streams[0])
                self.hardware_status['accelerometer'] = True
                self.logger.info("Accelerometer array connected")
            else:
                self.logger.warning("Accelerometer not found")
        except Exception as e:
            self.logger.error(f"Failed to connect accelerometer: {e}")
        
        connected_devices = sum(self.hardware_status.values())
        self.logger.info(f"Hardware initialization complete: {connected_devices}/3 devices connected")
        
        return self.hardware_status.copy()
    
    async def acquire_brain_data(self, duration: float = 60.0) -> BrainData:
        """
        Acquire synchronized multi-modal brain data
        
        Args:
            duration: Acquisition duration in seconds
            
        Returns:
            BrainData object containing all acquired data
        """
        self.logger.info(f"Starting brain data acquisition ({duration}s)...")
        
        all_data = {
            'meg': [],
            'optical': [],
            'accel': [],
            'timestamps': []
        }
        
        start_time = time.time()
        sample_count = 0
        
        try:
            while (time.time() - start_time) < duration:
                timestamp = time.time()
                
                # Collect from all streams with timeout
                for stream_type, inlet in self.data_streams.items():
                    if inlet:
                        try:
                            sample, ts = inlet.pull_sample(timeout=0.001)
                            if sample:
                                all_data[stream_type].append(sample)
                        except Exception as e:
                            self.logger.debug(f"Stream {stream_type} sample missed: {e}")
                
                all_data['timestamps'].append(timestamp)
                sample_count += 1
                
                # Small delay to prevent CPU overload
                await asyncio.sleep(0.001)  # 1ms sampling interval
                
                # Progress logging every 10 seconds
                if sample_count % 10000 == 0:
                    elapsed = time.time() - start_time
                    self.logger.info(f"Acquisition progress: {elapsed:.1f}/{duration}s")
            
            # Convert to numpy arrays
            brain_data = BrainData(
                meg_data=np.array(all_data['meg']) if all_data['meg'] else np.array([]),
                optical_data=np.array(all_data['optical']) if all_data['optical'] else np.array([]),
                accel_data=np.array(all_data['accel']) if all_data['accel'] else np.array([]),
                timestamps=np.array(all_data['timestamps']),
                sampling_rates={
                    'meg': 1000.0,  # Hz
                    'optical': 10.0,  # Hz
                    'accel': 1000.0  # Hz
                },
                metadata={
                    'duration': duration,
                    'sample_count': sample_count,
                    'hardware_status': self.hardware_status.copy()
                }
            )
            
            self.current_session_data = brain_data
            self.logger.info(f"Brain data acquisition complete: {sample_count} samples")
            
            return brain_data
            
        except Exception as e:
            self.logger.error(f"Data acquisition failed: {e}")
            raise BrainForgeError(f"Failed to acquire brain data: {e}")
    
    def preprocess_neural_data(self, brain_data: BrainData) -> mne.io.Raw:
        """
        Advanced preprocessing pipeline for neural data
        
        Args:
            brain_data: Raw brain data from acquisition
            
        Returns:
            Preprocessed MNE Raw object
        """
        self.logger.info("Starting neural data preprocessing...")
        
        try:
            if brain_data.meg_data.size == 0:
                raise BrainForgeError("No MEG data available for preprocessing")
            
            # Transpose data for MNE format (channels x time)
            meg_data = brain_data.meg_data.T
            
            # Create MNE Raw object
            n_channels = meg_data.shape[0]
            info = mne.create_info(
                ch_names=[f'MEG{i:03d}' for i in range(n_channels)],
                sfreq=brain_data.sampling_rates['meg'],
                ch_types='meg'
            )
            raw = mne.io.RawArray(meg_data, info)
            
            # Preprocessing pipeline
            self.logger.info("Applying preprocessing filters...")
            
            # Bandpass filter (1-100 Hz)
            filter_range = self.config.hardware.omp_helmet.filter_range
            raw.filter(filter_range[0], filter_range[1], fir_design='firwin')
            
            # Notch filter (remove power line noise)
            notch_freq = self.config.processing.filtering.notch_filter
            raw.notch_filter(notch_freq, fir_design='firwin')
            
            # ICA for artifact removal
            self.logger.info("Performing ICA artifact removal...")
            ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
            ica.fit(raw)
            
            # Apply ICA
            raw = ica.apply(raw)
            
            self.logger.info("Neural data preprocessing complete")
            return raw
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise BrainForgeError(f"Neural data preprocessing failed: {e}")
    
    def extract_brain_connectivity(self, processed_data: mne.io.Raw) -> np.ndarray:
        """
        Extract brain connectivity patterns from processed data
        
        Args:
            processed_data: Preprocessed MNE Raw object
            
        Returns:
            Connectivity matrix
        """
        self.logger.info("Extracting brain connectivity patterns...")
        
        try:
            # Create epochs for connectivity analysis
            events = mne.make_fixed_length_events(processed_data, duration=2.0)
            epochs = mne.Epochs(
                processed_data, 
                events, 
                tmin=0, 
                tmax=2.0, 
                baseline=None,
                preload=True
            )
            
            # Connectivity analysis
            connectivity_measure = ConnectivityMeasure(
                kind='correlation',
                vectorize=True,
                discard_diagonal=True
            )
            
            correlation_matrix = connectivity_measure.fit_transform([epochs.get_data()])[0]
            
            # Reshape to square matrix
            n_channels = epochs.info['nchan']
            connectivity_matrix = np.zeros((n_channels, n_channels))
            
            # Fill upper triangle
            triu_indices = np.triu_indices(n_channels, k=1)
            connectivity_matrix[triu_indices] = correlation_matrix
            
            # Make symmetric
            connectivity_matrix = connectivity_matrix + connectivity_matrix.T
            
            # Add diagonal (self-correlation = 1)
            np.fill_diagonal(connectivity_matrix, 1.0)
            
            self.connectivity_matrix = connectivity_matrix
            self.logger.info(f"Connectivity matrix computed: {connectivity_matrix.shape}")
            
            return connectivity_matrix
            
        except Exception as e:
            self.logger.error(f"Connectivity extraction failed: {e}")
            raise BrainForgeError(f"Failed to extract connectivity: {e}")
    
    def extract_neural_features(self, brain_data: mne.io.Raw) -> NeuralFeatures:
        """
        Extract comprehensive neural features for transfer learning
        
        Args:
            brain_data: Preprocessed brain data
            
        Returns:
            NeuralFeatures object containing extracted features
        """
        self.logger.info("Extracting neural features...")
        
        try:
            data = brain_data.get_data()
            sfreq = brain_data.info['sfreq']
            
            # Spectral features
            freqs, psd = signal.welch(data, fs=sfreq, nperseg=int(sfreq))
            
            # Temporal features
            firing_rates = np.mean(np.abs(data), axis=1)
            synchronization = np.corrcoef(data)
            
            # Spatial features using PCA
            pca = PCA(n_components=min(10, data.shape[0]))
            spatial_components = pca.fit_transform(data.T)
            
            features = NeuralFeatures(
                power_spectrum=psd,
                dominant_frequencies=freqs[np.argmax(psd, axis=1)],
                firing_rates=firing_rates,
                synchronization=synchronization,
                spatial_components=spatial_components,
                connectivity_matrix=self.connectivity_matrix if self.connectivity_matrix is not None else synchronization
            )
            
            self.logger.info("Neural feature extraction complete")
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            raise BrainForgeError(f"Failed to extract neural features: {e}")
    
    def create_brain_simulation(self, connectivity_matrix: np.ndarray, n_neurons: int = 100000) -> Dict[str, Any]:
        """
        Create detailed brain simulation using Brian2
        
        Args:
            connectivity_matrix: Brain connectivity pattern
            n_neurons: Number of neurons in simulation
            
        Returns:
            Dictionary containing simulation components
        """
        self.logger.info(f"Creating brain simulation with {n_neurons} neurons...")
        
        try:
            # Set Brian2 clock
            defaultclock.dt = 0.1*ms
            
            # Leaky integrate-and-fire neuron model
            eqs = '''
            dv/dt = (I - v)/tau : 1
            I : 1
            tau : second
            '''
            
            # Create neuron groups
            neurons = NeuronGroup(
                n_neurons, 
                eqs, 
                threshold='v > 1', 
                reset='v = 0'
            )
            neurons.tau = 10*ms
            neurons.v = 'rand()'
            neurons.I = 0.1
            
            # Create synapses based on connectivity
            synapses = Synapses(
                neurons, 
                neurons, 
                'w : 1', 
                on_pre='v_post += w'
            )
            
            # Connect based on connectivity matrix
            self._connect_from_connectivity(synapses, connectivity_matrix, n_neurons)
            
            # Monitoring
            spike_monitor = SpikeMonitor(neurons)
            rate_monitor = PopulationRateMonitor(neurons)
            
            self.simulation_network = {
                'neurons': neurons,
                'synapses': synapses,
                'spike_monitor': spike_monitor,
                'rate_monitor': rate_monitor
            }
            
            self.logger.info("Brain simulation created successfully")
            return self.simulation_network
            
        except Exception as e:
            self.logger.error(f"Simulation creation failed: {e}")
            raise BrainForgeError(f"Failed to create brain simulation: {e}")
    
    def _connect_from_connectivity(self, synapses, connectivity_matrix, n_neurons):
        """Connect neurons based on connectivity matrix"""
        try:
            # Scale connectivity matrix to neuron indices
            matrix_size = connectivity_matrix.shape[0]
            neurons_per_region = n_neurons // matrix_size
            
            connections_made = 0
            
            for i in range(matrix_size):
                for j in range(matrix_size):
                    if i != j and abs(connectivity_matrix[i, j]) > 0.3:  # Threshold
                        # Map matrix indices to neuron ranges
                        source_start = i * neurons_per_region
                        source_end = (i + 1) * neurons_per_region
                        target_start = j * neurons_per_region
                        target_end = (j + 1) * neurons_per_region
                        
                        # Connect with probability based on connectivity strength
                        connection_prob = min(abs(connectivity_matrix[i, j]), 0.1)
                        
                        synapses.connect(
                            i=range(source_start, source_end),
                            j=range(target_start, target_end),
                            p=connection_prob
                        )
                        
                        # Set weights
                        synapses.w[source_start:source_end, target_start:target_end] = abs(connectivity_matrix[i, j])
                        connections_made += 1
            
            self.logger.info(f"Neural connections created: {connections_made} region pairs")
            
        except Exception as e:
            self.logger.error(f"Neural connection failed: {e}")
            raise BrainForgeError(f"Failed to connect neurons: {e}")
    
    def transfer_brain_patterns(self, neural_features: NeuralFeatures, target_simulation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transfer extracted brain patterns to simulation
        
        Args:
            neural_features: Extracted neural features
            target_simulation: Target simulation network
            
        Returns:
            Updated simulation with transferred patterns
        """
        self.logger.info("Transferring brain patterns to simulation...")
        
        try:
            neurons = target_simulation['neurons']
            
            # Map firing rates to neural excitability
            normalized_rates = neural_features.firing_rates / np.max(neural_features.firing_rates)
            
            # Update neuron parameters based on features
            for i, rate in enumerate(normalized_rates[:len(neurons)]):
                # Adjust input current based on firing rate
                neurons.I[i] = 0.1 + rate * 0.5
                
                # Adjust time constant based on synchronization
                if i < len(neural_features.synchronization):
                    sync_factor = np.mean(neural_features.synchronization[i])
                    neurons.tau[i] = (10 + sync_factor * 5) * ms
            
            # Update synaptic weights based on connectivity
            if hasattr(target_simulation, 'synapses'):
                synapses = target_simulation['synapses']
                # Apply connectivity-based weight scaling
                weight_scale = np.mean(np.abs(neural_features.connectivity_matrix))
                synapses.w *= weight_scale
            
            self.logger.info("Brain pattern transfer complete")
            return target_simulation
            
        except Exception as e:
            self.logger.error(f"Pattern transfer failed: {e}")
            raise BrainForgeError(f"Failed to transfer brain patterns: {e}")
    
    async def run_brain_transfer_protocol(self, subject_id: str, scan_duration: float = 3600) -> Dict[str, Any]:
        """
        Execute complete brain scanning and transfer protocol
        
        Args:
            subject_id: Unique identifier for subject
            scan_duration: Scanning duration in seconds
            
        Returns:
            Complete brain simulation with transferred patterns
        """
        self.logger.info(f"Starting brain transfer protocol for subject {subject_id}")
        
        try:
            # Step 1: Initialize hardware
            self.logger.info("Phase 1: Initializing hardware...")
            hardware_status = await self.initialize_hardware_streams()
            
            if not any(hardware_status.values()):
                raise BrainForgeError("No hardware devices connected")
            
            # Step 2: Acquire brain data
            self.logger.info("Phase 2: Acquiring brain data...")
            raw_data = await self.acquire_brain_data(duration=scan_duration)
            
            # Step 3: Process data
            self.logger.info("Phase 3: Processing neural data...")
            processed_data = self.preprocess_neural_data(raw_data)
            
            # Step 4: Extract connectivity
            self.logger.info("Phase 4: Extracting brain connectivity...")
            connectivity = self.extract_brain_connectivity(processed_data)
            
            # Step 5: Extract features
            self.logger.info("Phase 5: Extracting neural features...")
            neural_features = self.extract_neural_features(processed_data)
            
            # Step 6: Create simulation
            self.logger.info("Phase 6: Creating brain simulation...")
            simulation = self.create_brain_simulation(connectivity)
            
            # Step 7: Transfer patterns
            self.logger.info("Phase 7: Transferring brain patterns...")
            final_simulation = self.transfer_brain_patterns(neural_features, simulation)
            
            self.logger.info("Brain transfer protocol complete!")
            
            return {
                'subject_id': subject_id,
                'simulation': final_simulation,
                'neural_features': neural_features,
                'connectivity_matrix': connectivity,
                'raw_data_summary': {
                    'duration': scan_duration,
                    'samples': len(raw_data.timestamps),
                    'hardware_used': [k for k, v in hardware_status.items() if v]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Brain transfer protocol failed: {e}")
            raise BrainForgeError(f"Brain transfer protocol failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'hardware_status': self.hardware_status.copy(),
            'data_streams_active': len([s for s in self.data_streams.values() if s is not None]),
            'current_session': self.current_session_data is not None,
            'simulation_ready': self.simulation_network is not None,
            'connectivity_computed': self.connectivity_matrix is not None,
            'brain_atlas_loaded': self.brain_atlas is not None,
            'compression_available': self.compression_module is not None
        }
    
    def visualize_results(self, save_plots: bool = False):
        """Create comprehensive visualization of results"""
        if self.connectivity_matrix is None:
            self.logger.warning("No connectivity data available for visualization")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot connectivity matrix
            im1 = axes[0, 0].imshow(self.connectivity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[0, 0].set_title('Brain Connectivity Matrix')
            axes[0, 0].set_xlabel('Brain Region')
            axes[0, 0].set_ylabel('Brain Region')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Plot eigenvalues of connectivity matrix
            eigenvals = np.linalg.eigvals(self.connectivity_matrix)
            axes[0, 1].plot(np.sort(eigenvals)[::-1], 'o-')
            axes[0, 1].set_title('Connectivity Matrix Eigenvalues')
            axes[0, 1].set_xlabel('Component')
            axes[0, 1].set_ylabel('Eigenvalue')
            
            # Plot network degree distribution
            degrees = np.sum(np.abs(self.connectivity_matrix) > 0.3, axis=1)
            axes[1, 0].hist(degrees, bins=20, alpha=0.7)
            axes[1, 0].set_title('Network Degree Distribution')
            axes[1, 0].set_xlabel('Degree')
            axes[1, 0].set_ylabel('Frequency')
            
            # Plot simulation activity (if available)
            if self.simulation_network:
                # Placeholder for simulation activity
                t = np.linspace(0, 1000, 1000)
                activity = np.random.exponential(0.1, 1000) * np.sin(2 * np.pi * t / 100)
                axes[1, 1].plot(t, activity)
                axes[1, 1].set_title('Simulated Neural Activity')
                axes[1, 1].set_xlabel('Time (ms)')
                axes[1, 1].set_ylabel('Activity')
            else:
                axes[1, 1].text(0.5, 0.5, 'No simulation data', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Simulation Status: Not Available')
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('brain_forge_results.png', dpi=300, bbox_inches='tight')
                self.logger.info("Visualization saved to brain_forge_results.png")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Example of running the integrated brain system
    async def main():
        try:
            # Initialize the system
            brain_system = IntegratedBrainSystem()
            
            # Run brain transfer protocol (short duration for testing)
            result = await brain_system.run_brain_transfer_protocol(
                subject_id="test_001",
                scan_duration=60  # 1 minute for testing
            )
            
            # Display results
            brain_system.visualize_results(save_plots=True)
            
            # Print system status
            status = brain_system.get_system_status()
            print("System Status:", status)
            
        except Exception as e:
            logger.error(f"Example run failed: {e}")
    
    # Run the example
    asyncio.run(main())
