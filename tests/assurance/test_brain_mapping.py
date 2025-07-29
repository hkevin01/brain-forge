"""
Assurance Tests for Advanced Brain Mapping & Digital Twins
Validates spatial connectivity analysis, interactive brain atlas, and digital brain simulation
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock
from dataclasses import dataclass
from typing import Dict, List, Tuple
import networkx as nx
from scipy import spatial, stats
from sklearn.metrics import silhouette_score
import time

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@dataclass
class BrainMappingSpecs:
    """Brain mapping and connectivity specifications"""
    atlas_resolution: float = 1.0  # mm voxel resolution
    connectivity_nodes: int = 400  # Schaefer 400-node parcellation
    temporal_resolution: float = 0.001  # 1ms temporal resolution
    spatial_accuracy: float = 0.95  # 95% spatial localization accuracy
    real_time_update_rate: int = 30  # 30 Hz visualization updates
    

@dataclass
class DigitalTwinSpecs:
    """Digital brain twin specifications"""
    neuron_models: int = 100000  # 100K neurons minimum
    synapse_models: int = 10000000  # 10M synapses minimum
    simulation_timestep: float = 0.0001  # 0.1ms simulation timestep
    real_time_factor: float = 0.1  # 10x slower than real-time acceptable
    pattern_fidelity: float = 0.90  # 90% pattern reproduction fidelity


class TestSpatialConnectivityAnalysis:
    """Test DTI/fMRI structural mapping with functional dynamics"""
    
    @pytest.fixture
    def connectivity_system(self):
        """Mock spatial connectivity analysis system"""
        system = Mock()
        system.specs = BrainMappingSpecs()
        system.atlas_nodes = 400  # Schaefer atlas
        system.structural_matrix = None
        system.functional_matrix = None
        return system
    
    def test_dti_structural_connectivity(self, connectivity_system):
        """Test DTI-based structural connectivity mapping"""
        n_nodes = connectivity_system.specs.connectivity_nodes
        
        # Generate realistic structural connectivity matrix
        def generate_dti_connectivity(n_nodes, sparsity=0.3):
            """Generate DTI-based structural connectivity"""
            # Create distance-based connectivity (nearby regions more connected)
            # Simulate brain regions in 3D space
            brain_coords = np.random.uniform(-50, 50, (n_nodes, 3))  # mm coordinates
            
            # Calculate pairwise distances
            distances = spatial.distance_matrix(brain_coords, brain_coords)
            
            # Connection probability decreases with distance
            connection_prob = np.exp(-distances / 20.0)  # 20mm characteristic length
            
            # Add anatomical constraints (interhemispheric connections are sparser)
            for i in range(n_nodes):
                for j in range(n_nodes):
                    # Assume left hemisphere (x < 0) and right hemisphere (x > 0)
                    if (brain_coords[i, 0] < 0) != (brain_coords[j, 0] < 0):
                        connection_prob[i, j] *= 0.1  # Reduce interhemispheric connections
            
            # Create binary connectivity matrix
            structural_matrix = (connection_prob > np.percentile(connection_prob, (1-sparsity)*100))
            
            # Add fiber tract strengths (fractional anisotropy values)
            fa_values = np.random.uniform(0.2, 0.8, (n_nodes, n_nodes))
            structural_matrix = structural_matrix.astype(float) * fa_values
            
            # Make symmetric
            structural_matrix = (structural_matrix + structural_matrix.T) / 2
            np.fill_diagonal(structural_matrix, 0)  # No self-connections
            
            return structural_matrix, brain_coords
        
        connectivity_system.generate_dti_connectivity = generate_dti_connectivity
        
        # Test structural connectivity generation
        structural_matrix, coords = connectivity_system.generate_dti_connectivity(n_nodes)
        connectivity_system.structural_matrix = structural_matrix
        connectivity_system.brain_coordinates = coords
        
        # Validate structural properties
        assert structural_matrix.shape == (n_nodes, n_nodes), "Connectivity matrix should be square"
        assert np.allclose(structural_matrix, structural_matrix.T), "Should be symmetric"
        assert np.all(np.diag(structural_matrix) == 0), "No self-connections"
        assert np.all(structural_matrix >= 0), "Connection strengths should be non-negative"
        
        # Test anatomical realism
        sparsity = np.sum(structural_matrix > 0) / (n_nodes * (n_nodes - 1))
        assert 0.1 <= sparsity <= 0.5, f"Sparsity {sparsity:.3f} should be realistic (10-50%)"
        
        # Test small-world properties
        G = nx.from_numpy_array(structural_matrix)
        if nx.is_connected(G):
            clustering = nx.average_clustering(G)
            path_length = nx.average_shortest_path_length(G)
            
            # Small-world networks have high clustering, short path length
            assert clustering > 0.1, "Should have significant clustering"
            assert path_length < np.log(n_nodes), "Should have short characteristic path length"
    
    def test_functional_connectivity_dynamics(self, connectivity_system):
        """Test functional connectivity from neural activity"""
        n_nodes = 400
        n_timepoints = 10000  # 10 seconds at 1kHz
        
        # Generate realistic neural time series
        def generate_neural_timeseries(n_nodes, n_timepoints, structural_matrix):
            """Generate functional connectivity from structural backbone"""
            # Create coupled oscillators based on structural connectivity
            dt = 0.001  # 1ms timestep
            
            # Neural mass model parameters
            tau = 0.02  # 20ms time constant
            noise_std = 0.1
            coupling_strength = 0.05
            
            # Initialize neural activity
            activity = np.random.normal(0, 0.1, (n_timepoints, n_nodes))
            
            # Integrate coupled dynamics
            for t in range(1, n_timepoints):
                # Local dynamics (damped oscillator)
                local_change = -activity[t-1] / tau + noise_std * np.random.randn(n_nodes)
                
                # Coupling through structural connections
                if structural_matrix is not None:
                    coupling_input = coupling_strength * np.dot(structural_matrix, activity[t-1])
                    local_change += coupling_input
                
                # Euler integration
                activity[t] = activity[t-1] + dt * local_change
            
            return activity
        
        connectivity_system.generate_neural_timeseries = generate_neural_timeseries
        
        # Generate structural connectivity first
        structural_matrix, _ = connectivity_system.generate_dti_connectivity(n_nodes)
        
        # Generate functional time series
        neural_activity = connectivity_system.generate_neural_timeseries(
            n_nodes, n_timepoints, structural_matrix
        )
        
        # Calculate functional connectivity
        def calculate_functional_connectivity(activity, method='correlation'):
            """Calculate functional connectivity matrix"""
            if method == 'correlation':
                # Pearson correlation
                functional_matrix = np.corrcoef(activity.T)
            elif method == 'coherence':
                # Spectral coherence (simplified)
                functional_matrix = np.abs(np.corrcoef(activity.T))
            
            # Remove self-connections
            np.fill_diagonal(functional_matrix, 0)
            return functional_matrix
        
        connectivity_system.calculate_functional_connectivity = calculate_functional_connectivity
        
        # Test functional connectivity calculation
        functional_matrix = connectivity_system.calculate_functional_connectivity(neural_activity)
        connectivity_system.functional_matrix = functional_matrix
        
        # Validate functional connectivity properties
        assert functional_matrix.shape == (n_nodes, n_nodes), "FC matrix should be square"
        assert np.allclose(functional_matrix, functional_matrix.T), "Should be symmetric"
        assert np.all(np.abs(functional_matrix) <= 1), "Correlations should be in [-1, 1]"
        
        # Test structure-function relationship
        if structural_matrix is not None:
            # Extract upper triangular elements (avoid double-counting)
            triu_indices = np.triu_indices(n_nodes, k=1)
            struct_connections = structural_matrix[triu_indices]
            func_connections = np.abs(functional_matrix[triu_indices])  # Use absolute correlation
            
            # Where structure exists, function should be stronger
            structural_edges = struct_connections > 0
            functional_with_struct = func_connections[structural_edges]
            functional_without_struct = func_connections[~structural_edges]
            
            if len(functional_with_struct) > 0 and len(functional_without_struct) > 0:
                struct_func_corr = stats.pearsonr(struct_connections, func_connections)[0]
                assert struct_func_corr > 0.1, "Structure-function correlation should be positive"
                
                mean_func_with_struct = np.mean(functional_with_struct)
                mean_func_without_struct = np.mean(functional_without_struct)
                assert mean_func_with_struct > mean_func_without_struct, \
                    "Functional connectivity should be stronger where structural connections exist"
    
    def test_network_topology_analysis(self, connectivity_system):
        """Test graph-theoretic network analysis"""
        # Use pre-generated connectivity matrices
        n_nodes = 400
        structural_matrix, _ = connectivity_system.generate_dti_connectivity(n_nodes)
        
        # Mock graph analysis functions
        def analyze_network_topology(connectivity_matrix, threshold=0.1):
            """Analyze graph-theoretic properties of brain networks"""
            # Threshold matrix to create binary network
            binary_matrix = (connectivity_matrix > threshold).astype(int)
            
            # Create NetworkX graph
            G = nx.from_numpy_array(binary_matrix)
            
            # Calculate network metrics
            if nx.is_connected(G):
                # Global efficiency
                global_efficiency = nx.global_efficiency(G)
                
                # Local efficiency
                local_efficiency = nx.local_efficiency(G)
                
                # Modularity
                communities = nx.community.greedy_modularity_communities(G)
                modularity = nx.community.modularity(G, communities)
                
                # Rich club coefficient
                degree_sequence = [d for n, d in G.degree()]
                rich_club_coeff = []
                for k in range(1, max(degree_sequence)):
                    subgraph = G.subgraph([n for n, d in G.degree() if d >= k])
                    if len(subgraph.nodes()) > 1:
                        actual_edges = subgraph.number_of_edges()
                        max_possible_edges = len(subgraph.nodes()) * (len(subgraph.nodes()) - 1) / 2
                        rich_club_coeff.append(actual_edges / max_possible_edges if max_possible_edges > 0 else 0)
                
                return {
                    'global_efficiency': global_efficiency,
                    'local_efficiency': local_efficiency,
                    'modularity': modularity,
                    'num_communities': len(communities),
                    'rich_club_coefficient': np.mean(rich_club_coeff) if rich_club_coeff else 0,
                    'is_connected': True
                }
            else:
                # Handle disconnected networks
                largest_cc = max(nx.connected_components(G), key=len)
                G_largest = G.subgraph(largest_cc)
                
                return {
                    'global_efficiency': nx.global_efficiency(G_largest),
                    'local_efficiency': nx.local_efficiency(G),
                    'modularity': 0,
                    'num_communities': 0,
                    'rich_club_coefficient': 0,
                    'is_connected': False,
                    'largest_component_size': len(largest_cc) / len(G.nodes())
                }
        
        connectivity_system.analyze_topology = analyze_network_topology
        
        # Test network analysis
        topology_results = connectivity_system.analyze_topology(structural_matrix)
        
        # Validate network properties
        assert 0 <= topology_results['global_efficiency'] <= 1, "Global efficiency should be in [0,1]"
        assert 0 <= topology_results['local_efficiency'] <= 1, "Local efficiency should be in [0,1]"
        assert -1 <= topology_results['modularity'] <= 1, "Modularity should be in [-1,1]"
        
        # Brain networks should have specific properties
        if topology_results['is_connected']:
            assert topology_results['global_efficiency'] > 0.1, "Brain networks should be efficient"
            assert topology_results['modularity'] > 0.1, "Brain networks should be modular"
            assert topology_results['num_communities'] >= 5, "Should detect multiple communities"
        else:
            assert topology_results['largest_component_size'] > 0.5, "Largest component should be substantial"


class TestInteractiveBrainAtlas:
    """Test real-time 3D visualization with multi-modal data overlay"""
    
    @pytest.fixture
    def atlas_system(self):
        """Mock interactive brain atlas system"""
        system = Mock()
        system.specs = BrainMappingSpecs()
        system.atlas_loaded = True
        system.real_time_data = {}
        return system
    
    def test_3d_brain_visualization(self, atlas_system):
        """Test 3D brain model rendering and updates"""
        # Mock 3D brain mesh data
        def load_brain_atlas(atlas_name='Schaefer400'):
            """Load 3D brain atlas"""
            if atlas_name == 'Schaefer400':
                # Simulate brain parcellation
                n_regions = 400
                
                # Generate brain region coordinates (simplified)
                brain_regions = []
                for i in range(n_regions):
                    # Distribute regions across brain volume
                    hemisphere = 'left' if i < n_regions // 2 else 'right'
                    x_coord = -30 + np.random.uniform(-20, 20) if hemisphere == 'left' else 30 + np.random.uniform(-20, 20)
                    y_coord = np.random.uniform(-40, 40)
                    z_coord = np.random.uniform(-20, 60)
                    
                    region = {
                        'id': i,
                        'name': f'Region_{i:03d}',
                        'hemisphere': hemisphere,
                        'coordinates': (x_coord, y_coord, z_coord),
                        'volume': np.random.uniform(100, 2000),  # mm³
                        'vertices': np.random.randint(100, 1000)  # Mesh complexity
                    }
                    brain_regions.append(region)
                
                return {
                    'name': atlas_name,
                    'regions': brain_regions,
                    'resolution': 1.0,  # mm
                    'coordinate_system': 'MNI152'
                }
            
            return None
        
        atlas_system.load_atlas = load_brain_atlas
        
        # Test atlas loading
        atlas_data = atlas_system.load_atlas('Schaefer400')
        assert atlas_data is not None, "Should load brain atlas"
        assert len(atlas_data['regions']) == 400, "Should load 400 regions"
        assert atlas_data['resolution'] <= atlas_system.specs.atlas_resolution, "Resolution should meet specs"
        
        # Test spatial organization
        left_regions = [r for r in atlas_data['regions'] if r['hemisphere'] == 'left']
        right_regions = [r for r in atlas_data['regions'] if r['hemisphere'] == 'right']
        
        assert len(left_regions) == len(right_regions), "Should have symmetric hemispheres"
        
        # Test coordinate validity
        for region in atlas_data['regions']:
            x, y, z = region['coordinates']
            assert -100 <= x <= 100, f"X coordinate {x} outside brain bounds"
            assert -100 <= y <= 100, f"Y coordinate {y} outside brain bounds"
            assert -50 <= z <= 100, f"Z coordinate {z} outside brain bounds"
    
    @pytest.mark.asyncio
    async def test_real_time_data_overlay(self, atlas_system):
        """Test real-time multi-modal data visualization"""
        # Load atlas first
        atlas_data = atlas_system.load_atlas('Schaefer400')
        n_regions = len(atlas_data['regions'])
        
        # Mock real-time data streams
        async def generate_omp_activity():
            """Generate OMP magnetometer activity"""
            while True:
                # Simulate neural activity across brain regions
                activity = np.random.exponential(1e-12, n_regions)  # Exponential distribution
                # Add some temporal structure (oscillations)
                t = time.time()
                oscillation = 0.5 * (1 + np.sin(2 * np.pi * 10 * t))  # 10 Hz alpha
                activity *= oscillation
                
                yield {
                    'modality': 'omp',
                    'timestamp': t,
                    'values': activity,
                    'units': 'Tesla',
                    'quality': np.random.uniform(0.8, 1.0)
                }
                await asyncio.sleep(1/30)  # 30 Hz update rate
        
        async def generate_fnirs_activity():
            """Generate fNIRS hemodynamic activity"""
            while True:
                # Slower hemodynamic changes
                hbo_change = np.random.normal(0, 1e-6, n_regions)  # μM
                hbr_change = np.random.normal(0, 1e-6, n_regions)  # μM
                
                yield {
                    'modality': 'fnirs',
                    'timestamp': time.time(),
                    'hbo': hbo_change,
                    'hbr': hbr_change,
                    'units': 'μM',
                    'quality': np.random.uniform(0.7, 0.95)
                }
                await asyncio.sleep(1/10)  # 10 Hz update rate (slower than OMP)
        
        atlas_system.generate_omp_stream = generate_omp_activity
        atlas_system.generate_fnirs_stream = generate_fnirs_activity
        
        # Mock visualization update system
        def update_brain_visualization(omp_data, fnirs_data, atlas_regions):
            """Update 3D brain visualization with real-time data"""
            visualization_data = []
            
            for i, region in enumerate(atlas_regions):
                # Combine multi-modal data for visualization
                omp_intensity = omp_data['values'][i] if i < len(omp_data['values']) else 0
                fnirs_intensity = (fnirs_data['hbo'][i] - fnirs_data['hbr'][i]) if i < len(fnirs_data['hbo']) else 0
                
                # Normalize for visualization (0-1 scale)
                omp_norm = min(1.0, omp_intensity / 1e-11)  # Normalize to 10 pT scale
                fnirs_norm = min(1.0, abs(fnirs_intensity) / 1e-5)  # Normalize to 10 μM scale
                
                # Combined intensity for color mapping
                combined_intensity = 0.7 * omp_norm + 0.3 * fnirs_norm
                
                visualization_data.append({
                    'region_id': region['id'],
                    'intensity': combined_intensity,
                    'omp_component': omp_norm,
                    'fnirs_component': fnirs_norm,
                    'coordinates': region['coordinates'],
                    'update_time': max(omp_data['timestamp'], fnirs_data['timestamp'])
                })
            
            return visualization_data
        
        atlas_system.update_visualization = update_brain_visualization
        
        # Test real-time updates
        omp_stream = atlas_system.generate_omp_stream()
        fnirs_stream = atlas_system.generate_fnirs_stream()
        
        update_times = []
        visualization_updates = []
        
        # Collect several updates
        for _ in range(10):
            try:
                omp_data = await asyncio.wait_for(omp_stream.__anext__(), timeout=0.1)
                fnirs_data = await asyncio.wait_for(fnirs_stream.__anext__(), timeout=0.1)
                
                start_time = time.time()
                viz_data = atlas_system.update_visualization(omp_data, fnirs_data, atlas_data['regions'])
                update_time = time.time() - start_time
                
                update_times.append(update_time)
                visualization_updates.append(viz_data)
                
            except asyncio.TimeoutError:
                break
        
        # Validate real-time performance
        assert len(visualization_updates) > 0, "Should generate visualization updates"
        
        # Check update rate performance
        max_update_time = max(update_times) if update_times else float('inf')
        mean_update_time = np.mean(update_times) if update_times else float('inf')
        target_frame_time = 1.0 / atlas_system.specs.real_time_update_rate  # 30 Hz = 33.3ms
        
        assert max_update_time <= target_frame_time, f"Max update time {max_update_time*1000:.1f}ms exceeds {target_frame_time*1000:.1f}ms"
        assert mean_update_time <= target_frame_time * 0.5, f"Mean update time {mean_update_time*1000:.1f}ms too high"
        
        # Validate visualization data quality
        for viz_update in visualization_updates:
            assert len(viz_update) == n_regions, "Should update all brain regions"
            
            for region_data in viz_update:
                assert 0 <= region_data['intensity'] <= 1, "Intensity should be normalized to [0,1]"
                assert len(region_data['coordinates']) == 3, "Should have 3D coordinates"
    
    def test_spatial_localization_accuracy(self, atlas_system):
        """Test accuracy of spatial localization"""
        # Generate known test sources
        test_sources = [
            {'name': 'Motor_L', 'true_coords': (-30, -20, 60), 'activity': 5e-12},  # Left motor cortex
            {'name': 'Visual_R', 'true_coords': (15, -80, 10), 'activity': 3e-12},  # Right visual cortex
            {'name': 'Frontal_L', 'true_coords': (-25, 40, 30), 'activity': 2e-12}   # Left frontal
        ]
        
        # Mock source localization algorithm
        def localize_sources(sensor_data, atlas_regions, method='minimum_norm'):
            """Localize neural sources from sensor data"""
            localized_sources = []
            
            for source in test_sources:
                true_coords = source['true_coords']
                
                # Find closest atlas region
                min_distance = float('inf')
                best_region = None
                
                for region in atlas_regions:
                    region_coords = region['coordinates']
                    distance = np.sqrt(sum((a - b)**2 for a, b in zip(true_coords, region_coords)))
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_region = region
                
                # Add localization error (realistic noise)
                localization_error = np.random.normal(0, 5)  # 5mm standard deviation
                estimated_coords = tuple(c + localization_error for c in best_region['coordinates'])
                
                localized_sources.append({
                    'name': source['name'],
                    'true_coordinates': true_coords,
                    'estimated_coordinates': estimated_coords,
                    'closest_region': best_region['name'],
                    'localization_error': min_distance + abs(localization_error),
                    'confidence': max(0, 1 - min_distance / 50)  # Confidence decreases with distance
                })
            
            return localized_sources
        
        atlas_system.localize_sources = localize_sources
        
        # Load atlas
        atlas_data = atlas_system.load_atlas('Schaefer400')
        
        # Test source localization
        mock_sensor_data = np.random.normal(0, 1e-12, (306, 1000))  # 306 OMP channels
        localization_results = atlas_system.localize_sources(mock_sensor_data, atlas_data['regions'])
        
        # Validate localization accuracy
        localization_errors = [result['localization_error'] for result in localization_results]
        mean_error = np.mean(localization_errors)
        max_error = max(localization_errors)
        
        assert mean_error <= 15.0, f"Mean localization error {mean_error:.1f}mm exceeds 15mm threshold"
        assert max_error <= 30.0, f"Max localization error {max_error:.1f}mm exceeds 30mm threshold"
        
        # Check confidence scores
        confidences = [result['confidence'] for result in localization_results]
        mean_confidence = np.mean(confidences)
        
        assert mean_confidence >= atlas_system.specs.spatial_accuracy * 0.8, \
            f"Mean confidence {mean_confidence:.3f} below acceptable threshold"


class TestDigitalBrainSimulation:
    """Test individual brain pattern mapping onto neural network models"""
    
    @pytest.fixture
    def digital_twin_system(self):
        """Mock digital brain twin system"""
        system = Mock()
        system.specs = DigitalTwinSpecs()
        system.neuron_models = {}
        system.synapse_models = {}
        system.simulation_active = False
        return system
    
    def test_brian2_integration(self, digital_twin_system):
        """Test Brian2 spiking neural network integration"""
        # Mock Brian2 neuron model creation
        def create_neuron_population(n_neurons, model_type='leaky_integrate_fire'):
            """Create Brian2-style neuron population"""
            if model_type == 'leaky_integrate_fire':
                # LIF neuron parameters
                neuron_params = {
                    'tau_m': 20e-3,  # Membrane time constant (20ms)
                    'tau_ref': 2e-3,  # Refractory period (2ms)
                    'v_rest': -70e-3,  # Resting potential (-70mV)
                    'v_thresh': -55e-3,  # Threshold (-55mV)
                    'v_reset': -75e-3,  # Reset potential (-75mV)
                    'resistance': 100e6  # Membrane resistance (100MΩ)
                }
                
                # Initialize neuron states
                neurons = {
                    'id': f'neurons_{n_neurons}',
                    'count': n_neurons,
                    'model': model_type,
                    'parameters': neuron_params,
                    'state': {
                        'voltage': np.random.uniform(-75e-3, -65e-3, n_neurons),
                        'last_spike': np.full(n_neurons, -np.inf),
                        'input_current': np.zeros(n_neurons)
                    }
                }
                
                return neurons
            
            return None
        
        def create_synapse_connections(pre_neurons, post_neurons, connection_prob=0.1):
            """Create synaptic connections between neuron populations"""
            n_pre = pre_neurons['count']
            n_post = post_neurons['count']
            
            # Generate random connections
            connections = []
            for i in range(n_pre):
                for j in range(n_post):
                    if np.random.random() < connection_prob:
                        # Synaptic parameters
                        weight = np.random.uniform(0.1e-9, 2.0e-9)  # 0.1-2.0 nA
                        delay = np.random.uniform(1e-3, 5e-3)  # 1-5 ms delay
                        
                        connections.append({
                            'pre_neuron': i,
                            'post_neuron': j,
                            'weight': weight,
                            'delay': delay,
                            'type': 'excitatory' if weight > 0 else 'inhibitory'
                        })
            
            return {
                'pre_population': pre_neurons['id'],
                'post_population': post_neurons['id'],
                'connections': connections,
                'total_synapses': len(connections)
            }
        
        digital_twin_system.create_neurons = create_neuron_population
        digital_twin_system.create_synapses = create_synapse_connections
        
        # Create test neural populations
        cortical_neurons = digital_twin_system.create_neurons(80000, 'leaky_integrate_fire')  # 80% excitatory
        inhibitory_neurons = digital_twin_system.create_neurons(20000, 'leaky_integrate_fire')  # 20% inhibitory
        
        # Validate neuron populations
        assert cortical_neurons['count'] >= digital_twin_system.specs.neuron_models * 0.8, \
            "Should have sufficient excitatory neurons"
        assert inhibitory_neurons['count'] >= digital_twin_system.specs.neuron_models * 0.2, \
            "Should have sufficient inhibitory neurons"
        
        # Create synaptic connections
        excitatory_synapses = digital_twin_system.create_synapses(cortical_neurons, cortical_neurons, 0.02)
        inhibitory_synapses = digital_twin_system.create_synapses(inhibitory_neurons, cortical_neurons, 0.05)
        
        total_synapses = excitatory_synapses['total_synapses'] + inhibitory_synapses['total_synapses']
        
        # Validate synaptic connectivity
        assert total_synapses >= digital_twin_system.specs.synapse_models, \
            f"Should have at least {digital_twin_system.specs.synapse_models} synapses, got {total_synapses}"
        
        # Test excitatory/inhibitory balance
        exc_connections = len([c for c in excitatory_synapses['connections'] if c['type'] == 'excitatory'])
        inh_connections = len([c for c in inhibitory_synapses['connections'] if c['type'] == 'inhibitory'])
        
        ei_ratio = exc_connections / (inh_connections + 1)  # Avoid division by zero
        assert 2 <= ei_ratio <= 8, f"E/I ratio {ei_ratio:.1f} should be between 2-8 for realistic networks"
    
    def test_real_time_simulation_performance(self, digital_twin_system):
        """Test real-time neural simulation performance"""
        # Mock neural simulation
        def simulate_neural_dynamics(neurons, synapses, duration_ms, timestep_ms=0.1):
            """Simulate neural network dynamics"""
            start_time = time.time()
            
            n_neurons = neurons['count']
            n_timesteps = int(duration_ms / timestep_ms)
            
            # Simulation state
            voltages = neurons['state']['voltage'].copy()
            spike_times = []
            
            # Integration loop
            for t_step in range(n_timesteps):
                current_time = t_step * timestep_ms * 1e-3  # Convert to seconds
                
                # Leaky integration
                tau_m = neurons['parameters']['tau_m']
                v_rest = neurons['parameters']['v_rest']
                
                # Membrane potential decay
                voltages += (timestep_ms * 1e-3 / tau_m) * (v_rest - voltages)
                
                # Add input currents (simplified)
                input_current = np.random.normal(0, 50e-12, n_neurons)  # 50 pA noise
                voltages += (timestep_ms * 1e-3 / tau_m) * input_current * neurons['parameters']['resistance']
                
                # Check for spikes
                spike_mask = voltages >= neurons['parameters']['v_thresh']
                if np.any(spike_mask):
                    spike_neurons = np.where(spike_mask)[0]
                    for neuron_id in spike_neurons:
                        spike_times.append((current_time, neuron_id))
                    
                    # Reset spiking neurons
                    voltages[spike_mask] = neurons['parameters']['v_reset']
            
            simulation_time = time.time() - start_time
            
            return {
                'simulation_duration_ms': duration_ms,
                'wall_clock_time_s': simulation_time,
                'real_time_factor': (duration_ms * 1e-3) / simulation_time,
                'spike_count': len(spike_times),
                'firing_rate_hz': len(spike_times) / (n_neurons * duration_ms * 1e-3),
                'final_voltages': voltages
            }
        
        digital_twin_system.simulate = simulate_neural_dynamics
        
        # Create test network
        neurons = digital_twin_system.create_neurons(100000)
        synapses = digital_twin_system.create_synapses(neurons, neurons, 0.01)
        
        # Test simulation performance
        test_durations = [10, 100, 1000]  # 10ms, 100ms, 1s
        performance_results = []
        
        for duration in test_durations:
            result = digital_twin_system.simulate(neurons, synapses, duration)
            performance_results.append(result)
            
            # Validate real-time performance
            real_time_factor = result['real_time_factor']
            min_acceptable_factor = digital_twin_system.specs.real_time_factor
            
            assert real_time_factor >= min_acceptable_factor, \
                f"Real-time factor {real_time_factor:.3f} below minimum {min_acceptable_factor}"
            
            # Validate neural activity
            firing_rate = result['firing_rate_hz']
            assert 0.1 <= firing_rate <= 50, f"Firing rate {firing_rate:.1f} Hz outside physiological range"
        
        # Performance should scale reasonably with duration
        short_sim = performance_results[0]  # 10ms
        long_sim = performance_results[-1]  # 1s
        
        time_scaling = long_sim['wall_clock_time_s'] / short_sim['wall_clock_time_s']
        duration_scaling = long_sim['simulation_duration_ms'] / short_sim['simulation_duration_ms']
        
        # Time scaling should be roughly linear with duration
        scaling_efficiency = time_scaling / duration_scaling
        assert 0.5 <= scaling_efficiency <= 2.0, f"Time scaling efficiency {scaling_efficiency:.2f} indicates poor scaling"
    
    def test_brain_pattern_reproduction_fidelity(self, digital_twin_system):
        """Test fidelity of reproducing individual brain patterns"""
        # Generate target brain pattern (from real neural data)
        pattern_duration = 1.0  # 1 second
        sampling_rate = 1000  # Hz
        n_timepoints = int(pattern_duration * sampling_rate)
        
        # Create realistic brain oscillation pattern
        time_points = np.linspace(0, pattern_duration, n_timepoints)
        
        # Multi-frequency brain pattern
        alpha_rhythm = np.sin(2 * np.pi * 10 * time_points)  # 10 Hz alpha
        beta_rhythm = 0.3 * np.sin(2 * np.pi * 20 * time_points)  # 20 Hz beta
        gamma_burst = 0.1 * np.sin(2 * np.pi * 40 * time_points) * np.exp(-((time_points - 0.5) / 0.1)**2)  # 40 Hz gamma burst
        
        target_pattern = alpha_rhythm + beta_rhythm + gamma_burst + 0.1 * np.random.randn(n_timepoints)
        
        # Mock digital twin calibration
        def calibrate_digital_twin(target_pattern, neural_network):
            """Calibrate digital twin to reproduce target brain pattern"""
            # Simulate parameter optimization process
            n_iterations = 100
            calibration_errors = []
            
            for iteration in range(n_iterations):
                # Simulate network response with current parameters
                simulated_pattern = self._simulate_network_response(neural_network, len(target_pattern))
                
                # Calculate pattern similarity
                correlation = np.corrcoef(target_pattern, simulated_pattern)[0, 1]
                mse = np.mean((target_pattern - simulated_pattern)**2)
                
                # Pattern fidelity metric
                fidelity = correlation * np.exp(-mse / np.var(target_pattern))
                calibration_errors.append(1 - fidelity)
                
                # Simulate parameter updates (simplified)
                if iteration > 10:  # Allow some optimization
                    break
            
            final_fidelity = 1 - calibration_errors[-1]
            
            return {
                'target_pattern': target_pattern,
                'final_fidelity': final_fidelity,
                'calibration_iterations': len(calibration_errors),
                'final_correlation': correlation,
                'final_mse': mse
            }
        
        def _simulate_network_response(self, network, n_timepoints):
            """Simulate network response pattern"""
            # Generate realistic network activity
            base_activity = np.random.normal(0, 0.5, n_timepoints)
            
            # Add structured oscillations (simplified)
            time_points = np.linspace(0, 1, n_timepoints)
            network_alpha = 0.8 * np.sin(2 * np.pi * 10 * time_points)
            network_beta = 0.2 * np.sin(2 * np.pi * 20 * time_points)
            
            return base_activity + network_alpha + network_beta
        
        digital_twin_system.calibrate_twin = calibrate_digital_twin
        digital_twin_system._simulate_network_response = _simulate_network_response
        
        # Create neural network
        neurons = digital_twin_system.create_neurons(50000)
        
        # Test pattern reproduction
        calibration_result = digital_twin_system.calibrate_twin(target_pattern, neurons)
        
        # Validate pattern fidelity
        fidelity = calibration_result['final_fidelity']
        correlation = calibration_result['final_correlation']
        
        assert fidelity >= digital_twin_system.specs.pattern_fidelity, \
            f"Pattern fidelity {fidelity:.3f} below specification {digital_twin_system.specs.pattern_fidelity}"
        
        assert correlation >= 0.7, f"Pattern correlation {correlation:.3f} too low"
        
        # Validate frequency content preservation
        target_fft = np.abs(np.fft.fft(target_pattern))
        simulated_pattern = digital_twin_system._simulate_network_response(neurons, len(target_pattern))
        simulated_fft = np.abs(np.fft.fft(simulated_pattern))
        
        # Compare frequency spectra
        freq_correlation = np.corrcoef(target_fft[:len(target_fft)//2], simulated_fft[:len(simulated_fft)//2])[0, 1]
        assert freq_correlation >= 0.5, f"Frequency content correlation {freq_correlation:.3f} too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
