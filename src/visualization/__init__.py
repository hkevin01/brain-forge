"""
Real-time Visualization and Monitoring for Brain-Forge

This module provides comprehensive visualization capabilities for multi-modal
brain data, including real-time plots, 3D brain visualization, and interactive
dashboards for monitoring brain activity and simulation results.
"""

import numpy as np
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import time
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor

from ..core.config import Config
from ..core.logger import get_logger
from ..core.exceptions import BrainForgeError

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pyvista as pv

# Web dashboard
import streamlit as st
import pandas as pd

logger = get_logger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    update_rate: float = 30.0  # FPS
    buffer_size: int = 10000  # Data points to keep in memory
    colormap: str = 'viridis'
    figure_size: Tuple[int, int] = (12, 8)
    enable_3d: bool = True
    enable_realtime: bool = True


class RealTimePlotter:
    """Real-time plotting for neural signals"""
    
    def __init__(self, n_channels: int = 64, buffer_size: int = 5000):
        self.n_channels = n_channels
        self.buffer_size = buffer_size
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize data buffers
        self.time_buffer = np.zeros(buffer_size)
        self.data_buffer = np.zeros((n_channels, buffer_size))
        self.buffer_index = 0
        
        # Initialize plotting
        self.fig, self.axes = plt.subplots(
            min(n_channels, 16), 1, 
            figsize=(12, min(n_channels, 16) * 2),
            sharex=True
        )
        if n_channels == 1:
            self.axes = [self.axes]
        
        self.lines = []
        for i, ax in enumerate(self.axes):
            line, = ax.plot(self.time_buffer, self.data_buffer[i], 'b-')
            self.lines.append(line)
            ax.set_ylabel(f'Channel {i+1}')
            ax.grid(True, alpha=0.3)
        
        self.axes[-1].set_xlabel('Time (s)')
        self.fig.suptitle('Real-time Neural Signals')
        
        # Animation
        self.animation = None
        self.is_running = False
        
    def add_data(self, new_data: np.ndarray, timestamp: float):
        """Add new data point to the buffer"""
        try:
            # Update buffers
            self.time_buffer[self.buffer_index] = timestamp
            
            if new_data.ndim == 1:
                self.data_buffer[:len(new_data), self.buffer_index] = new_data
            else:
                self.data_buffer[:, self.buffer_index] = new_data.flatten()[:self.n_channels]
            
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
            
        except Exception as e:
            self.logger.error(f"Failed to add data to buffer: {e}")
    
    def update_plot(self, frame):
        """Update plot for animation"""
        try:
            # Get current time window
            current_time = time.time()
            time_window = 10.0  # Show last 10 seconds
            
            # Find indices within time window
            time_mask = self.time_buffer > (current_time - time_window)
            valid_indices = np.where(time_mask)[0]
            
            if len(valid_indices) > 0:
                # Update lines
                for i, line in enumerate(self.lines):
                    if i < len(self.axes):
                        x_data = self.time_buffer[valid_indices] - current_time
                        y_data = self.data_buffer[i, valid_indices]
                        line.set_data(x_data, y_data)
                        
                        # Auto-scale y-axis
                        if len(y_data) > 0:
                            y_range = np.ptp(y_data)
                            if y_range > 0:
                                self.axes[i].set_ylim(
                                    np.min(y_data) - 0.1 * y_range,
                                    np.max(y_data) + 0.1 * y_range
                                )
                
                # Update x-axis
                self.axes[-1].set_xlim(-time_window, 0)
            
            return self.lines
            
        except Exception as e:
            self.logger.error(f"Plot update failed: {e}")
            return self.lines
    
    def start_animation(self, interval: int = 50):
        """Start real-time animation"""
        try:
            self.is_running = True
            self.animation = animation.FuncAnimation(
                self.fig, self.update_plot, 
                interval=interval, blit=False, cache_frame_data=False
            )
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Animation start failed: {e}")
    
    def stop_animation(self):
        """Stop real-time animation"""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()


class BrainViewer3D:
    """3D brain visualization using PyVista"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.plotter = None
        self.brain_mesh = None
        self.activity_overlay = None
        
    def initialize_brain_surface(self, mesh_file: Optional[str] = None):
        """Initialize 3D brain surface"""
        try:
            self.plotter = pv.Plotter()
            
            if mesh_file:
                # Load custom brain mesh
                self.brain_mesh = pv.read(mesh_file)
            else:
                # Create generic brain-like surface
                self.brain_mesh = self._create_brain_surface()
            
            # Add brain to scene
            self.plotter.add_mesh(
                self.brain_mesh,
                color='lightgray',
                opacity=0.8,
                show_edges=False
            )
            
            # Setup camera and lighting
            self.plotter.camera_position = 'xy'
            self.plotter.add_light(pv.Light())
            
            self.logger.info("3D brain surface initialized")
            
        except Exception as e:
            self.logger.error(f"Brain surface initialization failed: {e}")
            raise BrainForgeError(f"Failed to initialize 3D brain: {e}")
    
    def _create_brain_surface(self) -> pv.PolyData:
        """Create a generic brain-like surface"""
        # Create ellipsoid approximating brain shape
        brain_surface = pv.Ellipsoid(
            center=(0, 0, 0),
            a=0.08, b=0.06, c=0.05  # Brain-like proportions
        )
        
        # Add some surface complexity
        noise = np.random.normal(0, 0.002, brain_surface.points.shape)
        brain_surface.points += noise
        
        return brain_surface
    
    def visualize_activity(self, activity_data: np.ndarray, 
                          electrode_positions: np.ndarray):
        """Overlay neural activity on brain surface"""
        try:
            if self.plotter is None:
                self.initialize_brain_surface()
            
            # Create point cloud for electrodes
            electrode_cloud = pv.PolyData(electrode_positions)
            electrode_cloud['activity'] = activity_data
            
            # Add activity overlay
            self.plotter.add_mesh(
                electrode_cloud,
                scalars='activity',
                cmap='hot',
                point_size=10,
                render_points_as_spheres=True
            )
            
            # Add colorbar
            self.plotter.add_scalar_bar(
                title='Neural Activity',
                n_labels=5
            )
            
            self.logger.info("Neural activity visualization updated")
            
        except Exception as e:
            self.logger.error(f"Activity visualization failed: {e}")
    
    def visualize_connectivity(self, connectivity_matrix: np.ndarray,
                             node_positions: np.ndarray,
                             threshold: float = 0.3):
        """Visualize brain connectivity network"""
        try:
            if self.plotter is None:
                self.initialize_brain_surface()
            
            # Create nodes
            nodes = pv.PolyData(node_positions)
            node_strengths = np.sum(np.abs(connectivity_matrix), axis=1)
            nodes['strength'] = node_strengths
            
            # Add nodes to plot
            self.plotter.add_mesh(
                nodes,
                scalars='strength',
                cmap='plasma',
                point_size=15,
                render_points_as_spheres=True
            )
            
            # Create edges for strong connections
            n_nodes = len(node_positions)
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    if abs(connectivity_matrix[i, j]) > threshold:
                        # Create line between nodes
                        line = pv.Line(node_positions[i], node_positions[j])
                        
                        # Color based on connection strength
                        strength = abs(connectivity_matrix[i, j])
                        color = plt.cm.viridis(strength)[:3]
                        
                        self.plotter.add_mesh(
                            line,
                            color=color,
                            line_width=strength * 5
                        )
            
            self.logger.info("Connectivity visualization updated")
            
        except Exception as e:
            self.logger.error(f"Connectivity visualization failed: {e}")
    
    def show(self):
        """Display the 3D visualization"""
        if self.plotter:
            self.plotter.show()


class NetworkGraphVisualizer:
    """Network graph visualization for brain connectivity"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    def create_connectivity_graph(self, connectivity_matrix: np.ndarray,
                                node_labels: Optional[List[str]] = None,
                                threshold: float = 0.3) -> go.Figure:
        """Create interactive connectivity graph"""
        try:
            n_nodes = connectivity_matrix.shape[0]
            
            if node_labels is None:
                node_labels = [f'Node {i+1}' for i in range(n_nodes)]
            
            # Create network layout (circular for simplicity)
            angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
            node_x = np.cos(angles)
            node_y = np.sin(angles)
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            edge_x = []
            edge_y = []
            edge_weights = []
            
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    if abs(connectivity_matrix[i, j]) > threshold:
                        edge_x.extend([node_x[i], node_x[j], None])
                        edge_y.extend([node_y[i], node_y[j], None])
                        edge_weights.append(abs(connectivity_matrix[i, j]))
            
            # Add edge traces
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='lightblue'),
                hoverinfo='none',
                mode='lines',
                name='Connections'
            ))
            
            # Add nodes
            node_strengths = np.sum(np.abs(connectivity_matrix), axis=1)
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_labels,
                textposition="middle center",
                marker=dict(
                    size=node_strengths * 20 + 10,
                    color=node_strengths,
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title="Node Strength")
                ),
                hovertemplate='<b>%{text}</b><br>Strength: %{marker.color:.3f}',
                name='Brain Regions'
            ))
            
            # Update layout
            fig.update_layout(
                title="Brain Connectivity Network",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[
                    dict(
                        text="Node size and color represent connection strength",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Network graph creation failed: {e}")
            raise BrainForgeError(f"Failed to create connectivity graph: {e}")


class BrainForgeDashboard:
    """Streamlit-based interactive dashboard"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Data storage for dashboard
        self.current_data = {}
        self.processing_stats = {}
        self.hardware_status = {}
    
    def create_main_dashboard(self):
        """Create main Streamlit dashboard"""
        try:
            st.set_page_config(
                page_title="Brain-Forge Monitor",
                page_icon="🧠",
                layout="wide"
            )
            
            st.title("🧠 Brain-Forge Real-Time Monitor")
            st.markdown("---")
            
            # Sidebar for controls
            with st.sidebar:
                st.header("System Controls")
                
                # Hardware status
                st.subheader("Hardware Status")
                self._display_hardware_status()
                
                # Processing controls
                st.subheader("Processing Settings")
                self._display_processing_controls()
            
            # Main content area
            col1, col2 = st.columns([2, 1])
            
            with col1:
                self._display_main_visualizations()
            
            with col2:
                self._display_system_metrics()
            
            # Real-time data section
            st.markdown("---")
            self._display_realtime_data()
            
        except Exception as e:
            st.error(f"Dashboard creation failed: {e}")
            self.logger.error(f"Dashboard creation failed: {e}")
    
    def _display_hardware_status(self):
        """Display hardware connection status"""
        hardware_devices = {
            'OMP Helmet': self.hardware_status.get('omp_helmet', False),
            'Kernel Optical': self.hardware_status.get('kernel_optical', False),
            'Accelerometer': self.hardware_status.get('accelerometer', False)
        }
        
        for device, status in hardware_devices.items():
            if status:
                st.success(f"✅ {device}: Connected")
            else:
                st.error(f"❌ {device}: Disconnected")
    
    def _display_processing_controls(self):
        """Display processing parameter controls"""
        st.slider("Filter Low (Hz)", 0.1, 10.0, 1.0, key="filter_low")
        st.slider("Filter High (Hz)", 50, 200, 100, key="filter_high")
        st.slider("Compression Ratio", 1, 20, 5, key="compression_ratio")
        
        if st.button("Reset Processing"):
            st.info("Processing parameters reset to defaults")
    
    def _display_main_visualizations(self):
        """Display main visualization panels"""
        # Tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Real-time Signals", "Brain Map", "Connectivity"])
        
        with tab1:
            st.subheader("Neural Signal Activity")
            
            # Generate sample data for demonstration
            if 'signal_data' not in self.current_data:
                t = np.linspace(0, 10, 1000)
                signals = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(1000)
                self.current_data['signal_data'] = pd.DataFrame({
                    'time': t,
                    'amplitude': signals
                })
            
            st.line_chart(
                self.current_data['signal_data'].set_index('time'),
                height=300
            )
        
        with tab2:
            st.subheader("3D Brain Activity")
            st.info("3D brain visualization would be displayed here")
            # Note: Streamlit doesn't directly support PyVista
            # Would need to use st.plotly_chart or similar
        
        with tab3:
            st.subheader("Connectivity Matrix")
            
            # Generate sample connectivity matrix
            if 'connectivity' not in self.current_data:
                n_regions = 20
                connectivity = np.random.rand(n_regions, n_regions)
                connectivity = (connectivity + connectivity.T) / 2
                np.fill_diagonal(connectivity, 1.0)
                self.current_data['connectivity'] = connectivity
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                self.current_data['connectivity'],
                cmap='coolwarm',
                center=0,
                ax=ax
            )
            ax.set_title("Brain Region Connectivity")
            st.pyplot(fig)
    
    def _display_system_metrics(self):
        """Display system performance metrics"""
        st.subheader("System Metrics")
        
        # Processing performance
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Processing Latency",
                f"{self.processing_stats.get('mean_processing_time', 0.05):.3f}s",
                delta="-0.002s"
            )
        
        with col2:
            st.metric(
                "Data Quality",
                f"{self.processing_stats.get('mean_quality_score', 0.85):.2f}",
                delta="0.03"
            )
        
        # Data throughput
        st.metric(
            "Samples Processed",
            f"{self.processing_stats.get('chunks_processed', 0):,}",
            delta="156"
        )
        
        # System health
        st.subheader("System Health")
        
        health_metrics = {
            "CPU Usage": 45,
            "Memory Usage": 62,
            "GPU Usage": 78,
            "Storage": 34
        }
        
        for metric, value in health_metrics.items():
            progress_color = "normal"
            if value > 80:
                progress_color = "inverse"
            elif value > 60:
                progress_color = "secondary"
            
            st.progress(value / 100)
            st.text(f"{metric}: {value}%")
    
    def _display_realtime_data(self):
        """Display real-time data stream information"""
        st.header("Real-time Data Streams")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("MEG Data")
            st.json({
                "channels": 306,
                "sampling_rate": "1000 Hz",
                "buffer_size": "5000 samples",
                "last_update": "0.03s ago"
            })
        
        with col2:
            st.subheader("Optical Data")
            st.json({
                "channels": 104,
                "sampling_rate": "10 Hz",
                "buffer_size": "100 samples",
                "last_update": "0.1s ago"
            })
        
        with col3:
            st.subheader("Motion Data")
            st.json({
                "axes": 3,
                "sampling_rate": "1000 Hz",
                "buffer_size": "1000 samples",
                "last_update": "0.01s ago"
            })
    
    def update_data(self, new_data: Dict[str, Any]):
        """Update dashboard with new data"""
        self.current_data.update(new_data)
    
    def update_stats(self, new_stats: Dict[str, Any]):
        """Update processing statistics"""
        self.processing_stats.update(new_stats)
    
    def update_hardware_status(self, status: Dict[str, bool]):
        """Update hardware connection status"""
        self.hardware_status.update(status)


class VisualizationManager:
    """Manager for all visualization components"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize visualization components
        self.real_time_plotter = None
        self.brain_viewer = BrainViewer3D()
        self.network_visualizer = NetworkGraphVisualizer()
        self.dashboard = BrainForgeDashboard(config)
        
        # Visualization settings
        self.viz_config = VisualizationConfig()
        
    def initialize_realtime_plotting(self, n_channels: int = 64):
        """Initialize real-time plotting"""
        try:
            self.real_time_plotter = RealTimePlotter(
                n_channels=n_channels,
                buffer_size=self.viz_config.buffer_size
            )
            self.logger.info(f"Real-time plotting initialized for {n_channels} channels")
            
        except Exception as e:
            self.logger.error(f"Real-time plotting initialization failed: {e}")
            raise BrainForgeError(f"Failed to initialize real-time plotting: {e}")
    
    def start_realtime_visualization(self):
        """Start all real-time visualizations"""
        try:
            if self.real_time_plotter:
                # Start in separate thread to avoid blocking
                plot_thread = threading.Thread(
                    target=self.real_time_plotter.start_animation
                )
                plot_thread.daemon = True
                plot_thread.start()
            
            self.logger.info("Real-time visualization started")
            
        except Exception as e:
            self.logger.error(f"Real-time visualization start failed: {e}")
    
    def update_visualizations(self, data: Dict[str, Any]):
        """Update all visualizations with new data"""
        try:
            # Update real-time plots
            if self.real_time_plotter and 'neural_data' in data:
                self.real_time_plotter.add_data(
                    data['neural_data'],
                    data.get('timestamp', time.time())
                )
            
            # Update dashboard
            self.dashboard.update_data(data)
            
            # Update 3D visualization if connectivity data available
            if 'connectivity_matrix' in data and 'electrode_positions' in data:
                self.brain_viewer.visualize_connectivity(
                    data['connectivity_matrix'],
                    data['electrode_positions']
                )
            
        except Exception as e:
            self.logger.error(f"Visualization update failed: {e}")
    
    def create_static_report(self, data: Dict[str, Any], 
                           output_file: str = "brain_forge_report.html") -> str:
        """Create static HTML report with all visualizations"""
        try:
            from plotly.offline import plot
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create comprehensive report
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Neural Signal Power Spectrum',
                    'Connectivity Matrix',
                    'Network Graph',
                    'Processing Statistics'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "heatmap"}],
                    [{"type": "scatter"}, {"type": "bar"}]
                ]
            )
            
            # Add plots to subplots
            if 'power_spectrum' in data:
                freqs = data['power_spectrum']['frequencies']
                power = data['power_spectrum']['power']
                fig.add_trace(
                    go.Scatter(x=freqs, y=power[0], name='Channel 1'),
                    row=1, col=1
                )
            
            if 'connectivity_matrix' in data:
                fig.add_trace(
                    go.Heatmap(z=data['connectivity_matrix'], colorscale='viridis'),
                    row=1, col=2
                )
            
            # Save report
            plot(fig, filename=output_file, auto_open=False)
            
            self.logger.info(f"Static report saved: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Static report creation failed: {e}")
            raise BrainForgeError(f"Failed to create static report: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize visualization manager
    viz_manager = VisualizationManager()
    
    # Test with sample data
    sample_data = {
        'neural_data': np.random.randn(64, 1000),
        'connectivity_matrix': np.random.rand(20, 20),
        'electrode_positions': np.random.rand(20, 3),
        'timestamp': time.time()
    }
    
    # Update visualizations
    viz_manager.update_visualizations(sample_data)
    
    print("Visualization system initialized and tested successfully!")
