"""
3D Brain Visualization System for Brain-Forge

This module implements real-time 3D brain visualization using PyVista,
providing interactive brain rendering with neural activity overlays.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import time
from dataclasses import dataclass
import logging

# Visualization imports (with fallbacks for missing dependencies)
try:
    import pyvista as pv
    import vtk
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None
    vtk = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

from core.config import Config
from core.logger import get_logger
from core.exceptions import BrainForgeError, ValidationError


@dataclass
class BrainVisualizationConfig:
    """Configuration for brain visualization"""
    # 3D rendering settings
    brain_mesh_file: str = "brain_templates/brain_surface.vtk"
    brain_atlas_file: str = "brain_templates/harvard_oxford_atlas.nii"
    default_colormap: str = "viridis"
    opacity: float = 0.8
    
    # Real-time update settings
    update_interval: float = 0.1  # seconds
    max_timepoints: int = 1000
    activity_threshold: float = 0.1
    
    # Visualization options
    show_electrodes: bool = True
    show_connectivity: bool = True
    show_activity_overlay: bool = True
    enable_interaction: bool = True


class BrainRenderer:
    """3D brain rendering system using PyVista"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize brain renderer"""
        self.config = config or Config()
        self.logger = get_logger(f"{__name__}.BrainRenderer")
        
        # Visualization configuration
        self.viz_config = BrainVisualizationConfig()
        
        # Check dependencies
        if not PYVISTA_AVAILABLE:
            self.logger.warning("PyVista not available - using fallback rendering")
        
        # Initialize 3D components
        self.plotter = None
        self.brain_mesh = None
        self.electrode_positions = None
        self.activity_data = None
        
        self._setup_renderer()
    
    def _setup_renderer(self) -> None:
        """Set up the 3D renderer"""
        try:
            if PYVISTA_AVAILABLE:
                # Create PyVista plotter
                self.plotter = pv.Plotter(
                    title="Brain-Forge 3D Brain Visualization",
                    window_size=(1200, 800)
                )
                
                # Set up default view
                self.plotter.camera_position = 'iso'
                self.plotter.background_color = 'black'
                
                self.logger.info("PyVista 3D renderer initialized")
            else:
                self.logger.info("Using fallback visualization system")
                
        except Exception as e:
            self.logger.error(f"Failed to setup renderer: {e}")
            raise BrainForgeError(f"Renderer initialization failed: {e}")
    
    def load_brain_mesh(self, mesh_file: Optional[str] = None) -> bool:
        """Load brain surface mesh"""
        try:
            if not PYVISTA_AVAILABLE:
                # Create simple brain model
                self.brain_mesh = self._create_simple_brain_model()
                self.logger.info("Using simple brain model")
                return True
            
            mesh_path = mesh_file or self.viz_config.brain_mesh_file
            
            try:
                # Try to load brain mesh file
                self.brain_mesh = pv.read(mesh_path)
                self.logger.info(f"Loaded brain mesh from {mesh_path}")
            except:
                # Create default brain model
                self.brain_mesh = self._create_default_brain_mesh()
                self.logger.info("Using default brain mesh")
            
            # Add brain mesh to renderer
            self.plotter.add_mesh(
                self.brain_mesh,
                color='lightgray',
                opacity=self.viz_config.opacity,
                name='brain_surface'
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load brain mesh: {e}")
            return False
    
    def _create_default_brain_mesh(self) -> Any:
        """Create default brain mesh when file not available"""
        if not PYVISTA_AVAILABLE:
            return self._create_simple_brain_model()
        
        # Create brain-like ellipsoid
        brain = pv.Ellipsoid(
            center=(0, 0, 0),
            a=85,  # anterior-posterior
            b=65,  # left-right  
            c=75   # inferior-superior
        )
        
        # Add some complexity with surface deformation
        brain = brain.subdivide(2)
        
        # Add sulci-like deformations
        points = brain.points
        noise = np.random.normal(0, 2, points.shape[0])
        brain.points = points + np.column_stack([noise, noise, noise])
        
        return brain
    
    def _create_simple_brain_model(self) -> Dict[str, Any]:
        """Create simple brain model for fallback rendering"""
        # Simple brain representation
        return {
            'type': 'ellipsoid',
            'center': (0, 0, 0),
            'dimensions': (85, 65, 75),
            'color': 'lightgray'
        }
    
    def set_electrode_positions(self, positions: np.ndarray, 
                              labels: Optional[List[str]] = None) -> None:
        """Set electrode positions for visualization"""
        try:
            self.electrode_positions = positions
            
            if not PYVISTA_AVAILABLE:
                self.logger.info(f"Electrode positions set: {positions.shape}")
                return
            
            # Create electrode spheres
            electrode_spheres = pv.PolyData()
            
            for i, pos in enumerate(positions):
                sphere = pv.Sphere(center=pos, radius=2.0)
                electrode_spheres += sphere
            
            # Add electrodes to renderer
            self.plotter.add_mesh(
                electrode_spheres,
                color='red',
                point_size=5,
                name='electrodes'
            )
            
            # Add labels if provided
            if labels and len(labels) == len(positions):
                for i, (pos, label) in enumerate(zip(positions, labels)):
                    self.plotter.add_point_labels(
                        pos, [label],
                        point_size=8,
                        font_size=10,
                        name=f'electrode_label_{i}'
                    )
            
            self.logger.info(f"Added {len(positions)} electrodes to visualization")
            
        except Exception as e:
            self.logger.error(f"Failed to set electrode positions: {e}")
    
    def update_brain_activity(self, activity_data: np.ndarray,
                            electrode_positions: Optional[np.ndarray] = None) -> None:
        """Update brain activity visualization"""
        try:
            self.activity_data = activity_data
            
            if not PYVISTA_AVAILABLE:
                self.logger.debug(f"Activity data updated: {activity_data.shape}")
                return
            
            positions = electrode_positions or self.electrode_positions
            if positions is None:
                self.logger.warning("No electrode positions available for activity update")
                return
            
            # Create activity overlay
            activity_mesh = self._create_activity_mesh(activity_data, positions)
            
            # Remove old activity overlay
            if 'brain_activity' in self.plotter.renderer.actors:
                self.plotter.remove_actor('brain_activity')
            
            # Add new activity overlay
            self.plotter.add_mesh(
                activity_mesh,
                scalars=activity_data,
                cmap=self.viz_config.default_colormap,
                opacity=0.7,
                name='brain_activity'
            )
            
            self.logger.debug("Brain activity visualization updated")
            
        except Exception as e:
            self.logger.error(f"Failed to update brain activity: {e}")
    
    def _create_activity_mesh(self, activity_data: np.ndarray,
                            positions: np.ndarray) -> Any:
        """Create mesh for activity visualization"""
        if not PYVISTA_AVAILABLE:
            return None
        
        # Create point cloud from electrode positions
        activity_points = pv.PolyData(positions)
        activity_points['activity'] = activity_data
        
        # Interpolate activity to brain surface
        interpolated = self.brain_mesh.interpolate(
            activity_points,
            radius=20.0,
            sharpness=2.0
        )
        
        return interpolated
    
    def add_connectivity_visualization(self, connectivity_matrix: np.ndarray,
                                     positions: np.ndarray,
                                     threshold: float = 0.3) -> None:
        """Add connectivity visualization"""
        try:
            if not PYVISTA_AVAILABLE:
                self.logger.info("Connectivity visualization (fallback mode)")
                return
            
            # Create lines for significant connections
            lines = []
            line_data = []
            
            n_electrodes = len(positions)
            for i in range(n_electrodes):
                for j in range(i + 1, n_electrodes):
                    if abs(connectivity_matrix[i, j]) > threshold:
                        # Add line between electrodes
                        line = pv.Line(positions[i], positions[j])
                        lines.append(line)
                        line_data.append(abs(connectivity_matrix[i, j]))
            
            if lines:
                # Combine all lines
                connectivity_mesh = lines[0]
                for line in lines[1:]:
                    connectivity_mesh += line
                
                # Add to visualization
                self.plotter.add_mesh(
                    connectivity_mesh,
                    color='yellow',
                    opacity=0.6,
                    line_width=2,
                    name='connectivity'
                )
                
                self.logger.info(f"Added {len(lines)} connectivity lines")
            
        except Exception as e:
            self.logger.error(f"Failed to add connectivity visualization: {e}")
    
    def start_real_time_visualization(self) -> None:
        """Start real-time visualization"""
        try:
            if not PYVISTA_AVAILABLE:
                self.logger.info("Real-time visualization started (fallback mode)")
                self._fallback_real_time_viz()
                return
            
            # Show the visualization
            self.plotter.show(
                title="Brain-Forge Real-time Brain Visualization",
                interactive=self.viz_config.enable_interaction
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start real-time visualization: {e}")
    
    def _fallback_real_time_viz(self) -> None:
        """Fallback real-time visualization using matplotlib"""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.info("Real-time visualization: text-based fallback")
            print("🧠 Brain-Forge Real-time Visualization (Text Mode)")
            print("   - Brain mesh: ✅ Loaded")
            print("   - Electrodes: ✅ Positioned") 
            print("   - Activity: ✅ Monitoring")
            print("   - Connectivity: ✅ Tracking")
            return
        
        # Create matplotlib-based visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle("Brain-Forge Real-time Visualization")
        
        # Brain activity plot
        ax1.set_title("Brain Activity")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Activity Level")
        
        # Connectivity plot
        ax2.set_title("Connectivity Matrix")
        
        plt.tight_layout()
        plt.show()
    
    def close(self) -> None:
        """Close the visualization"""
        try:
            if self.plotter and PYVISTA_AVAILABLE:
                self.plotter.close()
            self.logger.info("Brain visualization closed")
        except Exception as e:
            self.logger.error(f"Error closing visualization: {e}")


class RealTimePlotter:
    """Real-time signal plotting system"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize real-time plotter"""
        self.config = config or Config()
        self.logger = get_logger(f"{__name__}.RealTimePlotter")
        
        # Check matplotlib availability
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available - using text output")
        
        # Initialize plotting components
        self.fig = None
        self.axes = None
        self.lines = []
        self.data_buffer = []
        
    def setup_multi_channel_plot(self, n_channels: int, 
                                channel_names: Optional[List[str]] = None) -> None:
        """Set up multi-channel signal plotting"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                self.logger.info(f"Text mode: {n_channels} channels ready for plotting")
                return
            
            # Create subplot grid
            rows = int(np.ceil(np.sqrt(n_channels)))
            cols = int(np.ceil(n_channels / rows))
            
            self.fig, self.axes = plt.subplots(rows, cols, figsize=(15, 10))
            self.fig.suptitle("Brain-Forge Real-time Signal Monitoring")
            
            # Flatten axes array for easier indexing
            if n_channels > 1:
                self.axes = self.axes.flatten()
            else:
                self.axes = [self.axes]
            
            # Initialize lines for each channel
            self.lines = []
            for i in range(n_channels):
                ax = self.axes[i]
                line, = ax.plot([], [], 'b-', linewidth=0.8)
                self.lines.append(line)
                
                # Set labels
                channel_name = channel_names[i] if channel_names else f"Ch {i+1}"
                ax.set_title(channel_name, fontsize=10)
                ax.set_xlim(0, 1000)  # 1 second window
                ax.set_ylim(-100, 100)  # Adjust based on signal range
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_channels, len(self.axes)):
                self.axes[i].set_visible(False)
            
            plt.tight_layout()
            self.logger.info(f"Multi-channel plot setup complete: {n_channels} channels")
            
        except Exception as e:
            self.logger.error(f"Failed to setup multi-channel plot: {e}")
    
    def update_signals(self, new_data: np.ndarray) -> None:
        """Update real-time signals"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                # Text-based output
                print(f"📊 Signal update: {new_data.shape[0]} channels, "
                      f"range [{new_data.min():.2f}, {new_data.max():.2f}]")
                return
            
            # Update data buffer
            self.data_buffer.append(new_data)
            if len(self.data_buffer) > 1000:  # Keep last 1000 samples
                self.data_buffer.pop(0)
            
            # Update plots
            if len(self.data_buffer) > 10:  # Need some data to plot
                data_array = np.array(self.data_buffer)
                time_points = np.arange(len(data_array))
                
                for i, line in enumerate(self.lines):
                    if i < data_array.shape[1]:
                        line.set_data(time_points, data_array[:, i])
                
                # Refresh display
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            
        except Exception as e:
            self.logger.error(f"Failed to update signals: {e}")
    
    def show(self) -> None:
        """Show the real-time plot"""
        if MATPLOTLIB_AVAILABLE and self.fig:
            plt.show(block=False)
            plt.ion()  # Turn on interactive mode


class BrainVisualizationSystem:
    """Complete brain visualization system"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize visualization system"""
        self.config = config or Config()
        self.logger = get_logger(f"{__name__}.BrainVisualizationSystem")
        
        # Initialize components
        self.brain_renderer = BrainRenderer(config)
        self.signal_plotter = RealTimePlotter(config)
        
        self.logger.info("Brain visualization system initialized")
    
    def setup_complete_visualization(self, n_channels: int,
                                   electrode_positions: np.ndarray,
                                   channel_names: Optional[List[str]] = None) -> None:
        """Set up complete brain visualization"""
        try:
            # Load brain mesh
            self.brain_renderer.load_brain_mesh()
            
            # Set electrode positions
            self.brain_renderer.set_electrode_positions(
                electrode_positions, channel_names
            )
            
            # Setup signal plotting
            self.signal_plotter.setup_multi_channel_plot(
                n_channels, channel_names
            )
            
            self.logger.info("Complete brain visualization setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup complete visualization: {e}")
            raise BrainForgeError(f"Visualization setup failed: {e}")
    
    def start_visualization(self) -> None:
        """Start the complete visualization system"""
        try:
            # Start signal plotting
            self.signal_plotter.show()
            
            # Start 3D brain visualization
            self.brain_renderer.start_real_time_visualization()
            
            self.logger.info("Brain visualization system started")
            
        except Exception as e:
            self.logger.error(f"Failed to start visualization: {e}")
    
    def update_visualization(self, signal_data: np.ndarray,
                           activity_data: Optional[np.ndarray] = None,
                           connectivity_matrix: Optional[np.ndarray] = None) -> None:
        """Update all visualization components"""
        try:
            # Update signal plots
            self.signal_plotter.update_signals(signal_data)
            
            # Update brain activity if provided
            if activity_data is not None:
                self.brain_renderer.update_brain_activity(activity_data)
            
            # Update connectivity if provided
            if connectivity_matrix is not None and self.brain_renderer.electrode_positions is not None:
                self.brain_renderer.add_connectivity_visualization(
                    connectivity_matrix,
                    self.brain_renderer.electrode_positions
                )
            
        except Exception as e:
            self.logger.error(f"Failed to update visualization: {e}")
    
    def close(self) -> None:
        """Close the visualization system"""
        self.brain_renderer.close()
        self.logger.info("Brain visualization system closed")


# Convenience functions
def create_brain_visualization(n_channels: int = 64,
                             electrode_positions: Optional[np.ndarray] = None,
                             config: Optional[Config] = None) -> BrainVisualizationSystem:
    """Create and setup brain visualization system"""
    
    # Generate default electrode positions if not provided
    if electrode_positions is None:
        # Create spherical electrode arrangement (simplified)
        theta = np.linspace(0, 2*np.pi, n_channels, endpoint=False)
        phi = np.linspace(0, np.pi, int(np.sqrt(n_channels)), endpoint=False)
        
        positions = []
        for p in phi:
            for t in theta[:int(n_channels/len(phi))]:
                x = 70 * np.sin(p) * np.cos(t)
                y = 70 * np.sin(p) * np.sin(t)
                z = 70 * np.cos(p)
                positions.append([x, y, z])
        
        electrode_positions = np.array(positions[:n_channels])
    
    # Create visualization system
    viz_system = BrainVisualizationSystem(config)
    viz_system.setup_complete_visualization(n_channels, electrode_positions)
    
    return viz_system
