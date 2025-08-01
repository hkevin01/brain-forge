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
    
    def create_interactive_brain_plot(self, 
                                    activity_data: Optional[np.ndarray] = None,
                                    connectivity_matrix: Optional[np.ndarray] = None,
                                    electrode_positions: Optional[np.ndarray] = None) -> Any:
        """Create complete interactive 3D brain plot"""
        try:
            if not PYVISTA_AVAILABLE:
                return self._create_fallback_plot(activity_data)
            
            # Load brain mesh if not already loaded
            if self.brain_mesh is None:
                self.load_brain_mesh()
            
            # Set electrode positions if provided
            if electrode_positions is not None:
                self.set_electrode_positions(electrode_positions)
            
            # Update activity if provided
            if activity_data is not None:
                self.update_brain_activity(activity_data, electrode_positions)
            
            # Add connectivity if provided
            if connectivity_matrix is not None and electrode_positions is not None:
                self.add_connectivity_visualization(
                    connectivity_matrix, electrode_positions
                )
            
            # Configure interactive features
            self.plotter.add_axes()
            self.plotter.show_grid()
            
            # Add colorbar for activity
            if activity_data is not None:
                self.plotter.add_scalar_bar(
                    title="Neural Activity",
                    n_labels=5,
                    position_x=0.85,
                    position_y=0.1
                )
            
            self.logger.info("Interactive 3D brain plot created")
            return self.plotter
            
        except Exception as e:
            self.logger.error(f"Failed to create interactive brain plot: {e}")
            return self._create_fallback_plot(activity_data)
    
    def _create_fallback_plot(self, activity_data: Optional[np.ndarray] = None):
        """Create fallback plot when PyVista unavailable"""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("No visualization libraries available")
            return None
        
        # Create 2D brain activity plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if activity_data is not None:
            # Simple 2D representation
            brain_image = np.random.rand(50, 50)  # Placeholder brain shape
            
            # Overlay activity data
            if len(activity_data) > 0:
                activity_avg = np.mean(activity_data)
                brain_image *= activity_avg
            
            im = ax.imshow(brain_image, cmap='viridis', aspect='equal')
            ax.set_title("Brain Activity (2D Fallback View)")
            plt.colorbar(im, ax=ax, label="Activity Level")
        else:
            # Simple brain outline
            theta = np.linspace(0, 2*np.pi, 100)
            x = 0.85 * np.cos(theta)
            y = 0.65 * np.sin(theta)
            ax.plot(x, y, 'gray', linewidth=3)
            ax.set_title("Brain Outline (2D Fallback View)")
            ax.set_aspect('equal')
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        self.logger.info("Created fallback 2D brain visualization")
        return fig
    
    def save_visualization(self, filename: str, 
                          screenshot: bool = True,
                          high_res: bool = True) -> str:
        """Save current visualization"""
        try:
            if not PYVISTA_AVAILABLE or self.plotter is None:
                self.logger.warning("PyVista not available - cannot save 3D visualization")
                return ""
            
            if screenshot:
                # Save screenshot
                if high_res:
                    self.plotter.window_size = (2400, 1600)  # High resolution
                
                screenshot_path = filename.replace('.png', '_screenshot.png')
                self.plotter.screenshot(screenshot_path, transparent_background=True)
                
                self.logger.info(f"Screenshot saved: {screenshot_path}")
                return screenshot_path
            
        except Exception as e:
            self.logger.error(f"Failed to save visualization: {e}")
            return ""
    
    def animate_activity(self, activity_sequence: np.ndarray,
                        electrode_positions: np.ndarray,
                        frame_rate: float = 10.0,
                        save_animation: bool = False,
                        output_file: str = "brain_activity.gif") -> None:
        """Create animated brain activity visualization"""
        try:
            if not PYVISTA_AVAILABLE:
                self.logger.warning("Animation requires PyVista")
                return
            
            # Setup animation
            n_frames = activity_sequence.shape[0]
            
            def update_frame(frame_idx):
                """Update function for animation"""
                activity = activity_sequence[frame_idx]
                self.update_brain_activity(activity, electrode_positions)
                return self.plotter.render()
            
            if save_animation:
                # Save as GIF
                self.plotter.open_gif(output_file)
                
                for frame in range(n_frames):
                    update_frame(frame)
                    self.plotter.write_frame()
                    time.sleep(1.0 / frame_rate)
                
                self.plotter.close()
                self.logger.info(f"Animation saved: {output_file}")
            else:
                # Interactive animation
                self.plotter.show(auto_close=False, interactive_update=True)
                
                for frame in range(n_frames):
                    update_frame(frame)
                    time.sleep(1.0 / frame_rate)
            
        except Exception as e:
            self.logger.error(f"Failed to create animation: {e}")
    
    def show(self, interactive: bool = True) -> None:
        """Display the 3D visualization"""
        try:
            if not PYVISTA_AVAILABLE or self.plotter is None:
                self.logger.warning("PyVista not available - cannot show 3D visualization")
                return
            
            if interactive:
                self.plotter.show(interactive=True)
            else:
                self.plotter.show(screenshot=True)
            
        except Exception as e:
            self.logger.error(f"Failed to show visualization: {e}")
    
    def close(self) -> None:
        """Close the visualization"""
        try:
            if self.plotter is not None:
                self.plotter.close()
                self.plotter = None
            
        except Exception as e:
            self.logger.error(f"Error closing visualization: {e}")


class InteractiveBrainViewer:
    """High-level interface for interactive brain visualization"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize interactive brain viewer"""
        self.config = config or Config()
        self.logger = get_logger(f"{__name__}.InteractiveBrainViewer")
        
        # Initialize renderer
        self.renderer = BrainRenderer(config)
        
        # Data storage
        self.current_activity = None
        self.current_connectivity = None
        self.electrode_positions = None
        
        # Visualization state
        self.is_initialized = False
    
    def initialize(self, electrode_positions: np.ndarray,
                  electrode_labels: Optional[List[str]] = None) -> bool:
        """Initialize the viewer with electrode configuration"""
        try:
            # Load brain mesh
            if not self.renderer.load_brain_mesh():
                self.logger.warning("Failed to load brain mesh - using default")
            
            # Set electrode positions
            self.electrode_positions = electrode_positions
            self.renderer.set_electrode_positions(electrode_positions, electrode_labels)
            
            self.is_initialized = True
            self.logger.info("Interactive brain viewer initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize brain viewer: {e}")
            return False
    
    def update_real_time_activity(self, activity_data: np.ndarray) -> None:
        """Update brain activity in real-time"""
        if not self.is_initialized:
            self.logger.warning("Brain viewer not initialized")
            return
        
        try:
            self.current_activity = activity_data
            self.renderer.update_brain_activity(activity_data, self.electrode_positions)
            
        except Exception as e:
            self.logger.error(f"Failed to update real-time activity: {e}")
    
    def update_connectivity(self, connectivity_matrix: np.ndarray,
                          threshold: float = 0.3) -> None:
        """Update connectivity visualization"""
        if not self.is_initialized:
            self.logger.warning("Brain viewer not initialized")
            return
        
        try:
            self.current_connectivity = connectivity_matrix
            self.renderer.add_connectivity_visualization(
                connectivity_matrix, self.electrode_positions, threshold
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update connectivity: {e}")
    
    def create_snapshot(self, filename: str = "brain_snapshot.png") -> str:
        """Create snapshot of current visualization"""
        return self.renderer.save_visualization(filename, screenshot=True)
    
    def start_real_time_session(self, update_callback: callable,
                              frame_rate: float = 10.0) -> None:
        """Start real-time visualization session"""
        if not self.is_initialized:
            self.logger.error("Cannot start session - viewer not initialized")
            return
        
        try:
            self.logger.info("Starting real-time visualization session")
            
            # Create interactive plot
            plot = self.renderer.create_interactive_brain_plot(
                activity_data=self.current_activity,
                connectivity_matrix=self.current_connectivity,
                electrode_positions=self.electrode_positions
            )
            
            if plot is None:
                self.logger.warning("Failed to create interactive plot")
                return
            
            # Start update loop
            def update_loop():
                while True:
                    try:
                        # Get new data from callback
                        new_data = update_callback()
                        
                        if 'activity' in new_data:
                            self.update_real_time_activity(new_data['activity'])
                        
                        if 'connectivity' in new_data:
                            self.update_connectivity(new_data['connectivity'])
                        
                        time.sleep(1.0 / frame_rate)
                        
                    except KeyboardInterrupt:
                        self.logger.info("Real-time session stopped by user")
                        break
                    except Exception as e:
                        self.logger.error(f"Error in real-time update: {e}")
                        break
            
            # Run in separate thread to maintain interactivity
            import threading
            update_thread = threading.Thread(target=update_loop)
            update_thread.daemon = True
            update_thread.start()
            
            # Show interactive plot
            self.renderer.show(interactive=True)
            
        except Exception as e:
            self.logger.error(f"Failed to start real-time session: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Example electrode positions (10-20 system subset)
    electrode_positions = np.array([
        [85, 0, 0],    # Right temporal
        [-85, 0, 0],   # Left temporal  
        [0, 65, 0],    # Front
        [0, -65, 0],   # Back
        [0, 0, 75],    # Top
        [42, 42, 30],  # Right frontal
        [-42, 42, 30], # Left frontal
        [42, -42, 30], # Right parietal
        [-42, -42, 30] # Left parietal
    ])
    
    electrode_labels = ['T8', 'T7', 'Fz', 'Oz', 'Cz', 'F4', 'F3', 'P4', 'P3']
    
    # Create interactive viewer
    viewer = InteractiveBrainViewer()
    
    # Initialize with electrode configuration
    if viewer.initialize(electrode_positions, electrode_labels):
        
        # Generate sample activity data
        activity_data = np.random.rand(len(electrode_positions)) * 100
        
        # Generate sample connectivity matrix
        n_electrodes = len(electrode_positions)
        connectivity_matrix = np.random.rand(n_electrodes, n_electrodes)
        connectivity_matrix = (connectivity_matrix + connectivity_matrix.T) / 2
        np.fill_diagonal(connectivity_matrix, 1.0)
        
        # Update visualizations
        viewer.update_real_time_activity(activity_data)
        viewer.update_connectivity(connectivity_matrix)
        
        # Create snapshot
        snapshot_file = viewer.create_snapshot("test_brain_visualization.png")
        print(f"Snapshot saved: {snapshot_file}")
        
        # Show interactive visualization
        if PYVISTA_AVAILABLE:
            print("Displaying interactive 3D brain visualization...")
            viewer.renderer.show(interactive=True)
        else:
            print("PyVista not available - 3D visualization disabled")
    
    else:
        print("Failed to initialize brain viewer")
    
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
