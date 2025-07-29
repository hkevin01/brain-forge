#!/usr/bin/env python3
"""
Brain-Forge Third-Party Libraries Integration Demo

This script demonstrates the integration of real third-party libraries:
- Braindecode for deep learning EEG models
- NeuroKit2 for neurophysiological signal processing
- PyVista for 3D brain visualization
- Additional neuroscience libraries

Author: Brain-Forge Development Team
Date: January 2025
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from specialized_tools import BrainDecodeIntegration, NeuroKit2Integration, SpecializedToolsManager
from core.logger import get_logger

logger = get_logger(__name__)

def demo_braindecode_integration():
    """Demonstrate Braindecode EEG models"""
    print("\n🧠 === BRAINDECODE INTEGRATION DEMO ===")
    
    try:
        braindecode = BrainDecodeIntegration()
        
        # Test EEGNet model creation
        print("\n1. Creating EEGNet classifier...")
        eegnet_model = braindecode.create_eegnet_classifier(n_channels=64, n_classes=4)
        print(f"   Model: {eegnet_model['name']}")
        print(f"   Framework: {eegnet_model['framework']}")
        print(f"   Channels: {eegnet_model['n_channels']}")
        print(f"   Classes: {eegnet_model['n_classes']}")
        print(f"   Ready for training: {eegnet_model['ready_for_training']}")
        
        # Test Shallow FBCSP Network
        print("\n2. Creating Shallow FBCSP Network...")
        fbcsp_model = braindecode.create_shallow_fbcsp_net(n_channels=64)
        print(f"   Model: {fbcsp_model['name']}")
        print(f"   Framework: {fbcsp_model['framework']}")
        print(f"   Channels: {fbcsp_model['n_channels']}")
        print(f"   Classes: {fbcsp_model['n_classes']}")
        print(f"   Ready for training: {fbcsp_model['ready_for_training']}")
        
        # Check if real models were created
        if eegnet_model['framework'] == 'braindecode':
            print("   ✅ Real Braindecode models successfully created!")
            if 'model' in eegnet_model:
                print(f"   📊 EEGNet model type: {type(eegnet_model['model'])}")
        else:
            print("   ⚠️  Using simulation (Braindecode not installed)")
            print("   💡 Install with: pip install braindecode torch")
            
    except Exception as e:
        print(f"   ❌ Error in Braindecode demo: {e}")

def demo_neurokit2_integration():
    """Demonstrate NeuroKit2 signal processing"""
    print("\n💓 === NEUROKIT2 INTEGRATION DEMO ===")
    
    try:
        nk2 = NeuroKit2Integration()
        
        # Generate sample EEG data
        print("\n1. Generating sample EEG data...")
        sampling_rate = 1000.0
        duration = 5.0  # 5 seconds
        n_channels = 32
        n_samples = int(sampling_rate * duration)
        
        # Create realistic EEG-like signal with multiple frequency components
        time = np.linspace(0, duration, n_samples)
        eeg_data = np.zeros((n_channels, n_samples))
        
        for ch in range(n_channels):
            # Add different frequency components
            alpha_wave = 2 * np.sin(2 * np.pi * 10 * time)  # 10 Hz alpha
            beta_wave = 1 * np.sin(2 * np.pi * 20 * time + ch * 0.1)  # 20 Hz beta
            noise = 0.5 * np.random.randn(n_samples)
            eeg_data[ch, :] = alpha_wave + beta_wave + noise
        
        print(f"   Generated EEG data: {eeg_data.shape} ({n_channels} channels, {n_samples} samples)")
        
        # Test EEG processing
        print("\n2. Processing EEG with NeuroKit2...")
        processed = nk2.process_eeg_signals(eeg_data, sampling_rate=sampling_rate)
        
        print(f"   Framework: {processed['framework']}")
        print(f"   Cleaned EEG shape: {processed['cleaned_eeg'].shape}")
        print(f"   Quality score: {processed['quality_score']:.3f}")
        
        print("\n3. Power band analysis:")
        for band, power in processed['power_bands'].items():
            print(f"   {band.capitalize():6s}: {power:.6f}")
        
        # Check if real NeuroKit2 was used
        if processed['framework'] == 'neurokit2':
            print("   ✅ Real NeuroKit2 processing successfully completed!")
        else:
            print("   ⚠️  Using simulation (NeuroKit2 not installed)")
            print("   💡 Install with: pip install neurokit2")
            
    except Exception as e:
        print(f"   ❌ Error in NeuroKit2 demo: {e}")

def demo_visualization_libraries():
    """Demonstrate 3D visualization capabilities"""
    print("\n🎨 === 3D VISUALIZATION DEMO ===")
    
    # Test PyVista availability
    print("\n1. Testing PyVista (3D brain visualization)...")
    try:
        import pyvista as pv
        print("   ✅ PyVista available!")
        
        # Create simple brain-like mesh
        sphere = pv.Sphere(radius=1.0, center=(0, 0, 0))
        print(f"   Created brain mesh: {sphere.n_points} points, {sphere.n_cells} cells")
        
        # Add some fake activity data
        activity = np.random.rand(sphere.n_points)
        sphere.point_data['neural_activity'] = activity
        print(f"   Added neural activity data (range: {activity.min():.3f} - {activity.max():.3f})")
        
    except ImportError:
        print("   ⚠️  PyVista not available")
        print("   💡 Install with: pip install pyvista")
    
    # Test Mayavi availability
    print("\n2. Testing Mayavi (advanced neuroimaging)...")
    try:
        import mayavi
        print("   ✅ Mayavi available!")
    except ImportError:
        print("   ⚠️  Mayavi not available")
        print("   💡 Install with: pip install mayavi")
    
    # Test Plotly availability
    print("\n3. Testing Plotly (interactive plotting)...")
    try:
        import plotly.graph_objects as go
        print("   ✅ Plotly available!")
        
        # Create sample brain connectivity plot
        n_nodes = 10
        x = np.random.randn(n_nodes)
        y = np.random.randn(n_nodes)
        z = np.random.randn(n_nodes)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=8, color='red'),
            name='Brain Regions'
        ))
        
        print(f"   Created 3D brain connectivity plot with {n_nodes} nodes")
        
    except ImportError:
        print("   ⚠️  Plotly not available")
        print("   💡 Install with: pip install plotly")

def demo_gpu_acceleration():
    """Demonstrate GPU acceleration capabilities"""
    print("\n🚀 === GPU ACCELERATION DEMO ===")
    
    # Test CuPy availability
    print("\n1. Testing CuPy (GPU arrays)...")
    try:
        import cupy as cp
        print("   ✅ CuPy available!")
        
        # Simple GPU computation
        gpu_array = cp.random.randn(1000, 1000)
        result = cp.mean(gpu_array ** 2)
        print(f"   GPU computation result: {float(result):.6f}")
        
        # Check GPU memory
        mempool = cp.get_default_memory_pool()
        print(f"   GPU memory used: {mempool.used_bytes() / 1024**2:.2f} MB")
        
    except ImportError:
        print("   ⚠️  CuPy not available")
        print("   💡 Install with: pip install cupy-cuda11x (for CUDA 11.x)")
    
    # Test PyTorch GPU
    print("\n2. Testing PyTorch GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            print(f"   ✅ PyTorch GPU available: {device}")
            
            # Simple GPU tensor operation
            gpu_tensor = torch.randn(1000, 1000, device='cuda')
            result = torch.mean(gpu_tensor ** 2)
            print(f"   GPU tensor computation: {result.item():.6f}")
        else:
            print("   ⚠️  PyTorch installed but no GPU available")
    except ImportError:
        print("   ⚠️  PyTorch not available")
        print("   💡 Install with: pip install torch torchvision")

def demo_specialized_tools_manager():
    """Demonstrate the full SpecializedToolsManager"""
    print("\n🔧 === SPECIALIZED TOOLS MANAGER DEMO ===")
    
    try:
        manager = SpecializedToolsManager()
        
        # Get capabilities
        print("\n1. Available capabilities:")
        capabilities = manager.get_capabilities()
        for category, tools in capabilities.items():
            print(f"   {category.capitalize()}:")
            for tool, info in tools.items():
                status = "✅" if info.get('available', False) else "⚠️"
                version = f" v{info.get('version', 'unknown')}" if info.get('version') else ""
                print(f"     {status} {tool}{version}")
        
        # Process some sample data
        print("\n2. Processing sample multi-modal data...")
        
        # Generate sample data
        eeg_data = np.random.randn(64, 5000)  # 64 channels, 5000 samples
        optical_data = np.random.randn(32, 5000)  # 32 optical channels
        motion_data = np.random.randn(3, 5000)  # 3-axis motion
        
        sample_data = {
            'eeg': eeg_data,
            'optical': optical_data,
            'motion': motion_data,
            'sampling_rate': 1000.0
        }
        
        results = manager.process_multimodal_data(sample_data)
        
        print(f"   Processing completed!")
        print(f"   Results keys: {list(results.keys())}")
        
        for key, result in results.items():
            if isinstance(result, dict):
                print(f"   {key}: {len(result)} items")
            elif isinstance(result, np.ndarray):
                print(f"   {key}: array shape {result.shape}")
            else:
                print(f"   {key}: {type(result)}")
                
    except Exception as e:
        print(f"   ❌ Error in SpecializedToolsManager demo: {e}")

def main():
    """Run all integration demos"""
    print("🧠 BRAIN-FORGE THIRD-PARTY LIBRARIES INTEGRATION DEMO")
    print("=" * 70)
    
    print("\nThis demo showcases the integration of specialized neuroscience libraries:")
    print("• Braindecode: Deep learning models for EEG classification")
    print("• NeuroKit2: Neurophysiological signal processing")
    print("• PyVista/Mayavi: 3D brain visualization")
    print("• CuPy/PyTorch: GPU acceleration")
    print("• MOABB: Benchmarking framework for EEG")
    
    # Run all demos
    demo_braindecode_integration()
    demo_neurokit2_integration()
    demo_visualization_libraries()
    demo_gpu_acceleration()
    demo_specialized_tools_manager()
    
    print("\n" + "=" * 70)
    print("🎉 THIRD-PARTY INTEGRATION DEMO COMPLETED!")
    
    # Summary
    print("\n📋 INTEGRATION STATUS SUMMARY:")
    print("✅ = Library available and working")
    print("⚠️  = Library not installed (using simulation)")
    print("❌ = Error occurred")
    
    print("\n💡 INSTALLATION COMMANDS:")
    print("pip install braindecode torch torchvision")
    print("pip install neurokit2") 
    print("pip install pyvista mayavi plotly")
    print("pip install cupy-cuda11x  # For NVIDIA GPUs")
    print("pip install moabb")
    
    print("\n🚀 Ready to enhance Brain-Forge with real neuroscience libraries!")

if __name__ == "__main__":
    main()
