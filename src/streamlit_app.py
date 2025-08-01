"""
Brain-Forge Streamlit Dashboard Application

Production-ready scientific interface for Brain-Forge BCI platform.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from visualization import BrainForgeDashboard
    from core.config import Config
    from hardware.integrated_system import IntegratedSystem
    BRAIN_FORGE_AVAILABLE = True
except ImportError as e:
    BRAIN_FORGE_AVAILABLE = False
    st.error(f"Brain-Forge modules not available: {e}")


def initialize_system():
    """Initialize Brain-Forge system components"""
    if not BRAIN_FORGE_AVAILABLE:
        return None, None
    
    try:
        # Initialize configuration
        config = Config()
        
        # Initialize dashboard
        dashboard = BrainForgeDashboard(config)
        
        # Initialize hardware system (mock mode for demo)
        hardware_system = IntegratedSystem(config)
        
        return dashboard, hardware_system
    
    except Exception as e:
        st.error(f"System initialization failed: {e}")
        return None, None


def simulate_hardware_data():
    """Simulate real-time hardware data"""
    return {
        'omp_helmet': np.random.rand() > 0.1,  # 90% uptime
        'kernel_optical': np.random.rand() > 0.05,  # 95% uptime
        'accelerometer': np.random.rand() > 0.02,  # 98% uptime
    }


def simulate_neural_data():
    """Simulate neural data for demonstration"""
    n_channels = 64
    n_samples = 1000
    
    # Generate realistic neural signals
    t = np.linspace(0, 10, n_samples)
    neural_data = {}
    
    # Alpha rhythm (8-13 Hz)
    alpha = np.sin(2 * np.pi * 10 * t) * np.exp(-t/5)
    
    # Beta rhythm (13-30 Hz)
    beta = 0.5 * np.sin(2 * np.pi * 20 * t) * np.exp(-t/8)
    
    # Gamma rhythm (30-100 Hz)
    gamma = 0.2 * np.sin(2 * np.pi * 40 * t) * np.exp(-t/3)
    
    # Combine and add noise
    for i in range(n_channels):
        signal = (alpha + beta + gamma +
                  0.1 * np.random.randn(n_samples))
        neural_data[f'channel_{i+1}'] = signal
    
    neural_data['timestamp'] = t
    
    return pd.DataFrame(neural_data)


def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Brain-Forge Scientific Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'dashboard' not in st.session_state:
        dashboard, hardware_system = initialize_system()
        st.session_state.dashboard = dashboard
        st.session_state.hardware_system = hardware_system
        st.session_state.is_acquiring = False
        st.session_state.start_time = time.time()
    
    # Main header
    st.title("🧠 Brain-Forge Scientific Dashboard")
    st.markdown("**Production Brain-Computer Interface Monitoring Platform**")
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ System Controls")
        
        # Acquisition control
        if st.button("🟢 Start Acquisition" if not st.session_state.is_acquiring 
                    else "🔴 Stop Acquisition"):
            st.session_state.is_acquiring = not st.session_state.is_acquiring
            if st.session_state.is_acquiring:
                st.session_state.start_time = time.time()
        
        # System status indicator
        if st.session_state.is_acquiring:
            st.success("✅ System Active")
            duration = time.time() - st.session_state.start_time
            st.info(f"⏱️ Acquisition Time: {duration:.1f}s")
        else:
            st.warning("⏸️ System Idle")
        
        st.markdown("---")
        
        # Hardware status
        st.subheader("🔌 Hardware Status")
        hardware_status = simulate_hardware_data()
        
        for device, status in hardware_status.items():
            if status:
                st.success(f"✅ {device.replace('_', ' ').title()}")
            else:
                st.error(f"❌ {device.replace('_', ' ').title()}")
        
        st.markdown("---")
        
        # System settings
        st.subheader("⚙️ Settings")
        
        # Processing parameters
        filter_low = st.slider("Low-pass Filter (Hz)", 1.0, 50.0, 1.0)
        filter_high = st.slider("High-pass Filter (Hz)", 50.0, 200.0, 100.0)
        compression_ratio = st.slider("Compression Ratio", 1.0, 10.0, 5.0)
        
        # Visualization settings
        st.subheader("📊 Visualization")
        refresh_rate = st.selectbox("Refresh Rate", [1, 2, 5, 10], index=2)
        enable_3d = st.checkbox("Enable 3D Visualization", True)
        show_connectivity = st.checkbox("Show Connectivity", True)
    
    # Main dashboard area
    if st.session_state.dashboard and BRAIN_FORGE_AVAILABLE:
        
        # Update hardware status in dashboard
        if st.session_state.is_acquiring:
            st.session_state.dashboard.update_hardware_status(hardware_status)
        
        # Create main dashboard
        st.session_state.dashboard.create_main_dashboard()
        
        # Real-time data updates
        if st.session_state.is_acquiring:
            
            # Auto-refresh for real-time updates
            if refresh_rate > 0:
                time.sleep(1.0 / refresh_rate)
                st.rerun()
    
    else:
        # Fallback dashboard when Brain-Forge modules unavailable
        st.warning("⚠️ Brain-Forge modules not available - Running in demo mode")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Processing Latency", "67 ms", "-12 ms")
            st.metric("Signal Quality", "94.2%", "+1.8%")
        
        with col2:
            st.metric("Data Throughput", "15.2 MB/s", "+0.8 MB/s")
            st.metric("Compression Ratio", "5.2x", "+0.3x")
        
        with col3:
            st.metric("Active Channels", "306", "0")
            st.metric("Artifacts Detected", "2", "-1")
        
        # Demo neural data visualization
        if st.session_state.is_acquiring:
            st.subheader("📈 Real-time Neural Signals")
            
            neural_data = simulate_neural_data()
            
            # Display first few channels
            st.line_chart(neural_data[['timestamp', 'channel_1', 'channel_2', 
                                     'channel_3', 'channel_4']].set_index('timestamp'))
            
            # Auto-refresh
            if refresh_rate > 0:
                time.sleep(1.0 / refresh_rate)
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("**Brain-Forge BCI Platform** | Scientific Dashboard v1.0")


if __name__ == "__main__":
    main()
