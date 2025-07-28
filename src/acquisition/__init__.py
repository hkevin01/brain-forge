"""
Data Acquisition Module

Handles multi-modal brain data acquisition from various hardware devices:
- OPM helmet magnetometer arrays
- Kernel optical imaging systems  
- Accelerometer motion tracking
- Multi-device streaming and synchronization
"""

from .stream_manager import StreamManager

__all__ = [
    'StreamManager'
]

# Hardware interface status
SUPPORTED_DEVICES = {
    'omp_helmet': 'OPM Magnetometer Array (306 channels)',
    'kernel_flow': 'Kernel Flow Hemodynamic Imaging', 
    'kernel_flux': 'Kernel Flux Neuron Speed Measurement',
    'accelerometer': "Brown's Accelo-hat Motion Tracking"
}

def get_available_devices():
    """Get list of available hardware devices"""
    # Implementation would check actual hardware availability
    # For now, return supported device types
    return SUPPORTED_DEVICES

def check_device_status(device_type: str):
    """Check status of specific device type"""
    # Implementation would query actual device status
    return device_type in SUPPORTED_DEVICES
