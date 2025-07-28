"""
Stream Manager for coordinating multi-modal brain data acquisition.

This module provides the main interface for managing multiple data streams
from various brain scanning devices with real-time synchronization.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import time
import threading
from collections import deque
import logging

from ..core.config import Config
from ..core.exceptions import StreamingError, HardwareError
from ..core.logger import get_logger, log_hardware_event, log_performance
from .opm_helmet import OPMHelmetInterface
from .kernel_optical import KernelOpticalInterface
from .accelerometer import AccelerometerInterface
from .synchronization import DataSynchronizer

logger = get_logger(__name__)


@dataclass
class StreamInfo:
    """Information about a data stream."""
    device_id: str
    device_type: str
    sampling_rate: int
    channels: int
    data_type: str
    is_active: bool = False
    last_timestamp: Optional[float] = None
    total_samples: int = 0


class StreamManager:
    """
    Manages multiple concurrent data streams from brain scanning devices.
    
    Coordinates data acquisition from OPM helmets, Kernel optical helmets,
    accelerometers, and other devices with real-time synchronization.
    """
    
    def __init__(self, config: Config):
        """
        Initialize stream manager.
        
        Args:
            config: System configuration object
        """
        self.config = config
        self.devices = {}
        self.streams = {}
        self.synchronizer = DataSynchronizer()
        self.is_acquiring = False
        self.acquisition_task = None
        
        # Stream buffers for real-time data
        self.stream_buffers = {}
        self.buffer_size = 10000  # samples per buffer
        
        # Event callbacks
        self.data_callbacks = []
        self.status_callbacks = []
        
        logger.info("Stream manager initialized")
        
    def add_device(self, device_type: str, device_config: Dict[str, Any]) -> str:
        """
        Add a device to the stream manager.
        
        Args:
            device_type: Type of device ('opm_helmet', 'kernel_optical', 'accelerometer')
            device_config: Device-specific configuration
            
        Returns:
            Device ID for the added device
        """
        try:
            device_id = device_config.get('device_id', f"{device_type}_{len(self.devices)}")
            
            # Create device interface based on type
            if device_type == 'omp_helmet':
                device = OPMHelmetInterface(device_config)
            elif device_type == 'kernel_optical':
                device = KernelOpticalInterface(device_config)
            elif device_type == 'accelerometer':
                device = AccelerometerInterface(device_config)
            else:
                raise StreamingError(f"Unknown device type: {device_type}")
                
            self.devices[device_id] = device
            
            # Initialize stream buffer
            self.stream_buffers[device_id] = deque(maxlen=self.buffer_size)
            
            # Create stream info
            self.streams[device_id] = StreamInfo(
                device_id=device_id,
                device_type=device_type,
                sampling_rate=device_config.get('sampling_rate', 1000),
                channels=device_config.get('channels', 1),
                data_type='float32'
            )
            
            log_hardware_event(device_type.upper(), 'DEVICE_ADDED', 
                             {'device_id': device_id})
            logger.info(f"Added {device_type} device: {device_id}")
            
            return device_id
            
        except Exception as e:
            raise StreamingError(f"Failed to add device: {str(e)}")
            
    async def connect_all_devices(self):
        """Connect to all registered devices."""
        connection_tasks = []
        
        for device_id, device in self.devices.items():
            if hasattr(device, 'connect'):
                connection_tasks.append(self._connect_device(device_id, device))
                
        if connection_tasks:
            await asyncio.gather(*connection_tasks, return_exceptions=True)
            
        connected_devices = [
            device_id for device_id, device in self.devices.items() 
            if getattr(device, 'is_connected', False)
        ]
        
        logger.info(f"Connected to {len(connected_devices)} devices: {connected_devices}")
        
    async def _connect_device(self, device_id: str, device):
        """Connect to a single device."""
        try:
            # Get connection parameters from config
            device_config = self.config.get_hardware_config(
                self.streams[device_id].device_type
            )
            
            await device.connect(**device_config.get('connection', {}))
            
            log_hardware_event(
                self.streams[device_id].device_type.upper(),
                'CONNECTED',
                {'device_id': device_id}
            )
            
        except Exception as e:
            logger.error(f"Failed to connect to device {device_id}: {str(e)}")
            
    async def calibrate_all_devices(self) -> Dict[str, Any]:
        """Calibrate all connected devices."""
        calibration_results = {}
        
        for device_id, device in self.devices.items():
            if hasattr(device, 'calibrate_sensors') and getattr(device, 'is_connected', False):
                try:
                    result = await device.calibrate_sensors()
                    calibration_results[device_id] = result
                    
                    log_hardware_event(
                        self.streams[device_id].device_type.upper(),
                        'CALIBRATED',
                        {'device_id': device_id, 'result': result}
                    )
                    
                except Exception as e:
                    logger.error(f"Calibration failed for device {device_id}: {str(e)}")
                    calibration_results[device_id] = {'error': str(e)}
                    
        logger.info(f"Calibration completed for {len(calibration_results)} devices")
        return calibration_results
        
    @log_performance
    async def start_acquisition(self):
        """Start data acquisition from all connected devices."""
        if self.is_acquiring:
            logger.warning("Acquisition already in progress")
            return
            
        try:
            # Start acquisition on all devices
            start_tasks = []
            for device_id, device in self.devices.items():
                if hasattr(device, 'start_acquisition') and getattr(device, 'is_connected', False):
                    start_tasks.append(device.start_acquisition())
                    
            if start_tasks:
                await asyncio.gather(*start_tasks, return_exceptions=True)
                
            # Start data collection task
            self.is_acquiring = True
            self.acquisition_task = asyncio.create_task(self._acquisition_loop())
            
            # Update stream status
            for stream_info in self.streams.values():
                stream_info.is_active = True
                
            log_hardware_event('STREAM_MANAGER', 'ACQUISITION_STARTED', 
                             {'active_streams': len(self.streams)})
            logger.info("Data acquisition started for all devices")
            
        except Exception as e:
            raise StreamingError(f"Failed to start acquisition: {str(e)}")
            
    async def _acquisition_loop(self):
        """Main data acquisition and synchronization loop."""
        logger.debug("Starting data acquisition loop")
        
        while self.is_acquiring:
            try:
                # Collect data from all active streams
                stream_data = {}
                
                for device_id, device in self.devices.items():
                    if self.streams[device_id].is_active:
                        # Get latest data from device
                        if hasattr(device, 'get_latest_data'):
                            data = device.get_latest_data(num_samples=10)
                            
                            if data.size > 0:
                                timestamp = time.time()
                                
                                # Add to stream buffer
                                for sample in data:
                                    self.stream_buffers[device_id].append((timestamp, sample))
                                    
                                stream_data[device_id] = {
                                    'data': data,
                                    'timestamp': timestamp,
                                    'device_type': self.streams[device_id].device_type
                                }
                                
                                # Update stream statistics
                                self.streams[device_id].last_timestamp = timestamp
                                self.streams[device_id].total_samples += len(data)
                                
                # Synchronize multi-modal data
                if stream_data:
                    synchronized_data = await self.synchronizer.synchronize_streams(stream_data)
                    
                    # Call data callbacks
                    for callback in self.data_callbacks:
                        try:
                            await callback(synchronized_data)
                        except Exception as e:
                            logger.error(f"Data callback error: {str(e)}")
                            
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)  # 10ms
                
            except Exception as e:
                logger.error(f"Error in acquisition loop: {str(e)}")
                await asyncio.sleep(0.1)
                
        logger.debug("Data acquisition loop stopped")
        
    async def stop_acquisition(self):
        """Stop data acquisition from all devices."""
        if not self.is_acquiring:
            return
            
        try:
            self.is_acquiring = False
            
            # Cancel acquisition task
            if self.acquisition_task:
                self.acquisition_task.cancel()
                try:
                    await self.acquisition_task
                except asyncio.CancelledError:
                    pass
                    
            # Stop acquisition on all devices
            stop_tasks = []
            for device_id, device in self.devices.items():
                if hasattr(device, 'stop_acquisition'):
                    stop_tasks.append(device.stop_acquisition())
                    
            if stop_tasks:
                await asyncio.gather(*stop_tasks, return_exceptions=True)
                
            # Update stream status
            for stream_info in self.streams.values():
                stream_info.is_active = False
                
            log_hardware_event('STREAM_MANAGER', 'ACQUISITION_STOPPED', 
                             {'total_samples': sum(s.total_samples for s in self.streams.values())})
            logger.info("Data acquisition stopped for all devices")
            
        except Exception as e:
            logger.error(f"Error stopping acquisition: {str(e)}")
            
    def add_data_callback(self, callback: Callable):
        """
        Add callback function for real-time data processing.
        
        Args:
            callback: Async function to call with synchronized data
        """
        self.data_callbacks.append(callback)
        logger.debug(f"Added data callback: {callback.__name__}")
        
    def remove_data_callback(self, callback: Callable):
        """Remove data callback function."""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
            logger.debug(f"Removed data callback: {callback.__name__}")
            
    def get_stream_data(self, device_id: str, num_samples: int = 1000) -> np.ndarray:
        """
        Get recent data from a specific stream.
        
        Args:
            device_id: ID of the device stream
            num_samples: Number of recent samples to return
            
        Returns:
            NumPy array of recent samples
        """
        if device_id not in self.stream_buffers:
            raise StreamingError(f"Unknown device: {device_id}")
            
        buffer = self.stream_buffers[device_id]
        
        if not buffer:
            return np.array([])
            
        # Get recent samples
        recent_samples = list(buffer)[-num_samples:]
        
        # Extract data (ignore timestamps)
        data = [sample[1] for sample in recent_samples]
        
        return np.array(data)
        
    def get_stream_statistics(self) -> Dict[str, Any]:
        """Get statistics for all active streams."""
        stats = {}
        
        for device_id, stream_info in self.streams.items():
            buffer = self.stream_buffers[device_id]
            
            stats[device_id] = {
                'device_type': stream_info.device_type,
                'is_active': stream_info.is_active,
                'sampling_rate': stream_info.sampling_rate,
                'channels': stream_info.channels,
                'total_samples': stream_info.total_samples,
                'buffer_size': len(buffer),
                'last_timestamp': stream_info.last_timestamp
            }
            
            # Calculate data rate if we have recent data
            if buffer and len(buffer) > 1:
                recent_data = list(buffer)[-100:]  # Last 100 samples
                if len(recent_data) > 1:
                    time_span = recent_data[-1][0] - recent_data[0][0]
                    if time_span > 0:
                        stats[device_id]['actual_rate'] = len(recent_data) / time_span
                        
        return stats
        
    async def disconnect_all_devices(self):
        """Disconnect from all devices."""
        try:
            # Stop acquisition if running
            if self.is_acquiring:
                await self.stop_acquisition()
                
            # Disconnect all devices
            disconnect_tasks = []
            for device_id, device in self.devices.items():
                if hasattr(device, 'disconnect'):
                    disconnect_tasks.append(device.disconnect())
                    
            if disconnect_tasks:
                await asyncio.gather(*disconnect_tasks, return_exceptions=True)
                
            log_hardware_event('STREAM_MANAGER', 'ALL_DEVICES_DISCONNECTED', {})
            logger.info("Disconnected from all devices")
            
        except Exception as e:
            logger.error(f"Error disconnecting devices: {str(e)}")
            
    def __del__(self):
        """Cleanup on object destruction."""
        if hasattr(self, 'is_acquiring') and self.is_acquiring:
            logger.warning("StreamManager destroyed while acquisition active")
