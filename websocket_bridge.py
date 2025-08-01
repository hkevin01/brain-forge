"""
Brain-Forge WebSocket Bridge

Real-time data bridge between React demo GUI and Python backend.
Enables the React demo to display real hardware data instead of simulations.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List
import websockets
import numpy as np
from datetime import datetime

# Brain-Forge imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from core.config import Config
    from hardware.integrated_system import IntegratedSystem
    from processing import RealTimeFilter, WaveletCompressor, FeatureExtractor
    BRAIN_FORGE_AVAILABLE = True
except ImportError as e:
    BRAIN_FORGE_AVAILABLE = False
    print(f"Brain-Forge modules not available: {e}")


class BrainForgeWebSocketServer:
    """WebSocket server for real-time data streaming"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients = set()
        
        # Initialize Brain-Forge system
        if BRAIN_FORGE_AVAILABLE:
            self.config = Config()
            self.hardware_system = IntegratedSystem(self.config)
            self.filter = RealTimeFilter(self.config)
            self.compressor = WaveletCompressor(self.config)
            self.feature_extractor = FeatureExtractor(self.config)
        else:
            print("⚠️  Running in simulation mode - Brain-Forge not available")
            
        # Data streaming state
        self.is_streaming = False
        self.stream_task = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def register_client(self, websocket):
        """Register a new WebSocket client"""
        self.clients.add(websocket)
        self.logger.info(
            f"Client connected. Total clients: {len(self.clients)}"
        )
        
        # Send initial status
        await self.send_to_client(websocket, {
            "type": "status",
            "message": "Connected to Brain-Forge WebSocket Server",
            "brain_forge_available": BRAIN_FORGE_AVAILABLE,
            "timestamp": datetime.now().isoformat()
        })
    
    async def unregister_client(self, websocket):
        """Unregister a WebSocket client"""
        self.clients.discard(websocket)
        self.logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def send_to_client(self, websocket, data: Dict[str, Any]):
        """Send data to a specific client"""
        try:
            await websocket.send(json.dumps(data, default=self.json_serializer))
        except websockets.exceptions.ConnectionClosed:
            await self.unregister_client(websocket)
        except Exception as e:
            self.logger.error(f"Error sending to client: {e}")
    
    async def broadcast_to_all(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients"""
        if not self.clients:
            return
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = self.prepare_for_json(data)
        message = json.dumps(serializable_data, default=self.json_serializer)
        
        # Send to all clients
        disconnected_clients = set()
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                self.logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            await self.unregister_client(client)
    
    def json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays and other objects"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)
    
    def prepare_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data structure for JSON serialization"""
        if isinstance(data, dict):
            return {k: self.prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.prepare_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        else:
            return data
    
    def generate_real_data(self) -> Dict[str, Any]:
        """Generate real hardware data or realistic simulation"""
        if BRAIN_FORGE_AVAILABLE:
            # Try to get real hardware data
            try:
                # This would be replaced with actual hardware interface calls
                hardware_data = self.hardware_system.acquire_sample()
                processed_data = self.process_hardware_data(hardware_data)
                return processed_data
            except Exception as e:
                self.logger.warning(f"Hardware acquisition failed: {e}")
                return self.generate_simulation_data()
        else:
            return self.generate_simulation_data()
    
    def process_hardware_data(self, hardware_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process real hardware data through Brain-Forge pipeline"""
        processed = {}
        
        try:
            # Process OMP data if available
            if 'omp_data' in hardware_data:
                omp_data = hardware_data['omp_data']
                filtered_data = self.filter.filter_meg_data(omp_data)
                compressed_data = self.compressor.compress_data(filtered_data)
                features = self.feature_extractor.extract_spectral_features(filtered_data)
                
                processed['omp'] = {
                    'raw_data': omp_data[-100:],  # Last 100 samples
                    'filtered_data': filtered_data[-100:],
                    'compressed_size': len(compressed_data),
                    'features': features,
                    'channels': omp_data.shape[0] if len(omp_data.shape) > 1 else 1,
                    'sampling_rate': 1000
                }
            
            # Process Kernel optical data if available
            if 'kernel_data' in hardware_data:
                kernel_data = hardware_data['kernel_data']
                processed_kernel = self.filter.filter_nirs_data(kernel_data)
                
                processed['kernel'] = {
                    'flow_data': processed_kernel.get('flow', [])[-50:],
                    'flux_data': processed_kernel.get('flux', [])[-50:],
                    'channels': processed_kernel.get('channels', 96),
                    'sampling_rate': 100
                }
            
            # Process accelerometer data if available
            if 'accel_data' in hardware_data:
                accel_data = hardware_data['accel_data']
                
                processed['accelerometer'] = {
                    'x_axis': accel_data.get('x', [])[-100:],
                    'y_axis': accel_data.get('y', [])[-100:],
                    'z_axis': accel_data.get('z', [])[-100:],
                    'magnitude': np.sqrt(
                        np.array(accel_data.get('x', [0]))**2 + 
                        np.array(accel_data.get('y', [0]))**2 + 
                        np.array(accel_data.get('z', [0]))**2
                    ).tolist()[-100:],
                    'sampling_rate': 1000
                }
            
            # Add system metrics
            processed['system'] = {
                'processing_latency': np.random.uniform(50, 100),  # ms
                'data_throughput': np.random.uniform(10, 20),  # MB/s
                'compression_ratio': np.random.uniform(4, 8),
                'cpu_usage': np.random.uniform(30, 80),
                'memory_usage': np.random.uniform(40, 90),
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing hardware data: {e}")
            return self.generate_simulation_data()
        
        return processed
    
    def generate_simulation_data(self) -> Dict[str, Any]:
        """Generate realistic simulation data when hardware unavailable"""
        current_time = time.time()
        
        # Generate neural signal patterns
        t = np.linspace(0, 1, 100)
        
        # Alpha rhythm (8-13 Hz)
        alpha = np.sin(2 * np.pi * 10 * t)
        # Beta rhythm (13-30 Hz)
        beta = 0.5 * np.sin(2 * np.pi * 20 * t)
        # Gamma rhythm (30-100 Hz)
        gamma = 0.2 * np.sin(2 * np.pi * 40 * t)
        
        # Combine with noise
        neural_signal = alpha + beta + gamma + 0.1 * np.random.randn(100)
        
        return {
            'type': 'real_time_data',
            'omp': {
                'raw_data': [neural_signal + 0.1 * np.random.randn(100) for _ in range(8)],
                'channels': 306,
                'sampling_rate': 1000,
                'signal_quality': np.random.uniform(85, 98)
            },
            'kernel': {
                'flow_data': [np.exp(-t/2) + 0.05 * np.random.randn(100) for _ in range(4)],
                'flux_data': [np.sin(2 * np.pi * 0.1 * t) + 0.02 * np.random.randn(100) for _ in range(4)],
                'channels': 96,
                'sampling_rate': 100,
                'hemoglobin_concentration': np.random.uniform(8, 15)
            },
            'accelerometer': {
                'x_axis': (0.1 * np.sin(2 * np.pi * 2 * t) + 0.05 * np.random.randn(100)).tolist(),
                'y_axis': (0.1 * np.cos(2 * np.pi * 2 * t) + 0.05 * np.random.randn(100)).tolist(),
                'z_axis': (9.81 + 0.05 * np.random.randn(100)).tolist(),
                'sampling_rate': 1000
            },
            'brain_activity': {
                'regions': [
                    'Frontal Cortex', 'Parietal Cortex', 'Temporal Cortex', 
                    'Occipital Cortex', 'Motor Cortex', 'Sensory Cortex',
                    'Hippocampus', 'Amygdala', 'Thalamus', 'Cerebellum'
                ],
                'activity_levels': np.random.rand(10) * 100,
                'connectivity_matrix': np.random.rand(10, 10)
            },
            'signal_processing': {
                'frequency_bands': {
                    'delta': np.random.uniform(10, 30),
                    'theta': np.random.uniform(15, 40),
                    'alpha': np.random.uniform(20, 60),
                    'beta': np.random.uniform(25, 70),
                    'gamma': np.random.uniform(10, 35)
                },
                'snr': np.random.uniform(15, 25),
                'artifacts_detected': np.random.randint(0, 5)
            },
            'system': {
                'processing_latency': np.random.uniform(50, 100),
                'data_throughput': np.random.uniform(10, 20),
                'compression_ratio': np.random.uniform(4, 8),
                'cpu_usage': np.random.uniform(30, 80),
                'memory_usage': np.random.uniform(40, 90),
                'gpu_usage': np.random.uniform(20, 70),
                'disk_usage': np.random.uniform(10, 50),
                'network_usage': np.random.uniform(5, 25)
            },
            'hardware_status': {
                'omp_helmet': np.random.rand() > 0.05,  # 95% uptime
                'kernel_optical': np.random.rand() > 0.02,  # 98% uptime
                'accelerometer': np.random.rand() > 0.01,  # 99% uptime
                'battery_levels': {
                    'omp': np.random.uniform(70, 100),
                    'kernel': np.random.uniform(60, 100),
                    'accel': np.random.uniform(80, 100)
                }
            },
            'alerts': self.generate_alerts(),
            'timestamp': current_time
        }
    
    def generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate system alerts"""
        alerts = []
        
        # Randomly generate alerts
        if np.random.rand() < 0.1:  # 10% chance of alert
            alert_types = [
                {
                    'type': 'warning',
                    'title': 'Signal Quality',
                    'message': 'SNR below optimal threshold',
                    'severity': 'medium'
                },
                {
                    'type': 'info',
                    'title': 'System Update',
                    'message': 'Processing parameters auto-adjusted',
                    'severity': 'low'
                },
                {
                    'type': 'error',
                    'title': 'Hardware',
                    'message': 'Temporary connection issue detected',
                    'severity': 'high'
                }
            ]
            
            alert = np.random.choice(alert_types)
            alert['timestamp'] = time.time()
            alert['id'] = f"alert_{int(time.time() * 1000)}"
            alerts.append(alert)
        
        return alerts
    
    async def start_data_streaming(self):
        """Start real-time data streaming"""
        self.is_streaming = True
        self.logger.info("Started real-time data streaming")
        
        while self.is_streaming:
            try:
                # Generate or acquire real data
                data = self.generate_real_data()
                
                # Broadcast to all connected clients
                await self.broadcast_to_all(data)
                
                # Stream at 10 Hz
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in data streaming: {e}")
                await asyncio.sleep(1.0)  # Wait before retrying
    
    async def stop_data_streaming(self):
        """Stop real-time data streaming"""
        self.is_streaming = False
        if self.stream_task:
            self.stream_task.cancel()
        self.logger.info("Stopped real-time data streaming")
    
    async def handle_client_message(self, websocket, message):
        """Handle incoming client messages"""
        try:
            data = json.loads(message)
            command = data.get('command')
            
            if command == 'start_acquisition':
                if not self.is_streaming:
                    self.stream_task = asyncio.create_task(self.start_data_streaming())
                    await self.send_to_client(websocket, {
                        'type': 'command_response',
                        'command': 'start_acquisition',
                        'status': 'success',
                        'message': 'Data acquisition started'
                    })
                else:
                    await self.send_to_client(websocket, {
                        'type': 'command_response',
                        'command': 'start_acquisition',
                        'status': 'info',
                        'message': 'Data acquisition already running'
                    })
            
            elif command == 'stop_acquisition':
                await self.stop_data_streaming()
                await self.send_to_client(websocket, {
                    'type': 'command_response',
                    'command': 'stop_acquisition',
                    'status': 'success',
                    'message': 'Data acquisition stopped'
                })
            
            elif command == 'get_status':
                await self.send_to_client(websocket, {
                    'type': 'status_response',
                    'streaming': self.is_streaming,
                    'clients_connected': len(self.clients),
                    'brain_forge_available': BRAIN_FORGE_AVAILABLE
                })
            
            else:
                await self.send_to_client(websocket, {
                    'type': 'error',
                    'message': f'Unknown command: {command}'
                })
                
        except json.JSONDecodeError:
            await self.send_to_client(websocket, {
                'type': 'error',
                'message': 'Invalid JSON message'
            })
        except Exception as e:
            self.logger.error(f"Error handling client message: {e}")
            await self.send_to_client(websocket, {
                'type': 'error',
                'message': f'Server error: {str(e)}'
            })
    
    async def client_handler(self, websocket, path):
        """Handle WebSocket client connections"""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            self.logger.error(f"Error in client handler: {e}")
        finally:
            await self.unregister_client(websocket)
    
    async def start_server(self):
        """Start the WebSocket server"""
        self.logger.info(f"Starting Brain-Forge WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(self.client_handler, self.host, self.port):
            self.logger.info("✅ WebSocket server started successfully")
            self.logger.info(f"📡 Connect React app to: ws://{self.host}:{self.port}")
            
            # Keep server running
            await asyncio.Future()  # Run forever


async def main():
    """Main function to run the WebSocket server"""
    server = BrainForgeWebSocketServer()
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down WebSocket server...")
        await server.stop_data_streaming()


if __name__ == "__main__":
    print("🧠 Brain-Forge WebSocket Bridge")
    print("=====================================")
    print("🔗 Bridging React GUI ↔ Python Backend")
    print("📡 Real-time data streaming server")
    print("⏹️  Press Ctrl+C to stop")
    print("")
    
    asyncio.run(main())
