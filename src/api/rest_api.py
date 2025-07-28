"""
Brain-Forge REST API

RESTful API interface for the Brain-Forge brain-computer interface platform.
Provides external access to brain data acquisition, processing, and analysis.
"""

from typing import Dict, List, Optional, Any
import json
import asyncio
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

# Web framework imports (with fallbacks)
try:
    from fastapi import FastAPI, HTTPException, WebSocket, Depends
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    HTTPException = None
    WebSocket = None

import numpy as np

from core.config import Config
from core.logger import get_logger
from core.exceptions import BrainForgeError


# API Data Models
class BrainDataRequest(BaseModel):
    """Request model for brain data acquisition"""
    duration: float = 10.0  # seconds
    channels: List[str] = []
    sampling_rate: float = 1000.0
    enable_compression: bool = True


class ProcessingRequest(BaseModel):
    """Request model for signal processing"""
    filter_low: float = 1.0
    filter_high: float = 100.0
    enable_artifact_removal: bool = True
    compression_ratio: float = 5.0


class TransferLearningRequest(BaseModel):
    """Request model for transfer learning operations"""
    source_subject_id: str
    target_subject_id: str
    pattern_type: str = "motor"
    adaptation_threshold: float = 0.8


@dataclass
class APIResponse:
    """Standard API response structure"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


class BrainForgeAPI:
    """Main Brain-Forge API application"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize Brain-Forge API"""
        self.config = config or Config()
        self.logger = get_logger(f"{__name__}.BrainForgeAPI")
        
        # Initialize API components
        self.app = None
        self.websocket_connections = []
        
        # Initialize Brain-Forge systems (simulated for API demo)
        self.acquisition_active = False
        self.processing_active = False
        
        if FASTAPI_AVAILABLE:
            self._setup_fastapi()
        else:
            self.logger.warning("FastAPI not available - using mock API")
            self._setup_mock_api()
    
    def _setup_fastapi(self):
        """Set up FastAPI application"""
        self.app = FastAPI(
            title="Brain-Forge API",
            description="RESTful API for Brain-Computer Interface Platform",
            version="1.0.0"
        )
        
        # Add routes
        self._add_routes()
        
        self.logger.info("FastAPI application initialized")
    
    def _setup_mock_api(self):
        """Set up mock API for demonstration"""
        self.app = MockAPI()
        self.logger.info("Mock API initialized")
    
    def _add_routes(self):
        """Add API routes"""
        
        @self.app.get("/")
        async def root():
            """API root endpoint"""
            return {
                "message": "Brain-Forge API",
                "version": "1.0.0",
                "status": "active",
                "capabilities": [
                    "multi_modal_acquisition",
                    "real_time_processing", 
                    "pattern_transfer_learning",
                    "3d_visualization"
                ]
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return APIResponse(
                success=True,
                message="Brain-Forge API is healthy",
                data={
                    "acquisition_active": self.acquisition_active,
                    "processing_active": self.processing_active,
                    "system_status": "operational"
                }
            ).__dict__
        
        @self.app.post("/acquisition/start")
        async def start_acquisition(request: BrainDataRequest):
            """Start brain data acquisition"""
            try:
                # Simulate acquisition start
                self.acquisition_active = True
                
                response_data = {
                    "acquisition_id": f"acq_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "duration": request.duration,
                    "channels": len(request.channels) if request.channels else 306,
                    "sampling_rate": request.sampling_rate,
                    "estimated_data_size": f"{request.duration * request.sampling_rate * 306 * 4 / 1024 / 1024:.1f} MB"
                }
                
                self.logger.info(f"Started acquisition: {response_data['acquisition_id']}")
                
                return APIResponse(
                    success=True,
                    message="Brain data acquisition started",
                    data=response_data
                ).__dict__
                
            except Exception as e:
                self.logger.error(f"Failed to start acquisition: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/acquisition/stop")
        async def stop_acquisition():
            """Stop brain data acquisition"""
            try:
                self.acquisition_active = False
                
                return APIResponse(
                    success=True,
                    message="Brain data acquisition stopped",
                    data={"status": "stopped"}
                ).__dict__
                
            except Exception as e:
                self.logger.error(f"Failed to stop acquisition: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/processing/analyze")
        async def analyze_data(request: ProcessingRequest):
            """Analyze brain data with processing pipeline"""
            try:
                self.processing_active = True
                
                # Simulate processing
                processing_results = {
                    "processing_id": f"proc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "filter_applied": f"{request.filter_low}-{request.filter_high} Hz",
                    "artifact_removal": request.enable_artifact_removal,
                    "compression_ratio": request.compression_ratio,
                    "processing_time": f"{np.random.uniform(0.05, 0.15):.3f} seconds",
                    "features_extracted": {
                        "spectral_power": True,
                        "connectivity_matrix": True,
                        "spatial_patterns": True
                    }
                }
                
                self.processing_active = False
                self.logger.info(f"Processing completed: {processing_results['processing_id']}")
                
                return APIResponse(
                    success=True,
                    message="Brain data processing completed",
                    data=processing_results
                ).__dict__
                
            except Exception as e:
                self.logger.error(f"Processing failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/transfer_learning/extract_patterns")
        async def extract_patterns(subject_id: str):
            """Extract brain patterns for transfer learning"""
            try:
                # Simulate pattern extraction
                patterns = {
                    "subject_id": subject_id,
                    "extraction_id": f"extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "patterns_found": {
                        "motor_patterns": 4,
                        "cognitive_patterns": 3,
                        "total_patterns": 7
                    },
                    "quality_scores": {
                        "average_quality": 0.87,
                        "min_quality": 0.74,
                        "max_quality": 0.95
                    }
                }
                
                self.logger.info(f"Patterns extracted for subject: {subject_id}")
                
                return APIResponse(
                    success=True,
                    message="Brain patterns extracted successfully",
                    data=patterns
                ).__dict__
                
            except Exception as e:
                self.logger.error(f"Pattern extraction failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/transfer_learning/transfer_pattern")
        async def transfer_pattern(request: TransferLearningRequest):
            """Transfer brain pattern between subjects"""
            try:
                # Simulate pattern transfer
                transfer_results = {
                    "transfer_id": f"transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "source_subject": request.source_subject_id,
                    "target_subject": request.target_subject_id,
                    "pattern_type": request.pattern_type,
                    "transfer_accuracy": np.random.uniform(0.8, 0.95),
                    "confidence_score": np.random.uniform(0.75, 0.9),
                    "adaptation_successful": True
                }
                
                self.logger.info(f"Pattern transfer completed: {transfer_results['transfer_id']}")
                
                return APIResponse(
                    success=True,
                    message="Pattern transfer completed successfully",
                    data=transfer_results
                ).__dict__
                
            except Exception as e:
                self.logger.error(f"Pattern transfer failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/visualization/brain_activity")
        async def get_brain_activity():
            """Get current brain activity for visualization"""
            try:
                # Simulate brain activity data
                activity_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "channels": 306,
                    "activity_levels": np.random.uniform(0, 1, 306).tolist(),
                    "connectivity_strength": np.random.uniform(0, 1),
                    "dominant_frequency": f"{np.random.uniform(8, 12):.1f} Hz",
                    "brain_state": "active"
                }
                
                return APIResponse(
                    success=True,
                    message="Current brain activity retrieved",
                    data=activity_data
                ).__dict__
                
            except Exception as e:
                self.logger.error(f"Failed to get brain activity: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws/realtime")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time data streaming"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Send real-time data
                    real_time_data = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "signal_data": np.random.randn(64, 10).tolist(),  # 64 channels, 10 samples
                        "quality_metrics": {
                            "signal_quality": np.random.uniform(0.8, 1.0),
                            "artifact_level": np.random.uniform(0, 0.2),
                            "processing_latency": f"{np.random.uniform(50, 150):.1f} ms"
                        }
                    }
                    
                    await websocket.send_text(json.dumps(real_time_data))
                    await asyncio.sleep(0.1)  # 10 Hz update rate
                    
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
            finally:
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
    
    async def broadcast_to_websockets(self, data: Dict[str, Any]):
        """Broadcast data to all connected WebSocket clients"""
        if not self.websocket_connections:
            return
        
        message = json.dumps(data)
        disconnected = []
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message)
            except:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            self.websocket_connections.remove(websocket)
    
    def get_app(self):
        """Get the API application"""
        return self.app


class MockAPI:
    """Mock API for demonstration when FastAPI is not available"""
    
    def __init__(self):
        self.routes = {}
        self.logger = get_logger(f"{__name__}.MockAPI")
    
    def get(self, path):
        def decorator(func):
            self.routes[f"GET {path}"] = func
            return func
        return decorator
    
    def post(self, path):
        def decorator(func):
            self.routes[f"POST {path}"] = func
            return func
        return decorator
    
    def websocket(self, path):
        def decorator(func):
            self.routes[f"WS {path}"] = func
            return func
        return decorator
    
    def simulate_request(self, method: str, path: str, data: Optional[Dict] = None):
        """Simulate API request for testing"""
        route_key = f"{method} {path}"
        if route_key in self.routes:
            self.logger.info(f"Mock API call: {route_key}")
            return {"status": "success", "mock": True, "data": data}
        else:
            return {"status": "error", "message": "Route not found"}


# API Application Factory
def create_brain_forge_api(config: Optional[Config] = None) -> BrainForgeAPI:
    """Create Brain-Forge API application"""
    return BrainForgeAPI(config)


# CLI for running the API server
def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the Brain-Forge API server"""
    if not FASTAPI_AVAILABLE:
        print("⚠️  FastAPI not available. Install with: pip install fastapi uvicorn")
        print("🔧 Running in mock API mode for demonstration")
        
        # Demonstrate mock API
        api = create_brain_forge_api()
        mock_api = api.get_app()
        
        print("📋 Available API endpoints (mock):")
        for route in mock_api.routes:
            print(f"   {route}")
        
        return
    
    try:
        import uvicorn
        
        api = create_brain_forge_api()
        app = api.get_app()
        
        print(f"🚀 Starting Brain-Forge API server on {host}:{port}")
        print("📋 Available endpoints:")
        print("   GET  /              - API information")
        print("   GET  /health        - Health check")
        print("   POST /acquisition/start - Start data acquisition")
        print("   POST /acquisition/stop  - Stop data acquisition")
        print("   POST /processing/analyze - Analyze brain data")
        print("   POST /transfer_learning/extract_patterns - Extract patterns")
        print("   POST /transfer_learning/transfer_pattern - Transfer patterns")
        print("   GET  /visualization/brain_activity - Get brain activity")
        print("   WS   /ws/realtime   - Real-time data stream")
        
        uvicorn.run(app, host=host, port=port)
        
    except ImportError:
        print("⚠️  uvicorn not available. Install with: pip install uvicorn")
        print("🔧 API application created but cannot start server")


if __name__ == "__main__":
    run_api_server()
