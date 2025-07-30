# Brain-Forge REST API Documentation

## Overview

The Brain-Forge REST API provides external access to the Brain-Computer Interface platform via HTTP endpoints and WebSocket connections. Built with FastAPI, it offers high-performance, async-capable brain data acquisition, processing, and streaming.

## Base URL

```
http://localhost:8000  # Default development server
```

## Authentication

Currently using development mode. Production deployments should implement OAuth2/JWT authentication.

## API Endpoints

### System Status

#### GET `/`
Get API information and capabilities.

**Response:**
```json
{
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
```

#### GET `/health`
System health check endpoint.

**Response:**
```json
{
  "success": true,
  "message": "Brain-Forge API is healthy",
  "data": {
    "acquisition_active": false,
    "processing_active": false,
    "system_status": "operational"
  },
  "timestamp": "2025-01-01T12:00:00.000Z"
}
```

### Brain Data Acquisition

#### POST `/acquisition/start`
Start brain data acquisition session.

**Request Body:**
```json
{
  "duration": 30.0,
  "channels": ["OPM_001", "OPM_002", "OPM_003"],
  "sampling_rate": 1000.0,
  "enable_compression": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Brain data acquisition started",
  "data": {
    "acquisition_id": "acq_20250101_120000",
    "duration": 30.0,
    "channels": 3,
    "sampling_rate": 1000.0,
    "estimated_data_size": "2.4 MB"
  },
  "timestamp": "2025-01-01T12:00:00.000Z"
}
```

#### POST `/acquisition/stop`
Stop current brain data acquisition.

**Response:**
```json
{
  "success": true,
  "message": "Brain data acquisition stopped", 
  "data": {
    "status": "stopped"
  },
  "timestamp": "2025-01-01T12:00:00.000Z"
}
```

### Signal Processing

#### POST `/processing/analyze`
Analyze brain data with processing pipeline.

**Request Body:**
```json
{
  "filter_low": 1.0,
  "filter_high": 100.0,
  "enable_artifact_removal": true,
  "compression_ratio": 5.0
}
```

**Response:**
```json
{
  "success": true,
  "message": "Brain data processing completed",
  "data": {
    "processing_id": "proc_20250101_120000",
    "filter_applied": "1.0-100.0 Hz", 
    "artifact_removal": true,
    "compression_ratio": 5.0,
    "processing_time": "0.087 seconds",
    "features_extracted": {
      "spectral_power": true,
      "connectivity_matrix": true,
      "spatial_patterns": true
    }
  },
  "timestamp": "2025-01-01T12:00:00.000Z"
}
```

### Transfer Learning

#### POST `/transfer_learning/extract_patterns`
Extract brain patterns for transfer learning.

**Request Body:**
```json
{
  "subject_id": "subject_001"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Brain patterns extracted successfully",
  "data": {
    "subject_id": "subject_001",
    "extraction_id": "extract_20250101_120000", 
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
  },
  "timestamp": "2025-01-01T12:00:00.000Z"
}
```

#### POST `/transfer_learning/transfer_pattern`
Transfer brain pattern between subjects.

**Request Body:**
```json
{
  "source_subject_id": "subject_001",
  "target_subject_id": "subject_002",
  "pattern_type": "motor",
  "adaptation_threshold": 0.8
}
```

**Response:**
```json
{
  "success": true,
  "message": "Pattern transfer completed successfully",
  "data": {
    "transfer_id": "transfer_20250101_120000",
    "source_subject": "subject_001",
    "target_subject": "subject_002", 
    "pattern_type": "motor",
    "transfer_accuracy": 0.89,
    "confidence_score": 0.84,
    "adaptation_successful": true
  },
  "timestamp": "2025-01-01T12:00:00.000Z"
}
```

### Visualization

#### GET `/visualization/brain_activity`
Get current brain activity for visualization.

**Response:**
```json
{
  "success": true,
  "message": "Current brain activity retrieved",
  "data": {
    "timestamp": "2025-01-01T12:00:00.000Z",
    "channels": 306,
    "activity_levels": [0.23, 0.45, 0.67, "..."],
    "connectivity_strength": 0.78,
    "dominant_frequency": "10.2 Hz",
    "brain_state": "active"
  },
  "timestamp": "2025-01-01T12:00:00.000Z"
}
```

## WebSocket Streaming

### `/ws/realtime`
Real-time brain data streaming via WebSocket.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/realtime');

ws.onopen = function(event) {
    console.log('Connected to Brain-Forge WebSocket');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received brain data:', data);
};
```

**Message Format:**
```json
{
  "timestamp": "2025-01-01T12:00:00.000Z",
  "signal_data": [[0.1, 0.2, 0.3], ["..."]],
  "quality_metrics": {
    "signal_quality": 0.95,
    "artifact_level": 0.05,
    "processing_latency": "75.2 ms"
  }
}
```

**Update Rate:** 10 Hz (every 100ms)  
**Data Format:** 64 channels × 10 samples per message

## Error Handling

All endpoints use consistent error response format:

```json
{
  "success": false,
  "message": "Error description",
  "detail": "Detailed error information",
  "timestamp": "2025-01-01T12:00:00.000Z"
}
```

**HTTP Status Codes:**
- `200` - Success
- `400` - Bad Request (invalid parameters)
- `404` - Not Found (invalid endpoint)
- `422` - Validation Error (invalid request body)
- `500` - Internal Server Error

## Data Models

### BrainDataRequest
```python
{
  "duration": float,          # Recording duration in seconds
  "channels": [str],          # List of channel names
  "sampling_rate": float,     # Sampling rate in Hz
  "enable_compression": bool  # Enable data compression
}
```

### ProcessingRequest
```python
{
  "filter_low": float,              # Low-pass filter frequency
  "filter_high": float,             # High-pass filter frequency  
  "enable_artifact_removal": bool,  # Enable artifact removal
  "compression_ratio": float        # Compression ratio (2.0-10.0)
}
```

### TransferLearningRequest
```python
{
  "source_subject_id": str,     # Source subject identifier
  "target_subject_id": str,     # Target subject identifier
  "pattern_type": str,          # Pattern type (motor, cognitive, etc.)
  "adaptation_threshold": float # Adaptation threshold (0.0-1.0)
}
```

## Python Client Example

```python
import requests
import asyncio
import websockets
import json

class BrainForgeClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self):
        """Check API health status"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
        
    def start_acquisition(self, duration=30.0, channels=None, sampling_rate=1000.0):
        """Start brain data acquisition"""
        if channels is None:
            channels = ["OPM_001", "OPM_002", "OPM_003"]
            
        data = {
            "duration": duration,
            "channels": channels,
            "sampling_rate": sampling_rate,
            "enable_compression": True
        }
        
        response = requests.post(f"{self.base_url}/acquisition/start", json=data)
        return response.json()
        
    def process_data(self, filter_low=1.0, filter_high=100.0):
        """Process brain data"""
        data = {
            "filter_low": filter_low,
            "filter_high": filter_high,
            "enable_artifact_removal": True,
            "compression_ratio": 5.0
        }
        
        response = requests.post(f"{self.base_url}/processing/analyze", json=data)
        return response.json()
        
    async def stream_realtime_data(self, callback):
        """Stream real-time brain data via WebSocket"""
        uri = f"ws://localhost:8000/ws/realtime"
        
        async with websockets.connect(uri) as websocket:
            async for message in websocket:
                data = json.loads(message)
                await callback(data)

# Usage example
client = BrainForgeClient()

# Check health
health = client.health_check()
print(f"System status: {health['data']['system_status']}")

# Start acquisition
result = client.start_acquisition(duration=10.0)
print(f"Acquisition started: {result['data']['acquisition_id']}")

# Process data
processing = client.process_data()
print(f"Processing completed: {processing['data']['processing_id']}")

# Stream real-time data
async def handle_data(data):
    print(f"Real-time data: {data['quality_metrics']['signal_quality']:.2%} quality")

asyncio.run(client.stream_realtime_data(handle_data))
```

## Performance Characteristics

- **Response Time**: <100ms for most endpoints
- **WebSocket Latency**: <50ms for real-time streaming  
- **Throughput**: 1000+ requests/second
- **Concurrent Connections**: 200+ WebSocket connections
- **Data Rate**: 100-500 MB/s streaming capability

## OpenAPI Documentation

The API provides automatic OpenAPI documentation:

- **Swagger UI**: `http://localhost:8000/docs`  
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## Deployment

### Development Server
```bash
python src/api/rest_api.py
```

### Production Server
```bash
uvicorn src.api.rest_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
```bash
docker build -t brain-forge-api .
docker run -p 8000:8000 brain-forge-api
```

## See Also

- [API Integration Demo](../../examples/api_integration_demo.py) - Complete API usage examples
- [WebSocket Client Examples](../../examples/websocket_examples/) - WebSocket integration patterns
- [Python SDK](../../sdk/python/) - Official Python client library
