# API Reference Documentation

## Table of Contents
- [Overview](#overview)
- [Authentication](#authentication)
- [REST API](#rest-api)
- [WebSocket API](#websocket-api)
- [Python API](#python-api)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

## Overview

Brain-Forge provides multiple APIs for different use cases:

- **REST API**: Standard HTTP endpoints for configuration and data management
- **WebSocket API**: Real-time data streaming and bidirectional communication
- **Python API**: Direct programmatic access to core functionality

### Base URLs

```
REST API:     http://localhost:8000/api/v1
WebSocket:    ws://localhost:8765
Streamlit:    http://localhost:8501
React GUI:    http://localhost:3000
```

### Content Types

```
Request:  application/json
Response: application/json
Stream:   application/octet-stream (binary data)
```

## Authentication

### API Key Authentication

```bash
# Include API key in headers
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     http://localhost:8000/api/v1/status
```

### WebSocket Authentication

```javascript
// Include token in connection
const ws = new WebSocket('ws://localhost:8765', [], {
    headers: {
        'Authorization': 'Bearer YOUR_API_KEY'
    }
});
```

### Token Management

```python
# Generate API token
from brain_forge.auth import AuthManager

auth = AuthManager()
token = auth.generate_token(user_id="researcher_001", role="researcher")
print(f"API Token: {token}")
```

## REST API

### System Management

#### Get System Status
```http
GET /api/v1/status
```

**Response**:
```json
{
    "status": "running",
    "version": "1.0.0",
    "uptime": 3600,
    "components": {
        "data_acquisition": "active",
        "signal_processing": "active",
        "visualization": "active",
        "storage": "healthy"
    },
    "hardware": {
        "omp_helmet": "connected",
        "kernel_optical": "connected",
        "accelerometer": "connected"
    },
    "performance": {
        "cpu_usage": 45.2,
        "memory_usage": 62.1,
        "processing_latency": 87
    }
}
```

#### System Configuration
```http
GET /api/v1/config
PUT /api/v1/config
```

**PUT Request Body**:
```json
{
    "data_acquisition": {
        "sampling_rate": 1000,
        "buffer_size": 10000
    },
    "signal_processing": {
        "filters": {
            "high_pass": 0.1,
            "low_pass": 100,
            "notch": 60
        }
    },
    "visualization": {
        "update_rate": 30,
        "color_map": "RdYlBu_r"
    }
}
```

### Data Management

#### List Sessions
```http
GET /api/v1/sessions
```

**Query Parameters**:
- `limit`: Number of sessions to return (default: 50)
- `offset`: Pagination offset (default: 0)
- `status`: Filter by status (active, completed, error)
- `date_from`: Start date filter (ISO 8601)
- `date_to`: End date filter (ISO 8601)

**Response**:
```json
{
    "sessions": [
        {
            "id": "session_20240101_001",
            "name": "Resting State Recording",
            "status": "completed",
            "start_time": "2024-01-01T10:00:00Z",
            "end_time": "2024-01-01T10:30:00Z",
            "duration": 1800,
            "participant_id": "P001",
            "data_size": 2147483648,
            "devices": ["omp_helmet", "kernel_optical"],
            "quality_score": 0.95
        }
    ],
    "total": 150,
    "limit": 50,
    "offset": 0
}
```

#### Create New Session
```http
POST /api/v1/sessions
```

**Request Body**:
```json
{
    "name": "Motor Task Experiment",
    "participant_id": "P002",
    "experiment_type": "motor_task",
    "duration": 1200,
    "devices": ["omp_helmet", "accelerometer"],
    "metadata": {
        "age": 25,
        "gender": "F",
        "handedness": "right"
    },
    "configuration": {
        "sampling_rate": 1000,
        "filters": {
            "high_pass": 0.1,
            "low_pass": 40
        }
    }
}
```

#### Get Session Details
```http
GET /api/v1/sessions/{session_id}
```

#### Start/Stop Session
```http
POST /api/v1/sessions/{session_id}/start
POST /api/v1/sessions/{session_id}/stop
```

### Data Export

#### Export Session Data
```http
POST /api/v1/sessions/{session_id}/export
```

**Request Body**:
```json
{
    "format": "bids",
    "data_types": ["raw", "processed", "features"],
    "compression": "gzip",
    "include_metadata": true
}
```

**Response**:
```json
{
    "export_id": "export_20240101_001",
    "status": "processing",
    "estimated_completion": "2024-01-01T11:05:00Z",
    "download_url": "/api/v1/exports/export_20240101_001/download"
}
```

#### Download Export
```http
GET /api/v1/exports/{export_id}/download
```

### Hardware Control

#### List Devices
```http
GET /api/v1/devices
```

**Response**:
```json
{
    "devices": [
        {
            "id": "omp_helmet_001",
            "type": "omp_helmet",
            "name": "OPM Helmet #1",
            "status": "connected",
            "channels": 306,
            "sampling_rate": 1000,
            "firmware_version": "2.1.3",
            "last_calibration": "2024-01-01T09:00:00Z"
        },
        {
            "id": "kernel_optical_001",
            "type": "kernel_optical",
            "name": "Kernel Flow System",
            "status": "connected",
            "channels": 64,
            "wavelengths": [760, 850],
            "ip_address": "192.168.1.100"
        }
    ]
}
```

#### Device Control
```http
POST /api/v1/devices/{device_id}/connect
POST /api/v1/devices/{device_id}/disconnect
POST /api/v1/devices/{device_id}/calibrate
GET  /api/v1/devices/{device_id}/status
```

### Signal Processing

#### Process Data
```http
POST /api/v1/processing/process
```

**Request Body**:
```json
{
    "session_id": "session_20240101_001",
    "processing_type": "real_time",
    "parameters": {
        "filters": {
            "high_pass": 0.1,
            "low_pass": 100,
            "notch": 60
        },
        "artifact_removal": {
            "method": "ica",
            "components": 20
        },
        "feature_extraction": {
            "spectral_bands": ["alpha", "beta", "gamma"],
            "connectivity": true
        }
    }
}
```

#### Get Processing Status
```http
GET /api/v1/processing/{job_id}/status
```

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onopen = function(event) {
    console.log('Connected to Brain-Forge WebSocket');

    // Authenticate
    ws.send(JSON.stringify({
        type: 'auth',
        token: 'YOUR_API_TOKEN'
    }));
};
```

### Message Types

#### Authentication
```json
{
    "type": "auth",
    "token": "YOUR_API_TOKEN"
}
```

#### Subscribe to Data Streams
```json
{
    "type": "subscribe",
    "streams": ["raw_data", "processed_data", "features"],
    "session_id": "session_20240101_001",
    "sample_rate": 30
}
```

#### Real-time Data
```json
{
    "type": "data",
    "stream": "raw_data",
    "timestamp": 1704110400000,
    "session_id": "session_20240101_001",
    "data": {
        "channels": 306,
        "samples": 100,
        "sampling_rate": 1000,
        "values": [/* binary data encoded as base64 */]
    }
}
```

#### Control Commands
```json
{
    "type": "control",
    "command": "start_recording",
    "session_id": "session_20240101_001"
}
```

#### System Events
```json
{
    "type": "event",
    "event": "device_disconnected",
    "device_id": "omp_helmet_001",
    "timestamp": 1704110400000,
    "severity": "warning"
}
```

### Python WebSocket Client

```python
import asyncio
import websockets
import json

class BrainForgeWebSocketClient:
    def __init__(self, url, token):
        self.url = url
        self.token = token
        self.websocket = None

    async def connect(self):
        self.websocket = await websockets.connect(self.url)
        await self.authenticate()

    async def authenticate(self):
        auth_message = {
            "type": "auth",
            "token": self.token
        }
        await self.websocket.send(json.dumps(auth_message))

    async def subscribe_to_data(self, streams, session_id):
        subscribe_message = {
            "type": "subscribe",
            "streams": streams,
            "session_id": session_id
        }
        await self.websocket.send(json.dumps(subscribe_message))

    async def listen(self):
        async for message in self.websocket:
            data = json.loads(message)
            await self.handle_message(data)

    async def handle_message(self, data):
        if data["type"] == "data":
            await self.process_data(data)
        elif data["type"] == "event":
            await self.handle_event(data)

# Usage
async def main():
    client = BrainForgeWebSocketClient(
        "ws://localhost:8765",
        "YOUR_API_TOKEN"
    )
    await client.connect()
    await client.subscribe_to_data(["raw_data"], "session_001")
    await client.listen()

asyncio.run(main())
```

## Python API

### Core Classes

#### BrainForge Main Interface
```python
from brain_forge import BrainForge
from brain_forge.config import Config

# Initialize with default configuration
bf = BrainForge()

# Initialize with custom configuration
config = Config.from_file("config/custom_config.yaml")
bf = BrainForge(config=config)

# System operations
status = bf.get_system_status()
bf.start_data_acquisition()
bf.stop_data_acquisition()
```

#### Data Acquisition
```python
from brain_forge.acquisition import DataAcquisitionManager
from brain_forge.hardware import IntegratedSystem

# Create acquisition manager
daq = DataAcquisitionManager()

# Configure devices
daq.add_device("omp_helmet", channels=306, sampling_rate=1000)
daq.add_device("kernel_optical", ip="192.168.1.100")

# Start acquisition
with daq.session("motor_task_001") as session:
    data = session.acquire(duration=60.0)  # 60 seconds
    print(f"Acquired {data.shape} samples")
```

#### Signal Processing
```python
from brain_forge.processing import SignalProcessor
from brain_forge.processing.filters import ButterworthFilter
from brain_forge.processing.artifacts import ICARemoval

# Create processing pipeline
processor = SignalProcessor()

# Add processing steps
processor.add_step(ButterworthFilter(
    low_cut=0.1, high_cut=100, fs=1000
))
processor.add_step(ICARemoval(n_components=20))

# Process data
processed_data = processor.process(raw_data)
```

#### Visualization
```python
from brain_forge.visualization import BrainVisualizer
from brain_forge.visualization.brain_models import load_brain_model

# Create visualizer
viz = BrainVisualizer()

# Load brain model
brain = load_brain_model("fsaverage")
viz.set_brain_model(brain)

# Visualize activity
viz.plot_activity(
    data=processed_data,
    time_point=5.0,  # seconds
    threshold=0.5,
    colormap="RdYlBu_r"
)

# Save visualization
viz.save("brain_activity.png", dpi=300)
```

#### Real-time Processing
```python
from brain_forge.realtime import RealTimeProcessor
import asyncio

class CustomProcessor(RealTimeProcessor):
    async def process_chunk(self, data_chunk):
        # Custom real-time processing
        filtered = self.apply_filters(data_chunk)
        features = self.extract_features(filtered)

        # Send to visualization
        await self.publish_data("features", features)

        return filtered

# Start real-time processing
processor = CustomProcessor(
    buffer_size=1000,
    overlap=100,
    sampling_rate=1000
)

asyncio.run(processor.start())
```

### Configuration Management

```python
from brain_forge.config import Config, DeviceConfig

# Load configuration
config = Config.from_file("config/config.yaml")

# Modify settings
config.data_acquisition.sampling_rate = 2000
config.signal_processing.filters.high_pass = 0.5

# Save configuration
config.save("config/modified_config.yaml")

# Device-specific configuration
device_config = DeviceConfig(
    device_type="omp_helmet",
    channels=306,
    sampling_rate=1000,
    port="/dev/ttyUSB0"
)
```

### Data Models

```python
from brain_forge.data import (
    Session, RawData, ProcessedData, Features
)
from datetime import datetime

# Create session
session = Session(
    id="session_20240101_001",
    name="Resting State",
    participant_id="P001",
    start_time=datetime.now(),
    devices=["omp_helmet", "accelerometer"]
)

# Raw data structure
raw_data = RawData(
    session_id=session.id,
    device_id="omp_helmet_001",
    channels=306,
    sampling_rate=1000,
    data=numpy_array,  # Shape: (channels, samples)
    timestamps=timestamp_array
)

# Processed data
processed_data = ProcessedData(
    session_id=session.id,
    raw_data_id=raw_data.id,
    processing_pipeline="standard_pipeline",
    data=processed_array,
    metadata={
        "filters_applied": ["high_pass", "low_pass"],
        "artifacts_removed": 15,
        "quality_score": 0.92
    }
)
```

## Error Handling

### HTTP Status Codes

```
200 OK - Request successful
201 Created - Resource created successfully
400 Bad Request - Invalid request parameters
401 Unauthorized - Authentication required
403 Forbidden - Insufficient permissions
404 Not Found - Resource not found
409 Conflict - Resource conflict
422 Unprocessable Entity - Validation error
429 Too Many Requests - Rate limit exceeded
500 Internal Server Error - Server error
503 Service Unavailable - Service temporarily unavailable
```

### Error Response Format

```json
{
    "error": {
        "code": "DEVICE_CONNECTION_FAILED",
        "message": "Failed to connect to OPM helmet",
        "details": {
            "device_id": "omp_helmet_001",
            "port": "/dev/ttyUSB0",
            "timestamp": "2024-01-01T10:00:00Z"
        },
        "suggestions": [
            "Check device connection",
            "Verify USB port permissions",
            "Restart device driver"
        ]
    }
}
```

### Python Exception Handling

```python
from brain_forge.exceptions import (
    BrainForgeException,
    DeviceConnectionError,
    ProcessingError,
    DataValidationError
)

try:
    bf = BrainForge()
    bf.start_acquisition()
except DeviceConnectionError as e:
    print(f"Device connection failed: {e}")
    print(f"Suggestions: {e.suggestions}")
except ProcessingError as e:
    print(f"Processing error: {e}")
    bf.reset_processing_pipeline()
except BrainForgeException as e:
    print(f"General error: {e}")
```

## Rate Limiting

### Limits

```
REST API:     100 requests/minute per API key
WebSocket:    1000 messages/minute per connection
Data Export:  5 concurrent exports per user
```

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704110460
```

## Examples

### Complete Real-time Processing Example

```python
import asyncio
from brain_forge import BrainForge
from brain_forge.realtime import RealTimeVisualization

async def real_time_demo():
    # Initialize system
    bf = BrainForge()

    # Start data acquisition
    session = bf.create_session("real_time_demo")
    await bf.start_acquisition(session.id)

    # Set up real-time visualization
    viz = RealTimeVisualization()
    viz.setup_brain_model("fsaverage")

    # Process data in real-time
    async for data_chunk in bf.stream_data(session.id):
        # Process chunk
        processed = bf.process_chunk(data_chunk)

        # Update visualization
        viz.update_activity(processed)

        # Check for events
        events = bf.detect_events(processed)
        if events:
            print(f"Detected events: {events}")

    # Cleanup
    await bf.stop_acquisition(session.id)

# Run demo
asyncio.run(real_time_demo())
```

### Batch Processing Example

```python
from brain_forge import BrainForge
from brain_forge.processing import BatchProcessor

def batch_processing_example():
    bf = BrainForge()

    # Load session data
    session = bf.load_session("session_20240101_001")
    raw_data = session.get_raw_data()

    # Create batch processor
    processor = BatchProcessor()
    processor.add_pipeline("preprocessing", [
        "butterworth_filter",
        "ica_removal",
        "bad_channel_interpolation"
    ])
    processor.add_pipeline("analysis", [
        "spectral_analysis",
        "connectivity_analysis",
        "source_localization"
    ])

    # Process data
    results = processor.process_session(session)

    # Export results
    bf.export_session(
        session.id,
        format="bids",
        include_processed=True,
        include_analysis=True
    )

batch_processing_example()
```

---

For more examples and detailed API documentation, visit our [GitHub repository](https://github.com/hkevin01/brain-forge) or [online documentation](https://brain-forge.readthedocs.io).
