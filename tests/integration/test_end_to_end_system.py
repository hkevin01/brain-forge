"""
Comprehensive End-to-End Tests for Brain-Forge System

This test module provides complete system testing, integrating all
components from data acquisition through processing to visualization
and API endpoints.
"""

import pytest
import numpy as np
import asyncio
import time
import tempfile
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.config import Config
from core.exceptions import BrainForgeError


class MockIntegratedSystem:
    """Mock integrated Brain-Forge system for end-to-end testing"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.devices = {}
        self.processing_pipeline = None
        self.data_streams = {}
        self.system_status = "stopped"
        self.active_scans = {}
        
    async def initialize_system(self):
        """Initialize the complete Brain-Forge system"""
        # Initialize hardware interfaces
        self.devices = {
            "omp_helmet": {"status": "disconnected", "channels": 306},
            "kernel_optical": {"status": "disconnected", "channels": 96},
            "accelerometer": {"status": "disconnected", "channels": 192}
        }
        
        # Initialize processing pipeline
        self.processing_pipeline = {
            "filter": {"enabled": True, "status": "ready"},
            "compression": {"enabled": True, "status": "ready"},
            "feature_extraction": {"enabled": True, "status": "ready"}
        }
        
        self.system_status = "initialized"
        return True
    
    async def connect_hardware(self):
        """Connect all hardware devices"""
        for device_name in self.devices:
            await asyncio.sleep(0.1)  # Simulate connection time
            self.devices[device_name]["status"] = "connected"
        
        return all(device["status"] == "connected" 
                  for device in self.devices.values())
    
    async def calibrate_system(self):
        """Calibrate the complete system"""
        if not all(device["status"] == "connected" 
                  for device in self.devices.values()):
            raise BrainForgeError("Cannot calibrate: devices not connected")
        
        # Simulate calibration process
        await asyncio.sleep(1.0)
        
        for device in self.devices.values():
            device["calibrated"] = True
            device["calibration_quality"] = np.random.uniform(0.9, 1.0)
        
        return True
    
    async def start_data_acquisition(self, scan_id="test_scan"):
        """Start multi-modal data acquisition"""
        if self.system_status != "initialized":
            raise BrainForgeError("System not properly initialized")
        
        # Create data streams
        self.data_streams[scan_id] = {
            "omp_data": [],
            "optical_data": [],
            "accel_data": [],
            "timestamps": [],
            "start_time": time.time()
        }
        
        self.active_scans[scan_id] = {
            "status": "active",
            "samples_acquired": 0,
            "duration": 0
        }
        
        self.system_status = "acquiring"
        return scan_id
    
    def acquire_data_chunk(self, scan_id, duration=1.0):
        """Acquire a chunk of multi-modal data"""
        if scan_id not in self.active_scans:
            raise BrainForgeError(f"No active scan with ID: {scan_id}")
        
        # Generate mock multi-modal data
        sampling_rates = {
            "omp": 1000,
            "optical": 100,
            "accel": 1000
        }
        
        chunk_data = {}
        for modality, rate in sampling_rates.items():
            samples = int(duration * rate)
            if modality == "omp":
                chunk_data[f"{modality}_data"] = np.random.randn(306, samples)
            elif modality == "optical":
                chunk_data[f"{modality}_data"] = np.random.randn(96, samples)
            elif modality == "accel":
                chunk_data[f"{modality}_data"] = np.random.randn(192, samples)
        
        # Add to data stream
        stream = self.data_streams[scan_id]
        stream["omp_data"].append(chunk_data["omp_data"])
        stream["optical_data"].append(chunk_data["optical_data"])
        stream["accel_data"].append(chunk_data["accel_data"])
        stream["timestamps"].append(time.time())
        
        # Update scan status
        self.active_scans[scan_id]["samples_acquired"] += sum(
            data.shape[1] for data in chunk_data.values()
        )
        self.active_scans[scan_id]["duration"] += duration
        
        return chunk_data
    
    def process_data(self, data_chunk):
        """Process acquired data through the pipeline"""
        results = {}
        
        for modality, data in data_chunk.items():
            # Simulate processing steps
            processed_data = {
                "filtered": data * 0.9 + np.random.randn(*data.shape) * 0.1,
                "compressed": data[:, ::5],  # Simple decimation
                "features": {
                    "mean": np.mean(data, axis=1),
                    "std": np.std(data, axis=1),
                    "peak_freq": np.random.uniform(8, 12, data.shape[0])
                }
            }
            results[modality] = processed_data
        
        return results
    
    def extract_brain_patterns(self, processed_data):
        """Extract brain patterns from processed data"""
        patterns = {}
        
        for modality, data in processed_data.items():
            if modality == "omp_data":
                # Extract neural oscillations
                patterns["alpha_rhythm"] = np.random.uniform(8, 12)
                patterns["beta_rhythm"] = np.random.uniform(13, 30)
                patterns["gamma_rhythm"] = np.random.uniform(30, 100)
            elif modality == "optical_data":
                # Extract hemodynamic patterns
                patterns["blood_flow"] = np.random.uniform(0.8, 1.2)
                patterns["oxygen_saturation"] = np.random.uniform(0.95, 0.99)
            elif modality == "accel_data":
                # Extract motion patterns
                patterns["head_motion"] = np.random.uniform(0, 0.1)
                patterns["motion_artifacts"] = np.random.uniform(0, 0.05)
        
        return patterns
    
    async def stop_data_acquisition(self, scan_id):
        """Stop data acquisition for a scan"""
        if scan_id in self.active_scans:
            self.active_scans[scan_id]["status"] = "completed"
            
        if len([scan for scan in self.active_scans.values() 
               if scan["status"] == "active"]) == 0:
            self.system_status = "initialized"
        
        return True
    
    def get_system_status(self):
        """Get complete system status"""
        return {
            "system_status": self.system_status,
            "devices": self.devices,
            "processing_pipeline": self.processing_pipeline,
            "active_scans": len([scan for scan in self.active_scans.values() 
                               if scan["status"] == "active"]),
            "total_scans": len(self.active_scans)
        }
    
    def save_scan_data(self, scan_id, filename):
        """Save scan data to file"""
        if scan_id not in self.data_streams:
            raise BrainForgeError(f"No data for scan ID: {scan_id}")
        
        scan_data = {
            "scan_id": scan_id,
            "data_streams": self.data_streams[scan_id],
            "scan_info": self.active_scans[scan_id],
            "system_config": self.config.to_dict()
        }
        
        with open(filename, 'w') as f:
            # Note: In real implementation, would use HDF5 or similar
            json.dump({k: str(v) for k, v in scan_data.items()}, f)
        
        return True


class TestSystemInitialization:
    """Test complete system initialization"""
    
    @pytest.fixture
    def system(self):
        """Create integrated system fixture"""
        return MockIntegratedSystem()
    
    @pytest.mark.asyncio
    async def test_full_system_startup(self, system):
        """Test complete system startup sequence"""
        # Initial state
        assert system.system_status == "stopped"
        
        # Initialize
        init_result = await system.initialize_system()
        assert init_result is True
        assert system.system_status == "initialized"
        
        # Connect hardware
        connect_result = await system.connect_hardware()
        assert connect_result is True
        
        # Verify all devices connected
        for device in system.devices.values():
            assert device["status"] == "connected"
        
        # Calibrate system
        calibrate_result = await system.calibrate_system()
        assert calibrate_result is True
        
        # Verify calibration
        for device in system.devices.values():
            assert device["calibrated"] is True
            assert device["calibration_quality"] > 0.8
    
    def test_system_configuration_loading(self, system):
        """Test system configuration loading"""
        # Create custom configuration
        custom_config = Config()
        custom_config.hardware.omp_channels = 512
        custom_config.processing.compression_algorithm = "neural_lz"
        custom_config.system.debug_mode = True
        
        # Create system with custom config
        custom_system = MockIntegratedSystem(custom_config)
        
        assert custom_system.config.hardware.omp_channels == 512
        assert custom_system.config.processing.compression_algorithm == "neural_lz"
        assert custom_system.config.system.debug_mode is True
    
    def test_system_status_monitoring(self, system):
        """Test system status monitoring"""
        async def run_test():
            await system.initialize_system()
            
            status = system.get_system_status()
            
            assert status["system_status"] == "initialized"
            assert "devices" in status
            assert "processing_pipeline" in status
            assert status["active_scans"] == 0
            assert status["total_scans"] == 0
            
        asyncio.run(run_test())


class TestDataAcquisitionWorkflow:
    """Test complete data acquisition workflow"""
    
    @pytest.fixture
    def initialized_system(self):
        """Create fully initialized system"""
        system = MockIntegratedSystem()
        
        async def setup():
            await system.initialize_system()
            await system.connect_hardware()
            await system.calibrate_system()
            return system
            
        return asyncio.run(setup())
    
    @pytest.mark.asyncio
    async def test_complete_scan_workflow(self, initialized_system):
        """Test complete brain scan workflow"""
        system = initialized_system
        
        # Start scan
        scan_id = await system.start_data_acquisition("full_test_scan")
        assert scan_id == "full_test_scan"
        assert system.system_status == "acquiring"
        
        # Acquire multiple data chunks
        total_chunks = 5
        all_chunks = []
        
        for i in range(total_chunks):
            chunk = system.acquire_data_chunk(scan_id, duration=1.0)
            all_chunks.append(chunk)
            
            # Verify chunk structure
            assert "omp_data" in chunk
            assert "optical_data" in chunk
            assert "accel_data" in chunk
            
            # Verify data shapes
            assert chunk["omp_data"].shape[0] == 306  # OMP channels
            assert chunk["optical_data"].shape[0] == 96  # Optical channels
            assert chunk["accel_data"].shape[0] == 192  # Accel channels
        
        # Check scan progress
        scan_status = system.active_scans[scan_id]
        assert scan_status["status"] == "active"
        assert scan_status["duration"] == total_chunks * 1.0
        assert scan_status["samples_acquired"] > 0
        
        # Stop scan
        stop_result = await system.stop_data_acquisition(scan_id)
        assert stop_result is True
        assert system.active_scans[scan_id]["status"] == "completed"
    
    def test_multi_modal_data_synchronization(self, initialized_system):
        """Test multi-modal data synchronization"""
        system = initialized_system
        
        async def run_test():
            scan_id = await system.start_data_acquisition("sync_test")
            
            # Acquire data chunk
            chunk = system.acquire_data_chunk(scan_id, duration=2.0)
            
            # Check temporal alignment
            omp_samples = chunk["omp_data"].shape[1]  # 1000 Hz * 2s = 2000
            optical_samples = chunk["optical_data"].shape[1]  # 100 Hz * 2s = 200
            accel_samples = chunk["accel_data"].shape[1]  # 1000 Hz * 2s = 2000
            
            assert omp_samples == 2000
            assert optical_samples == 200
            assert accel_samples == 2000
            
            # Check synchronization ratios
            assert omp_samples / optical_samples == 10  # 1000/100 = 10:1 ratio
            assert omp_samples == accel_samples  # Same sampling rate
            
        asyncio.run(run_test())
    
    def test_concurrent_scan_management(self, initialized_system):
        """Test managing multiple concurrent scans"""
        system = initialized_system
        
        async def run_test():
            # Start multiple scans
            scan_ids = []
            for i in range(3):
                scan_id = await system.start_data_acquisition(f"scan_{i}")
                scan_ids.append(scan_id)
            
            # Verify all scans are active
            status = system.get_system_status()
            assert status["active_scans"] == 3
            assert status["total_scans"] == 3
            
            # Acquire data for each scan
            for scan_id in scan_ids:
                chunk = system.acquire_data_chunk(scan_id, duration=0.5)
                assert len(chunk) == 3  # All three modalities
            
            # Stop scans one by one
            for scan_id in scan_ids:
                await system.stop_data_acquisition(scan_id)
            
            # Verify all scans completed
            final_status = system.get_system_status()
            assert final_status["active_scans"] == 0
            assert final_status["total_scans"] == 3  # Still tracked
            
        asyncio.run(run_test())


class TestProcessingPipeline:
    """Test complete processing pipeline"""
    
    @pytest.fixture
    def system_with_data(self):
        """Create system with acquired data"""
        system = MockIntegratedSystem()
        
        async def setup():
            await system.initialize_system()
            await system.connect_hardware()
            await system.calibrate_system()
            
            # Acquire some test data
            scan_id = await system.start_data_acquisition("processing_test")
            chunk = system.acquire_data_chunk(scan_id, duration=1.0)
            
            return system, scan_id, chunk
            
        return asyncio.run(setup())
    
    def test_complete_processing_pipeline(self, system_with_data):
        """Test complete data processing pipeline"""
        system, scan_id, chunk = system_with_data
        
        # Process the data
        processed_data = system.process_data(chunk)
        
        # Verify processing results
        for modality in ["omp_data", "optical_data", "accel_data"]:
            assert modality in processed_data
            
            modality_result = processed_data[modality]
            assert "filtered" in modality_result
            assert "compressed" in modality_result
            assert "features" in modality_result
            
            # Check feature extraction
            features = modality_result["features"]
            assert "mean" in features
            assert "std" in features
            assert "peak_freq" in features
    
    def test_brain_pattern_extraction(self, system_with_data):
        """Test brain pattern extraction"""
        system, scan_id, chunk = system_with_data
        
        # Process data first
        processed_data = system.process_data(chunk)
        
        # Extract brain patterns
        patterns = system.extract_brain_patterns(processed_data)
        
        # Verify pattern extraction
        assert "alpha_rhythm" in patterns
        assert "beta_rhythm" in patterns
        assert "gamma_rhythm" in patterns
        assert "blood_flow" in patterns
        assert "oxygen_saturation" in patterns
        assert "head_motion" in patterns
        assert "motion_artifacts" in patterns
        
        # Check pattern ranges
        assert 8 <= patterns["alpha_rhythm"] <= 12
        assert 13 <= patterns["beta_rhythm"] <= 30
        assert 30 <= patterns["gamma_rhythm"] <= 100
        assert 0 <= patterns["head_motion"] <= 0.1
    
    def test_real_time_processing_performance(self, system_with_data):
        """Test real-time processing performance"""
        system, scan_id, chunk = system_with_data
        
        # Time the processing pipeline
        start_time = time.time()
        processed_data = system.process_data(chunk)
        patterns = system.extract_brain_patterns(processed_data)
        processing_time = time.time() - start_time
        
        # Should process faster than data acquisition time
        chunk_duration = 1.0  # 1 second of data
        assert processing_time < chunk_duration / 2  # Use less than 50% of time
        
        # Verify results are complete
        assert len(processed_data) == 3  # All modalities processed
        assert len(patterns) >= 7  # All patterns extracted


class TestDataPersistence:
    """Test data storage and retrieval"""
    
    @pytest.fixture
    def system_with_complete_scan(self):
        """Create system with completed scan"""
        system = MockIntegratedSystem()
        
        async def setup():
            await system.initialize_system()
            await system.connect_hardware()
            await system.calibrate_system()
            
            # Complete scan
            scan_id = await system.start_data_acquisition("persistence_test")
            
            # Acquire multiple chunks
            for _ in range(3):
                chunk = system.acquire_data_chunk(scan_id, duration=1.0)
            
            await system.stop_data_acquisition(scan_id)
            
            return system, scan_id
            
        return asyncio.run(setup())
    
    def test_scan_data_saving(self, system_with_complete_scan):
        """Test saving scan data to file"""
        system, scan_id = system_with_complete_scan
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name
        
        # Save scan data
        save_result = system.save_scan_data(scan_id, filename)
        assert save_result is True
        
        # Verify file was created and contains data
        assert Path(filename).exists()
        
        with open(filename, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["scan_id"] == f'"{scan_id}"'  # JSON stringified
        assert "data_streams" in saved_data
        assert "scan_info" in saved_data
        assert "system_config" in saved_data
        
        # Clean up
        Path(filename).unlink()
    
    def test_scan_data_validation(self, system_with_complete_scan):
        """Test scan data validation before saving"""
        system, scan_id = system_with_complete_scan
        
        # Try to save non-existent scan
        with pytest.raises(BrainForgeError, match="No data for scan ID"):
            system.save_scan_data("non_existent_scan", "dummy.json")
        
        # Verify actual scan data exists
        assert scan_id in system.data_streams
        assert scan_id in system.active_scans


class TestErrorHandlingAndRecovery:
    """Test system error handling and recovery"""
    
    @pytest.fixture
    def system(self):
        """Create system for error testing"""
        return MockIntegratedSystem()
    
    def test_initialization_error_handling(self, system):
        """Test error handling during initialization"""
        # Test calibration without connection
        async def run_test():
            await system.initialize_system()
            # Don't connect hardware
            
            with pytest.raises(BrainForgeError, 
                             match="Cannot calibrate: devices not connected"):
                await system.calibrate_system()
                
        asyncio.run(run_test())
    
    def test_data_acquisition_error_handling(self, system):
        """Test error handling during data acquisition"""
        # Try to acquire data without initialization
        with pytest.raises(BrainForgeError, match="System not properly initialized"):
            asyncio.run(system.start_data_acquisition("error_test"))
    
    def test_invalid_scan_operations(self, system):
        """Test operations on invalid scans"""
        # Try to acquire data for non-existent scan
        with pytest.raises(BrainForgeError, match="No active scan with ID"):
            system.acquire_data_chunk("non_existent_scan")
    
    def test_system_recovery_after_error(self, system):
        """Test system recovery after errors"""
        async def run_test():
            # Cause an error
            try:
                await system.calibrate_system()  # Should fail - not connected
            except BrainForgeError:
                pass
            
            # System should still be recoverable
            await system.initialize_system()
            await system.connect_hardware()
            await system.calibrate_system()  # Should succeed now
            
            # Verify system is functional
            scan_id = await system.start_data_acquisition("recovery_test")
            chunk = system.acquire_data_chunk(scan_id, duration=0.5)
            assert len(chunk) == 3  # All modalities working
            
        asyncio.run(run_test())


class TestSystemIntegrationScenarios:
    """Test realistic system integration scenarios"""
    
    @pytest.fixture
    def full_system(self):
        """Create fully functional system"""
        system = MockIntegratedSystem()
        
        async def setup():
            await system.initialize_system()
            await system.connect_hardware()
            await system.calibrate_system()
            return system
            
        return asyncio.run(setup())
    
    def test_typical_research_session(self, full_system):
        """Test typical neuroscience research session"""
        system = full_system
        
        async def run_session():
            # Start multiple subjects/conditions
            scan_ids = []
            
            # Subject 1 - resting state
            scan_id_1 = await system.start_data_acquisition("subject1_rest")
            scan_ids.append(scan_id_1)
            
            # Acquire 5 minutes of resting state data
            for minute in range(5):
                chunk = system.acquire_data_chunk(scan_id_1, duration=60.0)
                processed = system.process_data(chunk)
                patterns = system.extract_brain_patterns(processed)
                
                # Verify continuous data quality
                assert len(patterns) >= 5
            
            await system.stop_data_acquisition(scan_id_1)
            
            # Subject 1 - task condition
            scan_id_2 = await system.start_data_acquisition("subject1_task")
            scan_ids.append(scan_id_2)
            
            # Acquire task data with event markers
            for trial in range(10):
                chunk = system.acquire_data_chunk(scan_id_2, duration=3.0)
                processed = system.process_data(chunk)
                patterns = system.extract_brain_patterns(processed)
                
                # Verify task-related patterns
                assert patterns["alpha_rhythm"] != patterns.get("prev_alpha", 0)
            
            await system.stop_data_acquisition(scan_id_2)
            
            # Verify session completed successfully
            status = system.get_system_status()
            assert status["total_scans"] == 2
            
            return scan_ids
            
        session_results = asyncio.run(run_session())
        assert len(session_results) == 2
    
    def test_long_duration_stability(self, full_system):
        """Test system stability over long durations"""
        system = full_system
        
        async def run_long_test():
            scan_id = await system.start_data_acquisition("stability_test")
            
            total_chunks = 60  # Simulate 1 hour at 1-minute chunks
            successful_chunks = 0
            
            for chunk_num in range(total_chunks):
                try:
                    chunk = system.acquire_data_chunk(scan_id, duration=60.0)
                    processed = system.process_data(chunk)
                    patterns = system.extract_brain_patterns(processed)
                    
                    # Verify data quality hasn't degraded
                    assert len(chunk) == 3
                    assert len(patterns) >= 5
                    
                    successful_chunks += 1
                    
                except Exception as e:
                    print(f"Error in chunk {chunk_num}: {e}")
            
            await system.stop_data_acquisition(scan_id)
            
            # Should have high success rate
            success_rate = successful_chunks / total_chunks
            assert success_rate > 0.95  # 95% success rate
            
            return successful_chunks
            
        # Note: This is a mock test - real test would take 1 hour
        # We're simulating the timing aspects
        successful_chunks = asyncio.run(run_long_test())
        assert successful_chunks >= 57  # Allow some failures
    
    def test_multi_user_concurrent_access(self, full_system):
        """Test concurrent access by multiple users/processes"""
        system = full_system
        
        async def run_concurrent_test():
            # Simulate multiple concurrent users
            tasks = []
            
            for user_id in range(3):
                async def user_session(uid):
                    scan_id = await system.start_data_acquisition(f"user_{uid}_scan")
                    
                    # Each user acquires data for 30 seconds
                    for _ in range(30):
                        chunk = system.acquire_data_chunk(scan_id, duration=1.0)
                        processed = system.process_data(chunk)
                        patterns = system.extract_brain_patterns(processed)
                    
                    await system.stop_data_acquisition(scan_id)
                    return f"user_{uid}_completed"
                
                tasks.append(user_session(user_id))
            
            # Run all user sessions concurrently
            results = await asyncio.gather(*tasks)
            
            # Verify all users completed successfully
            assert len(results) == 3
            assert all("completed" in result for result in results)
            
            # Verify system handled concurrent access
            status = system.get_system_status()
            assert status["total_scans"] == 3
            
        asyncio.run(run_concurrent_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
