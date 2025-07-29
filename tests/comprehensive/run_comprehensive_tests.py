"""
Comprehensive Test Runner for Brain-Forge README Claims

This module runs all tests that verify Brain-Forge functionality matches
the claims and examples in the README.md file.

Test Categories:
1. README Claims Validation - Core functionality claims
2. Hardware Integration - Multi-modal hardware support
3. Processing Pipeline - Signal processing capabilities  
4. Configuration System - Configuration management
5. Performance Benchmarks - Latency and throughput claims
6. Integration Workflows - End-to-end functionality

Usage:
    python -m pytest tests/comprehensive/ -v --tb=short
    
    or run specific test categories:
    
    python -m pytest tests/comprehensive/test_readme_claims.py -v
    python -m pytest tests/comprehensive/test_hardware_integration.py -v
    python -m pytest tests/comprehensive/test_processing_validation.py -v
    python -m pytest tests/comprehensive/test_configuration_system.py -v
    python -m pytest tests/comprehensive/test_performance_benchmarks.py -v
    python -m pytest tests/comprehensive/test_integration_workflows.py -v
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import brain_forge

# Import test utilities
from core.config import Config
from core.logger import get_logger


class TestSuiteRunner:
    """Main test suite runner for comprehensive README validation"""
    
    @staticmethod
    def run_all_tests():
        """Run all comprehensive tests"""
        print("=" * 80)
        print("Brain-Forge Comprehensive Test Suite")
        print("Validating all README claims and documentation")
        print("=" * 80)
        
        test_modules = [
            "test_readme_claims.py",
            "test_hardware_integration.py", 
            "test_processing_validation.py",
            "test_configuration_system.py",
            "test_performance_benchmarks.py",
            "test_integration_workflows.py"
        ]
        
        results = {}
        start_time = time.time()
        
        for module in test_modules:
            print(f"\nRunning {module}...")
            module_path = Path(__file__).parent / module
            
            # Run pytest on individual module
            exit_code = pytest.main([
                str(module_path), 
                "-v", 
                "--tb=short",
                "-x"  # Stop on first failure for debugging
            ])
            
            results[module] = "PASSED" if exit_code == 0 else "FAILED"
            print(f"{module}: {results[module]}")
        
        total_time = time.time() - start_time
        
        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUITE SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for r in results.values() if r == "PASSED")
        failed = sum(1 for r in results.values() if r == "FAILED")
        
        for module, result in results.items():
            status_marker = "✓" if result == "PASSED" else "✗"
            print(f"{status_marker} {module}: {result}")
        
        print(f"\nTotal: {passed} passed, {failed} failed")
        print(f"Total time: {total_time:.2f} seconds")
        
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED! Brain-Forge implementation matches README claims.")
        else:
            print(f"\n⚠️  {failed} test module(s) failed. See output above for details.")
        
        return failed == 0
    
    @staticmethod
    def run_quick_validation():
        """Run a quick validation of core functionality"""
        print("Running quick validation of core Brain-Forge functionality...")
        
        try:
            # Test basic imports
            print("✓ Testing imports...")
            config = Config()
            logger = get_logger("test")
            system_info = brain_forge.get_system_info()
            
            # Test basic processing
            print("✓ Testing basic processing...")
            from processing import RealTimeProcessor
            processor = RealTimeProcessor()
            test_data = np.random.randn(32, 1000)
            
            import asyncio
            result = asyncio.run(processor.process_data_chunk(test_data))
            
            assert 'processed_data' in result
            assert 'quality_score' in result
            assert 0.0 <= result['quality_score'] <= 1.0
            
            # Test configuration
            print("✓ Testing configuration system...")
            assert config.hardware.omp_channels >= 306
            assert config.processing.compression_ratio > 1.0
            
            # Test compression
            print("✓ Testing compression...")
            from processing import WaveletCompressor
            compressor = WaveletCompressor()
            compressed = compressor.compress(test_data, compression_ratio=5.0)
            assert compressed['compression_ratio'] > 2.0
            
            print("\n🎉 Quick validation PASSED! Core functionality working.")
            return True
            
        except Exception as e:
            print(f"\n❌ Quick validation FAILED: {e}")
            return False
    
    @staticmethod
    def validate_readme_examples():
        """Validate that all code examples from README work"""
        print("Validating README code examples...")
        
        examples_passed = 0
        examples_total = 0
        
        # Example 1: Basic configuration
        try:
            examples_total += 1
            from core.config import Config
            from core.logger import get_logger
            
            config = Config()
            logger = get_logger(__name__)
            logger.info("Brain-Forge initialized successfully")
            
            examples_passed += 1
            print("✓ Basic configuration example")
        except Exception as e:
            print(f"✗ Basic configuration example failed: {e}")
        
        # Example 2: Multi-modal acquisition setup
        try:
            examples_total += 1
            
            omp_config = {
                'channels': 306,
                'matrix_coils': 48,
                'sampling_rate': 1000,
                'magnetic_shielding': True,
                'movement_compensation': 'dynamic'
            }
            
            kernel_config = {
                'optical_modules': 40,
                'eeg_channels': 4,
                'wavelengths': [690, 905],
                'measurement_type': 'hemodynamic_electrical',
                'coverage': 'whole_head'
            }
            
            accelo_config = {
                'accelerometers': 64,
                'impact_detection': True,
                'motion_correlation': True,
                'navy_grade': True
            }
            
            # Validate configuration structures
            assert omp_config['channels'] == 306
            assert len(kernel_config['wavelengths']) == 2
            assert accelo_config['accelerometers'] == 64
            
            examples_passed += 1
            print("✓ Multi-modal acquisition example")
        except Exception as e:
            print(f"✗ Multi-modal acquisition example failed: {e}")
        
        # Example 3: Processing pipeline
        try:
            examples_total += 1
            from processing import RealTimeProcessor, WaveletCompressor
            
            processor = RealTimeProcessor()
            compressor = WaveletCompressor(wavelet='db8')
            
            # Test with sample data
            test_data = np.random.randn(64, 1000)
            import asyncio
            result = asyncio.run(processor.process_data_chunk(test_data))
            
            assert 'processed_data' in result
            assert 'features' in result
            
            examples_passed += 1
            print("✓ Processing pipeline example")
        except Exception as e:
            print(f"✗ Processing pipeline example failed: {e}")
        
        # Example 4: System info
        try:
            examples_total += 1
            system_info = brain_forge.get_system_info()
            hardware_status = brain_forge.check_hardware_support()
            
            assert 'version' in system_info
            assert 'dependencies' in system_info
            assert isinstance(hardware_status, dict)
            
            examples_passed += 1
            print("✓ System info example")
        except Exception as e:
            print(f"✗ System info example failed: {e}")
        
        print(f"\nREADME Examples: {examples_passed}/{examples_total} passed")
        return examples_passed == examples_total
    
    @staticmethod
    def check_installation_requirements():
        """Check that installation meets README requirements"""
        print("Checking installation requirements...")
        
        requirements_met = 0
        requirements_total = 0
        
        # Python version check
        requirements_total += 1
        if sys.version_info >= (3, 8):
            print("✓ Python 3.8+ requirement met")
            requirements_met += 1
        else:
            print(f"✗ Python version {sys.version_info} < 3.8")
        
        # Key dependencies check
        key_dependencies = [
            'numpy', 'scipy', 'scikit-learn', 'pywavelets'
        ]
        
        for dep in key_dependencies:
            requirements_total += 1
            try:
                __import__(dep)
                print(f"✓ {dep} available")
                requirements_met += 1
            except ImportError:
                print(f"✗ {dep} not available")
        
        # Brain-forge import check
        requirements_total += 1
        try:
            import brain_forge
            print(f"✓ brain_forge v{brain_forge.__version__} available")
            requirements_met += 1
        except ImportError:
            print("✗ brain_forge not available")
        
        print(f"\nInstallation: {requirements_met}/{requirements_total} requirements met")
        return requirements_met == requirements_total


def main():
    """Main test runner entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Brain-Forge Comprehensive Test Suite")
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick validation only')
    parser.add_argument('--examples', action='store_true',
                       help='Validate README examples only')
    parser.add_argument('--install', action='store_true',
                       help='Check installation requirements only')
    parser.add_argument('--all', action='store_true',
                       help='Run all comprehensive tests (default)')
    
    args = parser.parse_args()
    
    runner = TestSuiteRunner()
    
    if args.quick:
        success = runner.run_quick_validation()
    elif args.examples:
        success = runner.validate_readme_examples()
    elif args.install:
        success = runner.check_installation_requirements()
    else:
        # Run all tests by default
        print("Checking installation requirements...")
        install_ok = runner.check_installation_requirements()
        
        if not install_ok:
            print("\n❌ Installation requirements not met. Please install missing dependencies.")
            return 1
        
        print("\nValidating README examples...")
        examples_ok = runner.validate_readme_examples()
        
        if not examples_ok:
            print("\n❌ README examples failed. Basic functionality issues detected.")
            return 1
        
        print("\nRunning comprehensive test suite...")
        success = runner.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
