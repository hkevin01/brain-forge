#!/usr/bin/env python3
"""
Brain-Forge API Startup Test
Test script to verify API can start successfully
"""

import os
import subprocess
import sys
import time

import requests

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_api_imports():
    """Test that API imports work correctly"""
    print("Testing API imports...")
    try:
        from api.rest_api import BrainForgeAPI
        print("✓ API imports successful")
        return True
    except ImportError as e:
        print(f"✗ API import failed: {e}")
        return False

def test_api_initialization():
    """Test that API can be initialized"""
    print("Testing API initialization...")
    try:
        from api.rest_api import BrainForgeAPI
        api = BrainForgeAPI()
        print("✓ API initialization successful")
        return True, api
    except Exception as e:
        print(f"✗ API initialization failed: {e}")
        return False, None

def test_api_startup():
    """Test that API server can start"""
    print("Testing API server startup...")
    
    # Create a test script that starts the server for a short time
    test_script = '''
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from api.rest_api import BrainForgeAPI
import uvicorn
import threading
import time

def start_server():
    try:
        api = BrainForgeAPI()
        if api.app:
            uvicorn.run(api.app, host="127.0.0.1", port=8001, log_level="warning")
    except Exception as e:
        print(f"Server error: {e}")

# Start server in background thread
server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

# Give server time to start
time.sleep(3)

# Test basic endpoint
try:
    import requests
    response = requests.get("http://127.0.0.1:8001/health", timeout=5)
    if response.status_code == 200:
        print("✓ API server startup test successful")
    else:
        print(f"✗ API server returned status: {response.status_code}")
except Exception as e:
    print(f"✗ API server test failed: {e}")
'''
    
    # Write and execute test script
    test_file = os.path.join(os.path.dirname(__file__), 'temp_api_test.py')
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=10)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("✗ API server test timed out")
        return False
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)

def main():
    """Run all API tests"""
    print("=== Brain-Forge API Validation Tests ===\n")
    
    # Test 1: Imports
    import_success = test_api_imports()
    print()
    
    # Test 2: Initialization  
    init_success, api = test_api_initialization()
    print()
    
    # Test 3: Server startup (only if previous tests pass)
    startup_success = False
    if import_success and init_success:
        startup_success = test_api_startup()
    else:
        print("Skipping server startup test due to previous failures")
    print()
    
    # Summary
    print("=== Test Results Summary ===")
    print(f"Imports: {'✓ PASS' if import_success else '✗ FAIL'}")
    print(f"Initialization: {'✓ PASS' if init_success else '✗ FAIL'}")
    print(f"Server Startup: {'✓ PASS' if startup_success else '✗ FAIL'}")
    
    all_passed = import_success and init_success and startup_success
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
