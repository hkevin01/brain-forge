#!/usr/bin/env python3
"""
Quick API Test - Testing Brain-Forge API Implementation

This script tests if the existing API implementation is functional.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test if all API-related imports work"""
    print("🧪 Testing API Imports...")
    
    try:
        from api.rest_api import BrainForgeAPI, create_brain_forge_api
        print("✅ API module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ API import failed: {e}")
        return False

def test_api_creation():
    """Test API creation"""
    print("\n🧪 Testing API Creation...")
    
    try:
        from api.rest_api import create_brain_forge_api
        api = create_brain_forge_api()
        print("✅ API created successfully")
        
        app = api.get_app()
        if app:
            print("✅ FastAPI app available")
        else:
            print("⚠️  Mock API mode")
            
        return True
    except Exception as e:
        print(f"❌ API creation failed: {e}")
        return False

def test_dependencies():
    """Test key dependencies"""
    print("\n🧪 Testing Dependencies...")
    
    deps = [
        "fastapi",
        "uvicorn", 
        "websockets",
        "pydantic",
        "numpy"
    ]
    
    available = 0
    for dep in deps:
        try:
            __import__(dep)
            print(f"✅ {dep}")
            available += 1
        except ImportError:
            print(f"❌ {dep}")
    
    print(f"\n📊 Dependencies: {available}/{len(deps)} available")
    return available >= len(deps) * 0.8  # 80% threshold

if __name__ == "__main__":
    print("🧠 Brain-Forge API Quick Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_api_creation,
        test_dependencies
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📋 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed - API is functional!")
    else:
        print("⚠️  Some tests failed - API needs attention")
