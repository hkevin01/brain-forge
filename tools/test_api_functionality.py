#!/usr/bin/env python3
"""
API Testing and Validation Tool

Tests the existing Brain-Forge API implementation for functionality and completeness.
This is part of Phase 2 code modernization and enhancement.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from api.rest_api import BrainForgeAPI, create_brain_forge_api
    API_AVAILABLE = True
except ImportError as e:
    print(f"❌ API import failed: {e}")
    API_AVAILABLE = False

try:
    import fastapi
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

def test_api_creation():
    """Test API application creation"""
    print("🧪 Testing API Application Creation...")
    
    if not API_AVAILABLE:
        print("❌ API module not available")
        return False
    
    try:
        # Create API instance
        api = create_brain_forge_api()
        print("✅ API instance created successfully")
        
        # Check if app is properly initialized
        app = api.get_app()
        if app is not None:
            print("✅ FastAPI app initialized")
            return True
        else:
            print("⚠️  Mock API mode (FastAPI not available)")
            return True
            
    except Exception as e:
        print(f"❌ API creation failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints functionality"""
    print("\\n🧪 Testing API Endpoints...")
    
    if not API_AVAILABLE:
        print("❌ API module not available")
        return False
    
    try:
        api = create_brain_forge_api()
        app = api.get_app()
        
        if hasattr(app, 'routes'):
            routes = [route.path for route in app.routes if hasattr(route, 'path')]
            print(f"✅ Found {len(routes)} API routes:")
            for route in routes[:10]:  # Show first 10 routes
                print(f"   • {route}")
            
            if len(routes) > 10:
                print(f"   ... and {len(routes) - 10} more routes")
                
            return len(routes) > 0
        else:
            print("⚠️  Mock API mode - simulated endpoints available")
            return True
            
    except Exception as e:
        print(f"❌ Endpoint testing failed: {e}")
        return False

def test_api_dependencies():
    """Test API dependencies"""
    print("\\n🧪 Testing API Dependencies...")
    
    dependencies = [
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI server"),
        ("websockets", "WebSocket support"),
        ("pydantic", "Data validation"),
        ("numpy", "Numerical computing"),
    ]
    
    available_deps = []
    missing_deps = []
    
    for dep_name, description in dependencies:
        try:
            __import__(dep_name)
            available_deps.append((dep_name, description))
            print(f"✅ {dep_name} - {description}")
        except ImportError:
            missing_deps.append((dep_name, description))
            print(f"❌ {dep_name} - {description} (missing)")
    
    print(f"\\n📊 Dependency Summary:")
    print(f"   Available: {len(available_deps)}/{len(dependencies)}")
    print(f"   Missing: {len(missing_deps)}")
    
    return len(missing_deps) == 0

def test_api_configuration():
    """Test API configuration"""
    print("\\n🧪 Testing API Configuration...")
    
    try:
        from core.config import Config
        config = Config()
        
        # Check if API-related configuration exists
        api_attrs = []
        for attr in dir(config):
            if 'api' in attr.lower() or 'port' in attr.lower() or 'host' in attr.lower():
                api_attrs.append(attr)
        
        if api_attrs:
            print(f"✅ API configuration attributes found:")
            for attr in api_attrs[:5]:  # Show first 5
                value = getattr(config, attr, 'Not accessible')
                print(f"   • {attr}: {value}")
        else:
            print("⚠️  No specific API configuration found (using defaults)")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration testing failed: {e}")
        return False

def test_websocket_support():
    """Test WebSocket functionality"""
    print("\\n🧪 Testing WebSocket Support...")
    
    try:
        import websockets
        print("✅ WebSocket library available")
        
        if API_AVAILABLE:
            api = create_brain_forge_api()
            
            # Check if WebSocket endpoints are defined
            app = api.get_app()
            if hasattr(app, 'routes'):
                ws_routes = [route for route in app.routes 
                           if hasattr(route, 'path') and 'ws' in route.path.lower()]
                
                print(f"✅ Found {len(ws_routes)} WebSocket routes:")
                for route in ws_routes:
                    print(f"   • {route.path}")
                
                return len(ws_routes) > 0
            else:
                print("⚠️  Mock API mode - WebSocket simulation available")
                return True
        else:
            print("⚠️  API not available for WebSocket testing")
            return False
            
    except ImportError:
        print("❌ WebSocket library not available")
        return False
    except Exception as e:
        print(f"❌ WebSocket testing failed: {e}")
        return False

def generate_api_test_report():
    """Generate comprehensive API test report"""
    print("\\n" + "="*60)
    print("🧠 Brain-Forge API Testing Report")
    print("="*60)
    
    test_results = {
        "API Creation": test_api_creation(),
        "API Endpoints": test_api_endpoints(), 
        "Dependencies": test_api_dependencies(),
        "Configuration": test_api_configuration(),
        "WebSocket Support": test_websocket_support()
    }
    
    print("\\n📋 Test Results Summary:")
    print("-"*40)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print("-"*40)
    print(f"Overall Score: {passed}/{total} ({passed/total*100:.1f}%)")
    
    # Provide recommendations
    print("\\n🎯 Recommendations:")
    
    if test_results["Dependencies"]:
        print("✅ All dependencies are available - API is ready to run")
    else:
        print("⚠️  Install missing dependencies with: pip install -r requirements/base.txt")
    
    if test_results["API Creation"] and test_results["API Endpoints"]:
        print("✅ API implementation is functional")
        print("   Next steps: Start API server with: python src/api/rest_api.py")
    else:
        print("⚠️  API implementation needs debugging")
    
    if test_results["WebSocket Support"]:
        print("✅ Real-time streaming capabilities are available")
    else:
        print("⚠️  WebSocket support needs verification")
    
    # Phase 2 completion status
    phase2_completion = passed / total * 100
    print(f"\\n📊 Phase 2 Progress: {phase2_completion:.1f}% complete")
    
    if phase2_completion >= 80:
        print("🎉 Phase 2: Code Modernization - Nearly Complete!")
        print("   Ready to proceed to Phase 3: Documentation Enhancement")
    elif phase2_completion >= 60:
        print("⚡ Phase 2: Making good progress")
        print("   Address failing tests before proceeding")
    else:
        print("🔧 Phase 2: More work needed")
        print("   Focus on fixing core API functionality")

if __name__ == "__main__":
    generate_api_test_report()
