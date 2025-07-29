#!/usr/bin/env python3
"""
Test runner for Single Modality BCI Demo
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Import and run the demo
    from single_modality_bci_demo import main
    
    print("🚀 Starting Single Modality BCI Demo Test...")
    
    # Run the demo
    results = main()
    
    if results:
        print("\n✅ Single Modality BCI Demo completed successfully!")
        print(f"   Overall Performance: {results.get('overall_performance', 'UNKNOWN')}")
        print(f"   Session Accuracy: {results.get('session_accuracy', 0):.3f}")
        print(f"   Average Latency: {results.get('average_latency_ms', 0):.1f}ms")
    else:
        print("\n❌ Demo completed but no results returned")
        
except Exception as e:
    print(f"\n❌ Demo failed with error: {e}")
    import traceback
    traceback.print_exc()
