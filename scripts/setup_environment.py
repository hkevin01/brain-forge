#!/usr/bin/env python3
"""Environment setup script for Brain-Forge"""

import subprocess
import sys

def setup_environment():
    """Set up the Brain-Forge development environment"""
    print("🧠 Setting up Brain-Forge environment...")
    
    # Install requirements
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements/base.txt"])
        print("✅ Base requirements installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False
    
    print("✅ Environment setup complete!")
    return True

if __name__ == "__main__":
    setup_environment()
