"""
Main Application Entry Point
===========================

This file serves as the main entry point for the California Housing MLOps API.
It redirects to the proper API implementation in src/api/app.py

Author: Group 14
Date: August 2025
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import and run the main API
if __name__ == "__main__":
    from src.api.app import app
    app.run(host='0.0.0.0', port=5001, debug=False)
