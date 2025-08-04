"""Test configuration for pytest"""

import pytest
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def sample_housing_data():
    """Sample housing data for testing"""
    return {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY",
    }


@pytest.fixture
def api_client():
    """Flask test client fixture"""
    from api.app import app

    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
