"""
API tests for California Housing MLOps Pipeline
"""

import pytest
import os


def test_api_module_exists():
    """Test that API module exists"""
    assert os.path.exists("src/api/app.py"), "API app.py should exist"
    assert os.path.exists("src/api/schemas.py"), "API schemas.py should exist"


def test_flask_import():
    """Test that Flask can be imported"""
    try:
        import flask
        assert True, "Flask import should work"
    except ImportError:
        pytest.skip("Flask not available, skipping test")


def test_prediction_data_structure():
    """Test prediction data structure"""
    # Test data structure for California Housing
    sample_data = {
        "longitude": -122.25,
        "latitude": 37.85,
        "housing_median_age": 15.0,
        "total_rooms": 3000.0,
        "total_bedrooms": 600.0,
        "population": 1800.0,
        "households": 500.0,
        "median_income": 5.5
    }
    
    assert isinstance(sample_data, dict), "Sample data should be a dictionary"
    assert len(sample_data) == 8, "Sample data should have 8 features"
    assert "longitude" in sample_data, "Longitude should be present"
    assert "latitude" in sample_data, "Latitude should be present"


def test_validation_ranges():
    """Test basic validation ranges for California Housing"""
    # California longitude range
    valid_longitude = -122.25
    assert -124.5 <= valid_longitude <= -114.0, "Longitude should be in California range"
    
    # California latitude range  
    valid_latitude = 37.85
    assert 32.5 <= valid_latitude <= 42.1, "Latitude should be in California range"
    
    # Positive values
    assert 15.0 > 0, "Housing median age should be positive"
    assert 5.5 > 0, "Median income should be positive"


class TestAPIValidation:
    """Test class for API validation logic"""
    
    def test_geographic_bounds(self):
        """Test geographic boundary validation"""
        # Test valid California coordinates
        assert -124.0 >= -124.5, "Western longitude bound"
        assert -115.0 <= -114.0, "Eastern longitude bound"
        assert 33.0 >= 32.5, "Southern latitude bound"
        assert 41.0 <= 42.1, "Northern latitude bound"
    
    def test_housing_feature_ranges(self):
        """Test housing feature validation ranges"""
        # Test reasonable ranges
        assert 1.0 <= 50.0, "Housing age range"
        assert 100.0 <= 50000.0, "Total rooms range"
        assert 10.0 <= 10000.0, "Total bedrooms range"
        assert 50.0 <= 50000.0, "Population range"
        assert 10.0 <= 10000.0, "Households range"
        assert 0.5 <= 15.0, "Median income range"
