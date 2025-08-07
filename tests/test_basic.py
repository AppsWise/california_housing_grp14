"""
Basic tests for California Housing MLOps Pipeline
"""

import pytest
import os


def test_project_structure():
    """Test that essential project files exist"""
    assert os.path.exists("README.md"), "README.md should exist"
    assert os.path.exists("requirements.txt"), "requirements.txt should exist"
    assert os.path.exists("Dockerfile"), "Dockerfile should exist"
    assert os.path.exists("src/"), "src/ directory should exist"


def test_requirements_file():
    """Test that requirements.txt contains essential packages"""
    with open("requirements.txt", "r") as f:
        requirements = f.read().lower()
    
    assert "flask" in requirements, "Flask should be in requirements"
    assert "scikit-learn" in requirements, "scikit-learn should be in requirements"


def test_basic_math():
    """Test basic Python functionality"""
    assert 2 + 2 == 4, "Basic math should work"
    assert 10 / 2 == 5.0, "Division should work"
    assert 3 * 3 == 9, "Multiplication should work"


def test_data_directory():
    """Test that data directory exists"""
    assert os.path.exists("data/"), "data/ directory should exist"


def test_models_directory():
    """Test that models directory exists"""
    assert os.path.exists("models/"), "models/ directory should exist"


def test_config_directory():
    """Test that config directory exists"""
    assert os.path.exists("config/"), "config/ directory should exist"


class TestBasicFunctionality:
    """Test class for basic functionality"""
    
    def test_list_operations(self):
        """Test basic list operations"""
        test_list = [1, 2, 3, 4, 5]
        assert len(test_list) == 5
        assert sum(test_list) == 15
        assert max(test_list) == 5
    
    def test_string_operations(self):
        """Test basic string operations"""
        test_string = "California Housing"
        assert len(test_string) > 0
        assert "California" in test_string
        assert test_string.upper() == "CALIFORNIA HOUSING"
    
    def test_dict_operations(self):
        """Test basic dictionary operations"""
        test_dict = {"latitude": 37.85, "longitude": -122.25}
        assert "latitude" in test_dict
        assert len(test_dict) == 2
        assert test_dict["latitude"] == 37.85
