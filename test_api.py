"""
API Testing Script
==================

Simple test script to validate the California Housing API endpoints.

Author: Group 14
Date: August 2025
"""

import requests
import json
import time
import sys

# API base URL
BASE_URL = "http://localhost:5001"

def test_health_endpoint():
    """Test the health check endpoint."""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing health endpoint: {e}")
        return False

def test_predict_endpoint():
    """Test the prediction endpoint."""
    print("\nTesting /api/predict endpoint...")
    
    # Sample prediction data
    test_data = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing predict endpoint: {e}")
        return False

def test_metrics_endpoint():
    """Test the metrics endpoint."""
    print("\nTesting /metrics endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response length: {len(response.text)} characters")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing metrics endpoint: {e}")
        return False

def main():
    """Run all API tests."""
    print("=" * 50)
    print("CALIFORNIA HOUSING API TESTS")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Prediction", test_predict_endpoint),
        ("Metrics", test_metrics_endpoint),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} Test ---")
        results[test_name] = test_func()
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
