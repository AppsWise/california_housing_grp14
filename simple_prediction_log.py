#!/usr/bin/env python3
"""
Simple MLflow Prediction Logging Test
"""

import mlflow
import mlflow.sklearn
import json
from datetime import datetime
import requests

def test_simple_logging():
    """Test the simplest possible MLflow logging"""
    print("🚀 Testing Simple MLflow Logging...")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5002")
    
    # Set experiment
    mlflow.set_experiment("simple_prediction_logging")
    
    # Sample prediction data
    prediction_data = {
        "input_features": {
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252
        },
        "prediction": 371300.0,
        "model_version": "test_v1",
        "timestamp": datetime.now().isoformat(),
        "processing_time_ms": 25.5
    }
    
    # Start MLflow run and log data
    with mlflow.start_run(run_name=f"prediction_log_{datetime.now().strftime('%H%M%S')}"):
        # Log parameters
        mlflow.log_param("model_version", prediction_data["model_version"])
        mlflow.log_param("longitude", prediction_data["input_features"]["longitude"])
        mlflow.log_param("latitude", prediction_data["input_features"]["latitude"])
        mlflow.log_param("median_income", prediction_data["input_features"]["median_income"])
        
        # Log metrics
        mlflow.log_metric("prediction_value", prediction_data["prediction"])
        mlflow.log_metric("processing_time_ms", prediction_data["processing_time_ms"])
        
        # Log JSON artifact
        with open("temp_prediction.json", "w") as f:
            json.dump(prediction_data, f, indent=2)
        mlflow.log_artifact("temp_prediction.json")
        
        print("✅ Logged prediction data to MLflow successfully!")
        print(f"🔮 Prediction: ${prediction_data['prediction']:,.2f}")
        print(f"⏱️  Processing time: {prediction_data['processing_time_ms']}ms")
        print(f"🌐 View at: http://localhost:5002")

def test_api_integration():
    """Test integration with the running API"""
    print("\n🔗 Testing API Integration...")
    
    try:
        # Make a prediction request to the API
        api_url = "http://localhost:5001/api/predict"
        test_data = {
            "longitude": -117.12,
            "latitude": 32.75,
            "housing_median_age": 15.0,
            "total_rooms": 8.0,
            "total_bedrooms": 2.0,
            "population": 4000.0,
            "households": 3.8,
            "median_income": 7.5,
            "ocean_proximity": "NEAR OCEAN"
        }
        
        response = requests.post(api_url, json=test_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API prediction successful: ${result['prediction']:,.2f}")
            
            # Log this to MLflow
            mlflow.set_tracking_uri("http://localhost:5002")
            mlflow.set_experiment("api_prediction_test")
            
            with mlflow.start_run(run_name=f"api_test_{datetime.now().strftime('%H%M%S')}"):
                mlflow.log_param("source", "api_test")
                mlflow.log_param("api_endpoint", api_url)
                mlflow.log_metric("prediction_value", result['prediction'])
                mlflow.log_metric("processing_time_ms", result['processing_time_ms'])
                
                for key, value in test_data.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"input_{key}", value)
                    else:
                        mlflow.log_param(f"input_{key}", str(value))
                
                print("✅ Logged API prediction to MLflow!")
            
            return True
        else:
            print(f"❌ API request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API integration test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🎯 Simple MLflow Prediction Logging Test")
    print("="*45)
    
    # Test simple logging
    test_simple_logging()
    
    # Test API integration
    test_api_integration()
    
    print(f"\n🌐 Check results at: http://localhost:5002")
    print("✅ All tests completed!")
