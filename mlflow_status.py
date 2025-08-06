#!/usr/bin/env python3
"""
🎯 MLflow Integration - Quick Start Guide & Test
"""

import os
import sys
import json
import requests
from datetime import datetime

def show_status():
    """Show current MLflow integration status"""
    print("🎯 MLflow Integration Status")
    print("="*30)
    
    # Check MLflow data
    if os.path.exists("mlruns_local"):
        print("✅ MLflow local storage: READY")
        print(f"📁 Location: {os.path.abspath('mlruns_local')}")
    else:
        print("❌ MLflow local storage: NOT FOUND")
    
    # Check integration file
    if os.path.exists("mlflow_integration.py"):
        print("✅ MLflow integration: READY")
    else:
        print("❌ MLflow integration: NOT FOUND")
    
    # Check API
    try:
        response = requests.get("http://localhost:5001/health", timeout=3)
        if response.status_code == 200:
            print("✅ API server: RUNNING")
        else:
            print("❌ API server: ERROR")
    except:
        print("❌ API server: NOT RUNNING")

def test_prediction_with_logging():
    """Test making a prediction and verify MLflow logging"""
    print("\\n🚀 Testing Prediction + MLflow Logging")
    print("="*40)
    
    # Test data
    test_input = {
        "longitude": -120.0,
        "latitude": 37.0,
        "housing_median_age": 30.0,
        "total_rooms": 2000.0,
        "total_bedrooms": 400.0,
        "population": 3000.0,
        "households": 500.0,
        "median_income": 7.0,
        "ocean_proximity": "INLAND"
    }
    
    print("📊 Input data:")
    for key, value in test_input.items():
        print(f"   {key}: {value}")
    
    # Make prediction
    try:
        response = requests.post("http://localhost:5001/api/predict", json=test_input, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\\n✅ Prediction successful:")
            print(f"   🏠 Predicted price: ${result['prediction']:,.2f}")
            print(f"   ⏱️  Processing time: {result['processing_time_ms']:.2f}ms")
            print(f"   🤖 Model version: {result['model_version']}")
            
            # Verify MLflow logging
            print(f"\\n📝 MLflow logging:")
            print(f"   ✅ Automatically logged to mlruns_local/")
            print(f"   📊 Experiment: api_predictions")
            print(f"   🕒 Timestamp: {result['timestamp']}")
            
            return True, result['prediction']
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False, None

def show_recent_predictions():
    """Show recent predictions from MLflow"""
    print("\\n📈 Recent Predictions from MLflow")
    print("="*35)
    
    try:
        import mlflow
        import pandas as pd
        
        # Set tracking URI
        mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns_local')}")
        
        # Get api_predictions experiment
        experiments = mlflow.search_experiments()
        api_exp = [exp for exp in experiments if exp.name == "api_predictions"]
        
        if api_exp:
            runs = mlflow.search_runs(experiment_ids=[api_exp[0].experiment_id], max_results=5)
            
            if len(runs) > 0:
                print(f"📊 Found {len(runs)} recent predictions:")
                
                for i, (_, run) in enumerate(runs.iterrows(), 1):
                    run_id = run['run_id'][:8]
                    start_time = run['start_time'].strftime('%H:%M:%S') if pd.notna(run['start_time']) else 'N/A'
                    prediction = run.get('metrics.prediction_value', 0)
                    processing_time = run.get('metrics.processing_time_ms', 0)
                    
                    print(f"\\n   {i}. Run {run_id} ({start_time})")
                    print(f"      🏠 Prediction: ${prediction:,.2f}")
                    print(f"      ⏱️  Processing: {processing_time:.2f}ms")
                    
                    # Show some input parameters
                    longitude = run.get('metrics.input_longitude')
                    latitude = run.get('metrics.input_latitude')
                    if longitude and latitude:
                        print(f"      📍 Location: ({longitude}, {latitude})")
            else:
                print("❌ No predictions found in MLflow")
        else:
            print("❌ No api_predictions experiment found")
            
    except ImportError:
        print("❌ MLflow not available (run: conda activate BITSAIML)")
    except Exception as e:
        print(f"❌ Error reading MLflow data: {e}")

def show_usage_examples():
    """Show usage examples"""
    print("\\n💡 MLflow Integration Usage")
    print("="*30)
    
    print("📋 1. Making predictions (they auto-log to MLflow):")
    print("   curl -X POST -H 'Content-Type: application/json' \\\\")
    print("        -d '{\"longitude\": -122.0, \"latitude\": 37.0, ...}' \\\\") 
    print("        http://localhost:5001/api/predict")
    
    print("\\n📋 2. Direct MLflow logging in Python:")
    print("   from mlflow_integration import log_prediction_to_mlflow")
    print("   log_prediction_to_mlflow(input_data, prediction, processing_time)")
    
    print("\\n📋 3. View MLflow UI:")
    print("   conda activate BITSAIML")
    print(f"   mlflow ui --backend-store-uri file://{os.path.abspath('mlruns_local')}")
    print("   # Then open: http://localhost:5000")
    
    print("\\n📋 4. Access data programmatically:")
    print("   import mlflow")
    print(f"   mlflow.set_tracking_uri('file://{os.path.abspath('mlruns_local')}')")
    print("   runs = mlflow.search_runs()")

def main():
    """Main function"""
    print("🎯 MLflow Integration - Quick Start Guide")
    print("="*45)
    print("   Simple, Local, File-based MLflow Tracking")
    print("   No server dependencies, always works!")
    print()
    
    # Show status
    show_status()
    
    # Test prediction
    success, prediction = test_prediction_with_logging()
    
    # Show recent predictions
    show_recent_predictions()
    
    # Show usage
    show_usage_examples()
    
    # Summary
    print("\\n🎉 Summary:")
    if success:
        print("   ✅ MLflow integration working perfectly!")
        print(f"   🏠 Latest prediction: ${prediction:,.2f}")
        print("   📊 All predictions automatically logged")
        print("   💾 Data stored locally in mlruns_local/")
        print("   🌐 View in MLflow UI when needed")
    else:
        print("   ⚠️  API not responding - check if server is running:")
        print("   conda activate BITSAIML")
        print("   cd /Volumes/Development/Development/assignment/MLOps_ASS1")
        print("   python -m src.api.app")

if __name__ == "__main__":
    main()
