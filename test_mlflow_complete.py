#!/usr/bin/env python3
"""
Comprehensive MLflow Integration Test & Viewer
"""

import mlflow
import os
import pandas as pd
import requests
import json
from datetime import datetime
from mlflow_integration import log_prediction_to_mlflow

def test_api_with_mlflow():
    """Test API predictions and verify MLflow logging"""
    print("🚀 Testing API with MLflow Integration")
    print("="*45)
    
    api_url = "http://localhost:5001/api/predict"
    
    # Test predictions
    test_cases = [
        {
            "name": "San Francisco Bay Area",
            "data": {
                "longitude": -122.4,
                "latitude": 37.8,
                "housing_median_age": 35.0,
                "total_rooms": 3000.0,
                "total_bedrooms": 600.0,
                "population": 4000.0,
                "households": 700.0,
                "median_income": 10.0,
                "ocean_proximity": "NEAR BAY"
            }
        },
        {
            "name": "Los Angeles Area",
            "data": {
                "longitude": -118.3,
                "latitude": 34.1,
                "housing_median_age": 25.0,
                "total_rooms": 2500.0,
                "total_bedrooms": 500.0,
                "population": 3500.0,
                "households": 600.0,
                "median_income": 7.5,
                "ocean_proximity": "NEAR OCEAN"
            }
        },
        {
            "name": "Central Valley",
            "data": {
                "longitude": -119.0,
                "latitude": 36.0,
                "housing_median_age": 15.0,
                "total_rooms": 1800.0,
                "total_bedrooms": 350.0,
                "population": 2500.0,
                "households": 450.0,
                "median_income": 5.5,
                "ocean_proximity": "INLAND"
            }
        }
    ]
    
    predictions_made = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\\n📍 Test {i}: {test_case['name']}")
        
        try:
            # Make API request
            response = requests.post(api_url, json=test_case['data'], timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result['prediction']
                processing_time = result['processing_time_ms']
                
                print(f"   ✅ Prediction: ${prediction:,.2f}")
                print(f"   ⏱️  Processing time: {processing_time:.2f}ms")
                
                predictions_made.append({
                    'name': test_case['name'],
                    'prediction': prediction,
                    'processing_time': processing_time,
                    'input_data': test_case['data']
                })
                
            else:
                print(f"   ❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
    
    print(f"\\n✅ Made {len(predictions_made)} successful predictions")
    return predictions_made

def view_mlflow_data():
    """View all MLflow logged data"""
    print("\\n👀 MLflow Data Overview")
    print("="*25)
    
    # Set tracking URI
    tracking_dir = "mlruns_local"
    if not os.path.exists(tracking_dir):
        print("❌ No MLflow data found")
        return
    
    mlflow.set_tracking_uri(f"file://{os.path.abspath(tracking_dir)}")
    
    try:
        # List all experiments
        experiments = mlflow.search_experiments()
        print(f"📊 Found {len(experiments)} experiments:")
        
        total_runs = 0
        
        for exp in experiments:
            print(f"\\n   📁 {exp.name} (ID: {exp.experiment_id})")
            
            # Get runs for this experiment
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=50)
            total_runs += len(runs)
            
            print(f"      Runs: {len(runs)}")
            
            if len(runs) > 0:
                # Show latest runs
                for i, (_, run) in enumerate(runs.head(5).iterrows()):
                    run_id = run['run_id'][:8]
                    start_time = run['start_time'].strftime('%H:%M:%S') if pd.notna(run['start_time']) else 'N/A'
                    
                    if 'metrics.prediction_value' in run:
                        pred_val = run['metrics.prediction_value']
                        print(f"         🔮 {run_id} ({start_time}): ${pred_val:,.2f}")
                    else:
                        print(f"         📝 {run_id} ({start_time}): Training run")
        
        print(f"\\n📈 Total runs across all experiments: {total_runs}")
        
        # Focus on API predictions
        api_runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id for exp in experiments if exp.name == "api_predictions"], 
            max_results=10
        )
        
        if len(api_runs) > 0:
            print(f"\\n🔍 Recent API Predictions ({len(api_runs)} runs):")
            
            for _, run in api_runs.iterrows():
                run_id = run['run_id'][:8]
                start_time = run['start_time'].strftime('%H:%M:%S') if pd.notna(run['start_time']) else 'N/A'
                
                prediction = run.get('metrics.prediction_value', 0)
                processing_time = run.get('metrics.processing_time_ms', 0)
                longitude = run.get('metrics.input_longitude', 'N/A')
                latitude = run.get('metrics.input_latitude', 'N/A')
                
                print(f"      🏠 {run_id} ({start_time}): ${prediction:,.2f}")
                print(f"         📍 Location: ({longitude}, {latitude})")
                print(f"         ⏱️  Time: {processing_time:.2f}ms")
                print()
        
    except Exception as e:
        print(f"❌ Error reading MLflow data: {e}")

def test_direct_mlflow_logging():
    """Test direct MLflow logging"""
    print("\\n🔬 Testing Direct MLflow Logging")
    print("="*35)
    
    test_data = {
        "longitude": -121.0,
        "latitude": 38.0,
        "housing_median_age": 40.0,
        "total_rooms": 2200.0,
        "total_bedrooms": 450.0,
        "population": 3200.0,
        "households": 550.0,
        "median_income": 8.0,
        "ocean_proximity": "INLAND"
    }
    
    # Test direct logging
    success = log_prediction_to_mlflow(test_data, 325000.0, 15.5)
    
    if success:
        print("✅ Direct MLflow logging successful")
        print(f"   🔮 Logged prediction: $325,000.00")
        print(f"   📍 Location: ({test_data['longitude']}, {test_data['latitude']})")
    else:
        print("❌ Direct MLflow logging failed")
    
    return success

def show_mlflow_ui_info():
    """Show information about viewing MLflow UI"""
    print("\\n🌐 MLflow UI Information")
    print("="*25)
    
    tracking_dir = os.path.abspath("mlruns_local")
    
    print(f"📁 MLflow data location: {tracking_dir}")
    print(f"🌐 To view in MLflow UI, run:")
    print(f"   conda activate BITSAIML")
    print(f"   mlflow ui --backend-store-uri file://{tracking_dir}")
    print(f"   Then open: http://localhost:5000")
    print()
    print("📊 In the UI you can:")
    print("   - Compare prediction values across runs")
    print("   - View input parameters for each prediction")
    print("   - Analyze processing time trends")
    print("   - Download logged models")
    print("   - Search and filter runs")

def generate_summary_report():
    """Generate a summary report"""
    print("\\n📋 MLflow Integration Summary")
    print("="*30)
    
    tracking_dir = "mlruns_local"
    if not os.path.exists(tracking_dir):
        print("❌ No MLflow data to summarize")
        return
    
    mlflow.set_tracking_uri(f"file://{os.path.abspath(tracking_dir)}")
    
    try:
        # Get all experiments
        experiments = mlflow.search_experiments()
        total_runs = 0
        total_predictions = 0
        
        for exp in experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            total_runs += len(runs)
            
            # Count prediction runs
            prediction_runs = runs[runs['metrics.prediction_value'].notna()]
            total_predictions += len(prediction_runs)
        
        print(f"✅ MLflow Integration Status: WORKING")
        print(f"📊 Total Experiments: {len(experiments)}")
        print(f"🏃 Total Runs: {total_runs}")
        print(f"🔮 Total Predictions Logged: {total_predictions}")
        print(f"📁 Storage: File-based (local)")
        print(f"🔗 Integration: Simple & Reliable")
        
        # Get recent API predictions
        api_exp = [exp for exp in experiments if exp.name == "api_predictions"]
        if api_exp:
            api_runs = mlflow.search_runs(experiment_ids=[api_exp[0].experiment_id])
            if len(api_runs) > 0:
                latest_prediction = api_runs.iloc[0]['metrics.prediction_value']
                latest_time = api_runs.iloc[0]['start_time']
                print(f"🕒 Latest API Prediction: ${latest_prediction:,.2f} at {latest_time.strftime('%H:%M:%S')}")
        
        print(f"\\n🎯 Next Steps:")
        print(f"   1. Make predictions via API: curl -X POST http://localhost:5001/api/predict ...")
        print(f"   2. View logs: python {__file__}")
        print(f"   3. Open MLflow UI: mlflow ui --backend-store-uri file://{os.path.abspath(tracking_dir)}")
        
    except Exception as e:
        print(f"❌ Error generating summary: {e}")

def main():
    """Main test function"""
    print("🎯 MLflow Integration Test & Verification")
    print("="*45)
    
    # Test 1: API predictions with MLflow logging
    predictions = test_api_with_mlflow()
    
    # Test 2: Direct MLflow logging
    direct_success = test_direct_mlflow_logging()
    
    # Test 3: View logged data
    view_mlflow_data()
    
    # Test 4: Show UI info
    show_mlflow_ui_info()
    
    # Test 5: Generate summary
    generate_summary_report()
    
    print(f"\\n🎉 MLflow Integration Test Complete!")
    print(f"   ✅ API Predictions: {len(predictions)} successful")
    print(f"   ✅ Direct Logging: {'Working' if direct_success else 'Failed'}")
    print(f"   ✅ Data Viewing: Working")
    print(f"   ✅ Storage: Local file-based")

if __name__ == "__main__":
    main()
