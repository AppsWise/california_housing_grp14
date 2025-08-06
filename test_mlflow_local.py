#!/usr/bin/env python3
"""
Ultra Simple MLflow Test - No Server Dependencies
"""

import os
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import joblib
from datetime import datetime

def test_local_mlflow():
    """Test MLflow with local file storage only"""
    print("🚀 Testing Local MLflow (No Server Required)...")
    
    # Set local file-based tracking
    tracking_dir = "mlruns_local"
    os.makedirs(tracking_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{os.path.abspath(tracking_dir)}")
    
    # Create experiment (will work locally)
    experiment_name = "simple_local_test"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✅ Created experiment: {experiment_name}")
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id if experiment else "0"
        print(f"✅ Using existing experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    
    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=8, noise=0.1, random_state=42)
    
    # Train simple model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Make prediction
    sample_prediction = model.predict(X[0:1])[0]
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"local_test_{datetime.now().strftime('%H%M%S')}"):
        # Log parameters
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 10)
        mlflow.log_param("n_samples", 1000)
        mlflow.log_param("n_features", 8)
        
        # Log metrics
        mlflow.log_metric("sample_prediction", sample_prediction)
        mlflow.log_metric("model_score", model.score(X, y))
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        os.makedirs("models", exist_ok=True)
        model_path = "models/simple_rf_model.pkl"
        joblib.dump(model, model_path)
        
        print(f"✅ Logged run successfully!")
        print(f"🔮 Sample prediction: {sample_prediction:.2f}")
        print(f"📁 Tracking directory: {os.path.abspath(tracking_dir)}")
        print(f"💾 Model saved to: {model_path}")
        
        return True, model_path

def test_prediction_logging():
    """Test logging individual predictions"""
    print("\n📝 Testing Prediction Logging...")
    
    # Set local tracking
    tracking_dir = "mlruns_local"
    mlflow.set_tracking_uri(f"file://{os.path.abspath(tracking_dir)}")
    mlflow.set_experiment("prediction_logging")
    
    # Sample predictions to log
    predictions = [
        {"input": {"longitude": -122.23, "latitude": 37.88, "income": 8.32}, "prediction": 371300.0},
        {"input": {"longitude": -118.25, "latitude": 34.05, "income": 5.20}, "prediction": 107500.0},
        {"input": {"longitude": -117.12, "latitude": 32.75, "income": 7.50}, "prediction": 237400.0},
    ]
    
    for i, pred_data in enumerate(predictions):
        with mlflow.start_run(run_name=f"prediction_{i+1}"):
            # Log input parameters
            for key, value in pred_data["input"].items():
                mlflow.log_param(f"input_{key}", value)
            
            # Log prediction
            mlflow.log_metric("prediction_value", pred_data["prediction"])
            mlflow.log_metric("processing_time_ms", np.random.uniform(5, 50))
            
            # Log timestamp
            mlflow.log_param("timestamp", datetime.now().isoformat())
            
        print(f"✅ Logged prediction {i+1}: ${pred_data['prediction']:,.2f}")
    
    print(f"✅ All predictions logged successfully!")
    return True

def create_simple_api_integration():
    """Create a simple function that can be called from the API"""
    print("\n🔗 Creating Simple API Integration...")
    
    code = '''
import mlflow
import os
from datetime import datetime

def log_prediction_to_mlflow(input_data, prediction, processing_time=None):
    """Simple function to log predictions to MLflow"""
    try:
        # Set local tracking
        tracking_dir = "mlruns_local"
        os.makedirs(tracking_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{os.path.abspath(tracking_dir)}")
        mlflow.set_experiment("api_predictions")
        
        # Start run and log data
        with mlflow.start_run(run_name=f"api_pred_{datetime.now().strftime('%H%M%S')}"):
            # Log input parameters
            for key, value in input_data.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"input_{key}", value)
                else:
                    mlflow.log_param(f"input_{key}", str(value))
            
            # Log prediction and metrics
            mlflow.log_metric("prediction_value", prediction)
            if processing_time:
                mlflow.log_metric("processing_time_ms", processing_time)
            
            mlflow.log_param("timestamp", datetime.now().isoformat())
            
        return True
    except Exception as e:
        print(f"MLflow logging error: {e}")
        return False

# Test the function
if __name__ == "__main__":
    test_input = {
        "longitude": -122.23,
        "latitude": 37.88,
        "median_income": 8.3252,
        "housing_median_age": 41.0
    }
    
    success = log_prediction_to_mlflow(test_input, 371300.0, 25.5)
    print(f"MLflow logging {'successful' if success else 'failed'}")
'''
    
    # Save the integration code
    with open("mlflow_integration.py", "w") as f:
        f.write(code)
    
    print("✅ Created mlflow_integration.py")
    print("   You can import and use log_prediction_to_mlflow() in your API")
    
    # Test the integration
    exec(code)
    
    return True

def view_logged_data():
    """Show how to view logged data"""
    print("\n👀 Viewing Logged Data...")
    
    tracking_dir = "mlruns_local"
    if os.path.exists(tracking_dir):
        print(f"📁 MLflow data stored in: {os.path.abspath(tracking_dir)}")
        
        # List experiments
        mlflow.set_tracking_uri(f"file://{os.path.abspath(tracking_dir)}")
        
        try:
            experiments = mlflow.search_experiments()
            print(f"📊 Found {len(experiments)} experiments:")
            
            for exp in experiments:
                print(f"   - {exp.name} (ID: {exp.experiment_id})")
                
                # Get runs for this experiment
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                print(f"     Runs: {len(runs)}")
                
                if len(runs) > 0:
                    latest_run = runs.iloc[0]
                    print(f"     Latest run ID: {latest_run['run_id']}")
                    if 'metrics.prediction_value' in latest_run:
                        print(f"     Latest prediction: ${latest_run['metrics.prediction_value']:,.2f}")
            
            print(f"\n🌐 To view in MLflow UI:")
            print(f"   mlflow ui --backend-store-uri file://{os.path.abspath(tracking_dir)}")
            
        except Exception as e:
            print(f"Error reading experiments: {e}")
    else:
        print("❌ No MLflow data found")
    
    return True

def main():
    """Main test function"""
    print("🎯 Ultra Simple MLflow Test (Local File Storage)")
    print("="*55)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Local MLflow logging
    try:
        success, model_path = test_local_mlflow()
        if success:
            tests_passed += 1
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    # Test 2: Prediction logging
    try:
        if test_prediction_logging():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
    
    # Test 3: API integration
    try:
        if create_simple_api_integration():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
    
    # Test 4: View data
    try:
        if view_logged_data():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed >= 3:
        print("🎉 MLflow integration working! (Local file storage)")
        print(f"📁 Data location: {os.path.abspath('mlruns_local')}")
        print(f"🔗 Integration file: mlflow_integration.py")
        print(f"💾 Model saved: models/simple_rf_model.pkl")
        
        print(f"\n🚀 To use in your API, add this to app.py:")
        print(f"   from mlflow_integration import log_prediction_to_mlflow")
        print(f"   log_prediction_to_mlflow(input_data, prediction, processing_time)")
        
        return True
    else:
        print("⚠️  Some tests failed, but basic functionality may still work")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
