"""
MLflow Integration Check
========================

Script to verify MLflow setup and model tracking functionality.

Author: Group 14
Date: August 2025
"""

import mlflow
import mlflow.sklearn
import os
import sys
from pathlib import Path

def check_mlflow_tracking():
    """Check MLflow tracking setup."""
    print("🔍 Checking MLflow tracking setup...")
    
    # Set tracking URI
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    
    try:
        # Try to get experiments
        experiments = mlflow.search_experiments()
        print(f"✅ Found {len(experiments)} experiments")
        
        for exp in experiments:
            print(f"  - {exp.name} (ID: {exp.experiment_id})")
        
        return True
    except Exception as e:
        print(f"❌ Error connecting to MLflow: {e}")
        return False

def check_model_registry():
    """Check MLflow model registry."""
    print("\n🔍 Checking MLflow model registry...")
    
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Try to list registered models
        models = client.search_registered_models()
        print(f"✅ Found {len(models)} registered models")
        
        for model in models:
            print(f"  - {model.name}")
            latest_versions = client.get_latest_versions(model.name)
            for version in latest_versions:
                print(f"    Version {version.version} (Stage: {version.current_stage})")
        
        return True
    except Exception as e:
        print(f"❌ Error accessing model registry: {e}")
        return False

def test_experiment_logging():
    """Test experiment logging functionality."""
    print("\n🔍 Testing experiment logging...")
    
    try:
        # Set experiment
        experiment_name = "mlflow_test"
        mlflow.set_experiment(experiment_name)
        
        # Start a run
        with mlflow.start_run(run_name="connectivity_test"):
            # Log some dummy metrics
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.95)
            
            # Log a simple artifact
            with open("test_artifact.txt", "w") as f:
                f.write("This is a test artifact for MLflow connectivity check.")
            mlflow.log_artifact("test_artifact.txt")
            
            # Clean up
            os.remove("test_artifact.txt")
        
        print("✅ Successfully logged test experiment")
        return True
        
    except Exception as e:
        print(f"❌ Error logging experiment: {e}")
        return False

def main():
    """Run all MLflow checks."""
    print("=" * 60)
    print("MLFLOW INTEGRATION CHECK")
    print("=" * 60)
    
    checks = [
        ("MLflow Tracking", check_mlflow_tracking),
        ("Model Registry", check_model_registry),
        ("Experiment Logging", test_experiment_logging),
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        results[check_name] = check_func()
    
    # Summary
    print("\n" + "=" * 60)
    print("CHECK RESULTS SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL MLFLOW CHECKS PASSED!")
        print("MLflow integration is working correctly.")
    else:
        print("❌ Some MLflow checks failed!")
        print("Please check your MLflow setup.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
