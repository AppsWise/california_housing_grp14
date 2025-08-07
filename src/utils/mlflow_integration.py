
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
