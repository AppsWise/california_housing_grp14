"""Enhanced Flask API with Pydantic validation and bonus features"""

import os
import logging
import json
import time
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from prometheus_flask_exporter import PrometheusMetrics
from pydantic import ValidationError
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.database import PredictionDatabase
from utils.validation import (
    validate_prediction_input, 
    validate_batch_input,
    PredictionResponse,
    BatchPredictionResponse,
    ErrorResponse
)
from utils.model_retraining import ModelRetrainingTrigger

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Initialize database and retraining trigger
prediction_db = PredictionDatabase()
retrain_trigger = ModelRetrainingTrigger()

# Create logs directory
os.makedirs('logs', exist_ok=True)

# Configure comprehensive logging
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('logs/app.log')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

app.logger.addHandler(file_handler)
app.logger.addHandler(console_handler)
app.logger.setLevel(logging.INFO)

# Create separate logger for predictions
prediction_logger = logging.getLogger('predictions')
prediction_handler = logging.FileHandler('logs/predictions.log')
prediction_handler.setFormatter(log_formatter)
prediction_logger.addHandler(prediction_handler)
prediction_logger.setLevel(logging.INFO)

# Custom metrics
predictions_counter = metrics.counter(
    'predictions_total', 'Total number of predictions',
    labels={'endpoint': lambda: request.endpoint, 'method': lambda: request.method}
)

prediction_histogram = metrics.histogram(
    'prediction_duration_seconds', 'Time spent on prediction',
    labels={'method': lambda: request.method}
)

# Global variables
model = None

def load_model():
    """Load the trained model"""
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'model.pkl')
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            app.logger.info(f"Model loaded successfully from {model_path}")
            return True
        else:
            app.logger.error(f"Model file not found at {model_path}")
            return False
    except Exception as e:
        app.logger.error(f"Failed to load model: {e}")
        return False

def prepare_input(input_data):
    """Prepare input data for model prediction with feature engineering"""
    
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Feature engineering (same as training)
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    
    # Ensure correct column order (same as training)
    expected_columns = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income',
        'rooms_per_household', 'bedrooms_per_room', 'population_per_household', 
        'ocean_proximity'
    ]
    
    return df[expected_columns]

def log_prediction_request(endpoint, input_data, prediction=None, processing_time_ms=None, 
                          status_code=200, error_message=None):
    """Log prediction request to database and file"""
    try:
        # Log to database
        request_id = prediction_db.log_prediction_request(
            endpoint=endpoint,
            method=request.method,
            input_data=input_data,
            prediction=prediction,
            processing_time_ms=processing_time_ms,
            status_code=status_code,
            error_message=error_message,
            user_agent=request.headers.get('User-Agent'),
            ip_address=request.remote_addr
        )
        
        # Log to file
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": endpoint,
            "method": request.method,
            "input_data": input_data,
            "prediction": prediction,
            "processing_time_ms": processing_time_ms,
            "status_code": status_code,
            "error_message": error_message,
            "user_agent": request.headers.get('User-Agent'),
            "ip_address": request.remote_addr
        }
        
        prediction_logger.info(f"PREDICTION_REQUEST: {json.dumps(log_entry)}")
        
        return request_id
        
    except Exception as e:
        app.logger.error(f"Failed to log prediction request: {e}")
        return None

# Load model on startup
load_model()

@app.route('/')
def index():
    """Main page with API documentation"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Enhanced health check with model validation"""
    try:
        start_time = time.time()
        
        # Basic health info
        health_info = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'model_loaded': model is not None,
            'version': '1.0.0'
        }
        
        # Test model if loaded
        if model is not None:
            try:
                # Test prediction with sample data
                test_input = prepare_input({
                    'longitude': -122.23, 'latitude': 37.88, 'housing_median_age': 41.0,
                    'total_rooms': 880.0, 'total_bedrooms': 129.0, 'population': 322.0,
                    'households': 126.0, 'median_income': 8.3252, 'ocean_proximity': 'NEAR BAY'
                })
                
                test_prediction = model.predict(test_input)[0]
                health_info['model_status'] = 'functional'
                health_info['model_test_prediction'] = float(test_prediction)
                
            except Exception as e:
                health_info['model_status'] = f'error: {str(e)}'
        else:
            health_info['model_status'] = 'not_loaded'
        
        # Database status
        try:
            recent_count = prediction_db.get_prediction_count(hours=24)
            health_info['recent_predictions'] = recent_count
            health_info['database_status'] = 'connected'
        except Exception as e:
            health_info['database_status'] = f'error: {str(e)}'
        
        # Uptime
        health_info['uptime_seconds'] = time.time() - start_time
        
        app.logger.info("Health check completed: healthy")
        return jsonify(health_info)
        
    except Exception as e:
        app.logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/api/predict', methods=['POST'])
@predictions_counter
@prediction_histogram
def predict():
    """Make a single prediction with Pydantic validation"""
    start_time = time.time()
    input_data = None
    
    try:
        # Validate request has JSON data
        if not request.is_json:
            error_response = ErrorResponse(
                error="Invalid Content Type",
                message="Request must be JSON",
                timestamp=datetime.utcnow().isoformat()
            )
            log_prediction_request('/api/predict', {}, status_code=400, error_message=error_response.message)
            return jsonify(error_response.dict()), 400
        
        input_data = request.get_json()
        
        # Validate with Pydantic
        try:
            validated_data, warnings = validate_prediction_input(input_data)
            app.logger.info(f"Input validation successful with {len(warnings)} warnings")
            if warnings:
                app.logger.info(f"Validation warnings: {warnings}")
        except ValidationError as e:
            error_details = [{"field": str(err['loc'][-1]) if err['loc'] else 'unknown', "message": err['msg'], "type": err['type']} for err in e.errors()]
            error_message = "Input validation failed: " + "; ".join([f"{detail['field']}: {detail['message']}" for detail in error_details])
            
            response_data = {
                "error": "Validation Error",
                "message": error_message,
                "details": error_details,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            log_prediction_request('/api/predict', input_data, status_code=400, error_message=error_message)
            return jsonify(response_data), 400
        
        # Check if model is loaded
        if model is None:
            error_response = ErrorResponse(
                error="Model Not Available",
                message="Model is not loaded",
                timestamp=datetime.utcnow().isoformat()
            )
            log_prediction_request('/api/predict', input_data, status_code=500, error_message=error_response.message)
            return jsonify(error_response.dict()), 500
        
        # Convert validated data to dict for model prediction
        model_input = validated_data.dict()
        
        # Prepare input for model
        prepared_input = prepare_input(model_input)
        app.logger.info(f"Prepared input features: {list(prepared_input.columns)}")
        
        # Make prediction
        prediction = model.predict(prepared_input)[0]
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Create response with validation warnings
        response = PredictionResponse(
            prediction=float(prediction),
            model_version="1.0.0",
            timestamp=datetime.utcnow().isoformat(),
            processing_time_ms=processing_time_ms,
            validation_warnings=warnings
        )
        
        # Log successful prediction
        log_prediction_request(
            '/api/predict', 
            model_input, 
            prediction=float(prediction),
            processing_time_ms=processing_time_ms,
            status_code=200
        )
        
        app.logger.info(f"Successful prediction: {prediction} (processing time: {processing_time_ms/1000:.3f}s)")
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        error_response = ErrorResponse(
            error="Prediction Error",
            message=f"Internal server error: {str(e)}",
            timestamp=datetime.utcnow().isoformat()
        )
        
        app.logger.error(f"Prediction error: {e}")
        app.logger.error(f"Traceback: {traceback.format_exc()}")
        
        log_prediction_request('/api/predict', input_data or {}, 
                             processing_time_ms=processing_time_ms, status_code=500, 
                             error_message=error_response.message)
        
        return jsonify(error_response.dict()), 500

@app.route('/api/predict/batch', methods=['POST'])
@predictions_counter
def predict_batch():
    """Make batch predictions with Pydantic validation"""
    start_time = time.time()
    input_data = None
    
    try:
        if not request.is_json:
            error_response = ErrorResponse(
                error="Invalid Content Type",
                message="Request must be JSON",
                timestamp=datetime.utcnow().isoformat()
            )
            return jsonify(error_response.dict()), 400
        
        input_data = request.get_json()
        
        # Validate batch input with Pydantic
        try:
            validated_data, warnings = validate_batch_input(input_data)
            app.logger.info(f"Batch validation successful with {len(warnings)} warnings")
        except ValidationError as e:
            error_response = ErrorResponse(
                error="Validation Error",
                message="Batch input validation failed",
                details=e.errors(),
                timestamp=datetime.utcnow().isoformat()
            )
            return jsonify(error_response.dict()), 400
        
        if model is None:
            error_response = ErrorResponse(
                error="Model Not Available",
                message="Model is not loaded",
                timestamp=datetime.utcnow().isoformat()
            )
            return jsonify(error_response.dict()), 500
        
        # Process batch predictions
        predictions = []
        for prediction_input in validated_data.predictions:
            model_input = prediction_input.dict()
            prepared_input = prepare_input(model_input)
            prediction = model.predict(prepared_input)[0]
            predictions.append(float(prediction))
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        response = BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            model_version="1.0.0",
            timestamp=datetime.utcnow().isoformat(),
            processing_time_ms=processing_time_ms,
            validation_warnings=warnings
        )
        
        app.logger.info(f"Batch prediction completed: {len(predictions)} predictions (processing time: {processing_time_ms/1000:.3f}s)")
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        error_response = ErrorResponse(
            error="Batch Prediction Error",
            message=f"Internal server error: {str(e)}",
            timestamp=datetime.utcnow().isoformat()
        )
        
        app.logger.error(f"Batch prediction error: {e}")
        return jsonify(error_response.dict()), 500

@app.route('/api/retrain', methods=['POST'])
def trigger_retrain():
    """Manually trigger model retraining"""
    try:
        data = request.get_json() if request.is_json else {}
        reason = data.get('reason', 'manual_api_trigger')
        
        app.logger.info(f"Manual retrain triggered via API. Reason: {reason}")
        
        # Trigger retraining
        retrain_trigger.manual_retrain(reason)
        
        response = {
            'status': 'triggered',
            'message': 'Model retraining triggered successfully',
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        app.logger.error(f"Failed to trigger retraining: {e}")
        error_response = ErrorResponse(
            error="Retrain Trigger Error",
            message=f"Failed to trigger retraining: {str(e)}",
            timestamp=datetime.utcnow().isoformat()
        )
        return jsonify(error_response.dict()), 500

@app.route('/api/monitoring/performance', methods=['GET'])
def check_performance():
    """Check model performance and trigger retraining if needed"""
    try:
        needs_retrain = retrain_trigger.check_performance_degradation()
        needs_time_retrain = retrain_trigger.check_time_based_retrain()
        
        response = {
            'performance_degradation_detected': needs_retrain,
            'time_based_retrain_needed': needs_time_retrain,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'healthy' if not (needs_retrain or needs_time_retrain) else 'needs_attention'
        }
        
        if needs_retrain:
            response['recommendation'] = 'Model performance has degraded, consider retraining'
        elif needs_time_retrain:
            response['recommendation'] = 'Scheduled retraining window reached'
        
        return jsonify(response), 200
        
    except Exception as e:
        app.logger.error(f"Performance check failed: {e}")
        error_response = ErrorResponse(
            error="Performance Check Error",
            message=f"Failed to check performance: {str(e)}",
            timestamp=datetime.utcnow().isoformat()
        )
        return jsonify(error_response.dict()), 500

# Legacy endpoint for backward compatibility
@app.route('/predict', methods=['POST'])
def predict_legacy():
    """Legacy endpoint for form-based predictions"""
    try:
        if model is None:
            return render_template('index.html', 
                                 prediction_text='Error: Model not loaded')
        
        # Get form data and convert to API format
        form_data = request.form.to_dict()
        
        # Convert to proper format and validate
        api_data = {
            'longitude': float(form_data.get('longitude', 0)),
            'latitude': float(form_data.get('latitude', 0)),
            'housing_median_age': float(form_data.get('housing_median_age', 0)),
            'total_rooms': float(form_data.get('total_rooms', 0)),
            'total_bedrooms': float(form_data.get('total_bedrooms', 0)),
            'population': float(form_data.get('population', 0)),
            'households': float(form_data.get('households', 0)),
            'median_income': float(form_data.get('median_income', 0)),
            'ocean_proximity': form_data.get('ocean_proximity', 'INLAND')
        }
        
        # Use the same validation and prediction logic
        validated_data, warnings = validate_prediction_input(api_data)
        prepared_input = prepare_input(validated_data.dict())
        prediction = model.predict(prepared_input)[0]
        
        prediction_text = f'Predicted House Value: ${prediction:,.2f}'
        if warnings:
            prediction_text += f' (Warnings: {", ".join(warnings)})'
        
        return render_template('index.html', prediction_text=prediction_text)
        
    except Exception as e:
        app.logger.error(f"Legacy prediction error: {e}")
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    # Initialize logging
    app.logger.info("=" * 50)
    app.logger.info("California Housing Prediction API Starting...")
    app.logger.info(f"Model loaded: {model is not None}")
    app.logger.info(f"Database initialized: {prediction_db is not None}")
    app.logger.info(f"Retraining trigger initialized: {retrain_trigger is not None}")
    app.logger.info(f"Logs directory: logs/")
    app.logger.info("=" * 50)
    
    # Start monitoring (optional)
    # retrain_trigger.start_monitoring()
    
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=False)
