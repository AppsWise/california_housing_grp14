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

# Add utils to path for database access
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.database import PredictionDatabase
from utils.validation import (
    validate_prediction_input, 
    validate_batch_input,
    PredictionResponse,
    BatchPredictionResponse,
    ErrorResponse
)

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Initialize database for logging
prediction_db = PredictionDatabase()

# Create logs directory
os.makedirs('logs', exist_ok=True)

# Configure comprehensive logging
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler for application logs
file_handler = logging.FileHandler('logs/app.log')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# File handler for prediction logs
prediction_handler = logging.FileHandler('logs/predictions.log')
prediction_handler.setFormatter(log_formatter)
prediction_handler.setLevel(logging.INFO)

# Console handler for development
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# Configure main logger
app.logger.addHandler(file_handler)
app.logger.addHandler(console_handler)
app.logger.setLevel(logging.INFO)

# Create separate logger for predictions
prediction_logger = logging.getLogger('predictions')
prediction_logger.addHandler(prediction_handler)
prediction_logger.addHandler(console_handler)
prediction_logger.setLevel(logging.INFO)

# Construct the absolute path to the model file for robustness
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '..', '..', 'models', 'model.pkl')

# Load the trained model
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    app.logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    app.logger.error(f"Failed to load model: {e}")
    model = None

# Create metrics for monitoring
predictions_counter = metrics.counter(
    'predictions_total', 'Total number of predictions',
    labels={'method': lambda: request.method, 'endpoint': lambda: request.endpoint}
)

prediction_histogram = metrics.histogram(
    'prediction_duration_seconds', 'Time spent on prediction',
    labels={'method': lambda: request.method}
)

# Expected feature names for California Housing dataset
EXPECTED_FEATURES = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms', 
    'total_bedrooms', 'population', 'households', 'median_income',
    'rooms_per_household', 'bedrooms_per_room', 'population_per_household',
    'ocean_proximity'
]

def prepare_input(features):
    """
    Prepare input features for the model to match the expected format.
    The model expects the same feature engineering that was done during training.
    """
    try:
        # Create DataFrame for processing
        df = pd.DataFrame([features])
        
        # Handle missing values for total_bedrooms (if any)
        if 'total_bedrooms' in df.columns:
            df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
        
        # Feature engineering (same as in training)
        df['rooms_per_household'] = df['total_rooms'] / df['households']
        df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
        df['population_per_household'] = df['population'] / df['households']
        
        # Handle infinite or NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Ensure correct column order (same as training)
        expected_columns = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income',
            'rooms_per_household', 'bedrooms_per_room', 'population_per_household',
            'ocean_proximity'
        ]
        
        # Reorder columns to match training
        df = df[expected_columns]
        
        app.logger.info(f"Prepared input features: {df.columns.tolist()}")
        return df
        
    except Exception as e:
        app.logger.error(f"Error preparing input features: {e}")
        raise

def log_prediction_request(endpoint, input_data, prediction=None, processing_time=None, 
                         status_code=200, error_message=None):
    """Log prediction request to both file and database"""
    try:
        # Log to database
        prediction_db.log_prediction_request(
            endpoint=endpoint,
            method=request.method,
            input_data=input_data,
            prediction=prediction,
            processing_time_ms=processing_time * 1000 if processing_time else None,
            status_code=status_code,
            error_message=error_message,
            user_agent=request.headers.get('User-Agent', ''),
            ip_address=request.remote_addr
        )
        
        # Log to file
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'endpoint': endpoint,
            'method': request.method,
            'input_data': input_data,
            'prediction': prediction,
            'processing_time_ms': processing_time * 1000 if processing_time else None,
            'status_code': status_code,
            'error_message': error_message,
            'user_agent': request.headers.get('User-Agent', ''),
            'ip_address': request.remote_addr
        }
        
        prediction_logger.info(f"PREDICTION_REQUEST: {json.dumps(log_data)}")
        
    except Exception as e:
        app.logger.error(f"Failed to log prediction request: {e}")

@app.route('/')
def home():
    """Home page with prediction form"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Enhanced health check with detailed system information"""
    try:
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'model_loaded': model is not None,
            'version': '1.0.0',
            'uptime_seconds': time.time() - app.start_time if hasattr(app, 'start_time') else 0,
            'database_status': 'connected'
        }
        
        # Test database connection
        try:
            stats = prediction_db.get_prediction_stats(hours=1)
            health_data['recent_predictions'] = stats.get('total_predictions', 0)
            health_data['database_status'] = 'connected'
        except Exception as e:
            health_data['database_status'] = 'error'
            health_data['database_error'] = str(e)
        
        # Test model prediction capability
        try:
            if model is not None:
                # Test with dummy data
                test_data = {
                    'longitude': -122.23, 'latitude': 37.88, 'housing_median_age': 41.0,
                    'total_rooms': 880.0, 'total_bedrooms': 129.0, 'population': 322.0,
                    'households': 126.0, 'median_income': 8.3252, 'ocean_proximity': 'NEAR BAY'
                }
                test_input = prepare_input(test_data)
                test_prediction = model.predict(test_input)[0]
                health_data['model_test_prediction'] = float(test_prediction)
                health_data['model_status'] = 'functional'
            else:
                health_data['model_status'] = 'not_loaded'
        except Exception as e:
            health_data['model_status'] = 'error'
            health_data['model_error'] = str(e)
        
        app.logger.info(f"Health check completed: {health_data['status']}")
        return jsonify(health_data)
    
    except Exception as e:
        error_response = {
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }
        app.logger.error(f"Health check failed: {e}")
        return jsonify(error_response), 500

@app.route('/api/stats')
def prediction_stats():
    """Get prediction statistics and system metrics"""
    try:
        # Get prediction statistics
        stats_24h = prediction_db.get_prediction_stats(hours=24)
        stats_1h = prediction_db.get_prediction_stats(hours=1)
        recent_predictions = prediction_db.get_recent_predictions(limit=10)
        
        response_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'statistics': {
                'last_24_hours': stats_24h,
                'last_hour': stats_1h
            },
            'recent_requests': recent_predictions,
            'system_info': {
                'model_version': '1.0.0',
                'model_loaded': model is not None,
                'uptime_seconds': time.time() - app.start_time if hasattr(app, 'start_time') else 0
            }
        }
        
        app.logger.info("Statistics retrieved successfully")
        return jsonify(response_data)
    
    except Exception as e:
        app.logger.error(f"Failed to get statistics: {e}")
        return jsonify({'error': f'Failed to get statistics: {str(e)}'}), 500

@app.route('/api/logs')
def get_logs():
    """Get recent application logs (last 100 lines)"""
    try:
        logs = []
        
        # Read recent logs from file
        try:
            with open('logs/app.log', 'r') as f:
                lines = f.readlines()
                # Get last 100 lines
                recent_lines = lines[-100:] if len(lines) > 100 else lines
                logs.extend([line.strip() for line in recent_lines])
        except FileNotFoundError:
            logs.append("Log file not found")
        
        response_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'log_lines': logs,
            'total_lines': len(logs)
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        app.logger.error(f"Failed to get logs: {e}")
        return jsonify({'error': f'Failed to get logs: {str(e)}'}), 500

@app.route('/api/cleanup')
def cleanup_logs():
    """Clean up old logs (admin endpoint)"""
    try:
        # Clean up logs older than 30 days
        cleanup_result = prediction_db.cleanup_old_logs(days=30)
        
        response_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'cleanup_result': cleanup_result,
            'status': 'completed'
        }
        
        app.logger.info(f"Log cleanup completed: {cleanup_result}")
        return jsonify(response_data)
    
    except Exception as e:
        app.logger.error(f"Failed to cleanup logs: {e}")
        return jsonify({'error': f'Failed to cleanup logs: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
@predictions_counter
@prediction_histogram
def predict():
    """Make a single prediction with comprehensive logging"""
    start_time = time.time()
    input_data = None
    
    try:
        # Validate request has JSON data
        if not request.is_json:
            error_msg = "Request must be JSON"
            app.logger.warning(f"Invalid request format from {request.remote_addr}")
            log_prediction_request('/api/predict', {}, status_code=400, error_message=error_msg)
            return jsonify({'error': error_msg}), 400
        
        input_data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'
        ]
        
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            error_msg = f"Missing required fields: {missing_fields}"
            app.logger.warning(f"Missing fields in request: {missing_fields}")
            log_prediction_request('/api/predict', input_data, status_code=400, error_message=error_msg)
            return jsonify({'error': error_msg}), 400
        
        # Validate numeric fields
        numeric_fields = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income'
        ]
        
        for field in numeric_fields:
            try:
                input_data[field] = float(input_data[field])
            except (ValueError, TypeError):
                error_msg = f"Field '{field}' must be numeric"
                app.logger.warning(f"Invalid numeric value for {field}: {input_data.get(field)}")
                log_prediction_request('/api/predict', input_data, status_code=400, error_message=error_msg)
                return jsonify({'error': error_msg}), 400
        
        # Validate ocean_proximity
        valid_ocean_proximity = ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND']
        if input_data['ocean_proximity'] not in valid_ocean_proximity:
            error_msg = f"ocean_proximity must be one of: {valid_ocean_proximity}"
            app.logger.warning(f"Invalid ocean_proximity: {input_data['ocean_proximity']}")
            log_prediction_request('/api/predict', input_data, status_code=400, error_message=error_msg)
            return jsonify({'error': error_msg}), 400
        
        # Check if model is loaded
        if model is None:
            error_msg = "Model not loaded"
            app.logger.error("Model is not loaded")
            log_prediction_request('/api/predict', input_data, status_code=500, error_message=error_msg)
            return jsonify({'error': error_msg}), 500
        
        # Prepare input and make prediction
        processed_input = prepare_input(input_data)
        prediction = model.predict(processed_input)[0]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            'prediction': float(prediction),
            'model_version': '1.0.0',
            'timestamp': datetime.utcnow().isoformat(),
            'processing_time_ms': round(processing_time * 1000, 2)
        }
        
        # Log successful prediction
        log_prediction_request('/api/predict', input_data, prediction=float(prediction), 
                             processing_time=processing_time)
        
        # Log model performance
        prediction_db.log_model_performance(
            model_version='1.0.0',
            prediction_type='single_prediction',
            input_features=input_data,
            prediction_value=float(prediction)
        )
        
        app.logger.info(f"Successful prediction: {prediction} (processing time: {processing_time:.3f}s)")
        return jsonify(response_data)
    
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Prediction failed: {str(e)}"
        
        app.logger.error(f"Prediction error: {e}")
        app.logger.error(f"Traceback: {traceback.format_exc()}")
        
        log_prediction_request('/api/predict', input_data or {}, 
                             processing_time=processing_time, status_code=500, error_message=error_msg)
        
        return jsonify({'error': error_msg}), 500

@app.route('/predict', methods=['POST'])
def predict_legacy():
    """Legacy endpoint for form-based predictions"""
    try:
        if model is None:
            return render_template('index.html', 
                                 prediction_text='Error: Model not loaded')
        
        # Get the input values from the form  
        input_features = [float(x) for x in request.form.values()]
        features_value = [np.array(input_features)]
        
        # Log the input features
        app.logger.info(f"Input features: {features_value}")

        # Make prediction
        prediction = model.predict(features_value)
        
        return render_template('index.html', 
                             prediction_text=f'Predicted house price is ${prediction[0]:,.2f}')
    
    except Exception as e:
        app.logger.error(f"Form prediction error: {e}")
        return render_template('index.html', 
                             prediction_text='Error: Could not make prediction')

# Add batch prediction endpoint
@app.route('/api/predict/batch', methods=['POST'])
@predictions_counter  
@prediction_histogram
def batch_predict():
    """Make batch predictions with comprehensive logging"""
    start_time = time.time()
    input_data = None
    
    try:
        # Validate request has JSON data
        if not request.is_json:
            error_msg = "Request must be JSON"
            app.logger.warning(f"Invalid batch request format from {request.remote_addr}")
            log_prediction_request('/api/predict/batch', {}, status_code=400, error_message=error_msg)
            return jsonify({'error': error_msg}), 400
        
        input_data = request.get_json()
        
        # Validate input is a list
        if not isinstance(input_data, list):
            error_msg = "Batch input must be a list of prediction requests"
            app.logger.warning(f"Invalid batch format: expected list, got {type(input_data)}")
            log_prediction_request('/api/predict/batch', input_data, status_code=400, error_message=error_msg)
            return jsonify({'error': error_msg}), 400
        
        if len(input_data) == 0:
            error_msg = "Batch input cannot be empty"
            app.logger.warning("Empty batch request received")
            log_prediction_request('/api/predict/batch', input_data, status_code=400, error_message=error_msg)
            return jsonify({'error': error_msg}), 400
        
        # Limit batch size
        max_batch_size = 100
        if len(input_data) > max_batch_size:
            error_msg = f"Batch size too large. Maximum {max_batch_size} items allowed"
            app.logger.warning(f"Batch size {len(input_data)} exceeds limit {max_batch_size}")
            log_prediction_request('/api/predict/batch', {'batch_size': len(input_data)}, 
                                 status_code=400, error_message=error_msg)
            return jsonify({'error': error_msg}), 400
        
        # Check if model is loaded
        if model is None:
            error_msg = "Model not loaded"
            app.logger.error("Model is not loaded for batch prediction")
            log_prediction_request('/api/predict/batch', input_data, status_code=500, error_message=error_msg)
            return jsonify({'error': error_msg}), 500
        
        predictions = []
        errors = []
        
        # Process each item in the batch
        for i, item in enumerate(input_data):
            try:
                # Validate each item
                required_fields = [
                    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
                    'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'
                ]
                
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    error_msg = f"Item {i}: Missing required fields: {missing_fields}"
                    errors.append({'index': i, 'error': error_msg})
                    predictions.append(None)
                    continue
                
                # Validate numeric fields
                numeric_fields = [
                    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
                    'total_bedrooms', 'population', 'households', 'median_income'
                ]
                
                for field in numeric_fields:
                    try:
                        item[field] = float(item[field])
                    except (ValueError, TypeError):
                        error_msg = f"Item {i}: Field '{field}' must be numeric"
                        errors.append({'index': i, 'error': error_msg})
                        predictions.append(None)
                        continue
                
                # Validate ocean_proximity
                valid_ocean_proximity = ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND']
                if item['ocean_proximity'] not in valid_ocean_proximity:
                    error_msg = f"Item {i}: ocean_proximity must be one of: {valid_ocean_proximity}"
                    errors.append({'index': i, 'error': error_msg})
                    predictions.append(None)
                    continue
                
                # Make prediction
                processed_input = prepare_input(item)
                prediction = model.predict(processed_input)[0]
                predictions.append(float(prediction))
                
                # Log individual prediction performance
                prediction_db.log_model_performance(
                    model_version='1.0.0',
                    prediction_type='batch_prediction',
                    input_features=item,
                    prediction_value=float(prediction)
                )
                
            except Exception as e:
                error_msg = f"Item {i}: Prediction failed: {str(e)}"
                errors.append({'index': i, 'error': error_msg})
                predictions.append(None)
                app.logger.error(f"Batch item {i} failed: {e}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            'predictions': predictions,
            'errors': errors,
            'count': len(input_data),
            'successful_count': len([p for p in predictions if p is not None]),
            'error_count': len(errors),
            'model_version': '1.0.0',
            'timestamp': datetime.utcnow().isoformat(),
            'processing_time_ms': round(processing_time * 1000, 2)
        }
        
        # Log batch prediction request
        log_prediction_request('/api/predict/batch', 
                             {'batch_size': len(input_data), 'successful_count': response_data['successful_count']}, 
                             prediction=predictions, processing_time=processing_time)
        
        app.logger.info(f"Batch prediction completed: {len(input_data)} items, "
                       f"{response_data['successful_count']} successful, "
                       f"{response_data['error_count']} errors (processing time: {processing_time:.3f}s)")
        
        return jsonify(response_data)
    
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Batch prediction failed: {str(e)}"
        
        app.logger.error(f"Batch prediction error: {e}")
        app.logger.error(f"Traceback: {traceback.format_exc()}")
        
        log_prediction_request('/api/predict/batch', input_data or {}, 
                             processing_time=processing_time, status_code=500, error_message=error_msg)
        
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    # Record start time for uptime calculation
    app.start_time = time.time()
    
    # Log startup information
    app.logger.info("=" * 50)
    app.logger.info("California Housing Prediction API Starting...")
    app.logger.info(f"Model path: {model_path}")
    app.logger.info(f"Model loaded: {model is not None}")
    app.logger.info(f"Database initialized: {prediction_db is not None}")
    app.logger.info(f"Logs directory: logs/")
    app.logger.info("=" * 50)
    
    # Log system metrics at startup
    try:
        prediction_db.log_system_metric('app_startup', 1.0, {'version': '1.0.0', 'model_loaded': str(model is not None)})
    except Exception as e:
        app.logger.warning(f"Failed to log startup metric: {e}")
    
    # Use environment variable for port, default to 5001
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
