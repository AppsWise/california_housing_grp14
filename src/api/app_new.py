import os
import logging
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Configure logging to both file and console
log_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

# File handler
file_handler = logging.FileHandler('/var/log/app.log')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# Console handler for development
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# Configure logger
app.logger.addHandler(file_handler)
app.logger.addHandler(console_handler)
app.logger.setLevel(logging.INFO)

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
    Prepare input features for model prediction
    Features expected in order:
    longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
    population, households, median_income, rooms_per_household,
    bedrooms_per_room, population_per_household, ocean_proximity
    """
    try:
        # Extract base features from input
        longitude = float(features.get('longitude'))
        latitude = float(features.get('latitude'))
        housing_median_age = float(features.get('housing_median_age'))
        total_rooms = float(features.get('total_rooms'))
        total_bedrooms = float(features.get('total_bedrooms'))
        population = float(features.get('population'))
        households = float(features.get('households'))
        median_income = float(features.get('median_income'))
        ocean_proximity = str(features.get('ocean_proximity'))
        
        # Calculate derived features
        rooms_per_household = total_rooms / households if households > 0 else 0
        bedrooms_per_room = total_bedrooms / total_rooms if total_rooms > 0 else 0
        population_per_household = population / households if households > 0 else 0
        
        # Create DataFrame with proper column names and categorical encoding
        input_data = pd.DataFrame({
            'longitude': [longitude],
            'latitude': [latitude], 
            'housing_median_age': [housing_median_age],
            'total_rooms': [total_rooms],
            'total_bedrooms': [total_bedrooms],
            'population': [population],
            'households': [households],
            'median_income': [median_income],
            'rooms_per_household': [rooms_per_household],
            'bedrooms_per_room': [bedrooms_per_room],
            'population_per_household': [population_per_household],
            'ocean_proximity': [ocean_proximity]
        })
        
        # Ensure ocean_proximity is treated as categorical string
        input_data['ocean_proximity'] = input_data['ocean_proximity'].astype(str)
        
        app.logger.info(f"Prepared input data shape: {input_data.shape}")
        app.logger.info(f"Ocean proximity value: {ocean_proximity}")
        
        return input_data
        
    except Exception as e:
        app.logger.error(f"Error preparing input: {e}")
        raise ValueError(f"Invalid input data: {e}")

@app.route('/')
def home():
    """Home page with prediction form"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint for container monitoring"""
    health_data = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'version': '1.0.0'
    }
    return jsonify(health_data)

@app.route('/api/predict', methods=['POST'])
@predictions_counter
@prediction_histogram
def predict_api():
    """API endpoint for making predictions with JSON input/output"""
    try:
        # Check if model is loaded
        if model is None:
            app.logger.error("Model not loaded")
            return jsonify({
                'error': 'Model not available',
                'message': 'Machine learning model failed to load'
            }), 500
        
        # Get JSON data from request
        json_data = request.get_json()
        if not json_data:
            return jsonify({
                'error': 'No input data',
                'message': 'Please provide input data in JSON format'
            }), 400
        
        # Handle both single prediction and batch prediction
        if isinstance(json_data, list):
            # Batch prediction
            predictions = []
            for item in json_data:
                input_df = prepare_input(item)
                prediction = model.predict(input_df)[0]
                predictions.append(float(prediction))
            
            app.logger.info(f"Batch prediction completed: {len(predictions)} predictions")
            return jsonify({
                'predictions': predictions,
                'count': len(predictions),
                'model_version': '1.0.0',
                'timestamp': datetime.now().isoformat()
            })
        else:
            # Single prediction
            input_df = prepare_input(json_data)
            prediction = model.predict(input_df)[0]
            
            app.logger.info(f"Single prediction completed: {prediction}")
            return jsonify({
                'prediction': float(prediction),
                'model_version': '1.0.0', 
                'timestamp': datetime.now().isoformat()
            })
    
    except ValueError as ve:
        app.logger.warning(f"Validation error: {ve}")
        return jsonify({
            'error': 'Invalid input data',
            'message': str(ve)
        }), 400
    
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': 'Internal server error'
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
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
def predict_batch():
    """Dedicated batch prediction endpoint"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not available',
                'message': 'Machine learning model failed to load'
            }), 500
        
        json_data = request.get_json()
        if not json_data or not isinstance(json_data, list):
            return jsonify({
                'error': 'Invalid input',
                'message': 'Expected array of prediction objects'
            }), 400
        
        predictions = []
        for i, item in enumerate(json_data):
            try:
                input_df = prepare_input(item)
                prediction = model.predict(input_df)[0]
                predictions.append(float(prediction))
            except Exception as e:
                app.logger.error(f"Error processing item {i}: {e}")
                return jsonify({
                    'error': f'Error processing item {i}',
                    'message': str(e)
                }), 400
        
        app.logger.info(f"Batch prediction completed: {len(predictions)} predictions")
        return jsonify({
            'predictions': predictions,
            'count': len(predictions),
            'model_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        app.logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'error': 'Batch prediction failed',
            'message': 'Internal server error'
        }), 500

if __name__ == '__main__':
    # Use environment variable for port, default to 5001
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
