import os
import logging
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='/var/log/app.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')

# Construct the absolute path to the model file for robustness

#TODO: model.pkl is a dummy file. It needs to be replaced with the correct thing.Add
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '..', 'artifacts', 'model.pkl')

# Load the trained model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    # Log the input features
    app.logger.info(f"Input features: {features_value}")

    # Make prediction
    prediction = model.predict(features_value)
    
    return render_template('index.html', prediction_text='Predicted house price is ${}'.format(prediction))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)