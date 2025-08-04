"""
Model prediction and inference utilities
"""

import pickle
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class HousingPredictor:
    """California Housing Price Predictor"""
    
    def __init__(self, model_path: str = "models/model.pkl"):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to trained model file
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load trained model from file"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, features: Union[Dict[str, float], List[Dict[str, float]], pd.DataFrame, np.ndarray]) -> Union[float, List[float]]:
        """
        Make predictions on input features
        
        Args:
            features: Input features for prediction
            
        Returns:
            Predicted house price(s)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Convert input to appropriate format
        if isinstance(features, dict):
            # Single prediction from dictionary
            features_df = pd.DataFrame([features])
        elif isinstance(features, list) and all(isinstance(f, dict) for f in features):
            # Batch prediction from list of dictionaries
            features_df = pd.DataFrame(features)
        elif isinstance(features, pd.DataFrame):
            features_df = features
        elif isinstance(features, np.ndarray):
            # Assume preprocessed array
            prediction = self.model.predict(features)
            return prediction.tolist() if len(prediction) > 1 else float(prediction[0])
        else:
            raise ValueError("Invalid input format for features")
        
        # Make prediction
        try:
            prediction = self.model.predict(features_df)
            logger.info(f"Made prediction for {len(features_df)} samples")
            
            if len(prediction) == 1:
                return float(prediction[0])
            return prediction.tolist()
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def predict_proba(self, features: Union[Dict[str, float], pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities (for classification models)
        
        Args:
            features: Input features
            
        Returns:
            Prediction probabilities
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(features)
        else:
            raise NotImplementedError("Model does not support probability predictions")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from model
        
        Returns:
            Dictionary of feature names and importance scores
        """
        if hasattr(self.model, 'feature_importances_'):
            # For tree-based models
            if hasattr(self.model, 'feature_names_in_'):
                feature_names = self.model.feature_names_in_
            else:
                feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
            
            return dict(zip(feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            # For linear models
            if hasattr(self.model, 'feature_names_in_'):
                feature_names = self.model.feature_names_in_
            else:
                feature_names = [f"feature_{i}" for i in range(len(self.model.coef_))]
            
            return dict(zip(feature_names, np.abs(self.model.coef_)))
        else:
            raise NotImplementedError("Model does not support feature importance")

def create_prediction_input(longitude: float, latitude: float, housing_median_age: float,
                          total_rooms: float, total_bedrooms: float, population: float,
                          households: float, median_income: float, ocean_proximity: str) -> Dict[str, Any]:
    """
    Create prediction input dictionary
    
    Args:
        longitude: Longitude coordinate
        latitude: Latitude coordinate
        housing_median_age: Median age of houses in block
        total_rooms: Total number of rooms in block
        total_bedrooms: Total number of bedrooms in block
        population: Population of block
        households: Number of households in block
        median_income: Median income of households
        ocean_proximity: Ocean proximity category
        
    Returns:
        Dictionary with prediction input
    """
    return {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }
