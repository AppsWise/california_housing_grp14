"""
Model prediction and inference utilities with MLflow integration
"""

import pickle
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import sys

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.mlflow_tracking import get_model_registry, MLflowModelRegistry

logger = logging.getLogger(__name__)


class HousingPredictor:
    """California Housing Price Predictor with MLflow Registry Integration"""

    def __init__(self, 
                 model_path: str = "models/model.pkl",
                 use_mlflow_registry: bool = True,
                 model_name: str = "california_housing_best_model",
                 model_stage: str = "Production"):
        """
        Initialize predictor with trained model

        Args:
            model_path: Path to trained model file (fallback if MLflow fails)
            use_mlflow_registry: Whether to load model from MLflow registry
            model_name: Name of model in MLflow registry
            model_stage: Stage of model to load (Production, Staging, etc.)
        """
        self.model_path = model_path
        self.use_mlflow_registry = use_mlflow_registry
        self.model_name = model_name
        self.model_stage = model_stage
        self.model = None
        self.model_metadata = None
        self.mlflow_registry = None
        
        # Initialize MLflow registry if enabled
        if self.use_mlflow_registry:
            try:
                self.mlflow_registry = get_model_registry()
                logger.info("MLflow registry initialized for model loading")
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow registry: {e}")
                self.use_mlflow_registry = False
        
        self.load_model()

    def load_model(self) -> bool:
        """
        Load trained model from MLflow registry or file
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        # Try loading from MLflow registry first
        if self.use_mlflow_registry and self.mlflow_registry:
            try:
                model, metadata = self.mlflow_registry.load_model_from_registry(
                    self.model_name, self.model_stage
                )
                self.model = model
                self.model_metadata = metadata
                
                logger.info(f"Model loaded from MLflow registry: {self.model_name} "
                           f"version {metadata['version']} (stage: {metadata['stage']})")
                return True
                
            except Exception as e:
                logger.warning(f"Failed to load model from MLflow registry: {e}")
                logger.info("Falling back to local model file")
        
        # Fallback to loading from local file
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            
            # Create basic metadata for local model
            self.model_metadata = {
                "name": "local_model",
                "version": "file_based",
                "stage": "Local",
                "model_uri": self.model_path,
                "source": "local_file"
            }
            
            logger.info(f"Model loaded successfully from local file: {self.model_path}")
            return True
            
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model
        
        Returns:
            Dictionary with model information
        """
        if not self.model_metadata:
            return {"error": "No model metadata available"}
        
        info = {
            "model_loaded": self.model is not None,
            "model_metadata": self.model_metadata,
            "mlflow_enabled": self.use_mlflow_registry,
            "model_type": str(type(self.model).__name__) if self.model else None
        }
        
        # Add MLflow-specific information if available
        if self.use_mlflow_registry and self.mlflow_registry:
            try:
                registry_info = self.mlflow_registry.get_model_info(self.model_name)
                info["registry_info"] = registry_info
            except Exception as e:
                info["registry_error"] = str(e)
        
        return info

    def reload_model(self, force_mlflow: bool = False) -> bool:
        """
        Reload model from registry or file
        
        Args:
            force_mlflow: Force reload from MLflow registry even if disabled
            
        Returns:
            True if reload successful, False otherwise
        """
        logger.info("Reloading model...")
        
        if force_mlflow or self.use_mlflow_registry:
            self.use_mlflow_registry = True
            if not self.mlflow_registry:
                try:
                    self.mlflow_registry = get_model_registry()
                except Exception as e:
                    logger.error(f"Failed to initialize MLflow registry: {e}")
                    return False
        
        try:
            return self.load_model()
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
            return False

    def predict(
        self,
        features: Union[
            Dict[str, float], List[Dict[str, float]], pd.DataFrame, np.ndarray
        ],
    ) -> Union[float, List[float]]:
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
        elif isinstance(features, list) and all(
            isinstance(f, dict) for f in features
        ):
            # Batch prediction from list of dictionaries
            features_df = pd.DataFrame(features)
        elif isinstance(features, pd.DataFrame):
            features_df = features
        elif isinstance(features, np.ndarray):
            # Assume preprocessed array
            prediction = self.model.predict(features)
            return (
                prediction.tolist()
                if len(prediction) > 1
                else float(prediction[0])
            )
        else:
            raise ValueError("Invalid input format for features")

        # Apply feature engineering
        features_df = self._engineer_features(features_df)

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

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to input data
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with engineered features
        """
        df_copy = df.copy()
        
        # Create derived features that the model expects
        if "total_bedrooms" in df_copy.columns and "households" in df_copy.columns:
            df_copy["bedrooms_per_household"] = (
                df_copy["total_bedrooms"] / df_copy["households"]
            )

        if "population" in df_copy.columns and "households" in df_copy.columns:
            df_copy["population_per_household"] = (
                df_copy["population"] / df_copy["households"]
            )

        if "total_rooms" in df_copy.columns and "households" in df_copy.columns:
            df_copy["rooms_per_household"] = (
                df_copy["total_rooms"] / df_copy["households"]
            )
            
        if "total_bedrooms" in df_copy.columns and "total_rooms" in df_copy.columns:
            df_copy["bedrooms_per_room"] = (
                df_copy["total_bedrooms"] / df_copy["total_rooms"]
            )
        
        return df_copy

    def predict_proba(
        self, features: Union[Dict[str, float], pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Get prediction probabilities (for classification models)

        Args:
            features: Input features

        Returns:
            Prediction probabilities
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(features)
        else:
            raise NotImplementedError(
                "Model does not support probability predictions"
            )

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from model

        Returns:
            Dictionary of feature names and importance scores
        """
        if hasattr(self.model, "feature_importances_"):
            # For tree-based models
            if hasattr(self.model, "feature_names_in_"):
                feature_names = self.model.feature_names_in_
            else:
                feature_names = [
                    f"feature_{i}"
                    for i in range(len(self.model.feature_importances_))
                ]

            return dict(zip(feature_names, self.model.feature_importances_))
        elif hasattr(self.model, "coef_"):
            # For linear models
            if hasattr(self.model, "feature_names_in_"):
                feature_names = self.model.feature_names_in_
            else:
                feature_names = [
                    f"feature_{i}" for i in range(len(self.model.coef_))
                ]

            return dict(zip(feature_names, np.abs(self.model.coef_)))
        else:
            raise NotImplementedError(
                "Model does not support feature importance"
            )


def create_prediction_input(
    longitude: float,
    latitude: float,
    housing_median_age: float,
    total_rooms: float,
    total_bedrooms: float,
    population: float,
    households: float,
    median_income: float,
    ocean_proximity: str,
) -> Dict[str, Any]:
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
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }
