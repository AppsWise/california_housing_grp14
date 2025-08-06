"""
MLflow Model Registry and Tracking Integration
==============================================

Comprehensive MLflow integration for model versioning, tracking, and registry management.
Supports model loading from registry, experiment logging, and model deployment tracking.

Author: Group 14
Date: August 2025
"""

import os
import time
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class MLflowModelRegistry:
    """MLflow Model Registry Integration for production model management"""

    def __init__(self, tracking_uri: str = None, registry_uri: str = None):
        """
        Initialize MLflow Model Registry client
        
        Args:
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow registry URI (defaults to tracking_uri)
        """
        # Set MLflow URIs
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.registry_uri = registry_uri or self.tracking_uri
        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        
        self.client = MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.registry_uri)
        
        logger.info(f"MLflow Model Registry initialized with tracking URI: {self.tracking_uri}")

    def get_latest_model_version(self, model_name: str, stage: str = "Production") -> Optional[ModelVersion]:
        """
        Get the latest model version from registry
        
        Args:
            model_name: Name of the registered model
            stage: Model stage (Production, Staging, Archived, None)
        
        Returns:
            ModelVersion object or None if not found
        """
        try:
            latest_versions = self.client.get_latest_versions(
                name=model_name, 
                stages=[stage]
            )
            if latest_versions:
                return latest_versions[0]
            
            # If no model in specified stage, get latest version regardless of stage
            all_versions = self.client.search_model_versions(f"name='{model_name}'")
            if all_versions:
                # Sort by version number and return latest
                sorted_versions = sorted(all_versions, key=lambda x: int(x.version), reverse=True)
                logger.warning(f"No model in {stage} stage, using latest version: {sorted_versions[0].version}")
                return sorted_versions[0]
                
            return None
        except Exception as e:
            logger.error(f"Error getting latest model version for {model_name}: {e}")
            return None

    def load_model_from_registry(self, model_name: str, stage: str = "Production") -> Tuple[Any, Dict[str, Any]]:
        """
        Load model from MLflow registry
        
        Args:
            model_name: Name of the registered model
            stage: Model stage to load from
        
        Returns:
            Tuple of (model_object, model_metadata)
        """
        try:
            # Get latest model version
            model_version = self.get_latest_model_version(model_name, stage)
            if not model_version:
                raise ValueError(f"No model found for {model_name} in stage {stage}")
            
            # Load model
            model_uri = f"models:/{model_name}/{model_version.version}"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Get model metadata
            model_metadata = {
                "name": model_name,
                "version": model_version.version,
                "stage": model_version.current_stage,
                "run_id": model_version.run_id,
                "creation_timestamp": model_version.creation_timestamp,
                "last_updated_timestamp": model_version.last_updated_timestamp,
                "description": model_version.description,
                "status": model_version.status,
                "model_uri": model_uri
            }
            
            # Get run metadata
            run = self.client.get_run(model_version.run_id)
            model_metadata.update({
                "run_name": run.info.run_name,
                "experiment_id": run.info.experiment_id,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            })
            
            logger.info(f"Loaded model {model_name} version {model_version.version} from registry")
            return model, model_metadata
            
        except Exception as e:
            logger.error(f"Error loading model from registry: {e}")
            raise

    def register_model(self, model, model_name: str, run_id: str, description: str = None) -> str:
        """
        Register a trained model in MLflow registry
        
        Args:
            model: Trained model object
            model_name: Name for the registered model
            run_id: MLflow run ID
            description: Model description
        
        Returns:
            Model version URI
        """
        try:
            # Register model
            model_uri = mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=model_name
            )
            
            # Update model version description
            if description:
                latest_version = self.get_latest_model_version(model_name, stage=None)
                if latest_version:
                    self.client.update_model_version(
                        name=model_name,
                        version=latest_version.version,
                        description=description
                    )
            
            logger.info(f"Model registered as {model_name} with URI: {model_uri}")
            return model_uri
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    def transition_model_stage(self, model_name: str, version: str, stage: str, archive_existing: bool = True) -> bool:
        """
        Transition model to a different stage
        
        Args:
            model_name: Name of the registered model
            version: Model version to transition
            stage: Target stage (Production, Staging, Archived)
            archive_existing: Whether to archive existing models in target stage
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing
            )
            
            logger.info(f"Model {model_name} version {version} transitioned to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
            return False

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a registered model
        
        Args:
            model_name: Name of the registered model
        
        Returns:
            Dictionary with model information
        """
        try:
            # Get registered model
            model = self.client.get_registered_model(model_name)
            
            # Get all versions
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            model_info = {
                "name": model.name,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "description": model.description,
                "latest_versions": {},
                "all_versions": []
            }
            
            # Get latest versions by stage
            for stage in ["Production", "Staging", "Archived"]:
                latest = self.get_latest_model_version(model_name, stage)
                if latest:
                    model_info["latest_versions"][stage] = {
                        "version": latest.version,
                        "run_id": latest.run_id,
                        "status": latest.status,
                        "creation_timestamp": latest.creation_timestamp
                    }
            
            # Add all versions info
            for version in versions:
                version_info = {
                    "version": version.version,
                    "stage": version.current_stage,
                    "status": version.status,
                    "run_id": version.run_id,
                    "creation_timestamp": version.creation_timestamp,
                    "description": version.description
                }
                model_info["all_versions"].append(version_info)
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}

    def list_registered_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models
        
        Returns:
            List of model information dictionaries
        """
        try:
            models = self.client.search_registered_models()
            model_list = []
            
            for model in models:
                model_info = {
                    "name": model.name,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "description": model.description,
                    "latest_versions": {}
                }
                
                # Get latest versions by stage
                for stage in ["Production", "Staging", "Archived"]:
                    latest = self.get_latest_model_version(model.name, stage)
                    if latest:
                        model_info["latest_versions"][stage] = latest.version
                
                model_list.append(model_info)
            
            return model_list
            
        except Exception as e:
            logger.error(f"Error listing registered models: {e}")
            return []


class MLflowExperimentTracker:
    """MLflow Experiment Tracking for API predictions and model performance"""

    def __init__(self, experiment_name: str = "california_housing_api", tracking_uri: str = None):
        """
        Initialize experiment tracker
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
        """
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(self.tracking_uri)
        
        self.experiment_name = experiment_name
        self.experiment = mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow Experiment Tracker initialized for experiment: {experiment_name}")

    def log_prediction_batch(self, predictions: List[Dict[str, Any]], model_metadata: Dict[str, Any]):
        """
        Log a batch of predictions to MLflow
        
        Args:
            predictions: List of prediction dictionaries
            model_metadata: Model metadata from registry
        """
        try:
            with mlflow.start_run(run_name=f"prediction_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log model metadata
                mlflow.log_params({
                    "model_name": model_metadata.get("name"),
                    "model_version": model_metadata.get("version"),
                    "model_stage": model_metadata.get("stage"),
                    "batch_size": len(predictions)
                })
                
                # Log prediction statistics
                prediction_values = [p["prediction"] for p in predictions]
                mlflow.log_metrics({
                    "batch_size": len(predictions),
                    "avg_prediction": sum(prediction_values) / len(prediction_values),
                    "min_prediction": min(prediction_values),
                    "max_prediction": max(prediction_values),
                    "prediction_std": (sum((x - sum(prediction_values)/len(prediction_values))**2 for x in prediction_values) / len(prediction_values)) ** 0.5
                })
                
                # Log processing times
                processing_times = [p["processing_time_ms"] for p in predictions]
                mlflow.log_metrics({
                    "avg_processing_time_ms": sum(processing_times) / len(processing_times),
                    "max_processing_time_ms": max(processing_times),
                    "min_processing_time_ms": min(processing_times)
                })
                
                # Log timestamp
                mlflow.log_param("batch_timestamp", datetime.utcnow().isoformat())
                
        except Exception as e:
            logger.error(f"Error logging prediction batch to MLflow: {e}")

    def log_model_performance(self, model_name: str, metrics: Dict[str, float], 
                            evaluation_data: Dict[str, Any] = None):
        """
        Log model performance metrics
        
        Args:
            model_name: Name of the model
            metrics: Performance metrics dictionary
            evaluation_data: Optional evaluation data info
        """
        try:
            with mlflow.start_run(run_name=f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("evaluation_timestamp", datetime.utcnow().isoformat())
                
                # Log performance metrics
                mlflow.log_metrics(metrics)
                
                # Log evaluation data info if provided
                if evaluation_data:
                    mlflow.log_params(evaluation_data)
                
        except Exception as e:
            logger.error(f"Error logging model performance to MLflow: {e}")

    def log_api_metrics(self, endpoint: str, method: str, response_time: float, 
                       status_code: int, model_metadata: Dict[str, Any] = None):
        """
        Log API call metrics
        
        Args:
            endpoint: API endpoint called
            method: HTTP method
            response_time: Response time in milliseconds
            status_code: HTTP status code
            model_metadata: Optional model metadata
        """
        try:
            # Log to current run if one exists, otherwise create a new one
            if mlflow.active_run():
                mlflow.log_metrics({
                    "response_time_ms": response_time,
                    "status_code": status_code
                })
                mlflow.log_params({
                    "endpoint": endpoint,
                    "method": method,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                if model_metadata:
                    mlflow.log_params({
                        "model_name": model_metadata.get("name"),
                        "model_version": model_metadata.get("version")
                    })
            
        except Exception as e:
            logger.error(f"Error logging API metrics to MLflow: {e}")


# Global MLflow instances (initialized when needed)
_model_registry = None
_experiment_tracker = None


def get_model_registry() -> MLflowModelRegistry:
    """Get global MLflow model registry instance"""
    global _model_registry
    if _model_registry is None:
        _model_registry = MLflowModelRegistry()
    return _model_registry


def get_experiment_tracker(experiment_name: str = "california_housing_api") -> MLflowExperimentTracker:
    """Get global MLflow experiment tracker instance"""
    global _experiment_tracker
    if _experiment_tracker is None:
        _experiment_tracker = MLflowExperimentTracker(experiment_name)
    return _experiment_tracker


def initialize_mlflow_tracking():
    """Initialize MLflow tracking components"""
    try:
        # Initialize both components
        registry = get_model_registry()
        tracker = get_experiment_tracker()
        
        logger.info("MLflow tracking components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing MLflow tracking: {e}")
        return False
