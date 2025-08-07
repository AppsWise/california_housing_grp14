"""
California Housing Price Prediction API - Clean MLOps Implementation
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from flask import Flask, request, jsonify, render_template
from pydantic import ValidationError

# Import project modules
from src.api.schemas import (
    HousingPredictionInput,
    BatchPredictionInput,
    PredictionResponse,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
)
from src.models.predict import HousingPredictor
from src.utils.config import api_config
from src.utils.logging_config import setup_logging, StructuredLogger
from src.utils.monitoring import (
    metrics_collector,
    health_checker,
    performance_monitor,
)
from src.utils.database import PredictionDatabase
from src.utils.mlflow_integration import log_prediction_to_mlflow
from src.utils.mlflow_tracking import (
    get_model_registry,
    get_experiment_tracker,
    initialize_mlflow_tracking
)
from src.utils.prometheus_metrics import prometheus_metrics

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Setup logging
logger = setup_logging(
    log_level=api_config.get("logging.level", "INFO"),
    log_file=api_config.get("logging.file", "monitoring/logs/app.log"),
    log_format=api_config.get("logging.format"),
)
structured_logger = StructuredLogger("api")

# Global variables
model = None
db_manager = None
mlflow_registry = None
mlflow_tracker = None
start_time = time.time()
prediction_batch = []  # For batching predictions to MLflow


def load_model():
    """Load the trained model with MLflow integration"""
    global model

    model_path = api_config.get("model.path", "models/model.pkl")
    use_mlflow = api_config.get("mlflow.enabled", True)
    model_name = api_config.get("mlflow.model_name", "california_housing_best_model")
    model_stage = api_config.get("mlflow.model_stage", "Production")

    try:
        start_time_load = time.time()
        model = HousingPredictor(
            model_path=model_path,
            use_mlflow_registry=use_mlflow,
            model_name=model_name,
            model_stage=model_stage
        )
        load_time = (time.time() - start_time_load) * 1000

        # Get model info for logging
        model_info = model.get_model_info()
        
        structured_logger.log_model_load(
            model_path=model_info.get("model_metadata", {}).get("model_uri", model_path),
            model_version=model_info.get("model_metadata", {}).get("version", "unknown"),
            load_time=load_time,
        )

        logger.info(f"Model loaded successfully: {model_info.get('model_metadata', {}).get('name', 'unknown')}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def initialize_mlflow():
    """Initialize MLflow tracking components"""
    global mlflow_registry, mlflow_tracker

    if not api_config.get("mlflow.enabled", True):
        logger.info("MLflow integration disabled")
        return True

    try:
        # Initialize MLflow tracking
        success = initialize_mlflow_tracking()
        if success:
            mlflow_registry = get_model_registry()
            mlflow_tracker = get_experiment_tracker()
            logger.info("MLflow tracking components initialized successfully")
        return success
    except Exception as e:
        logger.error(f"Failed to initialize MLflow: {e}")
        return False


def check_mlflow_connection():
    """Health check for MLflow connection"""
    if not api_config.get("mlflow.enabled", True):
        return True
    
    try:
        if mlflow_registry:
            # Try to list models to check connection
            mlflow_registry.list_registered_models()
            return True
        return False
    except Exception:
        return False


def initialize_database():
    """Initialize database connection"""
    global db_manager

    try:
        db_path = api_config.get(
            "database.path", "logs/predictions.db"
        )
        db_manager = PredictionDatabase(db_path)
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


# Register health checks
def check_model_loaded():
    """Health check for model availability"""
    return model is not None


def check_database_connection():
    """Health check for database connection"""
    return db_manager is not None and db_manager.test_connection()


health_checker.register_check("model_loaded", check_model_loaded)
health_checker.register_check("database_connection", check_database_connection)
health_checker.register_check("mlflow_connection", check_mlflow_connection)


def startup():
    """Initialize application on startup"""
    logger.info("Initializing California Housing API...")

    # Initialize MLflow
    if not initialize_mlflow():
        logger.warning("MLflow initialization failed, continuing without MLflow")

    # Load model
    if not load_model():
        logger.error("Failed to load model during startup")

    # Initialize database
    if not initialize_database():
        logger.error("Failed to initialize database during startup")

    logger.info("California Housing API initialized successfully")


# Initialize on import
startup()


@app.route("/")
def home():
    """Home page with API documentation"""
    return render_template("index.html")


@app.route("/health")
def health_check():
    """Health check endpoint"""
    start_time_check = time.time()

    health_results = health_checker.run_checks()
    uptime = time.time() - start_time

    # Get model metadata if available
    model_version = "unknown"
    model_name = "unknown"
    if model and hasattr(model, 'model_metadata') and model.model_metadata:
        model_version = model.model_metadata.get("version", "unknown")
        model_name = model.model_metadata.get("name", "unknown")

    response = HealthResponse(
        status=health_results["status"],
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model is not None,
        model_version=model_version,
        uptime_seconds=uptime,
    )

    structured_logger.log_health_check(
        status=health_results["status"], checks=health_results["checks"]
    )

    processing_time = (time.time() - start_time_check) * 1000
    performance_monitor.record_request(processing_time, 200)

    return jsonify(response.model_dump()), 200


@app.route("/metrics")
def get_metrics():
    """Metrics endpoint for monitoring"""
    if not api_config.get("monitoring.enable_metrics", True):
        return jsonify({"error": "Metrics endpoint disabled"}), 404

    start_time_metrics = time.time()

    # Get performance stats
    perf_stats = performance_monitor.get_current_stats()

    # Get prediction metrics from database
    prediction_count_24h = 0
    if db_manager:
        try:
            stats = db_manager.get_prediction_stats(hours=24)
            prediction_count_24h = stats.get('total_predictions', 0)
        except Exception as e:
            logger.error(f"Error getting prediction stats: {e}")

    response = MetricsResponse(
        total_predictions=perf_stats["total_requests"],
        predictions_last_hour=prediction_count_24h,
        average_response_time_ms=perf_stats["average_response_time_ms"],
        error_rate_percentage=perf_stats["error_rate_percentage"],
        model_accuracy=None,  # Could be calculated from recent predictions
        last_updated=datetime.utcnow().isoformat(),
    )

    processing_time = (time.time() - start_time_metrics) * 1000
    performance_monitor.record_request(processing_time, 200)

    return jsonify(response.model_dump()), 200


@app.route("/metrics/prometheus")
def get_prometheus_metrics():
    """Prometheus-compatible metrics endpoint"""
    if not api_config.get("monitoring.enable_metrics", True):
        return "# Metrics disabled\n", 404, {'Content-Type': 'text/plain'}

    # Get performance stats
    perf_stats = performance_monitor.get_current_stats()

    # Get prediction metrics from database
    prediction_count_24h = 0
    if db_manager:
        try:
            stats = db_manager.get_prediction_stats(hours=24)
            prediction_count_24h = stats.get('total_predictions', 0)
        except Exception as e:
            logger.error(f"Error getting prediction stats: {e}")

    # Format metrics in Prometheus format
    metrics = f"""# HELP housing_api_predictions_total Total number of predictions made
# TYPE housing_api_predictions_total counter
housing_api_predictions_total {perf_stats["total_requests"]}

# HELP housing_api_predictions_24h Number of predictions in last 24 hours
# TYPE housing_api_predictions_24h gauge
housing_api_predictions_24h {prediction_count_24h}

# HELP housing_api_response_time_ms Average response time in milliseconds
# TYPE housing_api_response_time_ms gauge
housing_api_response_time_ms {perf_stats["average_response_time_ms"]}

# HELP housing_api_error_rate Error rate percentage
# TYPE housing_api_error_rate gauge
housing_api_error_rate {perf_stats["error_rate_percentage"]}

# HELP housing_api_model_loaded Whether the ML model is loaded
# TYPE housing_api_model_loaded gauge
housing_api_model_loaded {1 if model is not None else 0}
"""

    return metrics, 200, {'Content-Type': 'text/plain; version=0.0.4; charset=utf-8'}


@app.route("/api/monitoring/metrics")
def get_comprehensive_prometheus_metrics():
    """Comprehensive Prometheus metrics endpoint with full monitoring data"""
    if not api_config.get("monitoring.enable_metrics", True):
        return "# Metrics disabled\n", 404, {'Content-Type': 'text/plain'}

    try:
        # Get comprehensive metrics from our collector
        metrics_output = prometheus_metrics.get_metrics_output()
        return metrics_output, 200, {'Content-Type': 'text/plain; version=0.0.4; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error generating comprehensive metrics: {e}")
        return f"# Error generating metrics: {str(e)}\n", 500, {'Content-Type': 'text/plain'}


@app.route("/api/model/info")
def get_model_info():
    """Get current model information"""
    start_time_info = time.time()

    try:
        if model is None:
            error_response = ErrorResponse(
                error="Model Not Available",
                message="No model loaded",
                timestamp=datetime.utcnow().isoformat(),
            )
            return jsonify(error_response.model_dump()), 503

        model_info = model.get_model_info()
        
        response = {
            "model_info": model_info,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": (time.time() - start_time_info) * 1000
        }

        performance_monitor.record_request((time.time() - start_time_info) * 1000, 200)
        return jsonify(response), 200

    except Exception as e:
        error_response = ErrorResponse(
            error="Model Info Error",
            message=f"Error retrieving model info: {str(e)}",
            timestamp=datetime.utcnow().isoformat(),
        )
        performance_monitor.record_request((time.time() - start_time_info) * 1000, 500)
        return jsonify(error_response.model_dump()), 500


@app.route("/api/model/reload", methods=["POST"])
def reload_model():
    """Reload model from registry or file"""
    start_time_reload = time.time()

    try:
        force_mlflow = request.json.get("force_mlflow", False) if request.json else False
        
        if model is None:
            error_response = ErrorResponse(
                error="Model Not Available",
                message="No model instance available for reload",
                timestamp=datetime.utcnow().isoformat(),
            )
            return jsonify(error_response.model_dump()), 503

        success = model.reload_model(force_mlflow=force_mlflow)
        
        if success:
            model_info = model.get_model_info()
            response = {
                "status": "success",
                "message": "Model reloaded successfully",
                "model_info": model_info,
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time_ms": (time.time() - start_time_reload) * 1000
            }
            performance_monitor.record_request((time.time() - start_time_reload) * 1000, 200)
            return jsonify(response), 200
        else:
            error_response = ErrorResponse(
                error="Model Reload Failed",
                message="Failed to reload model",
                timestamp=datetime.utcnow().isoformat(),
            )
            performance_monitor.record_request((time.time() - start_time_reload) * 1000, 500)
            return jsonify(error_response.model_dump()), 500

    except Exception as e:
        error_response = ErrorResponse(
            error="Model Reload Error",
            message=f"Error reloading model: {str(e)}",
            timestamp=datetime.utcnow().isoformat(),
        )
        performance_monitor.record_request((time.time() - start_time_reload) * 1000, 500)
        return jsonify(error_response.model_dump()), 500


@app.route("/api/mlflow/models")
def list_registered_models():
    """List all registered models in MLflow registry"""
    start_time_list = time.time()

    if not api_config.get("mlflow.enabled", True):
        return jsonify({"error": "MLflow integration disabled"}), 404

    try:
        if mlflow_registry is None:
            error_response = ErrorResponse(
                error="MLflow Not Available",
                message="MLflow registry not initialized",
                timestamp=datetime.utcnow().isoformat(),
            )
            return jsonify(error_response.model_dump()), 503

        models = mlflow_registry.list_registered_models()
        
        response = {
            "registered_models": models,
            "count": len(models),
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": (time.time() - start_time_list) * 1000
        }

        performance_monitor.record_request((time.time() - start_time_list) * 1000, 200)
        return jsonify(response), 200

    except Exception as e:
        error_response = ErrorResponse(
            error="MLflow Registry Error",
            message=f"Error accessing MLflow registry: {str(e)}",
            timestamp=datetime.utcnow().isoformat(),
        )
        performance_monitor.record_request((time.time() - start_time_list) * 1000, 500)
        return jsonify(error_response.model_dump()), 500


@app.route("/api/mlflow/models/<model_name>")
def get_model_details(model_name: str):
    """Get detailed information about a specific registered model"""
    start_time_details = time.time()

    if not api_config.get("mlflow.enabled", True):
        return jsonify({"error": "MLflow integration disabled"}), 404

    try:
        if mlflow_registry is None:
            error_response = ErrorResponse(
                error="MLflow Not Available",
                message="MLflow registry not initialized",
                timestamp=datetime.utcnow().isoformat(),
            )
            return jsonify(error_response.model_dump()), 503

        model_info = mlflow_registry.get_model_info(model_name)
        
        if not model_info:
            error_response = ErrorResponse(
                error="Model Not Found",
                message=f"Model {model_name} not found in registry",
                timestamp=datetime.utcnow().isoformat(),
            )
            return jsonify(error_response.model_dump()), 404

        response = {
            "model_details": model_info,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": (time.time() - start_time_details) * 1000
        }

        performance_monitor.record_request((time.time() - start_time_details) * 1000, 200)
        return jsonify(response), 200

    except Exception as e:
        error_response = ErrorResponse(
            error="MLflow Registry Error",
            message=f"Error retrieving model details: {str(e)}",
            timestamp=datetime.utcnow().isoformat(),
        )
        performance_monitor.record_request((time.time() - start_time_details) * 1000, 500)
        return jsonify(error_response.model_dump()), 500


@app.route("/api/mlflow/models/<model_name>/transition", methods=["POST"])
def transition_model_stage(model_name: str):
    """Transition model to a different stage"""
    start_time_transition = time.time()

    if not api_config.get("mlflow.enabled", True):
        return jsonify({"error": "MLflow integration disabled"}), 404

    try:
        if mlflow_registry is None:
            error_response = ErrorResponse(
                error="MLflow Not Available",
                message="MLflow registry not initialized",
                timestamp=datetime.utcnow().isoformat(),
            )
            return jsonify(error_response.model_dump()), 503

        # Parse request data
        data = request.get_json()
        if not data:
            raise ValueError("No request data provided")

        version = data.get("version")
        stage = data.get("stage")
        archive_existing = data.get("archive_existing", True)

        if not version or not stage:
            error_response = ErrorResponse(
                error="Invalid Request",
                message="Both 'version' and 'stage' are required",
                timestamp=datetime.utcnow().isoformat(),
            )
            return jsonify(error_response.model_dump()), 400

        success = mlflow_registry.transition_model_stage(
            model_name, version, stage, archive_existing
        )

        if success:
            response = {
                "status": "success",
                "message": f"Model {model_name} version {version} transitioned to {stage}",
                "model_name": model_name,
                "version": version,
                "stage": stage,
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time_ms": (time.time() - start_time_transition) * 1000
            }
            performance_monitor.record_request((time.time() - start_time_transition) * 1000, 200)
            return jsonify(response), 200
        else:
            error_response = ErrorResponse(
                error="Transition Failed",
                message=f"Failed to transition model {model_name} to {stage}",
                timestamp=datetime.utcnow().isoformat(),
            )
            performance_monitor.record_request((time.time() - start_time_transition) * 1000, 500)
            return jsonify(error_response.model_dump()), 500

    except Exception as e:
        error_response = ErrorResponse(
            error="Model Transition Error",
            message=f"Error transitioning model stage: {str(e)}",
            timestamp=datetime.utcnow().isoformat(),
        )
        performance_monitor.record_request((time.time() - start_time_transition) * 1000, 500)
        return jsonify(error_response.model_dump()), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """Single prediction endpoint"""
    start_time_pred = time.time()

    try:
        # Parse input data
        input_data = request.get_json()
        if not input_data:
            raise ValueError("No input data provided")

        # Validate input using Pydantic
        validated_input = HousingPredictionInput(**input_data)

        # Check if model is loaded
        if model is None:
            error_response = ErrorResponse(
                error="Model Not Available",
                message="Model not loaded. Please check model configuration.",
                timestamp=datetime.utcnow().isoformat(),
            )
            performance_monitor.record_request(
                (time.time() - start_time_pred) * 1000, 503
            )
            return jsonify(error_response.model_dump()), 503

        # Make prediction
        prediction_input = validated_input.model_dump()
        prediction = model.predict(prediction_input)

        processing_time = (time.time() - start_time_pred) * 1000

        # Get model metadata for logging
        model_metadata = getattr(model, 'model_metadata', {})
        model_version = model_metadata.get('version', 'unknown')

        # Log prediction to database
        if db_manager:
            try:
                db_manager.log_prediction_request(
                    endpoint="/api/predict",
                    method="POST",
                    input_data=prediction_input,
                    prediction=prediction,
                    processing_time_ms=processing_time,
                )
            except Exception as e:
                logger.error(f"Error logging prediction to database: {e}")

        # Log prediction to MLflow (simple local approach)
        try:
            log_prediction_to_mlflow(
                input_data=prediction_input,
                prediction=prediction,
                processing_time=processing_time
            )
        except Exception as e:
            logger.error(f"Error logging prediction to MLflow: {e}")

        # Add prediction to MLflow batch for logging (fallback to complex approach)
        if mlflow_tracker and api_config.get("mlflow.enabled", True):
            try:
                prediction_data = {
                    "input_data": prediction_input,
                    "prediction": prediction,
                    "processing_time_ms": processing_time,
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_metadata": model_metadata
                }
                prediction_batch.append(prediction_data)
                
                # Log batch to MLflow if batch size reached
                batch_size = api_config.get("mlflow.batch_size", 10)
                if len(prediction_batch) >= batch_size:
                    mlflow_tracker.log_prediction_batch(prediction_batch, model_metadata)
                    prediction_batch.clear()
                    
            except Exception as e:
                logger.error(f"Error adding prediction to MLflow batch: {e}")

        structured_logger.log_prediction(
            input_data=prediction_input,
            prediction=prediction,
            processing_time=processing_time,
            model_version=model_version,
        )

        # Record metrics
        metrics_collector.record_metric("prediction_count", 1)
        metrics_collector.record_metric("prediction_value", prediction)

        # Record Prometheus metrics
        try:
            prometheus_metrics.record_prediction_metric(
                endpoint="/api/predict",
                status_code=200,
                processing_time=processing_time / 1000.0,  # Convert to seconds
                prediction_value=prediction,
                input_data=prediction_input
            )
        except Exception as e:
            logger.error(f"Error recording Prometheus metrics: {e}")

        response = PredictionResponse(
            prediction=prediction,
            model_version=model_version,
            timestamp=datetime.utcnow().isoformat(),
            processing_time_ms=processing_time,
            validation_warnings=[],
        )

        performance_monitor.record_request(processing_time, 200)
        return jsonify(response.model_dump()), 200

    except ValidationError as e:
        error_details = [
            {
                "field": str(err["loc"][-1]) if err["loc"] else "unknown",
                "message": err["msg"],
                "type": err["type"],
            }
            for err in e.errors()
        ]
        error_message = "Input validation failed: " + "; ".join(
            [
                f"{detail['field']}: {detail['message']}"
                for detail in error_details
            ]
        )

        error_response = ErrorResponse(
            error="Validation Error",
            message=error_message,
            details=error_details,
            timestamp=datetime.utcnow().isoformat(),
        )

        structured_logger.log_error(
            error_type="ValidationError",
            error_message=error_message,
            input_data=request.get_json(),
        )

        processing_time = (time.time() - start_time_pred) * 1000
        performance_monitor.record_request(processing_time, 400)

        # Record error in Prometheus metrics
        try:
            prometheus_metrics.record_prediction_metric(
                endpoint="/api/predict",
                status_code=400,
                processing_time=processing_time / 1000.0,
                prediction_value=0,
                input_data=request.get_json() or {}
            )
        except Exception as prom_e:
            logger.error(f"Error recording Prometheus error metrics: {prom_e}")

        return jsonify(error_response.model_dump()), 400

    except Exception as e:
        error_response = ErrorResponse(
            error="Prediction Error",
            message=f"Internal server error: {str(e)}",
            timestamp=datetime.utcnow().isoformat(),
        )

        structured_logger.log_error(
            error_type="PredictionError",
            error_message=str(e),
            input_data=request.get_json(),
        )

        processing_time = (time.time() - start_time_pred) * 1000
        performance_monitor.record_request(processing_time, 500)

        # Record error in Prometheus metrics
        try:
            prometheus_metrics.record_prediction_metric(
                input_data=request.get_json() or {},
                prediction=0,
                processing_time=processing_time / 1000.0,
                endpoint="/api/predict",
                status="server_error"
            )
        except Exception as prom_e:
            logger.error(f"Error recording Prometheus error metrics: {prom_e}")

        return jsonify(error_response.model_dump()), 500


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    """Batch prediction endpoint"""
    start_time_batch = time.time()

    try:
        # Parse input data
        input_data = request.get_json()
        if not input_data:
            raise ValueError("No input data provided")

        # Validate input using Pydantic
        validated_input = BatchPredictionInput(**input_data)

        # Check if model is loaded
        if model is None:
            error_response = ErrorResponse(
                error="Model Not Available",
                message="Model not loaded. Please check model configuration.",
                timestamp=datetime.utcnow().isoformat(),
            )
            performance_monitor.record_request(
                (time.time() - start_time_batch) * 1000, 503
            )
            return jsonify(error_response.model_dump()), 503

        # Make predictions
        predictions = []
        for pred_input in validated_input.predictions:
            prediction = model.predict(pred_input.model_dump())
            predictions.append(prediction)

        processing_time = (time.time() - start_time_batch) * 1000

        # Get model metadata for logging
        model_metadata = getattr(model, 'model_metadata', {})
        model_version = model_metadata.get('version', 'unknown')

        # Log batch prediction to database
        if db_manager:
            try:
                for i, pred_input in enumerate(validated_input.predictions):
                    db_manager.log_prediction_request(
                        endpoint="/api/predict/batch",
                        method="POST",
                        input_data=pred_input.model_dump(),
                        prediction=predictions[i],
                        processing_time_ms=processing_time / len(predictions),
                    )
            except Exception as e:
                logger.error(f"Error logging batch prediction to database: {e}")

        # Log batch to MLflow
        if mlflow_tracker and api_config.get("mlflow.enabled", True):
            try:
                batch_data = []
                for i, pred_input in enumerate(validated_input.predictions):
                    prediction_data = {
                        "input_data": pred_input.model_dump(),
                        "prediction": predictions[i],
                        "processing_time_ms": processing_time / len(predictions),
                        "timestamp": datetime.utcnow().isoformat(),
                        "model_metadata": model_metadata
                    }
                    batch_data.append(prediction_data)
                
                # Log immediately for batch predictions
                mlflow_tracker.log_prediction_batch(batch_data, model_metadata)
                    
            except Exception as e:
                logger.error(f"Error logging batch prediction to MLflow: {e}")

        # Record metrics
        metrics_collector.record_metric(
            "batch_prediction_count", len(predictions)
        )

        response = BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            model_version=model_version,
            timestamp=datetime.utcnow().isoformat(),
            processing_time_ms=processing_time,
            validation_warnings=[],
        )

        performance_monitor.record_request(processing_time, 200)
        return jsonify(response.model_dump()), 200

    except ValidationError as e:
        error_details = [
            {
                "field": str(err["loc"][-1]) if err["loc"] else "unknown",
                "message": err["msg"],
                "type": err["type"],
            }
            for err in e.errors()
        ]
        error_message = "Input validation failed: " + "; ".join(
            [
                f"{detail['field']}: {detail['message']}"
                for detail in error_details
            ]
        )

        error_response = ErrorResponse(
            error="Validation Error",
            message=error_message,
            details=error_details,
            timestamp=datetime.utcnow().isoformat(),
        )

        processing_time = (time.time() - start_time_batch) * 1000
        performance_monitor.record_request(processing_time, 400)
        return jsonify(error_response.model_dump()), 400

    except Exception as e:
        error_response = ErrorResponse(
            error="Batch Prediction Error",
            message=f"Internal server error: {str(e)}",
            timestamp=datetime.utcnow().isoformat(),
        )

        processing_time = (time.time() - start_time_batch) * 1000
        performance_monitor.record_request(processing_time, 500)
        return jsonify(error_response.model_dump()), 500


@app.route("/api/logs/predictions")
def get_prediction_logs():
    """Get prediction logs with filtering and pagination"""
    start_time_logs = time.time()

    try:
        if db_manager is None:
            error_response = ErrorResponse(
                error="Database Not Available",
                message="Database not initialized",
                timestamp=datetime.utcnow().isoformat(),
            )
            return jsonify(error_response.model_dump()), 503

        # Parse query parameters
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        hours = request.args.get('hours', 24, type=int)  # Last N hours
        endpoint = request.args.get('endpoint')  # Filter by endpoint
        min_price = request.args.get('min_price', type=float)
        max_price = request.args.get('max_price', type=float)

        # Get prediction logs with filters
        logs = db_manager.get_prediction_logs(
            limit=limit,
            offset=offset,
            hours=hours,
            endpoint=endpoint,
            min_price=min_price,
            max_price=max_price
        )

        # Get total count for pagination
        total_count = db_manager.get_prediction_count(
            hours=hours,
            endpoint=endpoint,
            min_price=min_price,
            max_price=max_price
        )

        processing_time = (time.time() - start_time_logs) * 1000

        response = {
            "logs": logs,
            "pagination": {
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total_count
            },
            "filters": {
                "hours": hours,
                "endpoint": endpoint,
                "min_price": min_price,
                "max_price": max_price
            },
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": processing_time
        }

        performance_monitor.record_request(processing_time, 200)
        return jsonify(response), 200

    except Exception as e:
        error_response = ErrorResponse(
            error="Logs Retrieval Error",
            message=f"Error retrieving prediction logs: {str(e)}",
            timestamp=datetime.utcnow().isoformat(),
        )
        performance_monitor.record_request((time.time() - start_time_logs) * 1000, 500)
        return jsonify(error_response.model_dump()), 500


@app.route("/api/logs/predictions/stats")
def get_prediction_stats():
    """Get prediction statistics and analytics"""
    start_time_stats = time.time()

    try:
        if db_manager is None:
            error_response = ErrorResponse(
                error="Database Not Available",
                message="Database not initialized",
                timestamp=datetime.utcnow().isoformat(),
            )
            return jsonify(error_response.model_dump()), 503

        # Parse query parameters
        hours = request.args.get('hours', 24, type=int)  # Last N hours
        endpoint = request.args.get('endpoint')  # Filter by endpoint

        # Get comprehensive statistics
        stats = db_manager.get_prediction_stats(hours=hours, endpoint=endpoint)
        
        # Get prediction distribution by hour
        hourly_stats = db_manager.get_hourly_prediction_stats(hours=hours)

        processing_time = (time.time() - start_time_stats) * 1000

        response = {
            "statistics": stats,
            "hourly_breakdown": hourly_stats,
            "filters": {
                "hours": hours,
                "endpoint": endpoint
            },
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": processing_time
        }

        performance_monitor.record_request(processing_time, 200)
        return jsonify(response), 200

    except Exception as e:
        error_response = ErrorResponse(
            error="Stats Retrieval Error",
            message=f"Error retrieving prediction statistics: {str(e)}",
            timestamp=datetime.utcnow().isoformat(),
        )
        performance_monitor.record_request((time.time() - start_time_stats) * 1000, 500)
        return jsonify(error_response.model_dump()), 500


@app.route("/api/logs/predictions/recent")
def get_recent_predictions():
    """Get recent predictions with quick filters"""
    start_time_recent = time.time()

    try:
        if db_manager is None:
            error_response = ErrorResponse(
                error="Database Not Available",
                message="Database not initialized",
                timestamp=datetime.utcnow().isoformat(),
            )
            return jsonify(error_response.model_dump()), 503

        # Parse query parameters
        count = request.args.get('count', 10, type=int)  # Number of recent predictions
        
        # Get recent predictions
        recent_logs = db_manager.get_recent_predictions(limit=count)

        processing_time = (time.time() - start_time_recent) * 1000

        response = {
            "recent_predictions": recent_logs,
            "count": len(recent_logs),
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": processing_time
        }

        performance_monitor.record_request(processing_time, 200)
        return jsonify(response), 200

    except Exception as e:
        error_response = ErrorResponse(
            error="Recent Predictions Error",
            message=f"Error retrieving recent predictions: {str(e)}",
            timestamp=datetime.utcnow().isoformat(),
        )
        performance_monitor.record_request((time.time() - start_time_recent) * 1000, 500)
        return jsonify(error_response.model_dump()), 500


@app.route("/api/logs/predictions/search")
def search_predictions():
    """Search predictions by input criteria"""
    start_time_search = time.time()

    try:
        if db_manager is None:
            error_response = ErrorResponse(
                error="Database Not Available",
                message="Database not initialized",
                timestamp=datetime.utcnow().isoformat(),
            )
            return jsonify(error_response.model_dump()), 503

        # Parse query parameters for search criteria
        longitude = request.args.get('longitude', type=float)
        latitude = request.args.get('latitude', type=float)
        median_income = request.args.get('median_income', type=float)
        housing_median_age = request.args.get('housing_median_age', type=float)
        total_rooms = request.args.get('total_rooms', type=int)
        total_bedrooms = request.args.get('total_bedrooms', type=int)
        population = request.args.get('population', type=int)
        households = request.args.get('households', type=int)
        ocean_proximity = request.args.get('ocean_proximity')
        
        limit = request.args.get('limit', 20, type=int)

        # Build search criteria
        search_criteria = {}
        if longitude is not None:
            search_criteria['longitude'] = longitude
        if latitude is not None:
            search_criteria['latitude'] = latitude
        if median_income is not None:
            search_criteria['median_income'] = median_income
        if housing_median_age is not None:
            search_criteria['housing_median_age'] = housing_median_age
        if total_rooms is not None:
            search_criteria['total_rooms'] = total_rooms
        if total_bedrooms is not None:
            search_criteria['total_bedrooms'] = total_bedrooms
        if population is not None:
            search_criteria['population'] = population
        if households is not None:
            search_criteria['households'] = households
        if ocean_proximity is not None:
            search_criteria['ocean_proximity'] = ocean_proximity

        if not search_criteria:
            error_response = ErrorResponse(
                error="No Search Criteria",
                message="At least one search criterion must be provided",
                timestamp=datetime.utcnow().isoformat(),
            )
            return jsonify(error_response.model_dump()), 400

        # Search predictions
        search_results = db_manager.search_predictions(search_criteria, limit=limit)

        processing_time = (time.time() - start_time_search) * 1000

        response = {
            "search_results": search_results,
            "count": len(search_results),
            "search_criteria": search_criteria,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": processing_time
        }

        performance_monitor.record_request(processing_time, 200)
        return jsonify(response), 200

    except Exception as e:
        error_response = ErrorResponse(
            error="Search Error",
            message=f"Error searching predictions: {str(e)}",
            timestamp=datetime.utcnow().isoformat(),
        )
        performance_monitor.record_request((time.time() - start_time_search) * 1000, 500)
        return jsonify(error_response.model_dump()), 500


if __name__ == "__main__":
    # Configuration
    host = api_config.get("server.host", "0.0.0.0")
    port = api_config.get("server.port", 5001)
    debug = api_config.get("server.debug", False)

    logger.info(f"Starting California Housing API on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Model path: {api_config.get('model.path')}")

    # Start Flask app
    app.run(host=host, port=port, debug=debug)
