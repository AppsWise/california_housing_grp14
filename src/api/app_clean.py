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
from src.utils.database import DatabaseManager

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
start_time = time.time()


def load_model():
    """Load the trained model"""
    global model

    model_path = api_config.get("model.path", "models/model.pkl")

    try:
        start_time_load = time.time()
        model = HousingPredictor(model_path)
        load_time = (time.time() - start_time_load) * 1000

        structured_logger.log_model_load(
            model_path=model_path,
            model_version=api_config.get("model.version", "1.0.0"),
            load_time=load_time,
        )

        return True
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return False


def initialize_database():
    """Initialize database connection"""
    global db_manager

    try:
        db_url = api_config.get(
            "database.url", "sqlite:///monitoring/logs/predictions.db"
        )
        db_manager = DatabaseManager(db_url)
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


@app.before_first_request
def startup():
    """Initialize application on first request"""
    logger.info("Initializing California Housing API...")

    # Load model
    if not load_model():
        logger.error("Failed to load model during startup")

    # Initialize database
    if not initialize_database():
        logger.error("Failed to initialize database during startup")

    logger.info("California Housing API initialized successfully")


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

    response = HealthResponse(
        status=health_results["status"],
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model is not None,
        model_version=(
            api_config.get("model.version", "1.0.0") if model else None
        ),
        uptime_seconds=uptime,
    )

    structured_logger.log_health_check(
        status=health_results["status"], checks=health_results["checks"]
    )

    processing_time = (time.time() - start_time_check) * 1000
    performance_monitor.record_request(processing_time, 200)

    return jsonify(response.dict()), 200


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
            prediction_count_24h = db_manager.get_prediction_count(hours=24)
        except Exception as e:
            logger.error(f"Error getting prediction count: {e}")

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

    return jsonify(response.dict()), 200


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
            return jsonify(error_response.dict()), 503

        # Make prediction
        prediction_input = validated_input.dict()
        prediction = model.predict(prediction_input)

        processing_time = (time.time() - start_time_pred) * 1000

        # Log prediction
        if db_manager:
            try:
                db_manager.log_prediction_request(
                    endpoint="/api/predict",
                    input_data=prediction_input,
                    prediction=prediction,
                    processing_time=processing_time,
                    model_version=api_config.get("model.version", "1.0.0"),
                )
            except Exception as e:
                logger.error(f"Error logging prediction: {e}")

        structured_logger.log_prediction(
            input_data=prediction_input,
            prediction=prediction,
            processing_time=processing_time,
            model_version=api_config.get("model.version", "1.0.0"),
        )

        # Record metrics
        metrics_collector.record_metric("prediction_count", 1)
        metrics_collector.record_metric("prediction_value", prediction)

        response = PredictionResponse(
            prediction=prediction,
            model_version=api_config.get("model.version", "1.0.0"),
            timestamp=datetime.utcnow().isoformat(),
            processing_time_ms=processing_time,
            validation_warnings=[],
        )

        performance_monitor.record_request(processing_time, 200)
        return jsonify(response.dict()), 200

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
        return jsonify(error_response.dict()), 400

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
        return jsonify(error_response.dict()), 500


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
            return jsonify(error_response.dict()), 503

        # Make predictions
        predictions = []
        for pred_input in validated_input.predictions:
            prediction = model.predict(pred_input.dict())
            predictions.append(prediction)

        processing_time = (time.time() - start_time_batch) * 1000

        # Log batch prediction
        if db_manager:
            try:
                for i, pred_input in enumerate(validated_input.predictions):
                    db_manager.log_prediction_request(
                        endpoint="/api/predict/batch",
                        input_data=pred_input.dict(),
                        prediction=predictions[i],
                        processing_time=processing_time / len(predictions),
                        model_version=api_config.get("model.version", "1.0.0"),
                    )
            except Exception as e:
                logger.error(f"Error logging batch prediction: {e}")

        # Record metrics
        metrics_collector.record_metric(
            "batch_prediction_count", len(predictions)
        )

        response = BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            model_version=api_config.get("model.version", "1.0.0"),
            timestamp=datetime.utcnow().isoformat(),
            processing_time_ms=processing_time,
            validation_warnings=[],
        )

        performance_monitor.record_request(processing_time, 200)
        return jsonify(response.dict()), 200

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
        return jsonify(error_response.dict()), 400

    except Exception as e:
        error_response = ErrorResponse(
            error="Batch Prediction Error",
            message=f"Internal server error: {str(e)}",
            timestamp=datetime.utcnow().isoformat(),
        )

        processing_time = (time.time() - start_time_batch) * 1000
        performance_monitor.record_request(processing_time, 500)
        return jsonify(error_response.dict()), 500


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
