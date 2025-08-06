"""
Prometheus Metrics Exporter for California Housing API
"""

import time
import sqlite3
import json
import mlflow
import os
from datetime import datetime, timedelta
from typing import Dict, List
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from flask import Response
import logging

logger = logging.getLogger(__name__)


class PrometheusMetricsExporter:
    """Export metrics to Prometheus format"""
    
    def __init__(self, db_path: str = "logs/predictions.db", mlflow_tracking_uri: str = None):
        self.db_path = db_path
        self.mlflow_tracking_uri = mlflow_tracking_uri or f"file://{os.path.abspath('mlruns_local')}"
        
        # Define Prometheus metrics
        self._init_metrics()
        
    def _init_metrics(self):
        """Initialize Prometheus metrics"""
        
        # Counters (always increasing)
        self.prediction_requests_total = Counter(
            'housing_api_prediction_requests_total',
            'Total number of prediction requests',
            ['endpoint', 'method', 'status_code']
        )
        
        self.prediction_errors_total = Counter(
            'housing_api_prediction_errors_total',
            'Total number of prediction errors',
            ['endpoint', 'error_type']
        )
        
        # Histograms (for distributions)
        self.prediction_processing_time = Histogram(
            'housing_api_prediction_processing_time_seconds',
            'Time spent processing predictions',
            ['endpoint'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        self.prediction_values = Histogram(
            'housing_api_prediction_values_dollars',
            'Distribution of housing price predictions',
            ['location_type'],
            buckets=(50000, 100000, 200000, 300000, 400000, 500000, 750000, 1000000, 1500000, 2000000)
        )
        
        # Gauges (current values)
        self.active_predictions_last_hour = Gauge(
            'housing_api_predictions_last_hour',
            'Number of predictions in the last hour'
        )
        
        self.active_predictions_last_24h = Gauge(
            'housing_api_predictions_last_24h',
            'Number of predictions in the last 24 hours'
        )
        
        self.average_prediction_value = Gauge(
            'housing_api_average_prediction_value_dollars',
            'Average housing price prediction in last 24h'
        )
        
        self.average_processing_time = Gauge(
            'housing_api_average_processing_time_seconds',
            'Average prediction processing time in last 24h'
        )
        
        self.error_rate_percentage = Gauge(
            'housing_api_error_rate_percentage',
            'Error rate percentage in last 24h'
        )
        
        self.database_size_bytes = Gauge(
            'housing_api_database_size_bytes',
            'Size of the prediction database in bytes'
        )
        
        self.mlflow_experiments_count = Gauge(
            'housing_api_mlflow_experiments_total',
            'Total number of MLflow experiments'
        )
        
        self.mlflow_runs_count = Gauge(
            'housing_api_mlflow_runs_total',
            'Total number of MLflow runs'
        )
        
        # Info metrics (metadata)
        self.api_info = Info(
            'housing_api_info',
            'Information about the Housing API'
        )
        
        self.model_info = Info(
            'housing_api_model_info', 
            'Information about the current model'
        )
        
        logger.info("Prometheus metrics initialized")
    
    def update_metrics_from_database(self):
        """Update metrics from database data"""
        try:
            if not os.path.exists(self.db_path):
                logger.warning(f"Database not found: {self.db_path}")
                return
                
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get predictions in last hour
                cursor.execute("""
                    SELECT COUNT(*) FROM prediction_requests 
                    WHERE datetime(timestamp) > datetime('now', '-1 hour')
                """)
                last_hour_count = cursor.fetchone()[0]
                self.active_predictions_last_hour.set(last_hour_count)
                
                # Get predictions in last 24 hours
                cursor.execute("""
                    SELECT COUNT(*) FROM prediction_requests 
                    WHERE datetime(timestamp) > datetime('now', '-24 hours')
                """)
                last_24h_count = cursor.fetchone()[0]
                self.active_predictions_last_24h.set(last_24h_count)
                
                # Get average prediction value (last 24h)
                cursor.execute("""
                    SELECT AVG(CAST(prediction as REAL)) FROM prediction_requests 
                    WHERE datetime(timestamp) > datetime('now', '-24 hours')
                    AND prediction IS NOT NULL
                """)
                avg_prediction = cursor.fetchone()[0]
                if avg_prediction:
                    self.average_prediction_value.set(avg_prediction)
                
                # Get average processing time (last 24h)  
                cursor.execute("""
                    SELECT AVG(processing_time_ms) FROM prediction_requests 
                    WHERE datetime(timestamp) > datetime('now', '-24 hours')
                    AND processing_time_ms IS NOT NULL
                """)
                avg_processing = cursor.fetchone()[0]
                if avg_processing:
                    self.average_processing_time.set(avg_processing / 1000.0)  # Convert to seconds
                
                # Get error rate (last 24h)
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(CASE WHEN status_code != 200 THEN 1 END) as errors
                    FROM prediction_requests 
                    WHERE datetime(timestamp) > datetime('now', '-24 hours')
                """)
                result = cursor.fetchone()
                total, errors = result[0], result[1]
                if total > 0:
                    error_rate = (errors / total) * 100
                    self.error_rate_percentage.set(error_rate)
                
                # Update counters from historical data
                cursor.execute("""
                    SELECT endpoint, method, status_code, COUNT(*) 
                    FROM prediction_requests 
                    GROUP BY endpoint, method, status_code
                """)
                for row in cursor.fetchall():
                    endpoint, method, status_code, count = row
                    # Note: We can't set counter values directly, this is for reference
                    # Counters should be incremented in real-time
                
                # Get database file size
                if os.path.exists(self.db_path):
                    db_size = os.path.getsize(self.db_path)
                    self.database_size_bytes.set(db_size)
                
                logger.info("Database metrics updated successfully")
                
        except Exception as e:
            logger.error(f"Error updating database metrics: {e}")
    
    def update_metrics_from_mlflow(self):
        """Update metrics from MLflow data"""
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            # Get experiments count
            experiments = mlflow.search_experiments()
            self.mlflow_experiments_count.set(len(experiments))
            
            # Get total runs count
            total_runs = 0
            for exp in experiments:
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                total_runs += len(runs)
            
            self.mlflow_runs_count.set(total_runs)
            
            logger.info("MLflow metrics updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating MLflow metrics: {e}")
    
    def record_prediction_request(self, endpoint: str, method: str, status_code: int, 
                                processing_time_ms: float, prediction_value: float = None,
                                location_type: str = "unknown"):
        """Record a prediction request in real-time metrics"""
        
        # Increment counters
        self.prediction_requests_total.labels(
            endpoint=endpoint, 
            method=method, 
            status_code=status_code
        ).inc()
        
        # Record processing time
        self.prediction_processing_time.labels(endpoint=endpoint).observe(
            processing_time_ms / 1000.0  # Convert to seconds
        )
        
        # Record prediction value distribution
        if prediction_value:
            self.prediction_values.labels(location_type=location_type).observe(prediction_value)
        
        # Record errors
        if status_code >= 400:
            error_type = "client_error" if status_code < 500 else "server_error"
            self.prediction_errors_total.labels(
                endpoint=endpoint,
                error_type=error_type
            ).inc()
    
    def set_api_info(self, version: str, model_version: str, uptime_seconds: float):
        """Set API information"""
        self.api_info.info({
            'version': version,
            'model_version': model_version,
            'uptime_seconds': str(uptime_seconds),
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def set_model_info(self, model_name: str, model_version: str, model_type: str, 
                      features_count: int):
        """Set model information"""
        self.model_info.info({
            'name': model_name,
            'version': model_version,
            'type': model_type,
            'features_count': str(features_count),
            'last_updated': datetime.utcnow().isoformat()
        })
    
    def update_all_metrics(self):
        """Update all metrics from various sources"""
        logger.info("Updating all Prometheus metrics...")
        
        self.update_metrics_from_database()
        self.update_metrics_from_mlflow()
        
        logger.info("All Prometheus metrics updated")
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        self.update_all_metrics()
        return generate_latest()
    
    def get_metrics_response(self) -> Response:
        """Get Flask response with Prometheus metrics"""
        metrics_data = self.get_metrics()
        return Response(metrics_data, mimetype=CONTENT_TYPE_LATEST)


# Global metrics exporter instance
metrics_exporter = None

def get_metrics_exporter(db_path: str = "logs/predictions.db", 
                        mlflow_tracking_uri: str = None) -> PrometheusMetricsExporter:
    """Get or create global metrics exporter instance"""
    global metrics_exporter
    
    if metrics_exporter is None:
        metrics_exporter = PrometheusMetricsExporter(db_path, mlflow_tracking_uri)
        
        # Set initial API info
        metrics_exporter.set_api_info(
            version="1.0.0",
            model_version="california_housing_v1",
            uptime_seconds=0
        )
        
        # Set model info
        metrics_exporter.set_model_info(
            model_name="california_housing_model",
            model_version="v1.0",
            model_type="RandomForestRegressor", 
            features_count=8
        )
        
    return metrics_exporter
