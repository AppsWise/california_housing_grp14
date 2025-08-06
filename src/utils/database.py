"""Database utilities for logging predictions and requests"""

import sqlite3
import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class PredictionDatabase:
    """SQLite database for storing prediction logs"""

    def __init__(self, db_path: str = "logs/predictions.db"):
        """Initialize database connection and create tables if needed"""
        self.db_path = db_path

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize database
        self._init_database()
        logger.info(f"Initialized prediction database at {db_path}")

    def _init_database(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Table for prediction requests
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    prediction TEXT,
                    processing_time_ms REAL,
                    status_code INTEGER,
                    error_message TEXT,
                    user_agent TEXT,
                    ip_address TEXT
                )
            """
            )

            # Table for system metrics
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    tags TEXT
                )
            """
            )

            # Table for model performance tracking
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_version TEXT,
                    prediction_type TEXT,
                    input_features TEXT,
                    prediction_value REAL,
                    confidence_score REAL,
                    feature_importance TEXT
                )
            """
            )

            conn.commit()
            logger.info("Database tables initialized successfully")

    def log_prediction_request(
        self,
        endpoint: str,
        method: str,
        input_data: Dict[Any, Any],
        prediction: Any = None,
        processing_time_ms: float = None,
        status_code: int = 200,
        error_message: str = None,
        user_agent: str = None,
        ip_address: str = None,
    ) -> int:
        """Log a prediction request to the database"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                timestamp = datetime.utcnow().isoformat()

                cursor.execute(
                    """
                    INSERT INTO prediction_requests
                    (timestamp, endpoint, method, input_data, prediction,
                     processing_time_ms, status_code, error_message, user_agent,
                     ip_address)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        timestamp,
                        endpoint,
                        method,
                        json.dumps(input_data),
                        (
                            json.dumps(prediction)
                            if prediction is not None
                            else None
                        ),
                        processing_time_ms,
                        status_code,
                        error_message,
                        user_agent,
                        ip_address,
                    ),
                )

                request_id = cursor.lastrowid
                conn.commit()

                logger.info(
                    f"Logged prediction request {request_id} for endpoint {endpoint}"
                )
                return request_id

        except Exception as e:
            logger.error(f"Failed to log prediction request: {e}")
            return -1

    def log_model_performance(
        self,
        model_version: str,
        prediction_type: str,
        input_features: Dict[str, Any],
        prediction_value: float,
        confidence_score: float = None,
        feature_importance: Dict[str, float] = None,
    ) -> int:
        """Log model performance metrics"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                timestamp = datetime.utcnow().isoformat()

                cursor.execute(
                    """
                    INSERT INTO model_performance
                    (timestamp, model_version, prediction_type, input_features,
                     prediction_value, confidence_score, feature_importance)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        timestamp,
                        model_version,
                        prediction_type,
                        json.dumps(input_features),
                        prediction_value,
                        confidence_score,
                        (
                            json.dumps(feature_importance)
                            if feature_importance
                            else None
                        ),
                    ),
                )

                performance_id = cursor.lastrowid
                conn.commit()

                logger.info(f"Logged model performance {performance_id}")
                return performance_id

        except Exception as e:
            logger.error(f"Failed to log model performance: {e}")
            return -1

    def log_system_metric(
        self,
        metric_name: str,
        metric_value: float,
        tags: Dict[str, str] = None,
    ) -> int:
        """Log system metrics"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                timestamp = datetime.utcnow().isoformat()

                cursor.execute(
                    """
                    INSERT INTO system_metrics (timestamp, metric_name, metric_value, tags)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        timestamp,
                        metric_name,
                        metric_value,
                        json.dumps(tags) if tags else None,
                    ),
                )

                metric_id = cursor.lastrowid
                conn.commit()

                return metric_id

        except Exception as e:
            logger.error(f"Failed to log system metric: {e}")
            return -1

    def get_prediction_stats(self, hours: int = 24, endpoint: str = None) -> Dict[str, Any]:
        """Get prediction statistics for the last N hours"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Build base query with optional endpoint filter
                base_filter = f"datetime(timestamp) > datetime('now', '-{hours} hours')"
                if endpoint:
                    base_filter += f" AND endpoint = '{endpoint}'"

                # Get total predictions in last N hours
                cursor.execute(f"""
                    SELECT COUNT(*) FROM prediction_requests
                    WHERE {base_filter}
                """)
                total_predictions = cursor.fetchone()[0]

                # Get successful predictions
                cursor.execute(f"""
                    SELECT COUNT(*) FROM prediction_requests
                    WHERE {base_filter} AND status_code = 200
                """)
                successful_predictions = cursor.fetchone()[0]

                # Get average processing time
                cursor.execute(f"""
                    SELECT AVG(processing_time_ms) FROM prediction_requests
                    WHERE {base_filter} AND processing_time_ms IS NOT NULL
                """)
                avg_processing_time = cursor.fetchone()[0] or 0

                # Get prediction value statistics
                cursor.execute(f"""
                    SELECT 
                        AVG(CAST(prediction as REAL)) as avg_prediction,
                        MIN(CAST(prediction as REAL)) as min_prediction,
                        MAX(CAST(prediction as REAL)) as max_prediction,
                        COUNT(CASE WHEN prediction IS NOT NULL THEN 1 END) as predictions_with_values
                    FROM prediction_requests
                    WHERE {base_filter}
                """)
                prediction_stats = cursor.fetchone()

                # Get error rate
                error_rate = 0
                if total_predictions > 0:
                    error_rate = (
                        (total_predictions - successful_predictions)
                        / total_predictions
                        * 100
                    )

                return {
                    "total_predictions": total_predictions,
                    "successful_predictions": successful_predictions,
                    "predictions_with_values": prediction_stats[3] if prediction_stats else 0,
                    "error_rate_percent": round(error_rate, 2),
                    "avg_processing_time_ms": round(avg_processing_time, 2),
                    "avg_prediction_value": round(prediction_stats[0], 2) if prediction_stats and prediction_stats[0] else None,
                    "min_prediction_value": round(prediction_stats[1], 2) if prediction_stats and prediction_stats[1] else None,
                    "max_prediction_value": round(prediction_stats[2], 2) if prediction_stats and prediction_stats[2] else None,
                    "time_window_hours": hours,
                }

        except Exception as e:
            logger.error(f"Failed to get prediction stats: {e}")
            return {}

    def get_recent_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent prediction requests with full details"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT id, timestamp, endpoint, method, input_data, prediction,
                           processing_time_ms, status_code, error_message
                    FROM prediction_requests
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (limit,),
                )

                results = []
                for row in cursor.fetchall():
                    # Parse input data JSON
                    input_data = {}
                    try:
                        input_data = json.loads(row[4]) if row[4] else {}
                    except:
                        pass
                    
                    results.append({
                        "id": row[0],
                        "timestamp": row[1],
                        "endpoint": row[2],
                        "method": row[3],
                        "input_data": input_data,
                        "prediction": float(row[5]) if row[5] else None,
                        "processing_time_ms": row[6],
                        "status_code": row[7],
                        "error_message": row[8],
                    })

                return results

        except Exception as e:
            logger.error(f"Failed to get recent predictions: {e}")
            return []

    def cleanup_old_logs(self, days: int = 30):
        """Clean up logs older than specified days"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Delete old prediction requests
                cursor.execute(
                    """
                    DELETE FROM prediction_requests
                    WHERE datetime(timestamp) < datetime('now', '-{} days')
                """.format(
                        days
                    )
                )
                deleted_requests = cursor.rowcount

                # Delete old system metrics
                cursor.execute(
                    """
                    DELETE FROM system_metrics
                    WHERE datetime(timestamp) < datetime('now', '-{} days')
                """.format(
                        days
                    )
                )
                deleted_metrics = cursor.rowcount

                # Delete old model performance logs
                cursor.execute(
                    """
                    DELETE FROM model_performance
                    WHERE datetime(timestamp) < datetime('now', '-{} days')
                """.format(
                        days
                    )
                )
                deleted_performance = cursor.rowcount

                conn.commit()

                logger.info(
                    f"Cleaned up {deleted_requests} requests, {deleted_metrics} "
                    f"metrics, {deleted_performance} performance logs older than "
                    f"{days} days"
                )

                return {
                    "deleted_requests": deleted_requests,
                    "deleted_metrics": deleted_metrics,
                    "deleted_performance": deleted_performance,
                }

        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")
            return {}

    def get_prediction_logs(self, limit: int = 50, offset: int = 0, hours: int = 24, 
                          endpoint: str = None, min_price: float = None, 
                          max_price: float = None) -> List[Dict[str, Any]]:
        """Get prediction logs with filtering and pagination"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query with filters
                query = """
                    SELECT id, timestamp, endpoint, method, input_data, prediction, 
                           processing_time_ms, status_code, error_message
                    FROM prediction_requests
                    WHERE datetime(timestamp) > datetime('now', '-{} hours')
                """.format(hours)
                
                params = []
                
                if endpoint:
                    query += " AND endpoint = ?"
                    params.append(endpoint)
                
                if min_price is not None:
                    query += " AND CAST(prediction as REAL) >= ?"
                    params.append(min_price)
                
                if max_price is not None:
                    query += " AND CAST(prediction as REAL) <= ?"
                    params.append(max_price)
                
                query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                
                results = []
                for row in cursor.fetchall():
                    # Parse input data JSON
                    input_data = {}
                    try:
                        input_data = json.loads(row[4]) if row[4] else {}
                    except:
                        pass
                    
                    results.append({
                        "id": row[0],
                        "timestamp": row[1],
                        "endpoint": row[2],
                        "method": row[3],
                        "input_data": input_data,
                        "prediction": float(row[5]) if row[5] else None,
                        "processing_time_ms": row[6],
                        "status_code": row[7],
                        "error_message": row[8]
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get prediction logs: {e}")
            return []

    def get_prediction_count(self, hours: int = 24, endpoint: str = None, 
                           min_price: float = None, max_price: float = None) -> int:
        """Get total count of predictions matching filters"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT COUNT(*) FROM prediction_requests
                    WHERE datetime(timestamp) > datetime('now', '-{} hours')
                """.format(hours)
                
                params = []
                
                if endpoint:
                    query += " AND endpoint = ?"
                    params.append(endpoint)
                
                if min_price is not None:
                    query += " AND CAST(prediction as REAL) >= ?"
                    params.append(min_price)
                
                if max_price is not None:
                    query += " AND CAST(prediction as REAL) <= ?"
                    params.append(max_price)
                
                cursor.execute(query, params)
                return cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Failed to get prediction count: {e}")
            return 0

    def get_hourly_prediction_stats(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get prediction statistics broken down by hour"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                        COUNT(*) as prediction_count,
                        AVG(CAST(prediction as REAL)) as avg_prediction,
                        MIN(CAST(prediction as REAL)) as min_prediction,
                        MAX(CAST(prediction as REAL)) as max_prediction,
                        AVG(processing_time_ms) as avg_processing_time
                    FROM prediction_requests
                    WHERE datetime(timestamp) > datetime('now', '-{} hours')
                    AND prediction IS NOT NULL
                    GROUP BY strftime('%Y-%m-%d %H:00:00', timestamp)
                    ORDER BY hour DESC
                """.format(hours))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        "hour": row[0],
                        "prediction_count": row[1],
                        "avg_prediction": round(row[2], 2) if row[2] else None,
                        "min_prediction": round(row[3], 2) if row[3] else None,
                        "max_prediction": round(row[4], 2) if row[4] else None,
                        "avg_processing_time_ms": round(row[5], 2) if row[5] else None
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get hourly prediction stats: {e}")
            return []

    def search_predictions(self, search_criteria: Dict[str, Any], 
                         limit: int = 20) -> List[Dict[str, Any]]:
        """Search predictions by input criteria"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build search query
                query = """
                    SELECT id, timestamp, endpoint, input_data, prediction, 
                           processing_time_ms
                    FROM prediction_requests
                    WHERE prediction IS NOT NULL
                """
                
                params = []
                
                # Add search criteria
                for field, value in search_criteria.items():
                    if isinstance(value, str):
                        query += f" AND json_extract(input_data, '$.{field}') = ?"
                    else:
                        query += f" AND CAST(json_extract(input_data, '$.{field}') as REAL) = ?"
                    params.append(value)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                
                results = []
                for row in cursor.fetchall():
                    # Parse input data JSON
                    input_data = {}
                    try:
                        input_data = json.loads(row[3]) if row[3] else {}
                    except:
                        pass
                    
                    results.append({
                        "id": row[0],
                        "timestamp": row[1],
                        "endpoint": row[2],
                        "input_data": input_data,
                        "prediction": float(row[4]) if row[4] else None,
                        "processing_time_ms": row[5]
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to search predictions: {e}")
            return []
