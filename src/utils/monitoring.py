"""Monitoring utilities for metrics collection and health checks"""

import time
import psutil
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and store application metrics"""

    def __init__(self, db_path: str = "monitoring/logs/metrics.db"):
        self.db_path = db_path
        self.setup_database()
        self.start_time = time.time()

    def setup_database(self):
        """Initialize metrics database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    tags TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
                ON metrics(timestamp)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_metrics_name
                ON metrics(metric_name)
            """
            )

    def record_metric(
        self, name: str, value: float, tags: Optional[str] = None
    ):
        """Record a metric value"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO metrics (metric_name, metric_value, tags) "
                    "VALUES (?, ?, ?)",
                    (name, value, tags),
                )
            logger.debug(f"Recorded metric: {name}={value}")
        except Exception as e:
            logger.error(f"Error recording metric {name}: {e}")

    def get_metric_stats(self, name: str, hours: int = 24) -> Dict[str, float]:
        """Get statistics for a metric over specified time period"""
        try:
            since = datetime.now() - timedelta(hours=hours)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as count,
                        AVG(metric_value) as avg,
                        MIN(metric_value) as min,
                        MAX(metric_value) as max,
                        SUM(metric_value) as sum
                    FROM metrics
                    WHERE metric_name = ? AND timestamp > ?
                """,
                    (name, since),
                )

                result = cursor.fetchone()
                if result and result[0] > 0:
                    return {
                        "count": result[0],
                        "average": result[1],
                        "minimum": result[2],
                        "maximum": result[3],
                        "total": result[4],
                    }
                else:
                    return {
                        "count": 0,
                        "average": 0,
                        "minimum": 0,
                        "maximum": 0,
                        "total": 0,
                    }
        except Exception as e:
            logger.error(f"Error getting metric stats for {name}: {e}")
            return {}

    def cleanup_old_metrics(self, days: int = 30):
        """Clean up metrics older than specified days"""
        try:
            cutoff = datetime.now() - timedelta(days=days)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM metrics WHERE timestamp < ?", (cutoff,)
                )
                deleted_count = cursor.rowcount
                logger.info(f"Cleaned up {deleted_count} old metrics")
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")


class HealthChecker:
    """System health monitoring"""

    def __init__(self):
        self.checks = {}

    def register_check(self, name: str, check_func):
        """Register a health check function"""
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")

    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "uptime_seconds": 0,
        }

        for name, check_func in self.checks.items():
            try:
                check_result = check_func()
                results["checks"][name] = {
                    "status": "healthy" if check_result else "unhealthy",
                    "details": (
                        check_result if isinstance(check_result, dict) else {}
                    ),
                }

                if not check_result:
                    results["status"] = "unhealthy"

            except Exception as e:
                results["checks"][name] = {"status": "error", "error": str(e)}
                results["status"] = "unhealthy"
                logger.error(f"Health check {name} failed: {e}")

        return results

    def check_disk_space(self, threshold_gb: float = 1.0) -> bool:
        """Check available disk space"""
        try:
            disk_usage = psutil.disk_usage("/")
            free_gb = disk_usage.free / (1024**3)
            return free_gb > threshold_gb
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return False

    def check_memory_usage(self, threshold_percent: float = 90.0) -> bool:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent < threshold_percent
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return False

    def check_cpu_usage(self, threshold_percent: float = 90.0) -> bool:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < threshold_percent
        except Exception as e:
            logger.error(f"Error checking CPU usage: {e}")
            return False


class PerformanceMonitor:
    """Monitor application performance metrics"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.request_count = 0
        self.error_count = 0
        self.response_times = []

    def record_request(self, response_time: float, status_code: int):
        """Record request metrics"""
        self.request_count += 1
        self.response_times.append(response_time)

        # Keep only last 1000 response times for memory efficiency
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]

        if status_code >= 400:
            self.error_count += 1

        # Record metrics
        self.metrics.record_metric("request_response_time", response_time)
        self.metrics.record_metric("request_count", 1)
        if status_code >= 400:
            self.metrics.record_metric("error_count", 1)

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(
                self.response_times
            )
            max_response_time = max(self.response_times)
            min_response_time = min(self.response_times)
        else:
            avg_response_time = max_response_time = min_response_time = 0

        error_rate = (
            (self.error_count / self.request_count * 100)
            if self.request_count > 0
            else 0
        )

        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate_percentage": error_rate,
            "average_response_time_ms": avg_response_time,
            "max_response_time_ms": max_response_time,
            "min_response_time_ms": min_response_time,
        }


# Global instances
metrics_collector = MetricsCollector()
health_checker = HealthChecker()
performance_monitor = PerformanceMonitor(metrics_collector)

# Register default health checks
health_checker.register_check("disk_space", health_checker.check_disk_space)
health_checker.register_check(
    "memory_usage", health_checker.check_memory_usage
)
health_checker.register_check("cpu_usage", health_checker.check_cpu_usage)
