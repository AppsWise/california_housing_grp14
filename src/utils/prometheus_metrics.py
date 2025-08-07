"""
Prometheus metrics module for California Housing API
"""

from .prometheus_exporter import get_metrics_exporter

# Initialize global prometheus metrics instance
prometheus_metrics = get_metrics_exporter()
