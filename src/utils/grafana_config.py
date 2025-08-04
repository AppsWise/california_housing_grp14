"""Grafana dashboard configuration for MLOps monitoring"""

import json


def create_mlops_dashboard():
    """
    Create a comprehensive Grafana dashboard for MLOps monitoring.

    This dashboard includes:
    - API performance metrics
    - Model prediction metrics
    - System health monitoring
    - Error tracking
    - Business metrics
    """

    dashboard = {
        "dashboard": {
            "id": None,
            "title": "California Housing MLOps Dashboard",
            "description": "Comprehensive monitoring for California Housing Prediction API",
            "tags": ["mlops", "machine-learning", "api", "monitoring"],
            "timezone": "browser",
            "panels": [
                # Panel 1: API Request Rate
                {
                    "id": 1,
                    "title": "API Request Rate",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "rate(flask_http_request_total[5m])",
                            "legendFormat": "Requests/sec",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "custom": {
                                "displayMode": "list",
                                "orientation": "horizontal",
                            },
                            "mappings": [],
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "red", "value": 10},
                                ]
                            },
                        }
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                },
                # Panel 2: Response Time
                {
                    "id": 2,
                    "title": "API Response Time",
                    "type": "timeseries",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, flask_http_request_duration_seconds_bucket)",
                            "legendFormat": "95th percentile",
                        },
                        {
                            "expr": "histogram_quantile(0.50, flask_http_request_duration_seconds_bucket)",
                            "legendFormat": "50th percentile",
                        },
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "custom": {
                                "drawStyle": "line",
                                "fillOpacity": 10,
                                "pointSize": 5,
                            },
                            "unit": "s",
                        }
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                },
                # Panel 3: Prediction Count
                {
                    "id": 3,
                    "title": "Model Predictions",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "predictions_total",
                            "legendFormat": "Total Predictions",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "thresholds"},
                            "mappings": [],
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 1000},
                                    {"color": "red", "value": 5000},
                                ]
                            },
                        }
                    },
                    "gridPos": {"h": 8, "w": 6, "x": 0, "y": 8},
                },
                # Panel 4: Error Rate
                {
                    "id": 4,
                    "title": "Error Rate",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": 'rate(flask_http_request_total{status=~"4..|5.."}[5m]) / rate(flask_http_request_total[5m]) * 100',
                            "legendFormat": "Error Rate %",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "thresholds"},
                            "unit": "percent",
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 1},
                                    {"color": "red", "value": 5},
                                ]
                            },
                        }
                    },
                    "gridPos": {"h": 8, "w": 6, "x": 6, "y": 8},
                },
                # Panel 5: Model Performance
                {
                    "id": 5,
                    "title": "Prediction Processing Time",
                    "type": "timeseries",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, prediction_duration_seconds_bucket)",
                            "legendFormat": "95th percentile",
                        },
                        {
                            "expr": "histogram_quantile(0.50, prediction_duration_seconds_bucket)",
                            "legendFormat": "50th percentile",
                        },
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "custom": {"drawStyle": "line", "fillOpacity": 10},
                            "unit": "s",
                        }
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                },
                # Panel 6: System Metrics
                {
                    "id": 6,
                    "title": "Python Memory Usage",
                    "type": "timeseries",
                    "targets": [
                        {
                            "expr": "python_gc_objects_collected_total",
                            "legendFormat": "Objects Collected",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "custom": {"drawStyle": "line", "fillOpacity": 10},
                        }
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                },
                # Panel 7: Request Status Breakdown
                {
                    "id": 7,
                    "title": "HTTP Status Codes",
                    "type": "piechart",
                    "targets": [
                        {
                            "expr": "flask_http_request_total",
                            "legendFormat": "Status {{status}}",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "custom": {
                                "hideFrom": {
                                    "tooltip": False,
                                    "vis": False,
                                    "legend": False,
                                }
                            },
                        }
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
                },
                # Panel 8: Business Metrics - Prediction Values Distribution
                {
                    "id": 8,
                    "title": "Prediction Values Over Time",
                    "type": "timeseries",
                    "targets": [
                        {
                            "expr": "avg_over_time(prediction_value[5m])",
                            "legendFormat": "Average Prediction",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "custom": {"drawStyle": "line", "fillOpacity": 10},
                            "unit": "currencyUSD",
                        }
                    },
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 24},
                },
            ],
            "time": {"from": "now-1h", "to": "now"},
            "timepicker": {},
            "templating": {"list": []},
            "annotations": {"list": []},
            "refresh": "5s",
            "schemaVersion": 30,
            "version": 1,
            "links": [],
        }
    }

    return dashboard


def create_alerting_rules():
    """
    Create alerting rules for the MLOps pipeline.

    Returns alert rules for:
    - High error rate
    - High response time
    - Model prediction anomalies
    - System resource issues
    """

    alerts = {
        "groups": [
            {
                "name": "mlops_alerts",
                "rules": [
                    {
                        "alert": "HighErrorRate",
                        "expr": 'rate(flask_http_request_total{status=~"4..|5.."}[5m]) / rate(flask_http_request_total[5m]) > 0.05',
                        "for": "2m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "High error rate detected",
                            "description": "Error rate is above 5% for more than 2 minutes",
                        },
                    },
                    {
                        "alert": "HighResponseTime",
                        "expr": "histogram_quantile(0.95, flask_http_request_duration_seconds_bucket) > 1.0",
                        "for": "5m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "High response time detected",
                            "description": "95th percentile response time is above 1 second",
                        },
                    },
                    {
                        "alert": "ModelPredictionAnomalies",
                        "expr": "histogram_quantile(0.95, prediction_duration_seconds_bucket) > 0.5",
                        "for": "3m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "Model prediction taking too long",
                            "description": "Model prediction time is above 500ms",
                        },
                    },
                    {
                        "alert": "NoRecentPredictions",
                        "expr": "increase(predictions_total[10m]) == 0",
                        "for": "10m",
                        "labels": {"severity": "info"},
                        "annotations": {
                            "summary": "No recent predictions",
                            "description": "No predictions made in the last 10 minutes",
                        },
                    },
                ],
            }
        ]
    }

    return alerts


# Save dashboard and alerting configurations
if __name__ == "__main__":
    import os

    # Create monitoring directory
    os.makedirs("deployment/monitoring/grafana", exist_ok=True)

    # Save dashboard configuration
    dashboard = create_mlops_dashboard()
    with open("deployment/monitoring/grafana/mlops_dashboard.json", "w") as f:
        json.dump(dashboard, f, indent=2)

    # Save alerting rules
    alerts = create_alerting_rules()
    with open("deployment/monitoring/grafana/alert_rules.json", "w") as f:
        json.dump(alerts, f, indent=2)

    print("✅ Grafana dashboard and alerting rules created successfully!")
    print("📊 Dashboard: deployment/monitoring/grafana/mlops_dashboard.json")
    print("🚨 Alerts: deployment/monitoring/grafana/alert_rules.json")
