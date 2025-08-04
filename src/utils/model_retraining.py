"""Model re-training trigger system for automated model updates"""

import os
import time
import json
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import sqlite3
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRetrainingTrigger:
    """
    Automated model re-training trigger system.
    
    This system monitors for:
    1. New data arrivals
    2. Model performance degradation
    3. Time-based re-training schedules
    4. Manual triggers
    """
    
    def __init__(self, 
                 data_path: str = "data/",
                 model_path: str = "models/",
                 logs_db_path: str = "logs/predictions.db",
                 config_path: str = "config/retraining_config.json"):
        
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.logs_db_path = logs_db_path
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize monitoring
        self.data_observer = None
        self.last_retrain_time = None
        self.performance_threshold = self.config.get("performance_threshold", 0.1)
        self.min_new_samples = self.config.get("min_new_samples", 100)
        self.retrain_interval_hours = self.config.get("retrain_interval_hours", 24)
        
        logger.info(f"Model retraining trigger initialized")
        logger.info(f"Monitoring data path: {self.data_path}")
        logger.info(f"Performance threshold: {self.performance_threshold}")
        logger.info(f"Min new samples for retrain: {self.min_new_samples}")
    
    def _load_config(self) -> Dict:
        """Load retraining configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default configuration
            default_config = {
                "performance_threshold": 0.1,
                "min_new_samples": 100,
                "retrain_interval_hours": 24,
                "enable_auto_retrain": True,
                "enable_data_monitoring": True,
                "enable_performance_monitoring": True,
                "backup_models": True,
                "notification_email": None
            }
            
            # Create config file
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
    
    def start_monitoring(self):
        """Start all monitoring systems"""
        logger.info("Starting model retraining monitoring...")
        
        if self.config.get("enable_data_monitoring", True):
            self._start_data_monitoring()
        
        if self.config.get("enable_performance_monitoring", True):
            self._start_performance_monitoring()
        
        logger.info("✅ Model retraining monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring systems"""
        if self.data_observer:
            self.data_observer.stop()
            self.data_observer.join()
        logger.info("Model retraining monitoring stopped")
    
    def _start_data_monitoring(self):
        """Monitor for new data files"""
        class DataFileHandler(FileSystemEventHandler):
            def __init__(self, trigger_instance):
                self.trigger = trigger_instance
            
            def on_created(self, event):
                if not event.is_directory and event.src_path.endswith('.csv'):
                    logger.info(f"New data file detected: {event.src_path}")
                    self.trigger._on_new_data_detected(event.src_path)
            
            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith('.csv'):
                    logger.info(f"Data file modified: {event.src_path}")
                    self.trigger._on_new_data_detected(event.src_path)
        
        self.data_observer = Observer()
        self.data_observer.schedule(
            DataFileHandler(self), 
            str(self.data_path), 
            recursive=True
        )
        self.data_observer.start()
        logger.info(f"Data monitoring started for: {self.data_path}")
    
    def _start_performance_monitoring(self):
        """Monitor model performance from prediction logs"""
        # This would typically run as a separate scheduled task
        logger.info("Performance monitoring initialized")
    
    def _on_new_data_detected(self, file_path: str):
        """Handle new data detection"""
        try:
            # Check if the new data meets criteria for retraining
            new_data = pd.read_csv(file_path)
            
            if len(new_data) >= self.min_new_samples:
                logger.info(f"New data file has {len(new_data)} samples (>= {self.min_new_samples})")
                
                if self.config.get("enable_auto_retrain", True):
                    self._trigger_retraining(reason="new_data", details=f"New data file: {file_path}")
                else:
                    logger.info("Auto-retrain disabled. Manual trigger required.")
            else:
                logger.info(f"New data file has only {len(new_data)} samples (< {self.min_new_samples})")
                
        except Exception as e:
            logger.error(f"Error processing new data file {file_path}: {e}")
    
    def check_performance_degradation(self) -> bool:
        """Check if model performance has degraded based on recent predictions"""
        try:
            if not os.path.exists(self.logs_db_path):
                logger.warning(f"Predictions database not found: {self.logs_db_path}")
                return False
            
            with sqlite3.connect(self.logs_db_path) as conn:
                # Get recent predictions (last 24 hours)
                query = """
                SELECT prediction, input_data, timestamp 
                FROM prediction_requests 
                WHERE timestamp > datetime('now', '-24 hours')
                AND prediction IS NOT NULL
                """
                
                recent_predictions = pd.read_sql_query(query, conn)
                
                if len(recent_predictions) < 10:
                    logger.info("Insufficient recent predictions for performance check")
                    return False
                
                # Simple performance check: look for anomalies
                predictions = pd.to_numeric(recent_predictions['prediction'], errors='coerce')
                
                # Check for unusual prediction patterns
                mean_pred = predictions.mean()
                std_pred = predictions.std()
                
                # Flag if too many predictions are outside 2 standard deviations
                outliers = predictions[(predictions < mean_pred - 2*std_pred) | 
                                    (predictions > mean_pred + 2*std_pred)]
                
                outlier_ratio = len(outliers) / len(predictions)
                
                if outlier_ratio > self.performance_threshold:
                    logger.warning(f"High outlier ratio detected: {outlier_ratio:.3f}")
                    return True
                
                logger.info(f"Performance check passed. Outlier ratio: {outlier_ratio:.3f}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking performance: {e}")
            return False
    
    def check_time_based_retrain(self) -> bool:
        """Check if it's time for scheduled retraining"""
        if not self.last_retrain_time:
            # Check last model modification time
            model_file = self.model_path / "model.pkl"
            if model_file.exists():
                self.last_retrain_time = datetime.fromtimestamp(model_file.stat().st_mtime)
            else:
                return True  # No model exists, should train
        
        time_since_retrain = datetime.now() - self.last_retrain_time
        hours_since_retrain = time_since_retrain.total_seconds() / 3600
        
        if hours_since_retrain >= self.retrain_interval_hours:
            logger.info(f"Time-based retrain triggered. Hours since last retrain: {hours_since_retrain:.1f}")
            return True
        
        return False
    
    def _trigger_retraining(self, reason: str, details: str = ""):
        """Trigger model retraining"""
        logger.info(f"🔄 Triggering model retraining. Reason: {reason}")
        logger.info(f"Details: {details}")
        
        try:
            # Backup current model if configured
            if self.config.get("backup_models", True):
                self._backup_current_model()
            
            # Log retraining event
            self._log_retraining_event(reason, details)
            
            # Execute retraining
            success = self._execute_retraining()
            
            if success:
                self.last_retrain_time = datetime.now()
                logger.info("✅ Model retraining completed successfully")
                
                # Send notification if configured
                if self.config.get("notification_email"):
                    self._send_notification(f"Model retrained successfully. Reason: {reason}")
            else:
                logger.error("❌ Model retraining failed")
                
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
    
    def _backup_current_model(self):
        """Backup current model before retraining"""
        model_file = self.model_path / "model.pkl"
        if model_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.model_path / f"model_backup_{timestamp}.pkl"
            shutil.copy2(model_file, backup_file)
            logger.info(f"Model backed up to: {backup_file}")
    
    def _execute_retraining(self) -> bool:
        """Execute the actual model retraining"""
        try:
            # Run the training script
            result = subprocess.run(
                ["python", "src/models/train.py"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info("Training script executed successfully")
                return True
            else:
                logger.error(f"Training script failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Training script timed out")
            return False
        except Exception as e:
            logger.error(f"Error executing training script: {e}")
            return False
    
    def _log_retraining_event(self, reason: str, details: str):
        """Log retraining events to database"""
        try:
            with sqlite3.connect(self.logs_db_path) as conn:
                cursor = conn.cursor()
                
                # Create retraining log table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS retraining_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        details TEXT,
                        status TEXT,
                        model_backup_path TEXT
                    )
                """)
                
                cursor.execute("""
                    INSERT INTO retraining_events 
                    (timestamp, reason, details, status)
                    VALUES (?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    reason,
                    details,
                    "triggered"
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error logging retraining event: {e}")
    
    def _send_notification(self, message: str):
        """Send notification about retraining events"""
        # This would integrate with email/Slack/etc.
        logger.info(f"📧 Notification: {message}")
    
    def manual_retrain(self, reason: str = "manual_trigger"):
        """Manually trigger model retraining"""
        self._trigger_retraining(reason, "Manually triggered retraining")
    
    def run_monitoring_cycle(self):
        """Run a single monitoring cycle (for scheduled execution)"""
        logger.info("Running monitoring cycle...")
        
        # Check performance degradation
        if self.check_performance_degradation():
            self._trigger_retraining("performance_degradation", "Model performance below threshold")
            return
        
        # Check time-based retraining
        if self.check_time_based_retrain():
            self._trigger_retraining("scheduled_retrain", "Time-based retraining schedule")
            return
        
        logger.info("No retraining triggers detected")


def create_monitoring_service():
    """Create monitoring service script"""
    service_script = '''#!/usr/bin/env python3
"""Model retraining monitoring service"""

import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.model_retraining import ModelRetrainingTrigger

def main():
    trigger = ModelRetrainingTrigger()
    
    try:
        # Start continuous monitoring
        trigger.start_monitoring()
        
        # Run monitoring cycles every hour
        while True:
            trigger.run_monitoring_cycle()
            time.sleep(3600)  # Sleep for 1 hour
            
    except KeyboardInterrupt:
        print("Stopping monitoring service...")
    finally:
        trigger.stop_monitoring()

if __name__ == "__main__":
    main()
'''
    
    os.makedirs("scripts", exist_ok=True)
    with open("scripts/monitoring_service.py", "w") as f:
        f.write(service_script)
    
    # Make executable
    os.chmod("scripts/monitoring_service.py", 0o755)
    
    logger.info("✅ Monitoring service script created: scripts/monitoring_service.py")


# Example usage
if __name__ == "__main__":
    # Create configuration and service
    trigger = ModelRetrainingTrigger()
    
    # Create monitoring service script
    create_monitoring_service()
    
    # Test performance check
    perf_issue = trigger.check_performance_degradation()
    print(f"Performance degradation detected: {perf_issue}")
    
    # Test time-based check
    time_retrain = trigger.check_time_based_retrain()
    print(f"Time-based retrain needed: {time_retrain}")
    
    print("✅ Model retraining trigger system initialized")
    print("🔄 Use trigger.start_monitoring() to begin monitoring")
    print("🏃 Use scripts/monitoring_service.py for continuous monitoring")
