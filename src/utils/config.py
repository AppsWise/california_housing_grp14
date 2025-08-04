"""
Configuration management for the MLOps project
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Config:
    """Base configuration class"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_data = {}
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """Load configuration from file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"Config file {config_file} not found, using defaults")
            return
        
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    self.config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    self.config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config_data
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value

class ModelConfig(Config):
    """Model configuration management"""
    
    def __init__(self, config_file: str = "config/model_config.yaml"):
        super().__init__(config_file)
        self.set_defaults()
    
    def set_defaults(self):
        """Set default model configuration"""
        defaults = {
            'model': {
                'type': 'decision_tree',
                'random_state': 42,
                'test_size': 0.2,
                'cv_folds': 5
            },
            'hyperparameters': {
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            },
            'features': {
                'categorical': ['ocean_proximity'],
                'numerical': [
                    'longitude', 'latitude', 'housing_median_age',
                    'total_rooms', 'total_bedrooms', 'population',
                    'households', 'median_income'
                ]
            },
            'preprocessing': {
                'scale_features': True,
                'handle_missing': 'median',
                'encode_categorical': 'onehot'
            }
        }
        
        for key, value in defaults.items():
            if not self.get(key):
                self.set(key, value)

class APIConfig(Config):
    """API configuration management"""
    
    def __init__(self, config_file: str = "config/api_config.yaml"):
        super().__init__(config_file)
        self.set_defaults()
    
    def set_defaults(self):
        """Set default API configuration"""
        defaults = {
            'server': {
                'host': '0.0.0.0',
                'port': int(os.getenv('PORT', 5001)),
                'debug': os.getenv('DEBUG', 'false').lower() == 'true',
                'workers': int(os.getenv('WORKERS', 1))
            },
            'model': {
                'path': os.getenv('MODEL_PATH', 'models/model.pkl'),
                'version': '1.0.0',
                'reload_interval': 3600  # seconds
            },
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'file': 'monitoring/logs/app.log',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'database': {
                'url': os.getenv('DATABASE_URL', 'sqlite:///monitoring/logs/predictions.db'),
                'echo': False
            },
            'monitoring': {
                'enable_metrics': True,
                'metrics_endpoint': '/metrics',
                'health_endpoint': '/health'
            },
            'validation': {
                'enable_input_validation': True,
                'max_batch_size': 1000,
                'timeout_seconds': 30
            }
        }
        
        for key, value in defaults.items():
            if not self.get(key):
                self.set(key, value)

class MonitoringConfig(Config):
    """Monitoring configuration management"""
    
    def __init__(self, config_file: str = "config/monitoring_config.yaml"):
        super().__init__(config_file)
        self.set_defaults()
    
    def set_defaults(self):
        """Set default monitoring configuration"""
        defaults = {
            'prometheus': {
                'port': 9090,
                'scrape_interval': '15s',
                'evaluation_interval': '15s'
            },
            'grafana': {
                'port': 3000,
                'admin_user': 'admin',
                'admin_password': 'admin'
            },
            'alerting': {
                'enable_alerts': True,
                'smtp_server': os.getenv('SMTP_SERVER', ''),
                'smtp_port': int(os.getenv('SMTP_PORT', 587)),
                'alert_email': os.getenv('ALERT_EMAIL', ''),
                'thresholds': {
                    'error_rate_percentage': 5.0,
                    'response_time_ms': 1000,
                    'accuracy_drop_percentage': 10.0
                }
            },
            'retention': {
                'logs_days': 30,
                'metrics_days': 90,
                'predictions_days': 365
            }
        }
        
        for key, value in defaults.items():
            if not self.get(key):
                self.set(key, value)

def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent.parent

def ensure_directories():
    """Ensure required directories exist"""
    directories = [
        'data/raw',
        'data/processed',
        'models/experiments',
        'monitoring/logs',
        'monitoring/prometheus',
        'monitoring/grafana/dashboards',
        'docs'
    ]
    
    project_root = get_project_root()
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep for empty directories
        gitkeep_path = dir_path / '.gitkeep'
        if not any(dir_path.iterdir()) and not gitkeep_path.exists():
            gitkeep_path.touch()

# Global configuration instances
model_config = ModelConfig()
api_config = APIConfig()
monitoring_config = MonitoringConfig()

# Ensure directories exist
ensure_directories()
