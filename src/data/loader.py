"""
Data loading utilities for the California Housing dataset
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def load_housing_data(data_path: str = "data/housing.csv") -> pd.DataFrame:
    """
    Load the California Housing dataset
    
    Args:
        data_path: Path to the housing dataset CSV file
        
    Returns:
        DataFrame with housing data
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded housing data with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Housing data file not found at {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading housing data: {e}")
        raise

def split_features_target(df: pd.DataFrame, target_column: str = "median_house_value") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset into features and target
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        
    Returns:
        Tuple of (features, target)
    """
    features = df.drop(columns=[target_column])
    target = df[target_column]
    
    logger.info(f"Split data into features {features.shape} and target {target.shape}")
    return features, target

def load_processed_data(data_path: str = "data/processed") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load processed training and testing data
    
    Args:
        data_path: Path to processed data directory
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    try:
        X_train = np.load(f"{data_path}/X_train.npy")
        X_test = np.load(f"{data_path}/X_test.npy")
        y_train = np.load(f"{data_path}/y_train.npy")
        y_test = np.load(f"{data_path}/y_test.npy")
        
        logger.info(f"Loaded processed data: train {X_train.shape}, test {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        logger.error(f"Processed data files not found in {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise
