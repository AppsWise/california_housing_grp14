"""Data processing utilities"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def load_housing_data(file_path: str) -> pd.DataFrame:
    """
    Load California housing dataset
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with housing data
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def prepare_features(df: pd.DataFrame, target_column: str = 'median_house_value'):
    """
    Prepare features for training
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle missing values
    X = X.fillna(X.median())
    
    logger.info(f"Prepared {X.shape[1]} features for {len(X)} samples")
    return X, y

def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into train and test sets
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def validate_input_data(data: dict) -> bool:
    """
    Validate input data for prediction
    
    Args:
        data: Dictionary with input features
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income',
        'ocean_proximity'
    ]
    
    # Check all required fields are present
    for field in required_fields:
        if field not in data:
            logger.warning(f"Missing required field: {field}")
            return False
    
    # Check numeric fields
    numeric_fields = required_fields[:-1]  # All except ocean_proximity
    for field in numeric_fields:
        try:
            float(data[field])
        except (ValueError, TypeError):
            logger.warning(f"Invalid numeric value for {field}: {data[field]}")
            return False
    
    # Check categorical field
    valid_ocean_proximity = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    if data['ocean_proximity'] not in valid_ocean_proximity:
        logger.warning(f"Invalid ocean_proximity value: {data['ocean_proximity']}")
        return False
    
    return True

def prepare_input_for_prediction(data: dict) -> pd.DataFrame:
    """
    Prepare input data for model prediction
    
    Args:
        data: Dictionary with input features
        
    Returns:
        DataFrame ready for prediction
    """
    if not validate_input_data(data):
        raise ValueError("Invalid input data")
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Ensure correct data types
    numeric_columns = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income'
    ]
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col])
    
    # Ensure ocean_proximity is string
    df['ocean_proximity'] = df['ocean_proximity'].astype(str)
    
    return df
