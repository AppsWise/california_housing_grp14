"""
Data Preparation Script
=======================

Script to prepare the California Housing dataset for training.
This script handles data loading, cleaning, and initial preprocessing.

Author: Group 14
Date: August 2025
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_housing_data(data_path="data/housing.csv"):
    """
    Load the California Housing dataset.
    
    Args:
        data_path (str): Path to the housing dataset
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def basic_data_info(df):
    """Print basic information about the dataset."""
    logger.info("Dataset Information:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        logger.info("Missing values:")
        for col, count in missing_values.items():
            if count > 0:
                logger.info(f"  {col}: {count}")
    else:
        logger.info("No missing values found")
    
    # Basic statistics
    logger.info("\nDataset Statistics:")
    print(df.describe())

def clean_data(df):
    """
    Clean the dataset by handling missing values and outliers.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    logger.info("Cleaning data...")
    
    df_clean = df.copy()
    
    # Handle missing values in total_bedrooms
    if 'total_bedrooms' in df_clean.columns:
        missing_bedrooms = df_clean['total_bedrooms'].isnull().sum()
        if missing_bedrooms > 0:
            logger.info(f"Filling {missing_bedrooms} missing values in total_bedrooms with median")
            df_clean['total_bedrooms'].fillna(df_clean['total_bedrooms'].median(), inplace=True)
    
    # Remove any rows with remaining missing values
    initial_rows = len(df_clean)
    df_clean.dropna(inplace=True)
    final_rows = len(df_clean)
    
    if initial_rows != final_rows:
        logger.info(f"Removed {initial_rows - final_rows} rows with missing values")
    
    # Basic outlier handling for target variable
    if 'median_house_value' in df_clean.columns:
        # Cap the target variable at 500,000 (as mentioned in original dataset)
        df_clean['median_house_value'] = df_clean['median_house_value'].clip(upper=500000)
    
    logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
    return df_clean

def save_processed_data(df, output_path="data/processed/housing_cleaned.csv"):
    """
    Save the processed dataset.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        output_path (str): Output file path
    """
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the data
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")

def main():
    """Main data preparation function."""
    logger.info("=" * 50)
    logger.info("CALIFORNIA HOUSING DATA PREPARATION")
    logger.info("=" * 50)
    
    # Load data
    df = load_housing_data()
    if df is None:
        logger.error("Failed to load data. Exiting.")
        sys.exit(1)
    
    # Show basic info
    basic_data_info(df)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Save processed data
    save_processed_data(df_clean)
    
    logger.info("=" * 50)
    logger.info("DATA PREPARATION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
