"""
Exploratory Data Analysis (EDA) Module
======================================

This module provides comprehensive EDA functionality for the California Housing dataset.
Converts the original Jupyter notebook into a production-ready Python module.

Author: Group 14
Date: August 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HousingEDA:
    """
    Exploratory Data Analysis class for California Housing dataset.
    
    Provides comprehensive analysis including:
    - Data structure analysis
    - Statistical summaries
    - Missing value analysis
    - Distribution analysis
    - Correlation analysis
    - Feature engineering insights
    """
    
    def __init__(self, data_path: str = "data/housing.csv"):
        """
        Initialize EDA with dataset path.
        
        Args:
            data_path (str): Path to the housing dataset
        """
        self.data_path = Path(data_path)
        self.df = None
        self.numeric_columns = None
        self.categorical_columns = None
        self.analysis_results = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the housing dataset.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            
            # Identify column types
            self.numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            self.categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
            
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def analyze_data_structure(self) -> Dict:
        """
        Analyze basic data structure and information.
        
        Returns:
            Dict: Data structure analysis results
        """
        if self.df is None:
            self.load_data()
            
        logger.info("Analyzing data structure...")
        
        structure_info = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'data_types': self.df.dtypes.to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns
        }
        
        self.analysis_results['structure'] = structure_info
        
        # Print summary
        print("=" * 50)
        print("DATA STRUCTURE ANALYSIS")
        print("=" * 50)
        print(f"Dataset Shape: {structure_info['shape']}")
        print(f"Total Columns: {len(structure_info['columns'])}")
        print(f"Numeric Columns: {len(self.numeric_columns)}")
        print(f"Categorical Columns: {len(self.categorical_columns)}")
        print(f"Memory Usage: {structure_info['memory_usage'] / 1024**2:.2f} MB")
        
        return structure_info
    
    def analyze_missing_values(self) -> Dict:
        """
        Analyze missing values in the dataset.
        
        Returns:
            Dict: Missing value analysis results
        """
        if self.df is None:
            self.load_data()
            
        logger.info("Analyzing missing values...")
        
        missing_count = self.df.isnull().sum()
        missing_percentage = (missing_count / len(self.df)) * 100
        
        missing_info = {
            'missing_count': missing_count.to_dict(),
            'missing_percentage': missing_percentage.to_dict(),
            'total_missing': missing_count.sum(),
            'columns_with_missing': missing_count[missing_count > 0].index.tolist()
        }
        
        self.analysis_results['missing_values'] = missing_info
        
        # Print summary
        print("\n" + "=" * 50)
        print("MISSING VALUES ANALYSIS")
        print("=" * 50)
        if missing_info['total_missing'] > 0:
            print("Columns with missing values:")
            for col in missing_info['columns_with_missing']:
                count = missing_info['missing_count'][col]
                pct = missing_info['missing_percentage'][col]
                print(f"  {col}: {count} ({pct:.2f}%)")
        else:
            print("No missing values found!")
        
        return missing_info
    
    def analyze_distributions(self) -> Dict:
        """
        Analyze distributions of numeric features.
        
        Returns:
            Dict: Distribution analysis results
        """
        if self.df is None:
            self.load_data()
            
        logger.info("Analyzing feature distributions...")
        
        # Statistical summary
        stats_summary = self.df[self.numeric_columns].describe()
        
        # Skewness analysis
        skewness = self.df[self.numeric_columns].skew().sort_values(ascending=False)
        
        # Kurtosis analysis
        kurtosis = self.df[self.numeric_columns].kurtosis().sort_values(ascending=False)
        
        distribution_info = {
            'statistical_summary': stats_summary.to_dict(),
            'skewness': skewness.to_dict(),
            'kurtosis': kurtosis.to_dict(),
            'highly_skewed_features': skewness[abs(skewness) > 1].index.tolist()
        }
        
        self.analysis_results['distributions'] = distribution_info
        
        # Print summary
        print("\n" + "=" * 50)
        print("DISTRIBUTION ANALYSIS")
        print("=" * 50)
        print("Statistical Summary:")
        print(stats_summary)
        
        print(f"\nSkewness Analysis:")
        print("Highly skewed features (|skew| > 1):")
        for feature in distribution_info['highly_skewed_features']:
            print(f"  {feature}: {skewness[feature]:.3f}")
        
        return distribution_info
    
    def analyze_correlations(self) -> Dict:
        """
        Analyze correlations between features.
        
        Returns:
            Dict: Correlation analysis results
        """
        if self.df is None:
            self.load_data()
            
        logger.info("Analyzing feature correlations...")
        
        # Correlation matrix
        correlation_matrix = self.df[self.numeric_columns].corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature_1': correlation_matrix.columns[i],
                        'feature_2': correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        # Target correlations (assuming last column is target)
        target_col = self.numeric_columns[-1]  # Assuming 'median_house_value'
        target_correlations = correlation_matrix[target_col].sort_values(ascending=False)
        
        correlation_info = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlation_pairs': high_corr_pairs,
            'target_correlations': target_correlations.to_dict(),
            'strong_predictors': target_correlations[abs(target_correlations) > 0.3].index.tolist()
        }
        
        self.analysis_results['correlations'] = correlation_info
        
        # Print summary
        print("\n" + "=" * 50)
        print("CORRELATION ANALYSIS")
        print("=" * 50)
        print("Strong predictors (|correlation| > 0.3):")
        for predictor in correlation_info['strong_predictors']:
            if predictor != target_col:
                corr = target_correlations[predictor]
                print(f"  {predictor}: {corr:.3f}")
        
        if high_corr_pairs:
            print(f"\nHighly correlated feature pairs:")
            for pair in high_corr_pairs:
                print(f"  {pair['feature_1']} - {pair['feature_2']}: {pair['correlation']:.3f}")
        
        return correlation_info
    
    def analyze_categorical_features(self) -> Dict:
        """
        Analyze categorical features.
        
        Returns:
            Dict: Categorical analysis results
        """
        if self.df is None:
            self.load_data()
            
        if not self.categorical_columns:
            return {'message': 'No categorical columns found'}
            
        logger.info("Analyzing categorical features...")
        
        categorical_info = {}
        
        for col in self.categorical_columns:
            value_counts = self.df[col].value_counts()
            categorical_info[col] = {
                'unique_values': self.df[col].nunique(),
                'value_counts': value_counts.to_dict(),
                'most_frequent': value_counts.index[0],
                'most_frequent_count': value_counts.iloc[0]
            }
        
        self.analysis_results['categorical'] = categorical_info
        
        # Print summary
        print("\n" + "=" * 50)
        print("CATEGORICAL FEATURES ANALYSIS")
        print("=" * 50)
        for col, info in categorical_info.items():
            print(f"{col}:")
            print(f"  Unique values: {info['unique_values']}")
            print(f"  Most frequent: {info['most_frequent']} ({info['most_frequent_count']} occurrences)")
            print(f"  Value distribution: {info['value_counts']}")
        
        return categorical_info
    
    def generate_feature_insights(self) -> Dict:
        """
        Generate insights for feature engineering.
        
        Returns:
            Dict: Feature engineering insights
        """
        if self.df is None:
            self.load_data()
            
        logger.info("Generating feature engineering insights...")
        
        insights = {
            'potential_new_features': [],
            'features_to_transform': [],
            'features_to_remove': [],
            'scaling_recommendations': {}
        }
        
        # Check for potential ratio features
        if 'total_bedrooms' in self.df.columns and 'households' in self.df.columns:
            insights['potential_new_features'].append('bedrooms_per_household')
        
        if 'population' in self.df.columns and 'households' in self.df.columns:
            insights['potential_new_features'].append('population_per_household')
        
        if 'total_rooms' in self.df.columns and 'households' in self.df.columns:
            insights['potential_new_features'].append('rooms_per_household')
        
        # Check for features that need transformation (highly skewed)
        if 'distributions' in self.analysis_results:
            skewed_features = self.analysis_results['distributions']['highly_skewed_features']
            insights['features_to_transform'] = skewed_features
        
        # Scaling recommendations
        for col in self.numeric_columns:
            col_range = self.df[col].max() - self.df[col].min()
            if col_range > 1000:
                insights['scaling_recommendations'][col] = 'StandardScaler or MinMaxScaler recommended'
        
        self.analysis_results['insights'] = insights
        
        # Print insights
        print("\n" + "=" * 50)
        print("FEATURE ENGINEERING INSIGHTS")
        print("=" * 50)
        
        if insights['potential_new_features']:
            print("Potential new features to create:")
            for feature in insights['potential_new_features']:
                print(f"  - {feature}")
        
        if insights['features_to_transform']:
            print("\nFeatures requiring transformation (highly skewed):")
            for feature in insights['features_to_transform']:
                print(f"  - {feature}")
        
        if insights['scaling_recommendations']:
            print("\nScaling recommendations:")
            for feature, recommendation in insights['scaling_recommendations'].items():
                print(f"  - {feature}: {recommendation}")
        
        return insights
    
    def run_full_analysis(self) -> Dict:
        """
        Run complete EDA analysis.
        
        Returns:
            Dict: Complete analysis results
        """
        logger.info("Running full EDA analysis...")
        
        # Run all analyses
        self.analyze_data_structure()
        self.analyze_missing_values()
        self.analyze_distributions()
        self.analyze_correlations()
        self.analyze_categorical_features()
        self.generate_feature_insights()
        
        logger.info("EDA analysis completed successfully!")
        return self.analysis_results
    
    def save_analysis_results(self, output_path: str = "reports/eda_results.json"):
        """
        Save analysis results to JSON file.
        
        Args:
            output_path (str): Path to save the results
        """
        import json
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            return obj
        
        # Recursively convert all numpy types
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(v) for v in obj]
            else:
                return convert_numpy_types(obj)
        
        converted_results = recursive_convert(self.analysis_results)
        
        with open(output_path, 'w') as f:
            json.dump(converted_results, f, indent=2, default=str)
        
        logger.info(f"Analysis results saved to {output_path}")


def main():
    """
    Main function to run EDA analysis.
    """
    # Initialize EDA
    eda = HousingEDA(data_path="data/housing.csv")
    
    # Run full analysis
    try:
        results = eda.run_full_analysis()
        
        # Save results
        eda.save_analysis_results("reports/eda_results.json")
        
        logger.info("EDA completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Error in EDA analysis: {e}")
        raise


if __name__ == "__main__":
    main()
