"""
Feature Engineering Module
==========================

This module provides feature engineering functionality based on EDA insights.
Converts exploratory analysis into production-ready feature transformations.

Author: Group 14
Date: August 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
import logging
from typing import List
import joblib
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature Engineering class for California Housing dataset.

    Provides comprehensive feature engineering including:
    - New feature creation
    - Feature transformations
    - Scaling and normalization
    - Feature selection
    """

    def __init__(self):
        """Initialize Feature Engineer."""
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.feature_names = []
        self.original_features = []

    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio-based features from existing features.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with new ratio features
        """
        logger.info("Creating ratio features...")

        df_copy = df.copy()

        # Bedrooms per household
        if "total_bedrooms" in df.columns and "households" in df.columns:
            df_copy["bedrooms_per_household"] = (
                df_copy["total_bedrooms"] / df_copy["households"]
            )
            logger.info("Created: bedrooms_per_household")

        # Population per household
        if "population" in df.columns and "households" in df.columns:
            df_copy["population_per_household"] = (
                df_copy["population"] / df_copy["households"]
            )
            logger.info("Created: population_per_household")

        # Rooms per household
        if "total_rooms" in df.columns and "households" in df.columns:
            df_copy["rooms_per_household"] = (
                df_copy["total_rooms"] / df_copy["households"]
            )
            logger.info("Created: rooms_per_household")

        # Income categories
        if "median_income" in df.columns:
            df_copy["income_category"] = pd.cut(
                df_copy["median_income"],
                bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                labels=["very_low", "low", "medium", "high", "very_high"],
            )
            logger.info("Created: income_category")

        # Housing age categories
        if "housing_median_age" in df.columns:
            df_copy["age_category"] = pd.cut(
                df_copy["housing_median_age"],
                bins=[0, 10, 25, 40, np.inf],
                labels=["new", "recent", "older", "old"],
            )
            logger.info("Created: age_category")

        return df_copy

    def apply_transformations(
        self, df: pd.DataFrame, skewed_features: List[str] = None
    ) -> pd.DataFrame:
        """
        Apply transformations to skewed features.

        Args:
            df (pd.DataFrame): Input dataframe
            skewed_features (List[str]):
            List of features that need transformation

        Returns:
            pd.DataFrame: Dataframe with transformed features
        """
        logger.info("Applying feature transformations...")

        df_copy = df.copy()

        if skewed_features is None:
            # Default skewed features for housing dataset
            skewed_features = [
                "total_rooms",
                "total_bedrooms",
                "population",
                "households",
            ]

        for feature in skewed_features:
            if feature in df_copy.columns:
                # Apply log transformation (add 1 to handle zeros)
                df_copy[f"{feature}_log"] = np.log1p(df_copy[feature])
                logger.info(f"Applied log transformation to: {feature}")

                # Apply square root transformation
                df_copy[f"{feature}_sqrt"] = np.sqrt(df_copy[feature])
                logger.info(f"Applied sqrt transformation to: {feature}")

        return df_copy

    def encode_categorical_features(
        self, df: pd.DataFrame, categorical_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Encode categorical features.

        Args:
            df (pd.DataFrame): Input dataframe
            categorical_columns (List[str]): List of categorical columns

        Returns:
            pd.DataFrame: Dataframe with encoded categorical features
        """
        logger.info("Encoding categorical features...")

        df_copy = df.copy()

        if categorical_columns is None:
            categorical_columns = df_copy.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        for col in categorical_columns:
            if col in df_copy.columns:
                # For ocean_proximity, use one-hot encoding
                if col == "ocean_proximity":
                    dummies = pd.get_dummies(
                        df_copy[col], prefix=col, drop_first=False
                    )
                    df_copy = pd.concat([df_copy, dummies], axis=1)
                    logger.info(f"Applied one-hot encoding to: {col}")

                # For ordinal categories, use label encoding
                elif "category" in col:
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                        df_copy[f"{col}_encoded"] = self.encoders[
                            col
                        ].fit_transform(df_copy[col].astype(str))
                    else:
                        df_copy[f"{col}_encoded"] = self.encoders[
                            col
                        ].transform(df_copy[col].astype(str))
                    logger.info(f"Applied label encoding to: {col}")

        return df_copy

    def scale_features(
        self,
        df: pd.DataFrame,
        numeric_columns: List[str] = None,
        scaler_type: str = "standard",
    ) -> pd.DataFrame:
        """
        Scale numeric features.

        Args:
            df (pd.DataFrame): Input dataframe
            numeric_columns (List[str]): List of numeric columns to scale
            scaler_type (str): Type of scaler ('standard' or 'minmax')

        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        logger.info(f"Scaling features with {scaler_type} scaler...")

        df_copy = df.copy()

        if numeric_columns is None:
            # Get numeric columns excluding target variable
            numeric_columns = df_copy.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            if "median_house_value" in numeric_columns:
                numeric_columns.remove(
                    "median_house_value"
                )  # Don't scale target

        # Initialize scaler
        scaler_key = f"{scaler_type}_scaler"
        if scaler_key not in self.scalers:
            if scaler_type == "standard":
                self.scalers[scaler_key] = StandardScaler()
            elif scaler_type == "minmax":
                self.scalers[scaler_key] = MinMaxScaler()
            else:
                raise ValueError("scaler_type must be 'standard' or 'minmax'")

        # Fit and transform
        scaler = self.scalers[scaler_key]

        # Create scaled feature names
        scaled_columns = [f"{col}_scaled" for col in numeric_columns]

        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == len(
            numeric_columns
        ):
            # Scaler is already fitted
            scaled_data = scaler.transform(df_copy[numeric_columns])
        else:
            # Fit and transform
            scaled_data = scaler.fit_transform(df_copy[numeric_columns])

        # Add scaled features to dataframe
        scaled_df = pd.DataFrame(
            scaled_data, columns=scaled_columns, index=df_copy.index
        )
        df_copy = pd.concat([df_copy, scaled_df], axis=1)

        logger.info(f"Scaled {len(numeric_columns)} features")

        return df_copy

    def select_features(
        self,
        df: pd.DataFrame,
        target_column: str = "median_house_value",
        method: str = "variance",
        threshold: float = 0.01,
    ) -> pd.DataFrame:
        """
        Select features based on various criteria.

        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Target column name
            method (str): Feature selection method
            threshold (float): Threshold for feature selection

        Returns:
            pd.DataFrame: Dataframe with selected features
        """
        logger.info(f"Selecting features using {method} method...")

        df_copy = df.copy()
        feature_columns = [
            col for col in df_copy.columns if col != target_column
        ]

        if method == "variance":
            # Remove low-variance features
            selector_key = f"variance_selector_{threshold}"
            if selector_key not in self.feature_selectors:
                self.feature_selectors[selector_key] = VarianceThreshold(
                    threshold=threshold
                )

                # Fit selector on numeric features only
                numeric_features = df_copy[feature_columns].select_dtypes(
                    include=["int64", "float64"]
                )
                self.feature_selectors[selector_key].fit(numeric_features)
                selected_feature_names = numeric_features.columns[
                    self.feature_selectors[selector_key].get_support()
                ].tolist()
            else:
                numeric_features = df_copy[feature_columns].select_dtypes(
                    include=["int64", "float64"]
                )
                selected_feature_names = numeric_features.columns[
                    self.feature_selectors[selector_key].get_support()
                ].tolist()

            # Keep selected numeric features and all categorical features
            categorical_features = (
                df_copy[feature_columns]
                .select_dtypes(exclude=["int64", "float64"])
                .columns.tolist()
            )
            final_features = (
                selected_feature_names + categorical_features + [target_column]
            )

            df_copy = df_copy[final_features]
            logger.info(
                f"Selected {len(selected_feature_names)} numeric "
                f"features out of "
                f"{len(numeric_features.columns)}"
            )

        return df_copy

    def create_polynomial_features(
        self, df: pd.DataFrame, features: List[str] = None, degree: int = 2
    ) -> pd.DataFrame:
        """
        Create polynomial features for specified columns.

        Args:
            df (pd.DataFrame): Input dataframe
            features (List[str]): Features to create polynomial features for
            degree (int): Degree of polynomial features

        Returns:
            pd.DataFrame: Dataframe with polynomial features
        """
        logger.info(f"Creating polynomial features of degree {degree}...")

        df_copy = df.copy()

        if features is None:
            # Default to key numeric features
            features = ["median_income", "housing_median_age"]

        for feature in features:
            if feature in df_copy.columns:
                for d in range(2, degree + 1):
                    df_copy[f"{feature}_poly_{d}"] = df_copy[feature] ** d
                    logger.info(f"Created: {feature}_poly_{d}")

        return df_copy

    def engineer_features(
        self,
        df: pd.DataFrame,
        target_column: str = "median_house_value",
        include_transformations: bool = True,
        include_scaling: bool = True,
        include_polynomial: bool = False,
    ) -> pd.DataFrame:
        """
        Apply complete feature engineering pipeline.

        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Target column name
            include_transformations (bool): Whether to include transformations
            include_scaling (bool): Whether to include scaling
            include_polynomial (bool): Whether to include polynomial features

        Returns:
            pd.DataFrame: Fully engineered dataframe
        """
        logger.info("Starting complete feature engineering pipeline...")

        # Store original feature names
        self.original_features = df.columns.tolist()

        # Step 1: Create ratio features
        df_engineered = self.create_ratio_features(df)

        # Step 2: Apply transformations to skewed features
        if include_transformations:
            skewed_features = [
                "total_rooms",
                "total_bedrooms",
                "population",
                "households",
            ]
            df_engineered = self.apply_transformations(
                df_engineered, skewed_features
            )

        # Step 3: Encode categorical features
        df_engineered = self.encode_categorical_features(df_engineered)

        # Step 4: Create polynomial features
        if include_polynomial:
            df_engineered = self.create_polynomial_features(df_engineered)

        # Step 5: Scale features
        if include_scaling:
            numeric_columns = df_engineered.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            if target_column in numeric_columns:
                numeric_columns.remove(target_column)
            df_engineered = self.scale_features(df_engineered, numeric_columns)

        # Step 6: Feature selection
        df_engineered = self.select_features(df_engineered, target_column)

        # Store final feature names
        self.feature_names = [
            col for col in df_engineered.columns if col != target_column
        ]

        logger.info(
            f"Feature engineering completed. Features: "
            f"{len(self.original_features)} -> {len(self.feature_names)}"
        )

        return df_engineered

    def save_transformers(self, output_dir: str = "models/transformers/"):
        """
        Save fitted transformers for later use.

        Args:
            output_dir (str): Directory to save transformers
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{output_dir}/{name}.pkl")
            logger.info(f"Saved scaler: {name}")

        # Save encoders
        for name, encoder in self.encoders.items():
            joblib.dump(encoder, f"{output_dir}/{name}.pkl")
            logger.info(f"Saved encoder: {name}")

        # Save feature selectors
        for name, selector in self.feature_selectors.items():
            joblib.dump(selector, f"{output_dir}/{name}.pkl")
            logger.info(f"Saved feature selector: {name}")

        # Save feature names
        import json

        with open(f"{output_dir}/feature_names.json", "w") as f:
            json.dump(
                {
                    "original_features": self.original_features,
                    "final_features": self.feature_names,
                },
                f,
                indent=2,
            )

        logger.info(f"All transformers saved to {output_dir}")

    def load_transformers(self, input_dir: str = "models/transformers/"):
        """
        Load previously fitted transformers.

        Args:
            input_dir (str): Directory to load transformers from
        """
        transformer_files = Path(input_dir).glob("*.pkl")

        for file_path in transformer_files:
            name = file_path.stem
            transformer = joblib.load(file_path)

            if "scaler" in name:
                self.scalers[name] = transformer
            elif "encoder" in name:
                self.encoders[name] = transformer
            elif "selector" in name:
                self.feature_selectors[name] = transformer

            logger.info(f"Loaded transformer: {name}")

        # Load feature names
        feature_names_path = Path(input_dir) / "feature_names.json"
        if feature_names_path.exists():
            import json

            with open(feature_names_path, "r") as f:
                feature_info = json.load(f)
                self.original_features = feature_info["original_features"]
                self.feature_names = feature_info["final_features"]

        logger.info(f"Transformers loaded from {input_dir}")


def main():
    """
    Main function to run feature engineering.
    """
    # Load data
    logger.info("Loading housing data...")
    df = pd.read_csv("data/housing.csv")

    # Initialize feature engineer
    engineer = FeatureEngineer()

    # Run feature engineering
    try:
        df_engineered = engineer.engineer_features(
            df,
            target_column="median_house_value",
            include_transformations=True,
            include_scaling=True,
            include_polynomial=False,
        )

        # Save engineered dataset
        output_path = "data/processed/housing_engineered.csv"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_engineered.to_csv(output_path, index=False)
        logger.info(f"Engineered dataset saved to {output_path}")

        # Save transformers
        engineer.save_transformers()

        logger.info("Feature engineering completed successfully!")
        return df_engineered

    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise


if __name__ == "__main__":
    main()
