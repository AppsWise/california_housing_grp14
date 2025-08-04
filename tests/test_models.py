"""Tests for model training and evaluation"""

import pytest
import os
import sys
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_model_exists():
    """Test that trained model exists"""
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "model.pkl"
    )
    assert os.path.exists(model_path), "Model file should exist"


def test_model_training_data():
    """Test that training data exists and has correct format"""
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "housing.csv"
    )
    assert os.path.exists(data_path), "Training data should exist"

    # Load and check data
    df = pd.read_csv(data_path)

    # Check required columns
    required_columns = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "median_house_value",
        "ocean_proximity",
    ]

    for col in required_columns:
        assert col in df.columns, f"Column {col} should be present in data"

    # Check data types
    assert (
        df["ocean_proximity"].dtype == "object"
    ), "ocean_proximity should be categorical"
    assert len(df) > 0, "Dataset should not be empty"


def test_model_prediction_format(sample_housing_data):
    """Test that model prediction returns correct format"""
    try:
        import pickle
        import pandas as pd

        # Load model
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "model.pkl"
        )
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Create test dataframe
        test_df = pd.DataFrame([sample_housing_data])
        # Add engineered features expected by the model
        test_df["rooms_per_household"] = test_df["total_rooms"] / test_df["households"]
        test_df["bedrooms_per_room"] = test_df["total_bedrooms"] / test_df["total_rooms"]
        test_df["population_per_household"] = test_df["population"] / test_df["households"]
        # Make prediction
        prediction = model.predict(test_df)

        assert len(prediction) == 1, "Should return single prediction"
        assert isinstance(
            prediction[0], (int, float)
        ), "Prediction should be numeric"
        assert prediction[0] > 0, "House price prediction should be positive"

    except FileNotFoundError:
        pytest.skip("Model file not found - train model first")
