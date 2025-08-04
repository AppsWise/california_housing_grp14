"""Input validation schemas using Pydantic for robust API data validation"""

from pydantic import BaseModel, Field, validator, ValidationError
from typing import Literal, List, Optional, Union
from enum import Enum
import pandas as pd
import numpy as np


class OceanProximity(str, Enum):
    """Valid ocean proximity values"""
    NEAR_BAY = "NEAR BAY"
    LESS_THAN_1H_OCEAN = "<1H OCEAN"
    INLAND = "INLAND"
    NEAR_OCEAN = "NEAR OCEAN"
    ISLAND = "ISLAND"


class HousingPredictionInput(BaseModel):
    """
    Pydantic model for validating California Housing prediction input data.
    
    This model ensures all input data meets the required format and constraints
    for the machine learning model to make accurate predictions.
    """
    
    # Geographic coordinates
    longitude: float = Field(
        ..., 
        description="Longitude coordinate",
        ge=-180.0, 
        le=180.0,
        example=-122.23
    )
    
    latitude: float = Field(
        ..., 
        description="Latitude coordinate", 
        ge=-90.0, 
        le=90.0,
        example=37.88
    )
    
    # Housing characteristics
    housing_median_age: float = Field(
        ..., 
        description="Median age of houses in the area",
        ge=0.0, 
        le=100.0,
        example=41.0
    )
    
    total_rooms: float = Field(
        ..., 
        description="Total number of rooms in the area",
        gt=0.0,
        example=880.0
    )
    
    total_bedrooms: float = Field(
        ..., 
        description="Total number of bedrooms in the area",
        gt=0.0,
        example=129.0
    )
    
    # Population and households
    population: float = Field(
        ..., 
        description="Population of the area",
        gt=0.0,
        example=322.0
    )
    
    households: float = Field(
        ..., 
        description="Number of households in the area",
        gt=0.0,
        example=126.0
    )
    
    # Economic indicator
    median_income: float = Field(
        ..., 
        description="Median income (in tens of thousands of dollars)",
        gt=0.0,
        example=8.3252
    )
    
    # Location category
    ocean_proximity: OceanProximity = Field(
        ..., 
        description="Proximity to ocean",
        example="NEAR BAY"
    )
    
    @validator('total_bedrooms')
    def validate_bedrooms_vs_rooms(cls, v, values):
        """Ensure total_bedrooms is reasonable compared to total_rooms"""
        if 'total_rooms' in values and v > values['total_rooms']:
            raise ValueError('total_bedrooms cannot exceed total_rooms')
        return v
    
    @validator('households')
    def validate_households_vs_population(cls, v, values):
        """Ensure reasonable household size"""
        if 'population' in values:
            avg_household_size = values['population'] / v
            if avg_household_size > 20.0:
                raise ValueError('Average household size seems unrealistic (>20 people per household)')
            if avg_household_size < 0.5:
                raise ValueError('Average household size seems unrealistic (<0.5 people per household)')
        return v
    
    @validator('longitude')
    def validate_california_longitude(cls, v):
        """Validate longitude is within California bounds (approximately)"""
        if not (-124.5 <= v <= -114.0):
            raise ValueError('Longitude should be within California bounds (-124.5 to -114.0)')
        return v
    
    @validator('latitude')
    def validate_california_latitude(cls, v):
        """Validate latitude is within California bounds (approximately)"""
        if not (32.0 <= v <= 42.0):
            raise ValueError('Latitude should be within California bounds (32.0 to 42.0)')
        return v
    
    class Config:
        """Pydantic configuration"""
        schema_extra = {
            "example": {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41.0,
                "total_rooms": 880.0,
                "total_bedrooms": 129.0,
                "population": 322.0,
                "households": 126.0,
                "median_income": 8.3252,
                "ocean_proximity": "NEAR BAY"
            }
        }


class BatchPredictionInput(BaseModel):
    """Model for batch prediction requests"""
    
    predictions: List[HousingPredictionInput] = Field(
        ..., 
        description="List of housing prediction inputs",
        min_items=1,
        max_items=1000  # Limit batch size for performance
    )
    
    @validator('predictions')
    def validate_batch_size(cls, v):
        """Ensure reasonable batch size"""
        if len(v) > 1000:
            raise ValueError('Batch size cannot exceed 1000 predictions')
        return v


class PredictionResponse(BaseModel):
    """Model for prediction response"""
    
    prediction: float = Field(..., description="Predicted house value")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    validation_warnings: Optional[List[str]] = Field(default_factory=list, description="Validation warnings")


class BatchPredictionResponse(BaseModel):
    """Model for batch prediction response"""
    
    predictions: List[float] = Field(..., description="List of predicted house values")
    count: int = Field(..., description="Number of predictions")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Total processing time in milliseconds")
    validation_warnings: Optional[List[str]] = Field(default_factory=list, description="Validation warnings")


class ErrorResponse(BaseModel):
    """Model for error responses"""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")


def validate_prediction_input(data: dict) -> tuple[HousingPredictionInput, List[str]]:
    """
    Validate prediction input data using Pydantic model.
    
    Args:
        data: Input data dictionary
        
    Returns:
        Tuple of (validated_data, warnings)
        
    Raises:
        ValidationError: If validation fails
    """
    warnings = []
    
    try:
        # Validate with Pydantic
        validated_data = HousingPredictionInput(**data)
        
        # Additional business logic warnings
        if validated_data.median_income > 15.0:
            warnings.append("High median income detected - ensure data quality")
        
        if validated_data.housing_median_age > 50.0:
            warnings.append("Very old housing detected - prediction may be less accurate")
        
        rooms_per_household = validated_data.total_rooms / validated_data.households
        if rooms_per_household > 10.0:
            warnings.append("High rooms per household ratio - verify data accuracy")
        
        return validated_data, warnings
        
    except ValidationError as e:
        raise e


def validate_batch_input(data: dict) -> tuple[BatchPredictionInput, List[str]]:
    """
    Validate batch prediction input data.
    
    Args:
        data: Input data dictionary with 'predictions' list
        
    Returns:
        Tuple of (validated_data, warnings)
        
    Raises:
        ValidationError: If validation fails
    """
    warnings = []
    
    try:
        validated_data = BatchPredictionInput(**data)
        
        # Aggregate warnings from individual predictions
        for i, prediction in enumerate(validated_data.predictions):
            _, pred_warnings = validate_prediction_input(prediction.dict())
            for warning in pred_warnings:
                warnings.append(f"Prediction {i+1}: {warning}")
        
        return validated_data, warnings
        
    except ValidationError as e:
        raise e


# Example usage and testing
if __name__ == "__main__":
    # Test valid input
    valid_input = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY"
    }
    
    try:
        validated, warnings = validate_prediction_input(valid_input)
        print("✅ Validation successful!")
        print(f"Warnings: {warnings}")
        print(f"Validated data: {validated}")
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
    
    # Test invalid input
    invalid_input = {
        "longitude": -200.0,  # Invalid longitude
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 1000.0,  # More bedrooms than rooms
        "population": 322.0,
        "households": 126.0,
        "median_income": -5.0,  # Negative income
        "ocean_proximity": "INVALID"  # Invalid ocean proximity
    }
    
    try:
        validated, warnings = validate_prediction_input(invalid_input)
        print("✅ Validation successful!")
    except ValidationError as e:
        print(f"❌ Expected validation failure: {e}")
