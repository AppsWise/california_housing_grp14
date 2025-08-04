"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union
from enum import Enum
from datetime import datetime

class OceanProximity(str, Enum):
    """Ocean proximity categories"""
    NEAR_BAY = "NEAR BAY"
    LESS_THAN_1H_OCEAN = "<1H OCEAN"
    INLAND = "INLAND"
    NEAR_OCEAN = "NEAR OCEAN"
    ISLAND = "ISLAND"

class HousingPredictionInput(BaseModel):
    """Input schema for housing price prediction"""
    
    longitude: float = Field(..., ge=-124.6, le=-114.1, description="Longitude coordinate (California bounds)")
    latitude: float = Field(..., ge=32.5, le=42.0, description="Latitude coordinate (California bounds)")
    housing_median_age: float = Field(..., ge=0, le=100, description="Median age of houses in the block")
    total_rooms: float = Field(..., ge=1, description="Total number of rooms in the block")
    total_bedrooms: float = Field(..., ge=1, description="Total number of bedrooms in the block")
    population: float = Field(..., ge=1, description="Population of the block")
    households: float = Field(..., ge=1, description="Number of households in the block")
    median_income: float = Field(..., ge=0, description="Median income of households in the block")
    ocean_proximity: OceanProximity = Field(..., description="Ocean proximity category")
    
    @validator('total_bedrooms')
    def bedrooms_less_than_rooms(cls, v, values):
        """Validate that bedrooms are less than total rooms"""
        if 'total_rooms' in values and v > values['total_rooms']:
            raise ValueError('Total bedrooms cannot exceed total rooms')
        return v
    
    @validator('households')
    def households_less_than_population(cls, v, values):
        """Validate that households are reasonable compared to population"""
        if 'population' in values and v > values['population']:
            raise ValueError('Number of households cannot exceed population')
        return v

class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions"""
    
    predictions: List[HousingPredictionInput] = Field(..., min_items=1, max_items=1000, 
                                                     description="List of prediction inputs")

class PredictionResponse(BaseModel):
    """Response schema for single prediction"""
    
    prediction: float = Field(..., description="Predicted house price")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    validation_warnings: List[str] = Field(default=[], description="Any validation warnings")

class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    
    predictions: List[float] = Field(..., description="List of predicted house prices")
    count: int = Field(..., description="Number of predictions made")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    validation_warnings: List[str] = Field(default=[], description="Any validation warnings")

class ErrorResponse(BaseModel):
    """Error response schema"""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[List[dict]] = Field(None, description="Error details")
    timestamp: str = Field(..., description="Error timestamp")

class HealthResponse(BaseModel):
    """Health check response schema"""
    
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Model version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")

class MetricsResponse(BaseModel):
    """Metrics response schema"""
    
    total_predictions: int = Field(..., description="Total number of predictions made")
    predictions_last_hour: int = Field(..., description="Predictions in the last hour")
    average_response_time_ms: float = Field(..., description="Average response time")
    error_rate_percentage: float = Field(..., description="Error rate percentage")
    model_accuracy: Optional[float] = Field(None, description="Current model accuracy")
    last_updated: str = Field(..., description="Last metrics update timestamp")

class RetrainingRequest(BaseModel):
    """Model retraining request schema"""
    
    trigger_reason: str = Field(..., description="Reason for retraining")
    force_retrain: bool = Field(default=False, description="Force retraining even if not needed")
    
class RetrainingResponse(BaseModel):
    """Model retraining response schema"""
    
    status: str = Field(..., description="Retraining status")
    message: str = Field(..., description="Retraining message")
    job_id: Optional[str] = Field(None, description="Retraining job ID")
    estimated_completion_time: Optional[str] = Field(None, description="Estimated completion time")
    timestamp: str = Field(..., description="Request timestamp")
