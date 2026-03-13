from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionResult(BaseModel):
    """
    Individual prediction result
    """
    class_id: int = Field(..., description="Predicted class ID")
    confidence: float = Field(..., description="Confidence score")


class PredictionResponse(BaseModel):
    """
    API response schema for /predict
    """
    predictions: List[PredictionResult]
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class ErrorResponse(BaseModel):
    """
    Error response schema
    """
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """
    Health check schema
    """
    status: str