from pydantic import BaseModel, Field, field_validator
from typing import Any


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description="Feature vector for inference",
        min_length=1,
        max_length=100,
    )
    model_version: str = Field(default="v1", description="Model version to use")

    @field_validator("features")
    @classmethod
    def features_must_be_finite(cls, v: list[float]) -> list[float]:
        import math
        for val in v:
            if not math.isfinite(val):
                raise ValueError("All feature values must be finite numbers")
        return v


class PredictionResponse(BaseModel):
    request_id: str
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    model_version: str
    inference_ms: float
    cache_hit: bool


class BatchPredictionRequest(BaseModel):
    requests: list[PredictionRequest] = Field(..., min_length=1, max_length=50)


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    total_ms: float


class PredictionRecord(BaseModel):
    id: int
    request_id: str
    predicted_class: str
    confidence: float
    model_version: str
    inference_ms: float
    cache_hit: bool
    created_at: str

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    status: str
    db: str
    cache: str
    model: str
    version: str
