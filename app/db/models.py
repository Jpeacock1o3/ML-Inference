from datetime import datetime
from sqlalchemy import (
    Integer, String, Float, DateTime, JSON, Index, Boolean,
    func, Text,
)
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    input_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    predicted_class: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    probabilities: Mapped[dict] = mapped_column(JSON, nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    inference_ms: Mapped[float] = mapped_column(Float, nullable=False)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )

    __table_args__ = (
        # Composite index for cache lookups by input hash + model version
        Index("ix_predictions_hash_model", "input_hash", "model_version"),
        # Index for time-range queries in dashboards/monitoring
        Index("ix_predictions_model_created", "model_version", "created_at"),
    )


class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    algorithm: Mapped[str] = mapped_column(String(100), nullable=False)
    feature_names: Mapped[dict] = mapped_column(JSON, nullable=False)
    class_names: Mapped[dict] = mapped_column(JSON, nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
