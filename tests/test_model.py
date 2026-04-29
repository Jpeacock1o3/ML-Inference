import pytest
from unittest.mock import MagicMock
import numpy as np

from app.ml.model import MLModel, compute_input_hash, _load_dummy_model


def test_compute_input_hash_deterministic():
    h1 = compute_input_hash([1.0, 2.0, 3.0], "v1")
    h2 = compute_input_hash([1.0, 2.0, 3.0], "v1")
    assert h1 == h2


def test_compute_input_hash_differs_by_version():
    h1 = compute_input_hash([1.0, 2.0], "v1")
    h2 = compute_input_hash([1.0, 2.0], "v2")
    assert h1 != h2


def test_compute_input_hash_differs_by_features():
    h1 = compute_input_hash([1.0, 2.0], "v1")
    h2 = compute_input_hash([1.0, 3.0], "v1")
    assert h1 != h2


@pytest.fixture
def loaded_model():
    model = MLModel()
    _load_dummy_model(model)
    return model


def test_dummy_model_loads(loaded_model):
    assert loaded_model._model is not None
    assert loaded_model.version == "v1"
    assert loaded_model.metadata["n_features"] == 4


@pytest.mark.asyncio
async def test_predict_returns_expected_shape(loaded_model):
    result = await loaded_model.predict([5.1, 3.5, 1.4, 0.2])
    assert "predicted_class" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "inference_ms" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5


@pytest.mark.asyncio
async def test_predict_raises_without_model():
    model = MLModel()
    with pytest.raises(RuntimeError, match="Model not loaded"):
        await model.predict([1.0, 2.0, 3.0, 4.0])
