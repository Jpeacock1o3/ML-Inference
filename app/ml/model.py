import asyncio
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from app.core.config import get_settings
from app.services.monitoring import PREDICTION_COUNT, PREDICTION_LATENCY

logger = logging.getLogger(__name__)
settings = get_settings()

# Thread pool for CPU-bound inference without blocking the event loop
_executor = ThreadPoolExecutor(max_workers=4)


class MLModel:
    def __init__(self) -> None:
        self._model = None
        self._metadata: dict = {}
        self._version: str = "v1"

    def load(self, path: str) -> None:
        artifact = joblib.load(path)
        self._model = artifact["model"]
        self._metadata = artifact.get("metadata", {})
        self._version = artifact.get("version", "v1")
        logger.info(
            "Model loaded",
            extra={"path": path, "version": self._version, "metadata": self._metadata},
        )

    @property
    def version(self) -> str:
        return self._version

    @property
    def metadata(self) -> dict:
        return self._metadata

    def _run_inference(self, features: list[float]) -> dict:
        X = np.array(features).reshape(1, -1)
        probas = self._model.predict_proba(X)[0]
        class_idx = int(np.argmax(probas))
        class_names = self._metadata.get("class_names", [str(i) for i in range(len(probas))])
        return {
            "predicted_class": class_names[class_idx],
            "confidence": float(probas[class_idx]),
            "probabilities": {class_names[i]: float(p) for i, p in enumerate(probas)},
        }

    async def predict(self, features: list[float]) -> dict:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        loop = asyncio.get_event_loop()
        start = time.perf_counter()

        result = await loop.run_in_executor(_executor, self._run_inference, features)

        inference_ms = (time.perf_counter() - start) * 1000
        result["inference_ms"] = round(inference_ms, 3)

        PREDICTION_COUNT.labels(model_version=self._version, status="success").inc()
        PREDICTION_LATENCY.labels(model_version=self._version).observe(inference_ms / 1000)

        return result


def compute_input_hash(features: list[float], model_version: str) -> str:
    payload = json.dumps({"features": features, "model_version": model_version}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


@lru_cache(maxsize=1)
def get_model() -> MLModel:
    model = MLModel()
    model_path = Path(settings.model_path)
    if model_path.exists():
        model.load(str(model_path))
    else:
        logger.warning(
            "Model file not found, serving dummy model",
            extra={"path": str(model_path)},
        )
        _load_dummy_model(model)
    return model


def _load_dummy_model(model: MLModel) -> None:
    """Load a simple iris classifier as the demo model."""
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X_train, _, y_train, _ = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    model._model = clf
    model._version = "v1"
    model._metadata = {
        "algorithm": "RandomForestClassifier",
        "feature_names": list(iris.feature_names),
        "class_names": list(iris.target_names),
        "n_features": 4,
        "description": "Iris flower species classifier (demo)",
    }
    logger.info("Dummy iris classifier loaded as demo model")
