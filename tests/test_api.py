import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock, patch

from app.main import app
from app.ml.model import MLModel
from app.db.database import get_db


# ---- Fixtures ----------------------------------------------------------------

@pytest.fixture
def mock_model():
    model = MagicMock(spec=MLModel)
    model.version = "v1"
    model._model = MagicMock()
    model.predict = AsyncMock(return_value={
        "predicted_class": "setosa",
        "confidence": 0.98,
        "probabilities": {"setosa": 0.98, "versicolor": 0.01, "virginica": 0.01},
        "inference_ms": 2.5,
    })
    return model


@pytest.fixture
def mock_db():
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
async def client(mock_model, mock_db):
    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db

    with patch("app.ml.model.get_model", return_value=mock_model), \
         patch("app.db.cache.cache_get", new_callable=AsyncMock, return_value=None), \
         patch("app.db.cache.cache_set", new_callable=AsyncMock):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac

    app.dependency_overrides.clear()


# ---- Tests -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_predict_success(client, mock_model):
    response = await client.post(
        "/api/v1/predictions/",
        json={"features": [5.1, 3.5, 1.4, 0.2]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == "setosa"
    assert data["confidence"] == pytest.approx(0.98)
    assert data["cache_hit"] is False
    assert "request_id" in data


@pytest.mark.asyncio
async def test_predict_cache_hit(mock_model, mock_db):
    cached = {
        "predicted_class": "setosa",
        "confidence": 0.98,
        "probabilities": {"setosa": 0.98, "versicolor": 0.01, "virginica": 0.01},
        "inference_ms": 2.5,
    }

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db

    with patch("app.ml.model.get_model", return_value=mock_model), \
         patch("app.db.cache.cache_get", new_callable=AsyncMock, return_value=cached):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(
                "/api/v1/predictions/",
                json={"features": [5.1, 3.5, 1.4, 0.2]},
            )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["cache_hit"] is True
    mock_model.predict.assert_not_called()


@pytest.mark.asyncio
async def test_predict_invalid_features(client):
    response = await client.post(
        "/api/v1/predictions/",
        json={"features": []},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_non_finite_features(client):
    response = await client.post(
        "/api/v1/predictions/",
        json={"features": [float("inf"), 3.5, 1.4, 0.2]},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_predict(client):
    response = await client.post(
        "/api/v1/predictions/batch",
        json={
            "requests": [
                {"features": [5.1, 3.5, 1.4, 0.2]},
                {"features": [6.7, 3.0, 5.2, 2.3]},
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    assert "total_ms" in data


@pytest.mark.asyncio
async def test_health_endpoint():
    with patch("app.api.routes.health.AsyncSessionLocal") as mock_session_cls, \
         patch("app.api.routes.health.get_redis") as mock_get_redis, \
         patch("app.api.routes.health.get_model") as mock_get_model:

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock()
        mock_session_cls.return_value = mock_session

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_get_redis.return_value = mock_redis

        mock_model_instance = MagicMock()
        mock_model_instance._model = MagicMock()
        mock_model_instance.version = "v1"
        mock_get_model.return_value = mock_model_instance

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "version" in data
