from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTION_COUNT = Counter(
    "ml_predictions_total",
    "Total ML predictions served",
    ["model_version", "status"],
)

PREDICTION_LATENCY = Histogram(
    "ml_prediction_duration_seconds",
    "ML model inference latency",
    ["model_version"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

CACHE_HITS = Counter(
    "cache_hits_total",
    "Total cache hits",
    ["cache_type"],
)

CACHE_MISSES = Counter(
    "cache_misses_total",
    "Total cache misses",
    ["cache_type"],
)

ACTIVE_CONNECTIONS = Gauge(
    "db_active_connections",
    "Active database connections",
)
