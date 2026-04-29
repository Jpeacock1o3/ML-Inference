-- MySQL initialization script
CREATE DATABASE IF NOT EXISTS ml_inference CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE ml_inference;

-- predictions table with optimized indexes
CREATE TABLE IF NOT EXISTS predictions (
    id             BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    request_id     VARCHAR(36)  NOT NULL,
    input_hash     VARCHAR(64)  NOT NULL,
    input_data     JSON         NOT NULL,
    predicted_class VARCHAR(100) NOT NULL,
    confidence     FLOAT        NOT NULL,
    probabilities  JSON         NOT NULL,
    model_version  VARCHAR(50)  NOT NULL,
    inference_ms   FLOAT        NOT NULL,
    cache_hit      TINYINT(1)   NOT NULL DEFAULT 0,
    created_at     DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY uq_request_id (request_id),
    -- Cache lookup: find existing results for same input + model
    INDEX ix_hash_model (input_hash, model_version),
    -- Dashboard / monitoring time-range queries
    INDEX ix_model_created (model_version, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- model_metadata table
CREATE TABLE IF NOT EXISTS model_metadata (
    id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    version       VARCHAR(50)  NOT NULL,
    algorithm     VARCHAR(100) NOT NULL,
    feature_names JSON         NOT NULL,
    class_names   JSON         NOT NULL,
    accuracy      FLOAT,
    description   TEXT,
    is_active     TINYINT(1)   NOT NULL DEFAULT 1,
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY uq_version (version),
    INDEX ix_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Seed the demo model metadata
INSERT IGNORE INTO model_metadata (version, algorithm, feature_names, class_names, accuracy, description, is_active)
VALUES (
    'v1',
    'RandomForestClassifier',
    '["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]',
    '["setosa", "versicolor", "virginica"]',
    0.967,
    'Iris flower species classifier (demo model)',
    1
);
