-- Weather Prediction System - Database Schema
-- PostgreSQL Database Setup

CREATE DATABASE weather_prediction;

-- Table: weather_data (historical weather observations)
CREATE TABLE IF NOT EXISTS weather_data (
    id SERIAL PRIMARY KEY,
    location VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    temperature REAL,
    min_temperature REAL,
    max_temperature REAL,
    humidity REAL,
    rainfall REAL DEFAULT 0,
    pressure REAL,
    wind_speed REAL,
    wind_direction REAL,
    cloud_cover REAL DEFAULT 0,
    weather_condition VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_weather_location_date ON weather_data (location, date);
CREATE INDEX IF NOT EXISTS idx_weather_date ON weather_data (date);

-- Table: predictions (ML model predictions)
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    location VARCHAR(100) NOT NULL,
    prediction_date DATE NOT NULL,
    predicted_temperature REAL,
    predicted_min_temperature REAL,
    predicted_max_temperature REAL,
    predicted_humidity REAL,
    predicted_rainfall REAL,
    rain_probability REAL,
    predicted_condition VARCHAR(100),
    predicted_rain BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_location_date ON predictions (location, prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions (prediction_date);

-- Table: model_metrics (evaluation results)
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    accuracy REAL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    total_days INTEGER,
    predicted_rain_days INTEGER,
    actual_rain_days INTEGER,
    true_positives INTEGER,
    true_negatives INTEGER,
    false_positives INTEGER,
    false_negatives INTEGER,
    training_samples INTEGER,
    test_samples INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_model_metrics_name ON model_metrics (model_name);
