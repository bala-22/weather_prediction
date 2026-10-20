"""
Prediction module for weather forecasting.
Uses trained ML model to predict weather for given areas.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from services.data_processor import (
    clean_weather_data,
    handle_missing_values,
    engineer_features,
    prepare_ml_features,
    classify_weather_condition,
)


def load_model():
    """Load the trained model from disk."""
    if not os.path.exists(Config.MODEL_PATH):
        return None

    model_data = joblib.load(Config.MODEL_PATH)
    return model_data


def predict_weather_for_area(area_name, forecast_data=None):
    """
    Generate 7-day weather prediction for a specific Tiruchirappalli sub-area.

    Args:
        area_name: Name of the sub-area (must be in Config.SUB_AREAS)
        forecast_data: Optional list of dicts with forecast data from API

    Returns:
        List of 7 daily prediction dicts
    """
    if area_name not in Config.SUB_AREAS:
        return None

    model_data = load_model()
    area_coords = Config.SUB_AREAS[area_name]

    # Generate date range for next 7 days (skip today)
    today = datetime.now().date()
    dates = [today + timedelta(days=i + 1) for i in range(7)]

    predictions = []

    if forecast_data and len(forecast_data) >= 7:
        # Use API forecast data as input features for ML model
        for i, day_data in enumerate(forecast_data[:7]):
            ml_prediction = _run_ml_prediction(model_data, day_data, area_coords, dates[i])
            predictions.append(ml_prediction)
    else:
        # Generate predictions using seasonal averages and ML model
        for i, date in enumerate(dates):
            day_data = _estimate_base_weather(date, area_coords)
            ml_prediction = _run_ml_prediction(model_data, day_data, area_coords, date)
            predictions.append(ml_prediction)

    return predictions


def _estimate_base_weather(date, coords):
    """
    Estimate base weather for a future date using historical climate averages.
    Used when no API forecast data is available.
    """
    month = date.month
    day_of_year = date.timetuple().tm_yday

    # Seasonal temperature pattern for Tiruchirappalli
    base_temp = {
        1: 26, 2: 28, 3: 31, 4: 34, 5: 35,
        6: 33, 7: 32, 8: 32, 9: 32, 10: 30,
        11: 28, 12: 26,
    }[month]

    np.random.seed(int(date.strftime("%Y%m%d")))
    temp = base_temp + np.random.normal(0, 1.5)
    temp_min = temp - np.random.uniform(5, 9)
    temp_max = temp + np.random.uniform(3, 7)

    base_humidity = {
        1: 55, 2: 50, 3: 45, 4: 50, 5: 55,
        6: 65, 7: 68, 8: 70, 9: 72, 10: 78,
        11: 75, 12: 65,
    }[month]
    humidity = base_humidity + np.random.normal(0, 8)
    humidity = np.clip(humidity, 25, 98)

    rain_prob_by_month = {
        1: 0.05, 2: 0.05, 3: 0.08, 4: 0.15, 5: 0.20,
        6: 0.30, 7: 0.35, 8: 0.40, 9: 0.45, 10: 0.55,
        11: 0.50, 12: 0.20,
    }
    rain_prob = rain_prob_by_month[month]

    is_rain = np.random.random() < rain_prob
    rainfall = round(np.random.exponential(5), 1) if is_rain else 0.0

    return {
        "temperature": round(temp, 1),
        "min_temperature": round(temp_min, 1),
        "max_temperature": round(temp_max, 1),
        "humidity": round(humidity, 1),
        "rainfall": rainfall,
        "pressure": round(1010 + np.random.normal(0, 3), 1),
        "wind_speed": round(abs(np.random.normal(8, 3)), 1),
        "cloud_cover": round(min(100, max(0, np.random.normal(40 if is_rain else 25, 15))), 1),
    }


def _run_ml_prediction(model_data, day_data, coords, pred_date=None):
    """
    Run ML model prediction for a single day's weather data.
    Combines ML model output with the weather data.
    """
    if model_data is None:
        # Fallback: use simple threshold-based prediction
        rain_prob = min(1.0, day_data["humidity"] / 100 * 0.8 +
                        day_data["cloud_cover"] / 100 * 0.2)
        predicted_rain = day_data["rainfall"] >= Config.RAIN_THRESHOLD

        return {
            "temperature": day_data["temperature"],
            "min_temperature": day_data["min_temperature"],
            "max_temperature": day_data["max_temperature"],
            "humidity": day_data["humidity"],
            "rainfall": day_data["rainfall"],
            "wind_speed": day_data["wind_speed"],
            "cloud_cover": day_data["cloud_cover"],
            "rain_probability": round(rain_prob * 100, 1),
            "predicted_rain": predicted_rain,
            "weather_condition": classify_weather_condition(day_data),
        }

    model = model_data["model"]
    feature_names = model_data["feature_names"]
    model_name = model_data["model_name"]
    scaler = model_data.get("scaler")

    # Reasonable defaults for features not in forecast data
    DEFAULTS = {
        "pressure": 1010,
        "wind_speed": 10,
        "cloud_cover": 50,
        "prev_temperature": day_data.get("temperature", 30),
        "prev_humidity": day_data.get("humidity", 60),
        "prev_rainfall": 0,
        "prev_wind_speed": day_data.get("wind_speed", 10),
        "temp_3day_avg": day_data.get("temperature", 30),
        "humidity_3day_avg": day_data.get("humidity", 60),
        "temp_7day_avg": day_data.get("temperature", 30),
        "humidity_7day_avg": day_data.get("humidity", 60),
    }

    # Build feature vector matching training features
    feature_values = {}
    for fname in feature_names:
        if fname in day_data and day_data[fname] is not None:
            feature_values[fname] = day_data[fname]
        elif fname in DEFAULTS:
            feature_values[fname] = DEFAULTS[fname]
        elif fname == "day_of_year":
            feature_values[fname] = pred_date.timetuple().tm_yday if pred_date else datetime.now().timetuple().tm_yday
        elif fname == "month":
            feature_values[fname] = pred_date.month if pred_date else datetime.now().month
        elif fname == "day_of_week":
            feature_values[fname] = pred_date.weekday() if pred_date else datetime.now().weekday()
        elif fname == "season":
            month = pred_date.month if pred_date else datetime.now().month
            if month in [1, 2]:
                feature_values[fname] = 0
            elif month in [3, 4, 5]:
                feature_values[fname] = 1
            elif month in [6, 7, 8, 9]:
                feature_values[fname] = 2
            else:
                feature_values[fname] = 3
        elif fname == "temp_range":
            feature_values[fname] = day_data["max_temperature"] - day_data["min_temperature"]
        elif fname in ["prev_temperature", "prev_humidity", "prev_rainfall", "prev_wind_speed"]:
            feature_values[fname] = day_data.get(fname.replace("prev_", ""), 0)
        elif fname in ["temp_3day_avg", "humidity_3day_avg", "rainfall_3day_sum",
                        "temp_7day_avg", "humidity_7day_avg"]:
            feature_values[fname] = day_data.get(fname.replace("_3day_avg", "")
                                                  .replace("_7day_avg", "")
                                                  .replace("_sum", ""), 0)
        else:
            feature_values[fname] = 0

    X = pd.DataFrame([feature_values], columns=feature_names)

    # Scale if needed
    if model_name in ["RandomForest", "LogisticRegression"] and scaler is not None:
        X_scaled = scaler.transform(X)
        rain_prediction = model.predict(X_scaled)[0]
        rain_probability = model.predict_proba(X_scaled)[0][1] * 100
    else:
        rain_prediction = model.predict(X)[0]
        rain_probability = model.predict_proba(X)[0][1] * 100

    predicted_rain = bool(rain_prediction)
    condition = classify_weather_condition(day_data)

    if predicted_rain and condition in ["Clear", "Fair"]:
        condition = "Rain Expected"

    return {
        "temperature": day_data["temperature"],
        "min_temperature": day_data["min_temperature"],
        "max_temperature": day_data["max_temperature"],
        "humidity": day_data["humidity"],
        "rainfall": day_data["rainfall"],
        "wind_speed": day_data["wind_speed"],
        "cloud_cover": day_data["cloud_cover"],
        "rain_probability": round(rain_probability, 1),
        "predicted_rain": predicted_rain,
        "weather_condition": condition,
    }


def predict_historical_period(start_date, end_date, historical_df=None):
    """
    Generate ML predictions for a historical period.
    Used for the 6-month analysis to compare predictions vs actuals.

    Args:
        start_date: Start date (datetime.date)
        end_date: End date (datetime.date)
        historical_df: DataFrame with actual historical weather data

    Returns:
        DataFrame with actual and predicted values
    """
    model_data = load_model()

    if historical_df is None or historical_df.empty:
        return pd.DataFrame()

    # Engineer features on the historical data
    df = engineer_features(historical_df.copy())
    X, y_true, feature_names = prepare_ml_features(df)

    if X.empty or model_data is None:
        return pd.DataFrame()

    model = model_data["model"]
    model_name = model_data["model_name"]
    scaler = model_data.get("scaler")

    if model_name in ["RandomForest", "LogisticRegression"] and scaler is not None:
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
    else:
        y_pred = model.predict(X)

    # Create results dataframe
    results = df.iloc[X.index].copy()
    results["predicted_rain"] = y_pred.tolist()
    if y_true is not None:
        results["actual_rain"] = y_true.values

    return results
