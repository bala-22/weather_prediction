"""
Data processor service for cleaning, transforming, and preparing weather data.
Handles missing values, feature engineering, and data validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import Config


def clean_weather_data(df):
    """
    Clean raw weather data:
    - Remove duplicates
    - Handle missing values
    - Validate ranges
    """
    if df.empty:
        return df

    df = df.copy()

    # Remove duplicates based on location and date
    df = df.drop_duplicates(subset=["location", "date"], keep="last")

    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Validate and clip temperature (Tiruchirappalli realistic range)
    df["temperature"] = df["temperature"].clip(-10, 55)
    df["min_temperature"] = df["min_temperature"].clip(-10, 55)
    df["max_temperature"] = df["max_temperature"].clip(-10, 55)

    # Validate humidity (0-100%)
    df["humidity"] = df["humidity"].clip(0, 100)

    # Validate rainfall (non-negative)
    df["rainfall"] = df["rainfall"].clip(lower=0)

    # Validate pressure (reasonable atmospheric range)
    df["pressure"] = df["pressure"].clip(900, 1100)

    # Validate wind speed (non-negative)
    df["wind_speed"] = df["wind_speed"].clip(lower=0)

    # Validate cloud cover (0-100%)
    df["cloud_cover"] = df["cloud_cover"].clip(0, 100)

    return df


def handle_missing_values(df):
    """
    Fill missing values using forward-fill then backward-fill strategy.
    Suitable for time-series weather data.
    """
    if df.empty:
        return df

    df = df.copy()
    numeric_cols = ["temperature", "min_temperature", "max_temperature",
                    "humidity", "rainfall", "pressure", "wind_speed",
                    "cloud_cover"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear")
            df[col] = df[col].ffill()
            df[col] = df[col].bfill()

    if "weather_condition" in df.columns:
        df["weather_condition"] = df["weather_condition"].fillna("Unknown")

    return df


def engineer_features(df):
    """
    Create additional features for the ML model from existing weather data.
    Includes temporal features and lag features.
    """
    if df.empty:
        return df

    df = df.copy()

    # Temporal features
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["year"] = df["date"].dt.year

    # Season encoding (Tiruchirappalli climate: hot/dry, monsoon, cool)
    # Jan-Feb: cool dry, Mar-May: hot dry, Jun-Sep: SW monsoon, Oct-Dec: NE monsoon
    df["season"] = df["month"].map({
        1: 0, 2: 0,  # Cool dry
        3: 1, 4: 1, 5: 1,  # Hot dry
        6: 2, 7: 2, 8: 2, 9: 2,  # SW Monsoon
        10: 3, 11: 3, 12: 3  # NE Monsoon
    })

    # Temperature range (diurnal variation)
    df["temp_range"] = df["max_temperature"] - df["min_temperature"]

    # Lag features (previous day values) for temporal patterns
    df["prev_temperature"] = df["temperature"].shift(1)
    df["prev_humidity"] = df["humidity"].shift(1)
    df["prev_rainfall"] = df["rainfall"].shift(1)
    df["prev_wind_speed"] = df["wind_speed"].shift(1)

    # Rolling averages (3-day and 7-day)
    df["temp_3day_avg"] = df["temperature"].rolling(window=3, min_periods=1).mean()
    df["humidity_3day_avg"] = df["humidity"].rolling(window=3, min_periods=1).mean()
    df["rainfall_3day_sum"] = df["rainfall"].rolling(window=3, min_periods=1).sum()
    df["temp_7day_avg"] = df["temperature"].rolling(window=7, min_periods=1).mean()
    df["humidity_7day_avg"] = df["humidity"].rolling(window=7, min_periods=1).mean()

    # Rain label based on threshold
    df["is_rain"] = (df["rainfall"] >= Config.RAIN_THRESHOLD).astype(int)

    return df


def classify_weather_condition(row):
    """
    Classify weather condition based on temperature, humidity, and rainfall.
    Returns a human-readable weather condition string.
    """
    rainfall = row.get("rainfall", 0)
    humidity = row.get("humidity", 50)
    temperature = row.get("temperature", 30)

    if rainfall >= 7.5:
        return "Heavy Rain"
    elif rainfall >= 2.5:
        return "Moderate Rain"
    elif rainfall >= Config.RAIN_THRESHOLD:
        return "Light Rain"
    elif humidity >= 80:
        return "Overcast"
    elif humidity >= 60:
        return "Partly Cloudy"
    elif temperature > 38:
        return "Hot and Clear"
    elif temperature > 30:
        return "Clear"
    else:
        return "Fair"


def prepare_ml_features(df):
    """
    Prepare feature matrix and target vector for ML model training.
    Returns X (features), y (target), and the list of feature names.

    IMPORTANT: rainfall is NOT included as a feature because the target
    (is_rain) is derived from rainfall. Using it would be data leakage.
    Only features available BEFORE the rain event are used.
    """
    feature_columns = [
        "temperature", "min_temperature", "max_temperature",
        "humidity", "pressure", "wind_speed", "cloud_cover",
        "day_of_year", "month", "day_of_week", "season", "temp_range",
        "prev_temperature", "prev_humidity", "prev_rainfall",
        "prev_wind_speed", "temp_3day_avg", "humidity_3day_avg",
        "temp_7day_avg", "humidity_7day_avg",
    ]

    # Only use columns that exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]

    X = df[available_features].copy()
    y = df["is_rain"].copy() if "is_rain" in df.columns else None

    # Drop rows where features or target are NaN
    if y is not None:
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
    else:
        X = X.dropna()

    return X, y, available_features


def get_monthly_summary(df):
    """
    Generate monthly weather summary statistics.
    Returns a DataFrame with monthly aggregates.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")

    monthly = df.groupby("month").agg(
        avg_temperature=("temperature", "mean"),
        avg_min_temperature=("min_temperature", "mean"),
        avg_max_temperature=("max_temperature", "mean"),
        total_rainfall=("rainfall", "sum"),
        avg_humidity=("humidity", "mean"),
        rain_days=("rainfall", lambda x: (x >= Config.RAIN_THRESHOLD).sum()),
        total_days=("rainfall", "count"),
        avg_wind_speed=("wind_speed", "mean"),
    ).round(2)

    monthly["rain_percentage"] = (
        (monthly["rain_days"] / monthly["total_days"] * 100).round(1)
    )

    return monthly.reset_index()
