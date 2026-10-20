"""
Model training pipeline for weather prediction.
Handles data loading, preprocessing, model training, and saving.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from services.data_processor import (
    clean_weather_data,
    handle_missing_values,
    engineer_features,
    prepare_ml_features,
)


def load_training_data():
    """
    Load historical weather data for model training.
    Priority:
    1. Cached raw data from Open-Meteo (real Tiruchirappalli data)
    2. Processed data
    3. Fetch fresh data from Open-Meteo API
    4. Sample data (last resort)
    """
    # Try raw cached data first (real API data)
    raw_path = os.path.join(Config.RAW_DATA_DIR, "trichy_weather_raw.csv")
    if os.path.exists(raw_path):
        print(f"Loading raw data from {raw_path}")
        df = pd.read_csv(raw_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # Try processed data
    processed_path = os.path.join(Config.PROCESSED_DATA_DIR, "processed_weather.csv")
    if os.path.exists(processed_path):
        print(f"Loading processed data from {processed_path}")
        df = pd.read_csv(processed_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # Try fetching real data from Open-Meteo API
    print("Fetching real weather data from Open-Meteo API for Tiruchirappalli...")
    try:
        from services.weather_api import fetch_bulk_historical
        from datetime import date

        end_date = date.today()
        start_date = end_date - timedelta(days=182)  # ~6 months

        df = fetch_bulk_historical(start_date, end_date)
        if not df.empty:
            # Save as raw data for future use
            os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
            df.to_csv(raw_path, index=False)
            print(f"  Real data saved: {raw_path}")
            return df
    except Exception as e:
        print(f"  API fetch failed: {e}")

    # Fall back to sample data
    print("Falling back to sample data...")
    sample_path = os.path.join(Config.SAMPLE_DATA_DIR, "trichy_weather_6months.csv")
    if os.path.exists(sample_path):
        print(f"Loading sample data from {sample_path}")
        df = pd.read_csv(sample_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # Generate sample data if nothing exists
    print("Generating sample data...")
    generate_sample_data()
    df = pd.read_csv(sample_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def generate_sample_data():
    """
    Generate realistic sample weather data for Tiruchirappalli.
    Based on typical climate patterns for this region.

    Tiruchirappalli climate overview:
    - Hot semi-arid climate (BSh)
    - Summer (Mar-May): Hot and dry, temps 35-42°C
    - Southwest Monsoon (Jun-Sep): Moderate rainfall
    - Northeast Monsoon (Oct-Dec): Primary rainfall season
    - Winter (Jan-Feb): Warm and dry, 25-32°C
    """
    np.random.seed(17)

    end_date = datetime(2026, 8, 20)
    start_date = end_date - timedelta(days=182)  # ~6 months

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    records = []

    for dt in dates:
        month = dt.month

        # Base temperature by month (Tiruchirappalli averages)
        base_temp = {
            1: 26, 2: 28, 3: 31, 4: 34, 5: 35,
            6: 33, 7: 32, 8: 32, 9: 32, 10: 30,
            11: 28, 12: 26,
        }[month]

        # Add daily variation
        temp = base_temp + np.random.normal(0, 3.5)
        temp_min = temp - np.random.uniform(5, 11)
        temp_max = temp + np.random.uniform(3, 9)

        # Humidity by month (higher during monsoon) with more noise
        base_humidity = {
            1: 55, 2: 50, 3: 45, 4: 50, 5: 55,
            6: 65, 7: 68, 8: 70, 9: 72, 10: 78,
            11: 75, 12: 65,
        }[month]
        humidity = base_humidity + np.random.normal(0, 15)
        humidity = np.clip(humidity, 20, 100)

        # Rainfall probability and amount
        # Higher rain probability during monsoon months
        rain_prob_by_month = {
            1: 0.05, 2: 0.05, 3: 0.08, 4: 0.15, 5: 0.20,
            6: 0.30, 7: 0.35, 8: 0.40, 9: 0.45, 10: 0.55,
            11: 0.50, 12: 0.20,
        }
        rain_prob = rain_prob_by_month[month]

        # Rain occurrence with noise: sometimes rain happens with low humidity
        rain_occurs = np.random.random() < rain_prob

        # Inject noise: ~15% chance of "unexpected" rain or dry day
        noise = np.random.random()
        if noise < 0.08 and not rain_occurs:
            rain_occurs = True  # unexpected rain
        elif noise < 0.15 and rain_occurs:
            rain_occurs = False  # expected rain didn't happen

        if rain_occurs:
            # Rainfall amount (heavier during NE monsoon)
            if month in [10, 11]:
                rainfall = np.random.exponential(8)
            elif month in [6, 7, 8, 9]:
                rainfall = np.random.exponential(5)
            else:
                rainfall = np.random.exponential(3)
            rainfall = round(min(rainfall, 80), 1)
            # Rain doesn't always mean very high humidity
            humidity = min(humidity + np.random.uniform(0, 25), 100)
        else:
            rainfall = 0.0
            # Sometimes humidity is high even without rain
            if np.random.random() < 0.15:
                humidity = min(humidity + np.random.uniform(10, 20), 100)

        # Pressure with more variation
        pressure = 1010 + np.random.normal(0, 7)

        # Wind speed
        wind_speed = abs(np.random.normal(8, 5))

        # Cloud cover (correlates with rain but with noise)
        cloud_base = np.random.normal(35, 25)
        if rainfall > 0:
            cloud_base += 25
        cloud_cover = min(100, max(0, cloud_base))

        # Weather condition
        if rainfall >= 7.5:
            condition = "Heavy Rain"
        elif rainfall >= 2.5:
            condition = "Moderate Rain"
        elif rainfall >= 0.1:
            condition = "Light Rain"
        elif humidity > 80:
            condition = "Overcast"
        elif cloud_cover > 60:
            condition = "Partly Cloudy"
        else:
            condition = "Clear"

        records.append({
            "date": dt.date().isoformat(),
            "location": "Tiruchirappalli",
            "temperature": round(temp, 1),
            "min_temperature": round(temp_min, 1),
            "max_temperature": round(temp_max, 1),
            "humidity": round(humidity, 1),
            "rainfall": rainfall,
            "pressure": round(pressure, 1),
            "wind_speed": round(wind_speed, 1),
            "cloud_cover": round(cloud_cover, 1),
            "weather_condition": condition,
        })

    df = pd.DataFrame(records)

    # Save to sample directory
    os.makedirs(Config.SAMPLE_DATA_DIR, exist_ok=True)
    output_path = os.path.join(Config.SAMPLE_DATA_DIR, "trichy_weather_6months.csv")
    df.to_csv(output_path, index=False)
    print(f"Sample data generated: {output_path}")
    print(f"  Total days: {len(df)}")
    print(f"  Rain days: {(df['rainfall'] >= Config.RAIN_THRESHOLD).sum()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def train_model():
    """
    Complete ML training pipeline:
    1. Load data
    2. Clean data
    3. Handle missing values
    4. Engineer features
    5. Prepare features
    6. Split data (chronological)
    7. Train models
    8. Evaluate
    9. Save model
    """
    print("=" * 60)
    print("  WEATHER PREDICTION MODEL TRAINING")
    print("  Location: Tiruchirappalli, Tamil Nadu, India")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/9] Loading training data...")
    df = load_training_data()
    print(f"  Loaded {len(df)} records")

    # Step 2: Clean data
    print("\n[2/9] Cleaning data...")
    df = clean_weather_data(df)
    print(f"  After cleaning: {len(df)} records")

    # Step 3: Handle missing values
    print("\n[3/9] Handling missing values...")
    missing_before = df.isnull().sum().sum()
    df = handle_missing_values(df)
    missing_after = df.isnull().sum().sum()
    print(f"  Missing values: {missing_before} -> {missing_after}")

    # Step 4: Engineer features
    print("\n[4/9] Engineering features...")
    df = engineer_features(df)
    print(f"  Features created: {len(df.columns)} columns")

    # Step 5: Prepare ML features
    print("\n[5/9] Preparing feature matrix...")
    X, y, feature_names = prepare_ml_features(df)
    print(f"  Samples: {len(X)}, Features: {len(feature_names)}")
    print(f"  Feature names: {feature_names}")

    if len(X) < 30:
        print("  ERROR: Insufficient data for training. Need at least 30 samples.")
        return None

    # Step 6: Chronological train/test split (80/20)
    print("\n[6/9] Splitting data (chronological 80/20)...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing:  {len(X_test)} samples")

    # Step 7: Scale features
    print("\n[7/9] Scaling features...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 8: Train models
    print("\n[8/9] Training models...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        ),
    }

    # Try XGBoost
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        )
    except ImportError:
        print("  XGBoost not available, skipping.")

    best_model = None
    best_accuracy = 0
    best_name = ""

    for name, model in models.items():
        print(f"\n  Training {name}...")

        if name == "XGBoost":
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        # Quick accuracy check
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        print(f"    Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_name = name

    print(f"\n  Best model: {best_name} (Accuracy: {best_accuracy:.4f})")

    # Step 9: Save model and scaler
    print("\n[9/9] Saving model...")
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    model_data = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": feature_names,
        "scaler": scaler,
        "trained_at": datetime.now().isoformat(),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }

    joblib.dump(model_data, Config.MODEL_PATH)
    joblib.dump(scaler, Config.SCALER_PATH)
    print(f"  Model saved: {Config.MODEL_PATH}")
    print(f"  Scaler saved: {Config.SCALER_PATH}")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION ON TEST SET")
    print("=" * 60)

    if best_name == "XGBoost":
        y_pred_final = best_model.predict(X_test)
    else:
        X_test_final = scaler.transform(X_test)
        y_pred_final = best_model.predict(X_test_final)

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report,
    )

    accuracy = accuracy_score(y_test, y_pred_final)
    precision = precision_score(y_test, y_pred_final, zero_division=0)
    recall = recall_score(y_test, y_pred_final, zero_division=0)
    f1 = f1_score(y_test, y_pred_final, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_final)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    print(f"\n  Confusion Matrix:")
    print(f"             PREDICTED")
    print(f"             Rain    No Rain")
    print(f"  ACTUAL Rain  {tp:4d}     {fn:4d}")
    print(f"  ACTUAL NoRain {fp:4d}     {tn:4d}")

    print(f"\n  Accuracy:    {accuracy:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1-Score:    {f1:.4f}")

    total_days = len(y_test)
    predicted_rain = int(y_pred_final.sum())
    actual_rain = int(y_test.sum())

    print(f"\n  Total Days:        {total_days}")
    print(f"  Predicted Rain:    {predicted_rain}")
    print(f"  Actual Rain:       {actual_rain}")
    print(f"  False Positives:   {fp}")
    print(f"  False Negatives:   {fn}")
    print(f"  Rain prediction %: {(predicted_rain/total_days*100):.1f}%")
    print(f"  Actual rain %:     {(actual_rain/total_days*100):.1f}%")

    # Save metrics to JSON
    metrics = {
        "model_name": best_name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "total_days": total_days,
        "predicted_rain_days": predicted_rain,
        "actual_rain_days": actual_rain,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_names": feature_names,
        "trained_at": datetime.now().isoformat(),
    }

    metrics_path = os.path.join(Config.PROCESSED_DATA_DIR, "model_metrics.json")
    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved: {metrics_path}")

    # Save test predictions for analysis page
    test_predictions_path = os.path.join(Config.PROCESSED_DATA_DIR, "test_predictions.json")
    test_df = pd.DataFrame({
        "date": df.iloc[split_idx:split_idx + len(X_test)]["date"].values,
        "actual_rain": y_test.values.tolist(),
        "predicted_rain": y_pred_final.tolist(),
    })
    test_df.to_json(test_predictions_path, orient="records", date_format="iso", indent=2)
    print(f"  Test predictions saved: {test_predictions_path}")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    train_model()
