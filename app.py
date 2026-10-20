"""
Weather Prediction System - Flask Application
Main application file with all routes and API endpoints.
Focus: Tiruchirappalli
"""

import os
import sys
import json
import webbrowser
import threading
import pandas as pd
from flask import Flask, render_template, jsonify, request

from config import Config

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY


@app.after_request
def add_no_cache(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_csv(path):
    """Load a CSV file into a DataFrame, return None if not found."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None


def _get_historical_df():
    """Load the best available historical weather DataFrame."""
    # Check for real API-cached data first
    import glob
    raw_files = glob.glob(os.path.join(Config.RAW_DATA_DIR, "trichy_*.csv"))
    if raw_files:
        # Use the most recent cache file
        latest = max(raw_files, key=os.path.getmtime)
        df = _load_csv(latest)
        if df is not None:
            return df

    for path in [
        os.path.join(Config.RAW_DATA_DIR, "trichy_weather_raw.csv"),
        os.path.join(Config.PROCESSED_DATA_DIR, "processed_weather.csv"),
        os.path.join(Config.SAMPLE_DATA_DIR, "trichy_weather_6months.csv"),
    ]:
        df = _load_csv(path)
        if df is not None:
            return df
    return None


def _load_analysis_results():
    """Load cached analysis results or compute them."""
    metrics_path = os.path.join(Config.PROCESSED_DATA_DIR, "analysis_results.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)

    model_metrics_path = os.path.join(Config.PROCESSED_DATA_DIR, "model_metrics.json")
    if os.path.exists(model_metrics_path):
        with open(model_metrics_path, "r") as f:
            return json.load(f)

    return None


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Home page."""
    current_weather = None
    # Try Open-Meteo (free) first, then OpenWeatherMap
    from services.weather_api import fetch_current_weather
    current_weather = fetch_current_weather()

    return render_template(
        "index.html",
        current_weather=current_weather,
        data_mode=Config.DATA_MODE,
    )


@app.route("/analysis")
def analysis():
    """6-Month Analysis page."""
    return render_template("analysis.html")


@app.route("/prediction")
def prediction():
    """1-Week Prediction page."""
    areas = list(Config.SUB_AREAS.keys())
    return render_template("prediction.html", areas=areas)


@app.route("/about")
def about():
    """About page."""
    return render_template("about.html")


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/api/current-weather")
def api_current_weather():
    """Get current weather for Tiruchirappalli."""
    from services.weather_api import fetch_current_weather
    weather = fetch_current_weather()
    if weather:
        return jsonify({"success": True, "data": weather})

    # Fallback if API fails
    return jsonify({
        "success": True,
        "data": {
            "temperature": 32.5,
            "min_temperature": 27.0,
            "max_temperature": 37.0,
            "humidity": 65,
            "pressure": 1010.2,
            "wind_speed": 12.3,
            "cloud_cover": 45,
            "weather_condition": "Partly Cloudy",
            "rainfall": 0,
            "location": "Tiruchirappalli",
            "note": "Fallback data (API unavailable)",
        },
    })


@app.route("/api/historical-analysis")
def api_historical_analysis():
    """
    Return the 6-month historical analysis data including daily records.
    Used by the analysis page charts.
    """
    df = _get_historical_df()
    if df is None:
        return jsonify({"success": False, "error": "No historical data available"}), 404

    # Load analysis results for metrics
    metrics = _load_analysis_results()

    # Load test set metrics from model_metrics.json (unseen test set evaluation)
    test_metrics = None
    model_metrics_path = os.path.join(Config.PROCESSED_DATA_DIR, "model_metrics.json")
    if os.path.exists(model_metrics_path):
        try:
            with open(model_metrics_path, "r") as f:
                test_metrics = json.load(f)
                if "confusion_matrix" not in test_metrics:
                    test_metrics["confusion_matrix"] = {
                        "tp": test_metrics.get("true_positives", 0),
                        "tn": test_metrics.get("true_negatives", 0),
                        "fp": test_metrics.get("false_positives", 0),
                        "fn": test_metrics.get("false_negatives", 0),
                    }
                td = test_metrics.get("total_days", 37)
                pred_r = test_metrics.get("predicted_rain_days", 0)
                act_r = test_metrics.get("actual_rain_days", 0)
                test_metrics["rain_prediction_percentage"] = round(pred_r / td * 100, 1) if td else 0
                test_metrics["actual_rain_percentage"] = round(act_r / td * 100, 1) if td else 0
        except Exception:
            pass

    # Load model comparison
    model_comparison = None
    comp_path = os.path.join(Config.PROCESSED_DATA_DIR, "model_comparison.json")
    if os.path.exists(comp_path):
        try:
            with open(comp_path, "r") as f:
                model_comparison = json.load(f)
        except Exception:
            pass

    # Extract feature importances from trained model
    feature_importances = []
    try:
        import joblib
        model_data = joblib.load(Config.MODEL_PATH)
        feat_names = model_data.get("feature_names", [])
        m = model_data.get("model")
        if hasattr(m, "feature_importances_"):
            raw_imp = sorted(zip(feat_names, m.feature_importances_), key=lambda x: x[1], reverse=True)
            feature_importances = [{"feature": f, "importance": round(float(val) * 100, 2)} for f, val in raw_imp]
    except Exception:
        pass

    # Get ML predictions per day
    predictions_by_date = {}
    try:
        import joblib
        from services.data_processor import engineer_features, prepare_ml_features
        df_feat = engineer_features(df.copy())
        X, y_true, feature_names = prepare_ml_features(df_feat)
        if not X.empty:
            model_data = joblib.load(Config.MODEL_PATH)
            model = model_data["model"]
            scaler = model_data["scaler"]
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            for i, idx in enumerate(X.index):
                d = df_feat.iloc[idx]["date"]
                date_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                predictions_by_date[date_str] = bool(y_pred[i])
    except Exception:
        pass

    # Build daily records
    daily_records = []
    for _, row in df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
        actual_rain = bool(row.get("rainfall", 0) >= Config.RAIN_THRESHOLD)
        has_prediction = date_str in predictions_by_date
        predicted_rain = predictions_by_date.get(date_str, actual_rain)
        daily_records.append({
            "date": date_str,
            "temperature": round(float(row.get("temperature", 0)), 1),
            "min_temperature": round(float(row.get("min_temperature", 0)), 1),
            "max_temperature": round(float(row.get("max_temperature", 0)), 1),
            "humidity": round(float(row.get("humidity", 0)), 1),
            "rainfall": round(float(row.get("rainfall", 0)), 1),
            "wind_speed": round(float(row.get("wind_speed", 0)), 1),
            "cloud_cover": round(float(row.get("cloud_cover", 0)), 1),
            "weather_condition": row.get("weather_condition", "Unknown"),
            "is_rain": actual_rain,
            "predicted_rain": predicted_rain,
            "has_prediction": has_prediction,
        })

    # Climate season metadata for Tiruchirappalli
    climate_meta = {
        "2026-02": {"season": "Late Winter", "desc": "Dry & fair weather, low humidity (0 rain days)"},
        "2026-03": {"season": "Pre-Monsoon Summer", "desc": "Rising heat, sporadic convective storms"},
        "2026-04": {"season": "Peak Summer Heat", "desc": "Extreme daytime heat (36-41°C), low rain"},
        "2026-05": {"season": "Pre-Monsoon Showers", "desc": "Convective thunderstorms, 9 rain days"},
        "2026-06": {"season": "SW Monsoon Onset", "desc": "Westerly winds, moderate showers, 10 rain days"},
        "2026-07": {"season": "Active SW Monsoon", "desc": "Frequent cloud cover & rain spells, 12 rain days"},
        "2026-08": {"season": "Mid-Monsoon Season", "desc": "Sustained monsoon rain, 13 rain days"},
    }

    # Monthly summary with predicted rain days and prediction accuracy
    monthly = []
    if not df.empty:
        df["month_period"] = df["date"].dt.to_period("M")
        for month, group in df.groupby("month_period"):
            scored = group[
                group["date"].dt.strftime("%Y-%m-%d").isin(predictions_by_date)
            ]
            rain_days = int((group["rainfall"] >= Config.RAIN_THRESHOLD).sum())
            pred_rain_days = int(
                sum(
                    1
                    for _, r in scored.iterrows()
                    if predictions_by_date.get(r["date"].strftime("%Y-%m-%d"))
                )
            )
            matches = int(
                sum(
                    1
                    for _, r in scored.iterrows()
                    if predictions_by_date.get(r["date"].strftime("%Y-%m-%d"))
                    == bool(r.get("rainfall", 0) >= Config.RAIN_THRESHOLD)
                )
            )

            m_str = str(month)
            meta = climate_meta.get(m_str, {"season": "Seasonal Climate", "desc": "Observed weather patterns"})
            match_acc = round((matches / len(scored)) * 100, 1) if len(scored) > 0 else 0.0

            monthly.append({
                "month": m_str,
                "season": meta["season"],
                "description": meta["desc"],
                "avg_temperature": round(float(group["temperature"].mean()), 1),
                "total_rainfall": round(float(group["rainfall"].sum()), 1),
                "avg_humidity": round(float(group["humidity"].mean()), 1),
                "rain_days": rain_days,
                "predicted_rain_days": pred_rain_days,
                "matches": matches,
                "accuracy": match_acc,
                "total_days": len(scored),
            })

    return jsonify({
        "success": True,
        "data": {
            "daily": daily_records,
            "monthly": monthly,
            "metrics": metrics,
            "test_metrics": test_metrics,
            "model_comparison": model_comparison,
            "feature_importances": feature_importances,
            "period": {
                "start": daily_records[0]["date"] if daily_records else "",
                "end": daily_records[-1]["date"] if daily_records else "",
                "total_days": metrics.get("total_days", len(daily_records)),
            },
        },
    })


@app.route("/api/analysis-metrics")
def api_analysis_metrics():
    """Return model evaluation metrics."""
    metrics = _load_analysis_results()
    if metrics:
        return jsonify({"success": True, "data": metrics})
    return jsonify({"success": False, "error": "No metrics available. Train the model first."}), 404


@app.route("/api/model-comparison")
def api_model_comparison():
    """Return model comparison results (all algorithms tested)."""
    path = os.path.join(Config.PROCESSED_DATA_DIR, "model_comparison.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        return jsonify({"success": True, "data": data})
    return jsonify({"success": False, "error": "No comparison data available."}), 404


@app.route("/api/failure-analysis")
def api_failure_analysis():
    """Return failed prediction days with reasons."""
    path = os.path.join(Config.PROCESSED_DATA_DIR, "failure_analysis.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            failures = json.load(f)
        return jsonify({"success": True, "data": failures})
    return jsonify({"success": False, "error": "No failure analysis available"}), 404


@app.route("/api/prediction/<area>")
def api_prediction(area):
    """Get 7-day prediction for a specific Tiruchirappalli sub-area."""
    if area not in Config.SUB_AREAS:
        return jsonify({"success": False, "error": f"Invalid area: {area}"}), 400

    from ml.predict import predict_weather_for_area

    # Try Open-Meteo forecast first (free, no key needed)
    from services.weather_api import fetch_7day_forecast
    coords = Config.SUB_AREAS[area]
    forecast_data = fetch_7day_forecast(coords["lat"], coords["lon"])

    predictions = predict_weather_for_area(area, forecast_data)

    if predictions is None:
        return jsonify({"success": False, "error": "Prediction failed"}), 500

    # Add date to each prediction (skip today, start from tomorrow)
    from datetime import datetime, timedelta
    today = datetime.now().date()
    for i, pred in enumerate(predictions):
        pred["date"] = (today + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        pred["area"] = area

    # Calculate summary stats
    temps = [p["temperature"] for p in predictions]
    humidities = [p["humidity"] for p in predictions]
    rainfalls = [p["rainfall"] for p in predictions]
    wind_speeds = [p["wind_speed"] for p in predictions]
    rain_probs = [p["rain_probability"] for p in predictions]

    summary = {
        "min_temperature": round(min(p["min_temperature"] for p in predictions), 1),
        "max_temperature": round(max(p["max_temperature"] for p in predictions), 1),
        "avg_temperature": round(sum(temps) / len(temps), 1),
        "avg_humidity": round(sum(humidities) / len(humidities), 1),
        "total_rainfall": round(sum(rainfalls), 1),
        "avg_wind_speed": round(sum(wind_speeds) / len(wind_speeds), 1),
        "avg_rain_probability": round(sum(rain_probs) / len(rain_probs), 1),
        "rain_days_predicted": sum(1 for p in predictions if p["predicted_rain"]),
    }

    return jsonify({
        "success": True,
        "data": {
            "area": area,
            "coordinates": Config.SUB_AREAS[area],
            "predictions": predictions,
            "summary": summary,
        },
    })


@app.route("/api/areas")
def api_areas():
    """Return list of available sub-areas."""
    areas = []
    for name, coords in Config.SUB_AREAS.items():
        areas.append({
            "name": name,
            "latitude": coords["lat"],
            "longitude": coords["lon"],
        })
    return jsonify({"success": True, "data": areas})


@app.route("/api/train-model", methods=["POST"])
def api_train_model():
    """Trigger model training via API."""
    try:
        from ml.train_model import train_model
        metrics = train_model()
        if metrics:
            return jsonify({"success": True, "data": metrics})
        return jsonify({"success": False, "error": "Training failed"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/model-metrics")
def api_model_metrics():
    """Return saved model training metrics."""
    metrics_path = os.path.join(Config.PROCESSED_DATA_DIR, "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
        return jsonify({"success": True, "data": data})
    return jsonify({"success": False, "error": "No model metrics found. Train the model first."}), 404


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template("index.html", error="Page not found"), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return render_template("index.html", error="Internal server error"), 500


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure data directories exist
    os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(Config.SAMPLE_DATA_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("  Weather Prediction Using Machine Learning")
    print("  Location: Tiruchirappalli, Tamil Nadu, India")
    print("=" * 60)
    print(f"  Data Mode: {Config.DATA_MODE}")
    print(f"  API Key: {'Configured' if Config.WEATHER_API_KEY else 'Not configured'}")
    print(f"  Debug: {Config.DEBUG}")
    print("=" * 60)

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=Config.DEBUG, host="0.0.0.0", port=5000)
