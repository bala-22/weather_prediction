"""Generate analysis results from real historical data."""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from config import Config
from ml.predict import predict_historical_period
from ml.evaluate import calculate_metrics, get_monthly_breakdown

# Load real data
df = pd.read_csv(os.path.join(Config.RAW_DATA_DIR, "trichy_weather_raw.csv"))
df["date"] = pd.to_datetime(df["date"])

print("Loaded {} days of real Tiruchirappalli weather data".format(len(df)))
print("Period: {} to {}".format(df["date"].min().date(), df["date"].max().date()))
print("Rain days (>=0.1mm): {}".format(
    (df["rainfall"] >= Config.RAIN_THRESHOLD).sum()))

start = df["date"].min().date()
end = df["date"].max().date()

results = predict_historical_period(start, end, df)
if not results.empty:
    actual = results["actual_rain"].values
    predicted = results["predicted_rain"].values
    metrics = calculate_metrics(actual, predicted)
    metrics["model_name"] = "RandomForest"

    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
    with open(os.path.join(Config.PROCESSED_DATA_DIR, "analysis_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    monthly = get_monthly_breakdown(results)
    with open(os.path.join(Config.PROCESSED_DATA_DIR, "monthly_breakdown.json"), "w") as f:
        json.dump(monthly, f, indent=2)

    print("\nAnalysis results saved for {} records".format(len(results)))
    print("Accuracy: {}".format(metrics["accuracy"]))
    print("Precision: {}".format(metrics["precision"]))
    print("Recall: {}".format(metrics["recall"]))
    print("F1-Score: {}".format(metrics["f1_score"]))
    print("TP: {}, TN: {}".format(metrics["true_positives"], metrics["true_negatives"]))
    print("FP: {}, FN: {}".format(metrics["false_positives"], metrics["false_negatives"]))
    print("Total days: {}".format(metrics["total_days"]))
    print("Predicted rain days: {}".format(metrics["predicted_rain_days"]))
    print("Actual rain days: {}".format(metrics["actual_rain_days"]))
else:
    print("No results generated")
