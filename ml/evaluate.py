"""
Model evaluation module.
Calculates and formats all evaluation metrics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


def calculate_metrics(actual_rain, predicted_rain):
    """
    Calculate all classification metrics for rain prediction.

    Args:
        actual_rain: array-like of actual rain labels (1=rain, 0=no rain)
        predicted_rain: array-like of predicted rain labels (1=rain, 0=no rain)

    Returns:
        Dictionary with all evaluation metrics
    """
    actual = np.array(actual_rain, dtype=int)
    predicted = np.array(predicted_rain, dtype=int)

    total_days = len(actual)

    # Confusion matrix components
    tp = int(np.sum((predicted == 1) & (actual == 1)))  # True Positive
    tn = int(np.sum((predicted == 0) & (actual == 0)))  # True Negative
    fp = int(np.sum((predicted == 1) & (actual == 0)))  # False Positive
    fn = int(np.sum((predicted == 0) & (actual == 1)))  # False Negative

    # Derived counts
    predicted_rain_days = int(np.sum(predicted == 1))
    actual_rain_days = int(np.sum(actual == 1))

    # Metrics with division-by-zero handling
    accuracy = (tp + tn) / total_days if total_days > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Percentages
    rain_prediction_pct = (predicted_rain_days / total_days * 100) if total_days > 0 else 0.0
    actual_rain_pct = (actual_rain_days / total_days * 100) if total_days > 0 else 0.0

    return {
        "total_days": total_days,
        "predicted_rain_days": predicted_rain_days,
        "actual_rain_days": actual_rain_days,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "rain_prediction_percentage": round(rain_prediction_pct, 1),
        "actual_rain_percentage": round(actual_rain_pct, 1),
        "confusion_matrix": {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        },
    }


def get_analysis_results():
    """
    Load or compute the 6-month analysis results.
    Returns a dictionary with all analysis data.
    """
    metrics_path = os.path.join(Config.PROCESSED_DATA_DIR, "analysis_results.json")

    # Try to load cached results
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)

    # Load test predictions if available
    predictions_path = os.path.join(Config.PROCESSED_DATA_DIR, "test_predictions.json")
    if os.path.exists(predictions_path):
        with open(predictions_path, "r") as f:
            test_preds = json.load(f)

        actual = [p["actual_rain"] for p in test_preds]
        predicted = [p["predicted_rain"] for p in test_preds]

        metrics = calculate_metrics(actual, predicted)

        # Save results
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    # Load model metrics if available
    model_metrics_path = os.path.join(Config.PROCESSED_DATA_DIR, "model_metrics.json")
    if os.path.exists(model_metrics_path):
        with open(model_metrics_path, "r") as f:
            return json.load(f)

    return None


def get_monthly_breakdown(df):
    """
    Calculate month-by-month rain prediction analysis.

    Returns:
        List of dicts with monthly metrics
    """
    if df is None or df.empty:
        return []

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    monthly_data = []
    for month, group in df.groupby("month"):
        actual = group["actual_rain"].values
        predicted = group["predicted_rain"].values

        metrics = calculate_metrics(actual, predicted)
        metrics["month"] = str(month)
        monthly_data.append(metrics)

    return monthly_data
