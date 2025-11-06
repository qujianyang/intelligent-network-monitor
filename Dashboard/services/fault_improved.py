"""
Improved Fault Detection Service - Consolidated and Simplified
==============================================================
Cleaner implementation with proper link relationships and essential features only.
"""

import os
import logging
import joblib
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from config import FAULT_MODEL_PATH

from .fault_db_mysql import (
    fetch_latest_metrics_mysql,
    fetch_recent_metrics_mysql,
    get_available_nodes_mysql
)

# Logging configuration
logger = logging.getLogger("qkdaio.fault_improved")
logger.setLevel(logging.INFO)

# QKD thresholds configuration
QKD_THRESHOLDS = {
    "qkdQber": ("max", 0.06),       # Max 6% error rate
    "qkdKeyRate": ("min", 900),     # Min 900 keys/sec
    "qkdVisibility": ("min", 0.85), # Min 85% visibility
    "qkdLaserPower": ("min", 0.5),  # Min 0.5 mW
    "neCpuLoad": ("max", 80),       # Max 80% CPU
    "neMemUsage": ("max", 90),      # Max 90% memory
    "neTemperature": ("max", 75)    # Max 75°C
}

def detect_fault(
    node_id: str,
    include_trends: bool = False,
    trend_hours: int = 3,
    custom_thresholds: Optional[Dict] = None
) -> Dict:
    """
    Unified fault detection function with optional trend analysis.

    Args:
        node_id: Node to diagnose (e.g., 'QKD_001')
        include_trends: Whether to include trend analysis
        trend_hours: Hours of history for trend analysis
        custom_thresholds: Override default thresholds

    Returns:
        Dictionary with complete fault analysis
    """
    # Validate node
    valid_nodes = get_available_nodes_mysql()
    if node_id not in valid_nodes:
        return {
            "status": "error",
            "message": f"Node '{node_id}' not found. Available: {', '.join(valid_nodes)}"
        }

    # Get latest metrics
    metrics = fetch_latest_metrics_mysql(node_id)
    if not metrics:
        return {
            "status": "error",
            "message": f"No data found for node '{node_id}'"
        }

    # Use custom or default thresholds
    thresholds = custom_thresholds or QKD_THRESHOLDS

    # Load model
    model_dir = os.path.dirname(FAULT_MODEL_PATH)
    model_path = os.path.join(model_dir, f"fault_detector_{node_id}.pkl")

    if not os.path.exists(model_path):
        return {
            "status": "error",
            "message": f"No trained model for '{node_id}'"
        }

    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data.get('scaler')
        feature_cols = model_data['feature_columns']
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {
            "status": "error",
            "message": f"Model loading failed: {e}"
        }

    # Prepare features
    feature_values = []
    for feature in feature_cols:
        if feature in metrics:
            feature_values.append(metrics[feature])
        else:
            # Use sensible defaults for missing features
            defaults = {
                "attenuation": 15.0,
                "detectorEfficiency": 0.1,
                "secureKeyRate": 1000.0
            }
            feature_values.append(defaults.get(feature, 0.0))

    # Run anomaly detection
    import numpy as np
    X = np.array(feature_values).reshape(1, -1)

    if scaler:
        X = scaler.transform(X)

    prediction = model.predict(X)[0]
    anomaly_score = float(model.decision_function(X)[0])

    # Check thresholds
    violations = []
    for metric, (check_type, threshold) in thresholds.items():
        value = metrics.get(metric)
        if value is not None:
            if check_type == "max" and value > threshold:
                violations.append({
                    "metric": metric,
                    "value": value,
                    "threshold": threshold,
                    "type": "exceeds"
                })
            elif check_type == "min" and value < threshold:
                violations.append({
                    "metric": metric,
                    "value": value,
                    "threshold": threshold,
                    "type": "below"
                })

    # Determine status
    is_anomaly = (prediction == -1) or len(violations) > 0

    if is_anomaly:
        if anomaly_score < -0.3 or len(violations) >= 3:
            severity = "HIGH"
        elif anomaly_score < -0.1 or len(violations) >= 1:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        status = "fault"
    else:
        severity = "NONE"
        status = "normal"

    # Build result
    result = {
        "status": status,
        "severity": severity,
        "node_id": node_id,
        "timestamp": metrics.get("timestamp", datetime.now()),
        "anomaly_score": anomaly_score,
        "threshold_violations": violations,
        "metrics": {
            "qkdQber": metrics.get("qkdQber"),
            "qkdKeyRate": metrics.get("qkdKeyRate"),
            "qkdVisibility": metrics.get("qkdVisibility"),
            "neCpuLoad": metrics.get("neCpuLoad"),
            "neMemUsage": metrics.get("neMemUsage"),
            "neTemperature": metrics.get("neTemperature")
        }
    }

    # Add link information if available
    if "connectedLinks" in metrics:
        result["connected_links"] = metrics["connectedLinks"]
        result["link_count"] = metrics.get("linkCount", 0)

    # Add trend analysis if requested
    if include_trends:
        trends = analyze_trends(node_id, trend_hours)
        result["trends"] = trends

    # Generate recommendations
    result["recommendations"] = generate_recommendations(result)

    return result

def analyze_trends(node_id: str, hours_back: int) -> Dict:
    """
    Simplified trend analysis.

    Args:
        node_id: Node to analyze
        hours_back: Hours of history

    Returns:
        Trend analysis results
    """
    recent_metrics = fetch_recent_metrics_mysql(node_id, hours_back)

    if not recent_metrics or len(recent_metrics) < 2:
        return {"status": "insufficient_data"}

    # Calculate trends for key metrics
    metrics_to_analyze = ["qkdQber", "qkdKeyRate", "neCpuLoad", "neTemperature"]
    trends = {}

    for metric in metrics_to_analyze:
        values = [m.get(metric, 0) for m in recent_metrics if metric in m]

        if len(values) >= 2:
            # Simple trend: compare first half vs second half average
            mid = len(values) // 2
            first_half = sum(values[:mid]) / mid
            second_half = sum(values[mid:]) / len(values[mid:])

            # Determine trend direction
            if metric == "qkdQber":  # Lower is better
                if second_half > first_half * 1.1:
                    trend = "deteriorating"
                elif second_half < first_half * 0.9:
                    trend = "improving"
                else:
                    trend = "stable"
            elif metric == "qkdKeyRate":  # Higher is better
                if second_half < first_half * 0.9:
                    trend = "deteriorating"
                elif second_half > first_half * 1.1:
                    trend = "improving"
                else:
                    trend = "stable"
            else:  # Stable is better
                change_ratio = abs(second_half - first_half) / (first_half + 0.001)
                trend = "deteriorating" if change_ratio > 0.2 else "stable"

            trends[metric] = {
                "direction": trend,
                "current": values[-1],
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values)
            }

    # Overall trend
    deteriorating = sum(1 for t in trends.values() if t["direction"] == "deteriorating")
    improving = sum(1 for t in trends.values() if t["direction"] == "improving")

    if deteriorating >= 2:
        overall = "deteriorating"
    elif improving >= 2:
        overall = "improving"
    else:
        overall = "stable"

    return {
        "hours_analyzed": hours_back,
        "data_points": len(recent_metrics),
        "metrics": trends,
        "overall": overall
    }

def generate_recommendations(analysis: Dict) -> List[str]:
    """
    Generate actionable recommendations based on fault analysis.

    Args:
        analysis: Fault analysis results

    Returns:
        List of recommendations
    """
    recommendations = []

    if analysis["status"] == "fault":
        # Check specific violations
        for violation in analysis.get("threshold_violations", []):
            metric = violation["metric"]

            if metric == "qkdQber":
                recommendations.append("Check fiber optic alignment and cleanliness")
                recommendations.append("Verify laser stability and power levels")

            elif metric == "qkdKeyRate":
                recommendations.append("Inspect photon detectors for degradation")
                recommendations.append("Check quantum channel attenuation")

            elif metric == "qkdVisibility":
                recommendations.append("Recalibrate optical components")
                recommendations.append("Check for environmental vibrations")

            elif metric == "neTemperature":
                recommendations.append("Check cooling system operation")
                recommendations.append("Verify thermal paste and heat sink contact")

            elif metric == "neCpuLoad" or metric == "neMemUsage":
                recommendations.append("Review running processes and optimize")
                recommendations.append("Consider system resource upgrade")

        # Add severity-based recommendations
        if analysis["severity"] == "HIGH":
            recommendations.insert(0, "IMMEDIATE ACTION REQUIRED")
            recommendations.append("Consider emergency maintenance")

        elif analysis["severity"] == "MEDIUM":
            recommendations.append("Schedule maintenance within 24 hours")

    else:  # Normal operation
        # Check trends if available
        if "trends" in analysis and analysis["trends"].get("overall") == "deteriorating":
            recommendations.append("Monitor system closely - trends showing degradation")
            recommendations.append("Schedule preventive maintenance")

    # Limit recommendations
    return recommendations[:5] if recommendations else ["System operating normally"]

