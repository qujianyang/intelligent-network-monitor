"""
Pure Machine Learning Fault Detection using Isolation Forest
=============================================================
This module focuses ONLY on ML-based anomaly detection without any rule-based logic.
Uses trained Isolation Forest models to detect anomalies based on patterns in data.
"""

import os
import logging
import joblib
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from .fault_db_mysql import (
    fetch_latest_metrics_mysql,
    fetch_recent_metrics_mysql,
    get_available_nodes_mysql
)

# Logging configuration
logger = logging.getLogger("qkd.ml_fault_detection")
logger.setLevel(logging.INFO)

# Global singleton instance for efficiency
_detector_instance = None


class MLFaultDetector:
    """Pure ML-based fault detection using Isolation Forest models"""

    def __init__(self, model_dir: str = None):
        """
        Initialize ML Fault Detector

        Args:
            model_dir: Directory containing trained Isolation Forest models
        """
        # Use absolute path to ensure models are found regardless of where script is run from
        if model_dir is None:
            # Get the absolute path to the models directory
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_dir = os.path.join(base_dir, "Dashboard", "models")

        self.model_dir = model_dir
        self.models = {}
        logger.info(f"Initializing MLFaultDetector with model_dir: {self.model_dir}")
        self._load_models()

    def _load_models(self):
        """Load all available Isolation Forest models"""
        available_nodes = get_available_nodes_mysql()
        logger.info(f"Available nodes from MySQL: {available_nodes}")

        for node_id in available_nodes:
            model_path = os.path.join(self.model_dir, f"fault_detector_{node_id}.pkl")
            logger.debug(f"Looking for model at: {model_path}")

            if os.path.exists(model_path):
                try:
                    model_data = joblib.load(model_path)

                    # Validate model structure
                    required_keys = ['model', 'feature_columns']
                    if all(key in model_data for key in required_keys):
                        self.models[node_id] = model_data
                        logger.info(f"Successfully loaded model for {node_id} from {model_path}")
                        logger.debug(f"Model for {node_id} has keys: {model_data.keys()}")
                    else:
                        logger.error(f"Model file for {node_id} missing required keys. Has: {model_data.keys() if hasattr(model_data, 'keys') else 'not a dict'}")
                except Exception as e:
                    logger.error(f"Failed to load model for {node_id} from {model_path}: {e}")
            else:
                logger.warning(f"Model file not found for {node_id} at {model_path}")

        logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")

    def detect_anomaly(self, node_id: str, metrics: Optional[Dict] = None) -> Dict:
        """
        Pure ML anomaly detection for a single data point.

        Args:
            node_id: Node identifier (e.g., 'QKD_001')
            metrics: Optional metrics dict. If None, fetches latest from database

        Returns:
            Dict containing:
                - is_anomaly: Boolean indicating if anomaly detected
                - anomaly_score: Float score from Isolation Forest
                - confidence: Confidence level of detection (0-1)
                - features_used: List of features used in detection
                - timestamp: Timestamp of the data point
        """
        # Check if model exists for this node
        if node_id not in self.models:
            return {
                "error": f"No trained model for node {node_id}",
                "available_nodes": list(self.models.keys())
            }

        # Get metrics if not provided
        if metrics is None:
            metrics = fetch_latest_metrics_mysql(node_id)
            if not metrics:
                return {
                    "error": f"No data available for node {node_id}",
                    "node_id": node_id
                }

        # Extract model components
        model_data = self.models[node_id]
        model = model_data['model']
        scaler = model_data.get('scaler')
        feature_columns = model_data['feature_columns']

        # Prepare feature vector
        feature_values = []
        for feature in feature_columns:
            value = metrics.get(feature, 0.0)  # Use 0 as default for missing features
            feature_values.append(value)

        # Convert to numpy array and scale
        X = np.array(feature_values).reshape(1, -1)

        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X

        # Get ML predictions
        prediction = model.predict(X_scaled)[0]
        anomaly_score = float(model.decision_function(X_scaled)[0])

        # Isolation Forest: -1 = anomaly, 1 = normal
        is_anomaly = (prediction == -1)

        # Calculate confidence based on score magnitude
        # Scores closer to 0 are less confident
        confidence = min(1.0, abs(anomaly_score) * 2)

        return {
            "node_id": node_id,
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "confidence": confidence,
            "prediction": int(prediction),
            "features_used": feature_columns,
            "timestamp": metrics.get("timestamp", datetime.now()),
            "data_point": {feature: metrics.get(feature) for feature in feature_columns}
        }

    def detect_anomalies_batch(self, node_id: str, hours_back: int = 24) -> Dict:
        """
        Detect anomalies in a batch of historical data points.

        Args:
            node_id: Node identifier
            hours_back: Hours of historical data to analyze

        Returns:
            Dict containing:
                - total_points: Total data points analyzed
                - anomalies_detected: Number of anomalies found
                - anomaly_rate: Percentage of anomalies
                - anomaly_scores: List of all scores
                - anomalies: List of anomalous data points with details
        """
        if node_id not in self.models:
            return {
                "error": f"No trained model for node {node_id}",
                "available_nodes": list(self.models.keys())
            }

        # Fetch historical data
        recent_metrics = fetch_recent_metrics_mysql(node_id, hours_back)

        if not recent_metrics:
            return {
                "error": f"No historical data for node {node_id}",
                "hours_requested": hours_back
            }

        # Extract model components
        model_data = self.models[node_id]
        model = model_data['model']
        scaler = model_data.get('scaler')
        feature_columns = model_data['feature_columns']

        # Process all data points
        anomalies = []
        all_scores = []

        for metric_data in recent_metrics:
            # Prepare features
            feature_values = [metric_data.get(col, 0.0) for col in feature_columns]
            X = np.array(feature_values).reshape(1, -1)

            if scaler:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X

            # Get predictions
            prediction = model.predict(X_scaled)[0]
            score = float(model.decision_function(X_scaled)[0])

            all_scores.append(score)

            if prediction == -1:  # Anomaly detected
                anomalies.append({
                    "timestamp": metric_data.get("timestamp"),
                    "anomaly_score": score,
                    "data_point": {col: metric_data.get(col) for col in feature_columns}
                })

        # Calculate statistics
        anomaly_rate = (len(anomalies) / len(recent_metrics)) * 100 if recent_metrics else 0

        return {
            "node_id": node_id,
            "time_range": {
                "hours_analyzed": hours_back,
                "start": recent_metrics[-1].get("timestamp") if recent_metrics else None,
                "end": recent_metrics[0].get("timestamp") if recent_metrics else None
            },
            "total_points": len(recent_metrics),
            "anomalies_detected": len(anomalies),
            "anomaly_rate": round(anomaly_rate, 2),
            "anomaly_scores": {
                "mean": float(np.mean(all_scores)),
                "std": float(np.std(all_scores)),
                "min": float(np.min(all_scores)),
                "max": float(np.max(all_scores))
            },
            "anomalies": anomalies[:10]  # Return first 10 anomalies for review
        }

    def get_model_info(self, node_id: str) -> Dict:
        """
        Get information about the trained model.

        Args:
            node_id: Node identifier

        Returns:
            Model information dictionary
        """
        if node_id not in self.models:
            return {"error": f"No model for {node_id}"}

        model_data = self.models[node_id]
        training_info = model_data.get('training_info', {})
        test_results = model_data.get('test_results', {})

        return {
            "node_id": node_id,
            "model_type": "Isolation Forest",
            "trained_at": training_info.get('trained_at', 'Unknown'),
            "training_samples": training_info.get('n_samples', 0),
            "features": training_info.get('features', []),
            "contamination": training_info.get('contamination', 0.05),
            "test_performance": {
                "test_samples": test_results.get('test_samples', 0),
                "anomalies_detected": test_results.get('outliers_detected', 0),
                "detection_rate": test_results.get('outlier_ratio', 0)
            }
        }


# Get or create singleton detector instance
def get_detector() -> MLFaultDetector:
    """Get or create the global MLFaultDetector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = MLFaultDetector()
    return _detector_instance


# Simplified functional interface
def detect_ml_anomaly(node_id: str) -> Dict:
    """
    Simple function to detect anomaly using ML model.

    Args:
        node_id: Node to check (e.g., 'QKD_001')

    Returns:
        Dict with is_anomaly, anomaly_score, and confidence
    """
    detector = get_detector()
    return detector.detect_anomaly(node_id)


def analyze_historical_anomalies(node_id: str, hours: int = 24) -> Dict:
    """
    Analyze historical data for anomalies.

    Args:
        node_id: Node to analyze
        hours: Hours of history to check

    Returns:
        Dict with anomaly statistics and details
    """
    detector = get_detector()
    return detector.detect_anomalies_batch(node_id, hours)


# Example usage
if __name__ == "__main__":
    # Example 1: Check current status
    result = detect_ml_anomaly("QKD_001")
    print(f"Anomaly: {result.get('is_anomaly')}, Score: {result.get('anomaly_score'):.3f}")

    # Example 2: Analyze last 24 hours
    history = analyze_historical_anomalies("QKD_001", 24)
    print(f"Anomaly rate (24h): {history.get('anomaly_rate')}%")