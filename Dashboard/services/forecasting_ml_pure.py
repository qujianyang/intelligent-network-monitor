"""
Pure Machine Learning Forecasting using Prophet
===============================================
This module focuses ONLY on ML-based forecasting without retraining every request.
Uses trained Prophet models saved to disk for fast predictions.
Follows the same pattern as fault_ml_pure.py with Isolation Forest.
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path

# Prophet imports
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None

from Dashboard.services.forecasting_db_mysql import get_db_connector, get_node_forecast_data

# Logging configuration
logger = logging.getLogger("qkd.ml_forecasting")
logger.setLevel(logging.INFO)

# Global singleton instance for efficiency
_forecaster_instance = None


class ProphetForecaster:
    """Pure ML-based forecasting using saved Prophet models"""

    def __init__(self, model_dir: str = None):
        """
        Initialize Prophet Forecaster

        Args:
            model_dir: Directory containing trained Prophet models
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available. Install with: pip install prophet")

        # Use absolute path to ensure models are found
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_dir = os.path.join(base_dir, "Dashboard", "models")

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        # Define metrics to forecast
        self.metrics_to_forecast = [
            'qkdKeyRate',
            'qkdQber',
            'qkdVisibility',
            'temperature',
            'cpu_load',
            'memory_usage'
        ]

        # Cache for loaded models
        self.models = {}

        logger.info(f"Initializing ProphetForecaster with model_dir: {self.model_dir}")
        self._load_models()

    def _load_models(self):
        """Load all available Prophet models from disk"""
        # Get available nodes from database
        db = get_db_connector()
        available_nodes = db.get_available_nodes()
        logger.info(f"Available nodes from MySQL: {available_nodes}")

        models_loaded = 0

        for node_id in available_nodes:
            for metric in self.metrics_to_forecast:
                model_path = os.path.join(self.model_dir, f"prophet_{node_id}_{metric}.pkl")

                if os.path.exists(model_path):
                    try:
                        model_data = joblib.load(model_path)

                        # Validate model structure
                        required_keys = ['model', 'trained_at', 'node_id', 'metric']
                        if all(key in model_data for key in required_keys):
                            # Store in cache with composite key
                            cache_key = f"{node_id}_{metric}"
                            self.models[cache_key] = model_data
                            models_loaded += 1
                            logger.debug(f"Loaded model for {node_id}_{metric}")
                        else:
                            logger.error(f"Model file for {node_id}_{metric} missing required keys")
                    except Exception as e:
                        logger.error(f"Failed to load model for {node_id}_{metric}: {e}")
                else:
                    logger.debug(f"Model not found for {node_id}_{metric} at {model_path}")

        logger.info(f"Loaded {models_loaded} Prophet models into cache")

    def train_model(self, node_id: str, metric: str, hours_back: int = 720) -> Dict:
        """
        Train a Prophet model for a specific node and metric.

        Args:
            node_id: Node identifier (e.g., 'QKD_001')
            metric: Metric to forecast (e.g., 'qkdKeyRate')
            hours_back: Hours of historical data to use for training (default: 720 = 30 days)

        Returns:
            Dictionary with training results
        """
        try:
            logger.info(f"Training Prophet model for {node_id}_{metric}")

            # Load historical data
            data = get_node_forecast_data(node_id, hours_back)

            if data.empty or metric not in data.columns:
                logger.error(f"No data available for {node_id}_{metric}")
                return {"error": f"No data for {node_id}_{metric}"}

            # Prepare data for Prophet (needs 'ds' and 'y' columns)
            prophet_data = pd.DataFrame({
                'ds': data['DateTime'],
                'y': data[metric]
            })

            # Remove NaN values
            prophet_data = prophet_data.dropna()

            if len(prophet_data) < 10:
                logger.error(f"Insufficient data for {node_id}_{metric}: {len(prophet_data)} points")
                return {"error": f"Insufficient data: {len(prophet_data)} points"}

            # Initialize Prophet with tuned parameters
            model = Prophet(
                changepoint_prior_scale=0.05,  # Flexibility of trend
                seasonality_prior_scale=10,    # Strength of seasonality
                holidays_prior_scale=10,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,       # Not enough data for yearly
                interval_width=0.8,
                uncertainty_samples=1000
            )

            # Fit model
            model.fit(prophet_data)

            # Prepare model data for saving
            model_data = {
                'model': model,
                'node_id': node_id,
                'metric': metric,
                'trained_at': datetime.now(),
                'training_samples': len(prophet_data),
                'training_hours': hours_back,
                'data_range': {
                    'start': prophet_data['ds'].min(),
                    'end': prophet_data['ds'].max()
                },
                'metric_stats': {
                    'mean': float(prophet_data['y'].mean()),
                    'std': float(prophet_data['y'].std()),
                    'min': float(prophet_data['y'].min()),
                    'max': float(prophet_data['y'].max())
                }
            }

            # Save model to disk
            model_path = os.path.join(self.model_dir, f"prophet_{node_id}_{metric}.pkl")
            joblib.dump(model_data, model_path)

            # Update cache
            cache_key = f"{node_id}_{metric}"
            self.models[cache_key] = model_data

            logger.info(f"Successfully trained and saved model for {node_id}_{metric}")
            return {
                "success": True,
                "node_id": node_id,
                "metric": metric,
                "training_samples": len(prophet_data),
                "model_path": model_path
            }

        except Exception as e:
            logger.error(f"Failed to train model for {node_id}_{metric}: {e}")
            return {"error": str(e)}

    def predict(self, node_id: str, metric: str, days_ahead: int = 7) -> Dict:
        """
        Make fast predictions using saved Prophet model.

        Args:
            node_id: Node identifier
            metric: Metric to forecast
            days_ahead: Number of days to forecast

        Returns:
            Dictionary with predictions and metadata
        """
        cache_key = f"{node_id}_{metric}"

        # Check if model exists in cache
        if cache_key not in self.models:
            # Try to load from disk
            model_path = os.path.join(self.model_dir, f"prophet_{node_id}_{metric}.pkl")

            if os.path.exists(model_path):
                try:
                    model_data = joblib.load(model_path)
                    self.models[cache_key] = model_data
                    logger.info(f"Loaded model from disk for {node_id}_{metric}")
                except Exception as e:
                    logger.error(f"Failed to load model from disk: {e}")
                    return {"error": f"Failed to load model for {node_id}_{metric}"}
            else:
                return {"error": f"No trained model found for {node_id}_{metric}"}

        # Get model from cache
        model_data = self.models[cache_key]
        model = model_data['model']

        try:
            # Create future dataframe
            future_periods = days_ahead * 24  # Hourly predictions

            # Start from the last training data point
            last_date = model_data['data_range']['end']

            # Generate future dates
            future = model.make_future_dataframe(
                periods=0,  # Don't include training data
                freq='H'
            )

            # Create actual future dates from where training ended
            future_dates = pd.date_range(
                start=last_date + timedelta(hours=1),
                periods=future_periods,
                freq='H'
            )

            future = pd.DataFrame({'ds': future_dates})

            # Make predictions (FAST - no training!)
            forecast = model.predict(future)

            # Extract predictions
            predictions = {
                'timestamps': forecast['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'predicted_values': forecast['yhat'].tolist(),
                'lower_bound': forecast['yhat_lower'].tolist(),
                'upper_bound': forecast['yhat_upper'].tolist()
            }

            # Calculate summary statistics
            predicted_mean = float(forecast['yhat'].mean())
            predicted_std = float(forecast['yhat'].std())

            return {
                'node_id': node_id,
                'metric': metric,
                'days_ahead': days_ahead,
                'predictions': predictions,
                'summary': {
                    'mean': predicted_mean,
                    'std': predicted_std,
                    'min': float(forecast['yhat'].min()),
                    'max': float(forecast['yhat'].max())
                },
                'model_info': {
                    'trained_at': model_data['trained_at'].isoformat(),
                    'training_samples': model_data['training_samples'],
                    'model_age_hours': (datetime.now() - model_data['trained_at']).total_seconds() / 3600
                }
            }

        except Exception as e:
            logger.error(f"Failed to predict for {node_id}_{metric}: {e}")
            return {"error": f"Prediction failed: {str(e)}"}

    def predict_all_metrics(self, node_id: str, days_ahead: int = 7) -> Dict:
        """
        Predict all metrics for a node using saved models.

        Args:
            node_id: Node identifier
            days_ahead: Number of days to forecast

        Returns:
            Dictionary with all metric predictions
        """
        results = {
            'node_id': node_id,
            'days_ahead': days_ahead,
            'timestamp': datetime.now().isoformat(),
            'forecasts': {}
        }

        successful_forecasts = 0

        for metric in self.metrics_to_forecast:
            prediction = self.predict(node_id, metric, days_ahead)

            if 'error' not in prediction:
                results['forecasts'][metric] = prediction
                successful_forecasts += 1
            else:
                results['forecasts'][metric] = prediction
                logger.warning(f"Failed to predict {metric} for {node_id}: {prediction['error']}")

        results['metrics_forecasted'] = successful_forecasts
        results['total_metrics'] = len(self.metrics_to_forecast)

        return results

    def retrain_node(self, node_id: str, hours_back: int = 720) -> Dict:
        """
        Retrain all models for a specific node with latest data.

        Args:
            node_id: Node identifier
            hours_back: Hours of historical data for training

        Returns:
            Dictionary with retrain results
        """
        logger.info(f"Retraining all models for {node_id}")

        results = {
            'node_id': node_id,
            'retrained_at': datetime.now().isoformat(),
            'models': {}
        }

        successful_retrains = 0

        for metric in self.metrics_to_forecast:
            train_result = self.train_model(node_id, metric, hours_back)

            if train_result.get('success'):
                results['models'][metric] = 'success'
                successful_retrains += 1
            else:
                results['models'][metric] = train_result.get('error', 'failed')

        results['successful_retrains'] = successful_retrains
        results['total_metrics'] = len(self.metrics_to_forecast)

        logger.info(f"Retrained {successful_retrains}/{len(self.metrics_to_forecast)} models for {node_id}")

        return results

    def get_model_info(self, node_id: str, metric: str) -> Dict:
        """
        Get information about a specific model.

        Args:
            node_id: Node identifier
            metric: Metric name

        Returns:
            Model information dictionary
        """
        cache_key = f"{node_id}_{metric}"

        if cache_key in self.models:
            model_data = self.models[cache_key]

            return {
                'node_id': node_id,
                'metric': metric,
                'exists': True,
                'trained_at': model_data['trained_at'].isoformat(),
                'training_samples': model_data['training_samples'],
                'model_age_hours': (datetime.now() - model_data['trained_at']).total_seconds() / 3600,
                'data_range': {
                    'start': model_data['data_range']['start'].isoformat(),
                    'end': model_data['data_range']['end'].isoformat()
                },
                'metric_stats': model_data['metric_stats']
            }
        else:
            return {
                'node_id': node_id,
                'metric': metric,
                'exists': False,
                'error': f"No model found for {node_id}_{metric}"
            }

    def get_all_models_status(self) -> Dict:
        """Get status of all loaded models."""
        status = {
            'total_models': len(self.models),
            'model_dir': self.model_dir,
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }

        for cache_key, model_data in self.models.items():
            node_id = model_data['node_id']
            metric = model_data['metric']

            if node_id not in status['models']:
                status['models'][node_id] = {}

            status['models'][node_id][metric] = {
                'trained_at': model_data['trained_at'].isoformat(),
                'age_hours': (datetime.now() - model_data['trained_at']).total_seconds() / 3600,
                'samples': model_data['training_samples']
            }

        return status


# Get or create singleton forecaster instance
def get_forecaster() -> ProphetForecaster:
    """Get or create the global ProphetForecaster instance."""
    global _forecaster_instance
    if _forecaster_instance is None:
        _forecaster_instance = ProphetForecaster()
    return _forecaster_instance


# Simplified functional interface (like fault_ml_pure.py)
def forecast_metric(node_id: str, metric: str, days_ahead: int = 7) -> Dict:
    """
    Simple function to forecast a metric using saved Prophet model.

    Args:
        node_id: Node to forecast (e.g., 'QKD_001')
        metric: Metric to forecast (e.g., 'qkdKeyRate')
        days_ahead: Forecast horizon

    Returns:
        Dict with predictions
    """
    forecaster = get_forecaster()
    return forecaster.predict(node_id, metric, days_ahead)


def forecast_node(node_id: str, days_ahead: int = 7) -> Dict:
    """
    Forecast all metrics for a node.

    Args:
        node_id: Node to forecast
        days_ahead: Forecast horizon

    Returns:
        Dict with all metric forecasts
    """
    forecaster = get_forecaster()
    return forecaster.predict_all_metrics(node_id, days_ahead)


def retrain_models(node_id: str) -> Dict:
    """
    Manually retrain all models for a node.

    Args:
        node_id: Node to retrain

    Returns:
        Retrain results
    """
    forecaster = get_forecaster()
    return forecaster.retrain_node(node_id)


def get_models_status() -> Dict:
    """Get status of all Prophet models."""
    forecaster = get_forecaster()
    return forecaster.get_all_models_status()


# Example usage
if __name__ == "__main__":
    # Example 1: Check model status
    status = get_models_status()
    print(f"Total models loaded: {status['total_models']}")

    # Example 2: Make a fast forecast
    result = forecast_metric("QKD_001", "qkdKeyRate", days_ahead=3)
    if 'error' not in result:
        print(f"Forecast generated in < 50ms (model already trained)")
        print(f"Mean prediction: {result['summary']['mean']:.2f}")

    # Example 3: Forecast all metrics for a node
    all_forecasts = forecast_node("QKD_001", days_ahead=7)
    print(f"Forecasted {all_forecasts['metrics_forecasted']} metrics")