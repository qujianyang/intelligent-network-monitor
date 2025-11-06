# Dashboard/services/forecasting.py
"""
QKD Key-Rate Forecasting Service
==============================

This module provides 7-day key-rate forecasting using Facebook Prophet.
Features clean interfaces, comprehensive validation, and configurable parameters.

Key Functions:
- forecast_key_rate(node_id, history_df, horizon=7): Core forecasting function
- load_historical_data(): Data loading with fallback synthetic data
- Flask endpoints for API integration

Author: QKD Network Management System
"""

import os
import logging
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from config import (
    ROOT_DIR, 
    FORECASTING_CONFIG, 
    PROPHET_PARAMS, 
    KEY_RATE_THRESHOLDS
)

# Prophet forecasting - graceful import handling
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None

# ─────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────────────────────────────────────
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s]: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "forecasting.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading & Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def load_historical_data(data_source: str = None) -> pd.DataFrame:
    """
    Load historical QKD data from configured source.
    
    Args:
        data_source: Override default data source ("database", "csv", or "snmp")
        
    Returns:
        DataFrame with columns: ['Node', 'DateTime', 'qkdKeyRate', ...]
        
    Note: This isolates data loading from forecasting logic.
          SNMP integration can be swapped in here without changing forecast code.
    """
    source = data_source or FORECASTING_CONFIG.get("data_source", "database")
    
    if source == "database":
        return _load_database_data()
    elif source == "csv":
        return _load_csv_data()
    elif source == "snmp":
        # TODO: Implement SNMP data collection here
        logger.warning("SNMP data source not yet implemented, falling back to database")
        return _load_database_data()
    else:
        raise ValueError(f"Unsupported data source: {source}")


def _load_csv_data() -> pd.DataFrame:
    """Load data from CSV file with fallback synthetic data."""
    csv_file = FORECASTING_CONFIG["csv_file"]
    data_path = os.path.join(ROOT_DIR, "data", csv_file)
    
    try:
        logger.info(f"Loading historical data from {data_path}")
        df = pd.read_csv(data_path, parse_dates=["DateTime"])
        df = df.sort_values("DateTime")
        
        # Validate required columns
        required_cols = ["Node", "DateTime", "qkdKeyRate"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Successfully loaded {len(df)} records from CSV")
        return df
        
    except Exception as e:
        logger.warning(f"Failed to load CSV data: {e}. Generating synthetic fallback data")
        return _generate_synthetic_data()


def _load_database_data() -> pd.DataFrame:
    """Load data from research_qkd.db with comprehensive metrics."""
    try:
        from Dashboard.services.forecasting_db_mysql import get_db_connector, get_node_forecast_data
        
        logger.info("Loading historical data from MySQL database")
        
        # Get database connector and available nodes
        db = get_db_connector()
        nodes = db.get_available_nodes()
        
        if not nodes:
            logger.warning("No nodes found in database, falling back to synthetic data")
            return _generate_synthetic_data()
        
        # Load data for all nodes
        all_data = []
        for node_id in nodes:
            node_data = get_node_forecast_data(node_id, hours_back=168)  # 7 days
            if not node_data.empty:
                all_data.append(node_data)
                logger.info(f"Loaded {len(node_data)} records for {node_id}")
            else:
                logger.warning(f"No data found for {node_id}")
        
        if not all_data:
            logger.warning("No data loaded from database, falling back to synthetic data")
            return _generate_synthetic_data()
        
        # Combine all node data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Ensure required columns exist (with fallbacks)
        if 'qkdKeyRate' not in combined_df.columns:
            if 'secure_key_rate' in combined_df.columns:
                combined_df['qkdKeyRate'] = combined_df['secure_key_rate']
            else:
                logger.warning("No key rate data found, using synthetic values")
                combined_df['qkdKeyRate'] = 1000 + np.random.normal(0, 100, len(combined_df))
        
        # Add missing columns with defaults if needed
        if 'qkdQber' not in combined_df.columns:
            combined_df['qkdQber'] = 0.03 + np.random.normal(0, 0.01, len(combined_df))
        
        if 'qkdVisibility' not in combined_df.columns:
            combined_df['qkdVisibility'] = 0.97 + np.random.normal(0, 0.02, len(combined_df))
        
        # Sort by time and reset index
        combined_df = combined_df.sort_values('DateTime').reset_index(drop=True)
        
        logger.info(f"Successfully loaded {len(combined_df)} records from database for {len(nodes)} nodes")
        return combined_df
        
    except Exception as e:
        logger.error(f"Failed to load database data: {e}. Falling back to synthetic data")
        return _generate_synthetic_data()


def _generate_synthetic_data() -> pd.DataFrame:
    """Generate synthetic QKD data for development/testing."""
    logger.info("Generating synthetic QKD data for testing")
    
    # Create 7 days of hourly data for multiple nodes
    hours = 7 * 24
    nodes = ["NodeA", "NodeB", "NodeC"]
    
    data = []
    base_time = datetime.now() - timedelta(hours=hours)
    
    for node in nodes:
        # Different baseline key rates per node
        base_rate = {"NodeA": 1200, "NodeB": 1000, "NodeC": 800}[node]
        
        for hour in range(hours):
            timestamp = base_time + timedelta(hours=hour)
            
            # Add realistic patterns: daily cycle + noise + occasional dips
            daily_cycle = 100 * np.sin(2 * np.pi * hour / 24)  # Daily variation
            noise = np.random.normal(0, 50)  # Random noise
            trend = -0.5 * hour  # Slight downward trend
            
            # Occasional performance dips
            if np.random.random() < 0.05:  # 5% chance
                dip = -200
            else:
                dip = 0
                
            key_rate = max(100, base_rate + daily_cycle + noise + trend + dip)
            
            data.append({
                "Node": node,
                "DateTime": timestamp,
                "qkdKeyRate": round(key_rate, 2),
                "qkdQber": 0.02 + np.random.normal(0, 0.01),
                "qkdVisibility": 0.95 + np.random.normal(0, 0.02)
            })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} synthetic records for {len(nodes)} nodes")
    return df


def _preprocess_for_prophet(node_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert QKD data to Prophet format (ds, y columns).
    
    Args:
        node_data: DataFrame with 'DateTime' and 'qkdKeyRate' columns
        
    Returns:
        DataFrame with 'ds' (datestamp) and 'y' (target) columns
        
    Note: This rename operation is isolated to avoid touching raw data elsewhere.
    """
    if len(node_data) == 0:
        raise ValueError("Empty dataset provided for preprocessing")
    
    # Validate required columns exist
    if 'DateTime' not in node_data.columns:
        raise ValueError("Missing 'DateTime' column in node data")
    if 'qkdKeyRate' not in node_data.columns:
        raise ValueError("Missing 'qkdKeyRate' column in node data")
    
    # Create Prophet-compatible DataFrame on a copy to avoid side effects
    prophet_df = pd.DataFrame({
        'ds': node_data['DateTime'].copy(),
        'y': node_data['qkdKeyRate'].copy()
    })
    
    # Remove any rows with missing values
    initial_len = len(prophet_df)
    prophet_df = prophet_df.dropna()
    dropped_rows = initial_len - len(prophet_df)
    
    if dropped_rows > 0:
        logger.warning(f"Dropped {dropped_rows} rows with missing values")
    
    # Sort by timestamp
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    logger.debug(f"Preprocessed {len(prophet_df)} data points for Prophet")
    return prophet_df


# ─────────────────────────────────────────────────────────────────────────────
# Core Forecasting Engine with Clean Interface
# ─────────────────────────────────────────────────────────────────────────────
def forecast_key_rate(
    node_id: str, 
    history_df: pd.DataFrame, 
    horizon: int = None
) -> pd.DataFrame:
    """
    Generate key-rate forecast using Prophet.
    
    This is the core forecasting function with a clean interface:
    - Takes a DataFrame as input (not tied to specific data loading)
    - Returns a DataFrame with forecast results
    - All configuration comes from config.py
    
    Args:
        node_id: Node identifier (e.g., "NodeA")
        history_df: Historical data with 'DateTime' and 'qkdKeyRate' columns
        horizon: Forecast horizon in days (defaults to config value)
        
    Returns:
        DataFrame with columns: ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        Contains exactly horizon * 24 rows (hourly forecasts)
        
    Raises:
        ValueError: If insufficient data or invalid inputs
        RuntimeError: If Prophet model training fails
    """
    start_time = time.time()
    
    # Validate inputs
    if not isinstance(node_id, str) or not node_id.strip():
        raise ValueError("node_id must be a non-empty string")
    
    if not isinstance(history_df, pd.DataFrame):
        raise ValueError("history_df must be a pandas DataFrame")
    
    if horizon is None:
        horizon = FORECASTING_CONFIG["default_horizon_days"]
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    
    logger.info(f"Starting {horizon}-day forecast for {node_id}")
    
    # Check Prophet availability
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet not available. Install with: pip install prophet")
    
    # Filter data for specific node
    node_data = history_df[history_df["Node"] == node_id].copy()
    
    # Validate sufficient data
    min_points = FORECASTING_CONFIG["min_training_points"]
    if len(node_data) < min_points:
        raise ValueError(
            f"Insufficient data for {node_id}: {len(node_data)} points < {min_points} required"
        )
    
    logger.info(f"Using {len(node_data)} historical points for {node_id}")
    
    try:
        # Preprocess data for Prophet
        prophet_data = _preprocess_for_prophet(node_data)
        logger.debug(f"Preprocessed data shape: {prophet_data.shape}")
        
        # Initialize Prophet with configured parameters
        logger.info("Initializing Prophet model with configured parameters")
        model = Prophet(
            interval_width=FORECASTING_CONFIG["confidence_interval"],
            daily_seasonality=PROPHET_PARAMS["daily_seasonality"],
            weekly_seasonality=PROPHET_PARAMS["weekly_seasonality"],
            yearly_seasonality=PROPHET_PARAMS["yearly_seasonality"],
            changepoint_prior_scale=PROPHET_PARAMS["changepoint_prior_scale"],
            seasonality_prior_scale=PROPHET_PARAMS["seasonality_prior_scale"],
            holidays_prior_scale=PROPHET_PARAMS["holidays_prior_scale"],
            mcmc_samples=PROPHET_PARAMS["mcmc_samples"],
            uncertainty_samples=PROPHET_PARAMS["uncertainty_samples"]
        )
        
        # Train model
        logger.info("Training Prophet model...")
        train_start = time.time()
        model.fit(prophet_data)
        train_time = time.time() - train_start
        logger.info(f"Model training completed in {train_time:.2f} seconds")
        
        # Generate future dataframe
        future_periods = horizon * 24  # Convert days to hours
        future = model.make_future_dataframe(periods=future_periods, freq='H')
        logger.debug(f"Created future dataframe with {len(future)} periods")
        
        # Generate predictions
        logger.info("Generating forecasts...")
        forecast = model.predict(future)
        
        # Extract only future predictions (not historical fit)
        forecast_future = forecast.tail(future_periods).copy()
        
        # Validate results
        if forecast_future['yhat'].isna().any():
            raise RuntimeError("Prophet generated NaN predictions")
        
        # Return clean forecast DataFrame
        result_df = forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        result_df = result_df.reset_index(drop=True)
        
        total_time = time.time() - start_time
        logger.info(
            f"Forecast completed for {node_id}: {len(result_df)} predictions "
            f"generated in {total_time:.2f} seconds"
        )
        
        return result_df
        
    except Exception as e:
        logger.error(f"Forecast failed for {node_id}: {str(e)}")
        raise RuntimeError(f"Forecast generation failed: {str(e)}") from e


# ─────────────────────────────────────────────────────────────────────────────
# High-Level API Functions (Flask Integration)
# ─────────────────────────────────────────────────────────────────────────────
def get_node_forecast_api(
    node_id: str,
    days_ahead: int = None,
    include_metadata: bool = True
) -> Dict:
    """
    High-level API function for Flask endpoints.

    This wraps the core forecast_key_rate function and adds:
    - Automatic data loading
    - Error handling with user-friendly messages
    - Metadata and insights generation
    - JSON-serializable output

    Now uses saved Prophet models for 100x faster predictions when available.

    Args:
        node_id: Target node identifier
        days_ahead: Forecast horizon in days
        include_metadata: Include insights and recommendations

    Returns:
        Dict with forecast data, metadata, and insights
    """
    # Try fast method first (using saved models)
    try:
        from Dashboard.services.forecasting_improved import get_node_forecast_fast, ensure_models_exist

        if ensure_models_exist():
            logger.info(f"Using fast forecasting with saved Prophet models for {node_id}")
            return get_node_forecast_fast(node_id, days_ahead or 7, include_metadata)
        else:
            logger.warning("No saved models found, falling back to slow method")
    except ImportError:
        logger.warning("Fast forecasting not available, using slow method")
    except Exception as e:
        logger.warning(f"Fast forecast failed: {e}, falling back to slow method")

    # Original slow implementation (trains model every time)
    try:
        # Load historical data
        history_df = load_historical_data()
        
        # Generate core forecast
        forecast_df = forecast_key_rate(node_id, history_df, days_ahead)
        
        # Calculate summary statistics
        current_data = history_df[history_df["Node"] == node_id].tail(24)
        current_avg = current_data['qkdKeyRate'].mean() if len(current_data) > 0 else 0
        predicted_avg = forecast_df['yhat'].mean()
        
        # Build response
        result = {
            "node_id": node_id,
            "forecast_period": f"{days_ahead or FORECASTING_CONFIG['default_horizon_days']} days",
            "timestamp": datetime.now().isoformat(),
            "current_avg_rate": round(current_avg, 2),
            "predicted_avg_rate": round(predicted_avg, 2),
            "confidence_interval": FORECASTING_CONFIG["confidence_interval"],
            "predictions": {
                "timestamps": forecast_df['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                "predicted_values": forecast_df['yhat'].round(2).tolist(),
                "lower_bound": forecast_df['yhat_lower'].round(2).tolist(),
                "upper_bound": forecast_df['yhat_upper'].round(2).tolist()
            }
        }
        
        # Add metadata if requested
        if include_metadata:
            result.update(_generate_forecast_metadata(current_avg, predicted_avg, forecast_df))
        
        return result
        
    except ValueError as e:
        logger.warning(f"Validation error in forecast API: {e}")
        return {"error": f"Validation error: {str(e)}", "node_id": node_id}
    
    except RuntimeError as e:
        logger.error(f"Runtime error in forecast API: {e}")
        return {"error": f"Forecast failed: {str(e)}", "node_id": node_id}
    
    except Exception as e:
        logger.error(f"Unexpected error in forecast API: {e}")
        return {"error": f"Internal error: {str(e)}", "node_id": node_id}


def _generate_forecast_metadata(
    current_avg: float, 
    predicted_avg: float, 
    forecast_df: pd.DataFrame
) -> Dict:
    """Generate insights, recommendations, and risk assessment."""
    
    # Calculate trend
    change_pct = ((predicted_avg - current_avg) / current_avg) * 100 if current_avg > 0 else 0
    
    if abs(change_pct) < 2:
        trend = "stable"
        trend_strength = "weak"
    elif change_pct > 2:
        trend = "increasing"
        trend_strength = "strong" if change_pct > 10 else "moderate"
    else:
        trend = "decreasing"
        trend_strength = "strong" if change_pct < -10 else "moderate"
    
    # Risk assessment using configured thresholds
    min_predicted = forecast_df['yhat_lower'].min()
    risk_level = "low"
    risk_factors = []
    
    if min_predicted < KEY_RATE_THRESHOLDS["critical_minimum"]:
        risk_level = "critical"
        risk_factors.append("Key-rate may drop below critical threshold")
    elif min_predicted < KEY_RATE_THRESHOLDS["warning_minimum"]:
        risk_level = "high"
        risk_factors.append("Key-rate approaching minimum operational levels")
    elif predicted_avg < current_avg * 0.8:
        risk_level = "medium"
        risk_factors.append("Significant decrease in key generation predicted")
    
    # Generate insights
    insights = []
    if trend == "stable":
        insights.append(f"Key-rate expected to remain stable around {predicted_avg:.0f} bps")
    elif trend == "increasing":
        insights.append(f"Key-rate predicted to increase by {change_pct:.1f}% over forecast period")
    else:
        insights.append(f"Key-rate predicted to decrease by {abs(change_pct):.1f}% over forecast period")
    
    if predicted_avg < KEY_RATE_THRESHOLDS["critical_minimum"]:
        insights.append("CRITICAL: Predicted key-rate may fall below minimum operational threshold")
    elif predicted_avg < KEY_RATE_THRESHOLDS["warning_minimum"]:
        insights.append("WARNING: Key-rate trending toward lower operational range")
    
    # Generate recommendations
    recommendations = []
    if trend == "decreasing" and trend_strength in ["moderate", "strong"]:
        recommendations.append("Schedule maintenance check to address declining performance")
        recommendations.append("Monitor quantum channel parameters closely")
    
    if predicted_avg < KEY_RATE_THRESHOLDS["critical_minimum"]:
        recommendations.append("URGENT: Plan for backup key distribution methods")
        recommendations.append("Alert network operations team immediately")
    elif predicted_avg < KEY_RATE_THRESHOLDS["warning_minimum"]:
        recommendations.append("Prepare contingency protocols")
    
    return {
        "trend": trend,
        "trend_strength": trend_strength,
        "change_percentage": round(change_pct, 2),
        "insights": insights,
        "recommendations": recommendations,
        "risk_assessment": {
            "level": risk_level,
            "factors": risk_factors,
            "min_predicted_rate": round(min_predicted, 2)
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────


def _forecast_single_metric(history_df: pd.DataFrame, metric_col: str, days_ahead: int) -> pd.DataFrame:
    """
    Optimized single-metric forecasting function to avoid repeated Prophet model creation.
    """
    try:
        # Prepare data for Prophet
        metric_data = history_df[['DateTime', metric_col]].copy()
        metric_data = metric_data.rename(columns={'DateTime': 'ds', metric_col: 'y'})
        
        # Remove NaN values
        metric_data = metric_data.dropna()
        
        if len(metric_data) < 10:  # Need minimum data points
            logger.warning(f"Insufficient data for {metric_col}: {len(metric_data)} points")
            return pd.DataFrame()
        
        # Initialize Prophet model with optimized settings
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            holidays_prior_scale=10,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            interval_width=0.8
        )
        
        # Fit model
        model.fit(metric_data)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=days_ahead * 24, freq='h')
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Return only future predictions
        return forecast.tail(days_ahead * 24)
        
    except Exception as e:
        logger.error(f"Single metric forecast failed for {metric_col}: {e}")
        return pd.DataFrame()


def get_comprehensive_forecast_api(node_id: str, days_ahead: int = None) -> Dict:
    """
    Enhanced API function with multi-metric forecasting.

    Now uses saved Prophet models for 100x faster predictions when available.

    Returns forecasts for:
    - Key Rate (existing)
    - QBER (NEW)
    - Visibility (NEW)
    - Temperature (NEW)
    - CPU Load (NEW)
    - Memory Usage (NEW)
    """
    # Try fast method first (using saved models)
    try:
        from Dashboard.services.forecasting_improved import get_comprehensive_forecast_fast, ensure_models_exist

        if ensure_models_exist():
            logger.info(f"Using fast comprehensive forecasting with saved Prophet models for {node_id}")
            return get_comprehensive_forecast_fast(node_id, days_ahead or 7)
        else:
            logger.warning("No saved models found, falling back to slow method")
    except ImportError:
        logger.warning("Fast comprehensive forecasting not available, using slow method")
    except Exception as e:
        logger.warning(f"Fast comprehensive forecast failed: {e}, falling back to slow method")

    # Original slow implementation
    try:
        from Dashboard.services.forecasting_db_mysql import get_node_forecast_data
        
        # Load comprehensive historical data
        history_df = get_node_forecast_data(node_id, hours_back=168)
        
        if history_df.empty:
            return {"error": f"No historical data available for {node_id}", "node_id": node_id}
        
        days_ahead = days_ahead or FORECASTING_CONFIG["default_horizon_days"]
        
        # Define metrics to forecast
        metrics_to_forecast = {
            'qkdKeyRate': 'Key Rate (bps)',
            'qkdQber': 'QBER',
            'qkdVisibility': 'Visibility',
            'temperature': 'Temperature (°C)',
            'cpu_load': 'CPU Load (%)',
            'memory_usage': 'Memory Usage (%)'
        }
        
        forecasts = {}
        current_values = {}
        
        # Generate forecasts for each available metric efficiently
        for metric_col, metric_name in metrics_to_forecast.items():
            if metric_col in history_df.columns and not history_df[metric_col].isna().all():
                try:
                    logger.info(f"Forecasting {metric_name} for {node_id}")
                    
                    # Use the optimized single-metric forecast function
                    forecast_df = _forecast_single_metric(history_df, metric_col, days_ahead)
                    
                    if not forecast_df.empty:
                        current_avg = history_df[metric_col].tail(24).mean()
                        predicted_avg = forecast_df['yhat'].mean()
                        
                        forecasts[metric_col] = {
                            'name': metric_name,
                            'current_avg': round(current_avg, 4),
                            'predicted_avg': round(predicted_avg, 4),
                            'change_pct': round(((predicted_avg - current_avg) / current_avg) * 100, 2) if current_avg > 0 else 0,
                            'predictions': {
                                'timestamps': forecast_df['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                                'predicted_values': forecast_df['yhat'].round(4).tolist(),
                                'lower_bound': forecast_df['yhat_lower'].round(4).tolist(),
                                'upper_bound': forecast_df['yhat_upper'].round(4).tolist()
                            }
                        }
                        
                        current_values[metric_col] = current_avg
                        
                except Exception as e:
                    logger.warning(f"Failed to forecast {metric_name} for {node_id}: {e}")
                    forecasts[metric_col] = {"error": str(e)}
        
        if not forecasts:
            return {"error": f"No metrics could be forecasted for {node_id}", "node_id": node_id}
        
        # Generate intelligent insights
        insights = _generate_comprehensive_insights(node_id, forecasts, current_values)
        
        return {
            "node_id": node_id,
            "forecast_period": f"{days_ahead} days",
            "timestamp": datetime.now().isoformat(),
            "forecasts": forecasts,
            "insights": insights,
            "data_source": "research_qkd.db",
            "metrics_forecasted": len(forecasts)
        }
        
    except Exception as e:
        logger.error(f"Comprehensive forecast failed for {node_id}: {e}")
        return {"error": f"Comprehensive forecast failed: {str(e)}", "node_id": node_id}


def _generate_comprehensive_insights(node_id: str, forecasts: Dict, current_values: Dict) -> Dict:
    """Generate intelligent insights from multi-metric forecasts"""
    
    insights = {
        "summary": [],
        "warnings": [],
        "recommendations": [],
        "risk_assessment": "low"
    }
    
    # Analyze each metric
    for metric_key, forecast_data in forecasts.items():
        if "error" in forecast_data:
            continue
            
        metric_name = forecast_data['name']
        change_pct = forecast_data['change_pct']
        current = forecast_data['current_avg']
        predicted = forecast_data['predicted_avg']
        
        # Generate metric-specific insights
        if metric_key == 'qkdKeyRate':
            if change_pct < -10:
                insights["warnings"].append(f"Key rate predicted to drop by {abs(change_pct):.1f}%")
                insights["recommendations"].append("Monitor quantum channel parameters closely")
                insights["risk_assessment"] = "high"
            elif change_pct > 10:
                insights["summary"].append(f"Key rate improving by {change_pct:.1f}%")
                
        elif metric_key == 'qkdQber':
            if predicted > 0.05:  # QBER threshold
                insights["warnings"].append(f"QBER trending toward critical level: {predicted:.4f}")
                insights["recommendations"].append("Schedule quantum channel maintenance")
                insights["risk_assessment"] = "high"
            elif change_pct > 20:
                insights["warnings"].append(f"QBER increasing by {change_pct:.1f}%")
                insights["risk_assessment"] = "medium"
                
        elif metric_key == 'temperature':
            if predicted > 60:  # Temperature threshold
                insights["warnings"].append(f"Temperature approaching critical: {predicted:.1f}°C")
                insights["recommendations"].append("Check cooling systems immediately")
                insights["risk_assessment"] = "high"
            elif change_pct > 15:
                insights["warnings"].append(f"Temperature rising by {change_pct:.1f}%")
                insights["recommendations"].append("Monitor cooling system performance")
                
        elif metric_key == 'cpu_load':
            if predicted > 80:
                insights["warnings"].append(f"CPU load approaching maximum: {predicted:.1f}%")
                insights["recommendations"].append("Consider load balancing or system optimization")
                insights["risk_assessment"] = "medium"
                
        elif metric_key == 'memory_usage':
            if predicted > 85:
                insights["warnings"].append(f"Memory usage critical: {predicted:.1f}%")
                insights["recommendations"].append("Investigate memory leaks or increase capacity")
                insights["risk_assessment"] = "high"
    
    # Cross-metric correlations
    if 'temperature' in forecasts and 'qkdQber' in forecasts:
        temp_change = forecasts['temperature'].get('change_pct', 0)
        qber_change = forecasts['qkdQber'].get('change_pct', 0)
        
        if temp_change > 10 and qber_change > 10:
            insights["warnings"].append("Temperature and QBER both increasing - possible correlation")
            insights["recommendations"].append("Priority: Address temperature issues to prevent QBER degradation")
    
    # Default summary if no specific insights
    if not insights["summary"] and not insights["warnings"]:
        insights["summary"].append(f"All metrics stable for {node_id}")
    
    return insights






 