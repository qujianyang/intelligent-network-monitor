"""
Improved Fault Model Training with Essential Features Only
===========================================================
Train Isolation Forest models using only the 10 essential features.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path for Dashboard imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from Dashboard.services.fault_db_mysql import (
    get_training_data_mysql,
    get_available_nodes_mysql,
    get_fault_db_connector
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the 10 essential features (no derived features)
ESSENTIAL_FEATURES = [
    # Core QKD metrics (4)
    'qkdQber',          # Quantum bit error rate
    'qkdKeyRate',       # Key generation rate
    'qkdVisibility',    # Visibility
    'qkdLaserPower',    # Laser power

    # Node health metrics (3)
    'neCpuLoad',        # CPU usage
    'neMemUsage',       # Memory usage
    'neTemperature',    # Temperature

    # Additional QKD parameters (3)
    'attenuation',      # Fiber attenuation
    'detectorEfficiency', # Detector efficiency
    'secureKeyRate'     # Secure key rate after privacy amplification
]

def prepare_features(df: pd.DataFrame, node_id: str) -> pd.DataFrame:
    """
    Prepare features for a specific node using only essential features.

    Args:
        df: Full training dataframe
        node_id: Node to filter for

    Returns:
        Feature dataframe ready for training
    """
    # Filter for specific node
    node_data = df[df['Node'] == node_id].copy()

    if len(node_data) == 0:
        logger.warning(f"No data found for node {node_id}")
        return pd.DataFrame()

    # Select only essential features that exist in the data
    available_features = [col for col in ESSENTIAL_FEATURES if col in node_data.columns]
    missing_features = [col for col in ESSENTIAL_FEATURES if col not in node_data.columns]

    if missing_features:
        logger.warning(f"Missing features for {node_id}: {missing_features}")

    if len(available_features) < 7:  # Need at least 7 features for good detection
        logger.error(f"Insufficient features for {node_id}: only {len(available_features)} available")
        return pd.DataFrame()

    X = node_data[available_features].copy()

    # Handle any missing values with forward fill then backward fill
    X = X.ffill().bfill()

    # If still NaN, use column mean
    X = X.fillna(X.mean())

    logger.info(f"Prepared {len(X)} samples with {len(available_features)} features for {node_id}")
    logger.info(f"Features used: {available_features}")

    # Log feature statistics
    logger.info(f"Feature ranges for {node_id}:")
    for feature in available_features[:5]:  # Show first 5 features
        logger.info(f"  {feature}: [{X[feature].min():.3f}, {X[feature].max():.3f}] (mean: {X[feature].mean():.3f})")

    return X

def train_node_model(
    node_id: str,
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
    model_dir: str = "Dashboard/models"
) -> Dict:
    """
    Train Isolation Forest model for a specific node with essential features.

    Args:
        node_id: Node identifier
        df: Training data
        contamination: Expected proportion of anomalies
        random_state: Random seed
        model_dir: Directory to save models

    Returns:
        Dictionary with training results
    """
    logger.info(f"Training improved model for {node_id}")

    # Prepare features
    X = prepare_features(df, node_id)

    if X.empty:
        return {"error": f"No valid training data for {node_id}"}

    # Split data (80/20 split)
    if len(X) >= 100:
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=random_state)
        logger.info(f"Split data: {len(X_train)} train, {len(X_test)} test samples")
    else:
        X_train = X
        X_test = None
        logger.warning(f"Small dataset ({len(X)} samples), training on full data")

    # Standardize features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train Isolation Forest with optimized parameters
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,           # Number of trees
        max_samples='auto',          # Subsample size
        max_features=1.0,            # Use all features
        bootstrap=False,             # No bootstrap sampling
        n_jobs=-1,                   # Use all CPU cores
        warm_start=False
    )

    logger.info("Training Isolation Forest...")
    model.fit(X_train_scaled)

    # Evaluate on test set if available
    test_results = {}
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)
        scores = model.decision_function(X_test_scaled)

        n_outliers = (predictions == -1).sum()
        outlier_ratio = n_outliers / len(predictions)

        test_results = {
            "test_samples": len(X_test),
            "outliers_detected": int(n_outliers),
            "outlier_ratio": float(outlier_ratio),
            "avg_anomaly_score": float(scores.mean()),
            "min_anomaly_score": float(scores.min()),
            "max_anomaly_score": float(scores.max()),
            "std_anomaly_score": float(scores.std())
        }

        logger.info(f"Test results: {n_outliers}/{len(X_test)} outliers ({outlier_ratio:.2%})")
        logger.info(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Create model directory
    os.makedirs(model_dir, exist_ok=True)

    # Save model and metadata
    model_filename = f"fault_detector_{node_id}.pkl"
    model_path = os.path.join(model_dir, model_filename)

    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': list(X.columns),
        'training_info': {
            'node_id': node_id,
            'trained_at': datetime.now().isoformat(),
            'n_samples': len(X_train),
            'n_features': len(X.columns),
            'contamination': contamination,
            'features': list(X.columns),
            'feature_ranges': {
                col: {'min': float(X[col].min()), 'max': float(X[col].max()), 'mean': float(X[col].mean())}
                for col in X.columns
            }
        },
        'test_results': test_results
    }

    joblib.dump(model_data, model_path)
    logger.info(f"Model saved to {model_path}")

    return {
        "node_id": node_id,
        "model_path": model_path,
        "n_samples": len(X_train),
        "n_features": len(X.columns),
        "test_results": test_results,
        "features": list(X.columns)
    }

def train_all_models(hours_back: int = 720, contamination: float = 0.05):
    """
    Train improved models for all available nodes.

    Args:
        hours_back: Hours of historical data to use
        contamination: Expected anomaly ratio
    """
    print("=" * 60)
    print("IMPROVED FAULT DETECTION MODEL TRAINING")
    print("Using Essential Features Only")
    print("=" * 60)

    # Get database stats
    db = get_fault_db_connector()
    stats = db.get_database_stats()
    print(f"\nDatabase Statistics:")
    print(f"  Total node metrics: {stats.get('total_node_metrics', 0)}")
    print(f"  Total link metrics: {stats.get('total_link_metrics', 0)}")
    print(f"  Time range: {stats.get('time_range', {}).get('start', 'N/A')} to {stats.get('time_range', {}).get('end', 'N/A')}")

    # Load training data
    print(f"\nLoading {hours_back} hours of training data from MySQL...")
    df = get_training_data_mysql(hours_back)

    if df.empty:
        print("ERROR: No training data available in MySQL!")
        return

    print(f"Loaded {len(df)} total records")

    # Check which features are available
    available_in_data = [f for f in ESSENTIAL_FEATURES if f in df.columns]
    missing_in_data = [f for f in ESSENTIAL_FEATURES if f not in df.columns]

    print(f"\nFeature Analysis:")
    print(f"  Essential features available: {len(available_in_data)}/10")
    if missing_in_data:
        print(f"  Missing features: {missing_in_data}")

    # Get available nodes
    nodes = df['Node'].unique()
    print(f"\nFound {len(nodes)} nodes: {list(nodes)}")

    # Train model for each node
    results = []
    for node_id in nodes:
        print(f"\n" + "=" * 40)
        print(f"Training model for {node_id}")
        print("=" * 40)

        result = train_node_model(
            node_id=node_id,
            df=df,
            contamination=contamination,
            model_dir="Dashboard/models"
        )

        results.append(result)

        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"SUCCESS: Model trained")
            print(f"  - Samples: {result['n_samples']}")
            print(f"  - Features: {result['n_features']}")
            if result['test_results']:
                print(f"  - Test outliers: {result['test_results']['outliers_detected']}/{result['test_results']['test_samples']}")
                print(f"  - Score std: {result['test_results']['std_anomaly_score']:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"Successfully trained: {len(successful)}/{len(nodes)} models")

    if successful:
        print("\nTrained models:")
        for r in successful:
            print(f"  - {r['node_id']}: {r['n_features']} features, {r['n_samples']} samples")

    if failed:
        print("\nFailed models:")
        for r in failed:
            print(f"  - {r['node_id']}: {r['error']}")

    print(f"\nModels saved in 'Dashboard/models' directory")
    print("Ready for improved fault detection!")

def compare_with_old_models():
    """Compare new models with old models if they exist."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    old_dir = "Dashboard/models_old"  # Compare with old backups if they exist
    new_dir = "Dashboard/models"

    if not os.path.exists(old_dir):
        print("No old models found for comparison")
        return

    old_models = [f for f in os.listdir(old_dir) if f.startswith("fault_detector_QKD") and f.endswith(".pkl")]
    new_models = [f for f in os.listdir(new_dir) if f.startswith("fault_detector_QKD") and f.endswith(".pkl")]

    print(f"Old models: {len(old_models)}")
    print(f"New models: {len(new_models)}")

    for model_file in new_models:
        old_path = os.path.join(old_dir, model_file)
        new_path = os.path.join(new_dir, model_file)

        if os.path.exists(old_path):
            old_data = joblib.load(old_path)
            new_data = joblib.load(new_path)

            old_features = old_data.get('feature_columns', [])
            new_features = new_data.get('feature_columns', [])

            print(f"\n{model_file}:")
            print(f"  Old: {len(old_features)} features")
            print(f"  New: {len(new_features)} features")
            print(f"  Feature reduction: {len(old_features) - len(new_features)} features removed")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train improved fault detection models')
    parser.add_argument('--hours', type=int, default=720,
                       help='Hours of historical data to use (default: 720 = 30 days)')
    parser.add_argument('--contamination', type=float, default=0.05,
                       help='Expected anomaly ratio (default: 0.05 = 5%%)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with old models after training')

    args = parser.parse_args()

    # Train models
    train_all_models(
        hours_back=args.hours,
        contamination=args.contamination
    )

    # Compare if requested
    if args.compare:
        compare_with_old_models()