"""
Train Prophet Models Script
===========================
Manual script to train all Prophet models for QKD forecasting.
Run this whenever you want to retrain models with new data.

Usage:
    python Dashboard/services/train_prophet_models.py

This will:
1. Load historical data from MySQL (qkd_ml database)
2. Train Prophet models for each node and metric
3. Save models to Dashboard/models/prophet_{node_id}_{metric}.pkl
4. Display training results
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Dashboard.services.forecasting_ml_pure import ProphetForecaster, get_forecaster
from Dashboard.services.forecasting_db_mysql import get_db_connector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s]: %(message)s'
)
logger = logging.getLogger(__name__)


def train_all_models(hours_back: int = 720, force_retrain: bool = False):
    """
    Train Prophet models for all nodes and metrics.

    Args:
        hours_back: Hours of historical data to use (default: 720 = 30 days)
        force_retrain: If True, retrain even if models exist
    """
    print("=" * 60)
    print("PROPHET MODEL TRAINING SCRIPT")
    print("=" * 60)
    print(f"Training with {hours_back} hours ({hours_back/24:.1f} days) of data")
    print(f"Force retrain: {force_retrain}")
    print()

    # Initialize forecaster
    forecaster = ProphetForecaster()

    # Get available nodes from database
    db = get_db_connector()
    nodes = db.get_available_nodes()

    if not nodes:
        print("ERROR: No nodes found in database!")
        return

    print(f"Found {len(nodes)} nodes: {', '.join(nodes)}")
    print(f"Will train {len(forecaster.metrics_to_forecast)} metrics per node")
    print(f"Total models to train: {len(nodes) * len(forecaster.metrics_to_forecast)}")
    print("-" * 60)

    # Track results
    total_models = 0
    successful_models = 0
    failed_models = []
    training_times = []

    start_time = time.time()

    # Train models for each node
    for node_idx, node_id in enumerate(nodes, 1):
        print(f"\n[{node_idx}/{len(nodes)}] Training models for {node_id}")
        print("-" * 40)

        node_start = time.time()
        node_successes = 0

        for metric_idx, metric in enumerate(forecaster.metrics_to_forecast, 1):
            total_models += 1

            # Check if model exists
            model_info = forecaster.get_model_info(node_id, metric)
            model_exists = model_info.get('exists', False)

            if model_exists and not force_retrain:
                age_hours = model_info.get('model_age_hours', 0)
                print(f"  [{metric_idx}/{len(forecaster.metrics_to_forecast)}] {metric}: "
                      f"EXISTS (age: {age_hours:.1f}h) - Skipping")
                successful_models += 1
                continue

            # Train model
            print(f"  [{metric_idx}/{len(forecaster.metrics_to_forecast)}] {metric}: ", end="")
            sys.stdout.flush()

            metric_start = time.time()
            result = forecaster.train_model(node_id, metric, hours_back)
            metric_time = time.time() - metric_start
            training_times.append(metric_time)

            if result.get('success'):
                samples = result.get('training_samples', 0)
                print(f"SUCCESS ({samples} samples, {metric_time:.2f}s)")
                successful_models += 1
                node_successes += 1
            else:
                error = result.get('error', 'Unknown error')
                print(f"FAILED - {error}")
                failed_models.append(f"{node_id}_{metric}")

        node_time = time.time() - node_start
        print(f"\nNode {node_id} complete: {node_successes}/{len(forecaster.metrics_to_forecast)} "
              f"models trained in {node_time:.1f}s")

    # Training complete
    total_time = time.time() - start_time
    avg_train_time = sum(training_times) / len(training_times) if training_times else 0

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total models: {total_models}")
    print(f"Successful: {successful_models}")
    print(f"Failed: {len(failed_models)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time per model: {avg_train_time:.2f}s")

    if failed_models:
        print(f"\nFailed models:")
        for model in failed_models:
            print(f"  - {model}")

    # Show model status
    print("\n" + "-" * 60)
    print("MODEL STATUS")
    print("-" * 60)

    status = forecaster.get_all_models_status()
    print(f"Models loaded in cache: {status['total_models']}")
    print(f"Model directory: {status['model_dir']}")

    for node_id, metrics in status['models'].items():
        print(f"\n{node_id}:")
        for metric, info in metrics.items():
            age = info['age_hours']
            if age < 1:
                age_str = f"{age*60:.1f} minutes"
            elif age < 24:
                age_str = f"{age:.1f} hours"
            else:
                age_str = f"{age/24:.1f} days"
            print(f"  - {metric}: {info['samples']} samples, age: {age_str}")

    print("\n" + "=" * 60)
    print("Models saved to: Dashboard/models/prophet_*.pkl")
    print("To use: Models will be loaded automatically for fast predictions")
    print("=" * 60)


def train_single_node(node_id: str, hours_back: int = 720):
    """
    Train models for a single node.

    Args:
        node_id: Node to train (e.g., 'QKD_001')
        hours_back: Hours of historical data
    """
    print(f"Training models for {node_id} only...")
    forecaster = ProphetForecaster()
    result = forecaster.retrain_node(node_id, hours_back)

    print(f"\nResults for {node_id}:")
    print(f"  Successful: {result['successful_retrains']}/{result['total_metrics']}")

    for metric, status in result['models'].items():
        print(f"  - {metric}: {status}")


def check_data_availability():
    """Check data availability before training."""
    print("Checking data availability...")
    print("-" * 40)

    db = get_db_connector()
    nodes = db.get_available_nodes()

    print(f"Nodes available: {len(nodes)}")

    for node_id in nodes:
        # Get comprehensive data
        data = db.get_comprehensive_node_data(node_id, hours_back=720)

        node_metrics = data['node_metrics']
        link_metrics = data['link_metrics']

        print(f"\n{node_id}:")
        print(f"  Node metrics: {len(node_metrics)} records")
        print(f"  Link metrics: {len(link_metrics)} records")

        if not node_metrics.empty:
            print(f"  Date range: {node_metrics['timestamp'].min()} to {node_metrics['timestamp'].max()}")


def main():
    """Main function with command-line argument handling."""
    import argparse

    parser = argparse.ArgumentParser(description='Train Prophet forecasting models')
    parser.add_argument('--node', type=str, help='Train specific node only (e.g., QKD_001)')
    parser.add_argument('--hours', type=int, default=720,
                        help='Hours of historical data to use (default: 720)')
    parser.add_argument('--force', action='store_true',
                        help='Force retrain even if models exist')
    parser.add_argument('--check', action='store_true',
                        help='Check data availability only')

    args = parser.parse_args()

    if args.check:
        check_data_availability()
    elif args.node:
        train_single_node(args.node, args.hours)
    else:
        train_all_models(args.hours, args.force)


if __name__ == "__main__":
    main()