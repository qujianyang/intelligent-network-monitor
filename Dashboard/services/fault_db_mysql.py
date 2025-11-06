"""
MySQL Database Connector for Fault Detection System
====================================================
Provides MySQL data access for Isolation Forest anomaly detection.
Replaces SQLite research database with production MySQL data.
"""

import pymysql
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class MySQLFaultConnector:
    """MySQL connector for fault detection with real QKD data"""

    def __init__(self):
        """Initialize MySQL connector for ML database"""
        # Use ML-specific database configuration
        self.host = os.getenv('ML_DB_HOST', 'localhost')
        self.port = int(os.getenv('ML_DB_PORT', '3307'))
        self.database = os.getenv('ML_DB_NAME', 'qkd_ml')
        self.user = os.getenv('ML_DB_USER', 'root')
        self.password = os.getenv('ML_DB_PASSWORD', '')

        logger.info(f"Initialized MySQL fault connector: {self.host}:{self.port}/{self.database}")

    def _get_connection(self):
        """Create a new MySQL connection"""
        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database
        )

    def get_available_nodes(self) -> List[str]:
        """Get list of nodes that have metrics data"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Get nodes that have metrics (using node_name as primary key)
            cursor.execute("""
                SELECT DISTINCT n.node_name
                FROM nodes n
                INNER JOIN node_metrics nm ON n.node_name = nm.node_name
                WHERE n.node_type = 'QKD'
                ORDER BY n.node_name
            """)

            nodes = [row[0] for row in cursor.fetchall()]
            conn.close()

            logger.info(f"Found {len(nodes)} nodes with metrics: {nodes}")
            return nodes

        except Exception as e:
            logger.error(f"Failed to get available nodes: {e}")
            return []

    def get_training_data(self, hours_back: int = 720) -> pd.DataFrame:
        """
        Get training data for Isolation Forest models.
        Combines node_metrics and link_metrics data.

        Args:
            hours_back: Hours of historical data to retrieve (default: 720 = 30 days)

        Returns:
            DataFrame with columns mapped to fault detection format
        """
        try:
            conn = self._get_connection()
            time_threshold = datetime.now() - timedelta(hours=hours_back)

            # Query using new schema with node_name as primary key
            query = """
            SELECT
                n.node_name as Node,
                nm.timestamp as DateTime,
                nm.cpu_usage as neCpuLoad,
                nm.memory_usage as neMemUsage,
                nm.temperature as neTemperature,
                nm.aggregated_key_rate as nodeKeyRate,
                -- Aggregate link metrics for nodes with multiple links
                AVG(lm.qber) as qkdQber,
                AVG(lm.key_rate) as qkdKeyRate,
                AVG(lm.visibility) as qkdVisibility,
                AVG(lm.laser_power) as qkdLaserPower,
                AVG(lm.attenuation) as attenuation,
                AVG(lm.photon_loss) as nodePhotonLoss,
                AVG(lm.detector_efficiency) as detectorEfficiency,
                AVG(lm.secure_key_rate) as secureKeyRate
            FROM node_metrics nm
            INNER JOIN nodes n ON nm.node_name = n.node_name
            LEFT JOIN links l ON (l.source_node = n.node_name OR l.destination_node = n.node_name) AND l.link_type = 'Quantum'
            LEFT JOIN link_metrics lm ON lm.link_name = l.link_name AND lm.timestamp = nm.timestamp
            WHERE nm.timestamp >= %s
            AND n.node_type = 'QKD'
            GROUP BY n.node_name, nm.timestamp, nm.cpu_usage, nm.memory_usage,
                     nm.temperature, nm.aggregated_key_rate
            ORDER BY nm.timestamp DESC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[time_threshold],
                parse_dates=['DateTime']
            )
            conn.close()

            if len(df) == 0:
                logger.warning("No training data found in MySQL")
                return pd.DataFrame()

            # No derived features needed - Isolation Forest will learn relationships

            logger.info(f"Loaded {len(df)} training records from MySQL for {df['Node'].nunique()} nodes")
            return df

        except Exception as e:
            logger.error(f"Failed to load training data from MySQL: {e}")
            return pd.DataFrame()

    def fetch_latest_metrics(self, node_id: str) -> Optional[Dict[str, float]]:
        """
        Fetch the latest metrics for a specific node.

        Args:
            node_id: Node identifier (e.g., 'QKD_001')

        Returns:
            Dictionary with latest metrics or None if not found
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Check if node exists (node_name is now the primary key)
            cursor.execute("SELECT node_name FROM nodes WHERE node_name = %s", (node_id,))
            node_result = cursor.fetchone()

            if not node_result:
                logger.warning(f"Node {node_id} not found")
                conn.close()
                return None

            node_name = node_result[0]

            # Get latest node metrics using new schema
            cursor.execute("""
                SELECT
                    timestamp,
                    cpu_usage,
                    memory_usage,
                    temperature,
                    aggregated_key_rate,
                    power_consumption,
                    error_count
                FROM node_metrics
                WHERE node_name = %s
                ORDER BY timestamp DESC
                LIMIT 1
            """, (node_name,))

            node_metrics = cursor.fetchone()

            if not node_metrics:
                logger.warning(f"No metrics found for node {node_id}")
                conn.close()
                return None

            # Get average of latest link metrics using new schema
            cursor.execute("""
                SELECT
                    AVG(lm.qber) as qber,
                    AVG(lm.key_rate) as key_rate,
                    AVG(lm.visibility) as visibility,
                    AVG(lm.laser_power) as laser_power,
                    AVG(lm.attenuation) as attenuation,
                    AVG(lm.detector_efficiency) as detector_efficiency,
                    AVG(lm.secure_key_rate) as secure_key_rate,
                    AVG(lm.dark_count_rate) as dark_count_rate,
                    AVG(lm.photon_loss) as photon_loss,
                    COUNT(DISTINCT l.link_name) as link_count,
                    GROUP_CONCAT(l.link_name) as link_names
                FROM links l
                INNER JOIN link_metrics lm ON lm.link_name = l.link_name
                WHERE (l.source_node = %s OR l.destination_node = %s)
                AND l.link_type = 'Quantum'
                AND lm.timestamp = (SELECT MAX(timestamp) FROM link_metrics WHERE link_name = l.link_name)
                GROUP BY lm.timestamp
                ORDER BY lm.timestamp DESC
                LIMIT 1
            """, (node_name, node_name))

            link_metrics = cursor.fetchone()
            conn.close()

            # Combine metrics into expected format (adjusted for new schema)
            metrics = {
                "timestamp": node_metrics[0],
                "neCpuLoad": float(node_metrics[1]),
                "neMemUsage": float(node_metrics[2]),
                "neTemperature": float(node_metrics[3]),
                "nodeKeyRate": float(node_metrics[4]),
                "powerConsumption": float(node_metrics[5]) if node_metrics[5] else 100.0,
                "errorCount": int(node_metrics[6]) if node_metrics[6] else 0
            }

            # Add link metrics if available (updated for new schema)
            if link_metrics:
                metrics.update({
                    "qkdQber": float(link_metrics[0]) if link_metrics[0] else 0.025,
                    "qkdKeyRate": float(link_metrics[1]) if link_metrics[1] else metrics["nodeKeyRate"],
                    "qkdVisibility": float(link_metrics[2]) if link_metrics[2] else 0.97,
                    "qkdLaserPower": float(link_metrics[3]) if link_metrics[3] else 10.0,
                    "attenuation": float(link_metrics[4]) if link_metrics[4] else 15.0,
                    "detectorEfficiency": float(link_metrics[5]) if link_metrics[5] else 0.1,
                    "secureKeyRate": float(link_metrics[6]) if link_metrics[6] else metrics["nodeKeyRate"] * 0.9,
                    "darkCountRate": float(link_metrics[7]) if link_metrics[7] else 50.0,
                    "nodePhotonLoss": float(link_metrics[8]) if link_metrics[8] else 20.0,  # Average photon loss
                    "linkCount": int(link_metrics[9]) if link_metrics[9] else 0,
                    "connectedLinks": link_metrics[10] if link_metrics[10] else ""
                })
            else:
                # Use default QKD metrics if no link data
                metrics.update({
                    "qkdQber": 0.025,
                    "qkdKeyRate": metrics["nodeKeyRate"],
                    "qkdVisibility": 0.97,
                    "qkdLaserPower": 10.0,
                    "attenuation": 15.0,
                    "detectorEfficiency": 0.1,
                    "secureKeyRate": metrics["nodeKeyRate"] * 0.9,
                    "darkCountRate": 50.0,
                    "nodePhotonLoss": 20.0
                })

            logger.info(f"Fetched latest metrics for {node_id}")
            return metrics

        except Exception as e:
            logger.error(f"Failed to fetch latest metrics for {node_id}: {e}")
            return None

    def fetch_recent_metrics(self, node_id: str, hours_back: int = 1) -> List[Dict[str, float]]:
        """
        Fetch recent metrics for trend analysis.

        Args:
            node_id: Node identifier
            hours_back: Hours of data to retrieve

        Returns:
            List of metrics dictionaries, newest first
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # First, get the latest timestamp for this node to use as reference
            cursor.execute("""
                SELECT MAX(timestamp) as latest_time
                FROM node_metrics
                WHERE node_name = %s
            """, (node_id,))

            result = cursor.fetchone()
            if not result or not result[0]:
                logger.warning(f"No data found for node {node_id}")
                conn.close()
                return []

            latest_timestamp = result[0]
            logger.debug(f"Latest data for {node_id} is from {latest_timestamp}")

            # Calculate time threshold relative to the latest available data
            time_threshold = latest_timestamp - timedelta(hours=hours_back)

            # Get recent metrics with proper schema
            query = """
            SELECT
                nm.timestamp,
                nm.cpu_usage as cpu_load,
                nm.memory_usage,
                nm.temperature,
                nm.aggregated_key_rate,
                0 as photon_loss,
                nm.power_consumption,
                nm.error_count,
                AVG(lm.qber) as qkd_qber,
                AVG(lm.key_rate) as qkd_key_rate,
                AVG(lm.visibility) as qkd_visibility,
                AVG(lm.laser_power) as qkd_laser_power,
                AVG(lm.attenuation) as attenuation,
                AVG(lm.detector_efficiency) as detector_efficiency,
                AVG(lm.secure_key_rate) as secure_key_rate
            FROM node_metrics nm
            LEFT JOIN links l ON (l.source_node = %s OR l.destination_node = %s)
                AND l.link_type = 'Quantum'
            LEFT JOIN link_metrics lm ON lm.link_name = l.link_name
                AND ABS(TIMESTAMPDIFF(SECOND, nm.timestamp, lm.timestamp)) < 60
            WHERE nm.node_name = %s
            AND nm.timestamp >= %s
            GROUP BY nm.timestamp
            ORDER BY nm.timestamp DESC
            """

            cursor.execute(query, (node_id, node_id, node_id, time_threshold))
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                logger.warning(f"No recent metrics found for {node_id}")
                return []

            # Convert to list of dictionaries
            metrics_list = []
            for row in rows:
                metrics = {
                    "timestamp": row[0],
                    "neCpuLoad": float(row[1]),
                    "neMemUsage": float(row[2]),
                    "neTemperature": float(row[3]),
                    "nodeKeyRate": float(row[4]),
                    "nodePhotonLoss": float(row[5]) if row[5] else 0.1,
                    "powerConsumption": float(row[6]) if row[6] else 100.0,
                    "errorCount": int(row[7]) if row[7] else 0,
                    "qkdQber": float(row[8]) if row[8] else 0.025,
                    "qkdKeyRate": float(row[9]) if row[9] else float(row[4]),
                    "qkdVisibility": float(row[10]) if row[10] else 0.97,
                    "qkdLaserPower": float(row[11]) if row[11] else 10.0,
                    "attenuation": float(row[12]) if row[12] else 15.0,
                    "detectorEfficiency": float(row[13]) if row[13] else 0.1,
                    "secureKeyRate": float(row[14]) if row[14] else float(row[4]) * 0.9
                }
                metrics_list.append(metrics)

            logger.info(f"Fetched {len(metrics_list)} recent metrics for {node_id} (from {time_threshold} to {latest_timestamp})")
            return metrics_list

        except Exception as e:
            logger.error(f"Failed to fetch recent metrics for {node_id}: {e}")
            return []

    def get_database_stats(self) -> Dict[str, any]:
        """Get database statistics for validation"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            stats = {}

            # Count metrics
            cursor.execute("SELECT COUNT(*) FROM node_metrics")
            stats['total_node_metrics'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM link_metrics")
            stats['total_link_metrics'] = cursor.fetchone()[0]

            # Count unique nodes with metrics
            cursor.execute("""
                SELECT COUNT(DISTINCT nm.node_name)
                FROM node_metrics nm
            """)
            stats['nodes_with_metrics'] = cursor.fetchone()[0]

            # Time range
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM node_metrics")
            time_range = cursor.fetchone()
            stats['time_range'] = {
                'start': str(time_range[0]) if time_range[0] else None,
                'end': str(time_range[1]) if time_range[1] else None
            }

            conn.close()

            logger.info(f"Database stats: {stats['total_node_metrics']} node records, "
                       f"{stats['total_link_metrics']} link records")

            return stats

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}


# Global connector instance
_global_fault_connector = None

def get_fault_db_connector() -> MySQLFaultConnector:
    """Get global MySQL fault database connector instance"""
    global _global_fault_connector

    if _global_fault_connector is None:
        _global_fault_connector = MySQLFaultConnector()
        logger.info("Initialized global MySQL fault database connector")

    return _global_fault_connector


# Convenience functions for easy integration with fault.py
def fetch_latest_metrics_mysql(node_id: str) -> Optional[Dict[str, float]]:
    """Fetch latest metrics for a node (compatible with fault.py)"""
    db = get_fault_db_connector()
    return db.fetch_latest_metrics(node_id)

def fetch_recent_metrics_mysql(node_id: str, hours_back: int = 1) -> List[Dict[str, float]]:
    """Fetch recent metrics for trend analysis (compatible with fault.py)"""
    db = get_fault_db_connector()
    return db.fetch_recent_metrics(node_id, hours_back)

def get_training_data_mysql(hours_back: int = 720) -> pd.DataFrame:
    """Get training data from MySQL (compatible with fault.py)"""
    db = get_fault_db_connector()
    return db.get_training_data(hours_back)

def get_available_nodes_mysql() -> List[str]:
    """Get list of available nodes from MySQL"""
    db = get_fault_db_connector()
    return db.get_available_nodes()