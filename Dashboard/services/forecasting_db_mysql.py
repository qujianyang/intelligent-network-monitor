"""
MySQL Database Connector for QKD Forecasting
============================================
Connects forecasting system to MySQL database for production use.

Features:
- Connects to MySQL instead of SQLite
- Compatible with existing forecasting.py
- Multi-metric support (QBER, visibility, temperature, CPU, memory)
- Efficient time-series queries with proper indexing
"""

import pymysql
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class MySQLQKDConnector:
    """MySQL database connector for forecasting with real QKD data"""

    def __init__(self):
        """Initialize MySQL connector for ML database"""
        # Use ML-specific database configuration (same as fault_db_mysql.py)
        self.host = os.getenv('ML_DB_HOST', 'localhost')
        self.port = int(os.getenv('ML_DB_PORT', '3307'))
        self.database = os.getenv('ML_DB_NAME', 'qkd_ml')
        self.user = os.getenv('ML_DB_USER', 'root')
        self.password = os.getenv('ML_DB_PASSWORD', '')

        logger.info(f"Initialized MySQL connector: {self.host}:{self.port}/{self.database}")

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
        """Get list of all available QKD nodes"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Get QKD nodes only (using qkd_ml schema)
            cursor.execute("""
                SELECT DISTINCT node_name
                FROM nodes
                WHERE node_type = 'QKD'
                ORDER BY node_name
            """)
            nodes = [row[0] for row in cursor.fetchall()]

            conn.close()
            logger.info(f"Found {len(nodes)} QKD nodes: {nodes}")
            return nodes

        except Exception as e:
            logger.error(f"Failed to get available nodes: {e}")
            return []

    def get_available_links(self) -> List[str]:
        """Get list of all available links"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT DISTINCT link_name
                FROM links
                ORDER BY link_name
            """)
            links = [row[0] for row in cursor.fetchall()]

            conn.close()
            logger.info(f"Found {len(links)} links: {links}")
            return links

        except Exception as e:
            logger.error(f"Failed to get available links: {e}")
            return []

    def get_node_metrics_history(self, node_id: str, hours_back: int = 168) -> pd.DataFrame:
        """
        Get historical node metrics from MySQL.

        Args:
            node_id: Target node (e.g., 'QKD_001')
            hours_back: Hours of history to retrieve (default: 168 = 7 days)

        Returns:
            DataFrame with node metrics
        """
        try:
            conn = self._get_connection()

            # In qkd_ml, node_name is the primary key (no separate id)
            cursor = conn.cursor()
            cursor.execute("SELECT node_name FROM nodes WHERE node_name = %s", (node_id,))
            node_result = cursor.fetchone()

            if not node_result:
                logger.warning(f"Node {node_id} not found in database")
                conn.close()
                return pd.DataFrame()

            node_name = node_result[0]

            # Calculate time threshold
            time_threshold = datetime.now() - timedelta(hours=hours_back)

            query = """
            SELECT
                timestamp,
                cpu_usage as cpu_load,
                memory_usage,
                temperature,
                power_consumption,
                aggregated_key_rate,
                key_generation_rate as consumption_rate,
                0 as uptime_hours,
                error_count,
                warning_count
            FROM node_metrics
            WHERE node_name = %s
            AND timestamp >= %s
            ORDER BY timestamp ASC
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=[node_name, time_threshold],
                parse_dates=['timestamp']
            )
            conn.close()

            if len(df) == 0:
                logger.warning(f"No node metrics found for {node_id} in last {hours_back} hours")
                return pd.DataFrame()

            logger.info(f"Retrieved {len(df)} node metric records for {node_id}")
            return df

        except Exception as e:
            logger.error(f"Failed to get node metrics for {node_id}: {e}")
            return pd.DataFrame()

    def get_link_metrics_history(self, node_id: str, hours_back: int = 168) -> pd.DataFrame:
        """
        Get historical link metrics for all links connected to a node.

        Args:
            node_id: Target node (e.g., 'QKD_001')
            hours_back: Hours of history to retrieve

        Returns:
            DataFrame with QKD link metrics
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # In qkd_ml, node_name is the primary key
            cursor.execute("SELECT node_name FROM nodes WHERE node_name = %s", (node_id,))
            node_result = cursor.fetchone()

            if not node_result:
                logger.warning(f"Node {node_id} not found in database")
                conn.close()
                return pd.DataFrame()

            node_name = node_result[0]

            # Find all links connected to this node (using node_name)
            cursor.execute("""
                SELECT link_name
                FROM links
                WHERE source_node = %s OR destination_node = %s
            """, (node_name, node_name))

            link_names = [row[0] for row in cursor.fetchall()]

            if not link_names:
                logger.warning(f"No links found for node {node_id}")
                conn.close()
                return pd.DataFrame()

            # Get metrics for all connected links
            time_threshold = datetime.now() - timedelta(hours=hours_back)

            # Create placeholders for IN clause
            placeholders = ','.join(['%s'] * len(link_names))

            query = f"""
            SELECT
                link_name,
                timestamp,
                qber as qkd_qber,
                key_rate as qkd_key_rate,
                visibility as qkd_visibility,
                laser_power as qkd_laser_power,
                secure_key_rate,
                photon_loss,
                attenuation,
                detector_efficiency,
                0 as temperature_fiber,
                90 as link_quality_score
            FROM link_metrics
            WHERE link_name IN ({placeholders})
            AND timestamp >= %s
            ORDER BY timestamp ASC
            """

            params = link_names + [time_threshold]
            df = pd.read_sql_query(
                query,
                conn,
                params=params,
                parse_dates=['timestamp']
            )
            conn.close()

            if len(df) == 0:
                logger.warning(f"No link metrics found for {node_id} links in last {hours_back} hours")
                return pd.DataFrame()

            # Add node_id for consistency
            df['node_id'] = node_id

            # Add laser power column (it's already included in the main query)
            # Just rename it if it wasn't renamed already
            if 'laser_power' in df.columns and 'qkd_laser_power' not in df.columns:
                df = df.rename(columns={'laser_power': 'qkd_laser_power'})
            elif 'qkd_laser_power' not in df.columns:
                df['qkd_laser_power'] = 10.0  # Default value

            logger.info(f"Retrieved {len(df)} link metric records for {node_id}")
            return df

        except Exception as e:
            logger.error(f"Failed to get link metrics for {node_id}: {e}")
            return pd.DataFrame()

    def get_comprehensive_node_data(self, node_id: str, hours_back: int = 168) -> Dict[str, pd.DataFrame]:
        """
        Get all available data for a node (both node metrics and link metrics).

        Returns:
            Dict with 'node_metrics' and 'link_metrics' DataFrames
        """
        logger.info(f"Getting comprehensive data for {node_id} ({hours_back} hours)")

        node_metrics = self.get_node_metrics_history(node_id, hours_back)
        link_metrics = self.get_link_metrics_history(node_id, hours_back)

        result = {
            'node_metrics': node_metrics,
            'link_metrics': link_metrics
        }

        logger.info(f"Comprehensive data for {node_id}: "
                   f"{len(node_metrics)} node records, {len(link_metrics)} link records")

        return result

    def get_database_stats(self) -> Dict[str, any]:
        """Get database statistics for monitoring and validation"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            stats = {}

            # Node metrics stats
            cursor.execute("SELECT COUNT(*) as count FROM node_metrics")
            stats['total_node_metrics'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT node_name) as count FROM node_metrics")
            stats['unique_nodes_with_metrics'] = cursor.fetchone()[0]

            # Link metrics stats
            cursor.execute("SELECT COUNT(*) as count FROM link_metrics")
            stats['total_link_metrics'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT link_name) as count FROM link_metrics")
            stats['unique_links_with_metrics'] = cursor.fetchone()[0]

            # Time range
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM node_metrics")
            node_time = cursor.fetchone()
            stats['node_metrics_time_range'] = {
                'start': str(node_time[0]) if node_time[0] else None,
                'end': str(node_time[1]) if node_time[1] else None
            }

            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM link_metrics")
            link_time = cursor.fetchone()
            stats['link_metrics_time_range'] = {
                'start': str(link_time[0]) if link_time[0] else None,
                'end': str(link_time[1]) if link_time[1] else None
            }

            conn.close()

            logger.info(f"Database stats: {stats['total_node_metrics']} node records, "
                       f"{stats['total_link_metrics']} link records")

            return stats

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}


# Import numpy for simulated data
import numpy as np

# Global connector instance (singleton pattern)
_global_mysql_connector = None

def get_db_connector() -> MySQLQKDConnector:
    """Get global MySQL database connector instance"""
    global _global_mysql_connector

    if _global_mysql_connector is None:
        _global_mysql_connector = MySQLQKDConnector()
        logger.info("Initialized global MySQL database connector")

    return _global_mysql_connector

def reset_db_connector():
    """Reset global database connector (useful for testing)"""
    global _global_mysql_connector
    _global_mysql_connector = None
    logger.info("Reset global MySQL database connector")


# Convenience functions for easy integration
def get_node_forecast_data(node_id: str, hours_back: int = 168) -> pd.DataFrame:
    """
    Get forecast-ready data for a node combining all available metrics.
    This function maintains compatibility with forecasting.py

    Returns unified DataFrame with columns:
    - DateTime: timestamp
    - Node: node_id
    - qkdKeyRate: from link_metrics.key_rate
    - qkdQber: from link_metrics.qber
    - qkdVisibility: from link_metrics.visibility
    - temperature: from node_metrics.temperature_system
    - cpu_load: from node_metrics.cpu_load
    - memory_usage: from node_metrics.memory_usage
    """
    try:
        db = get_db_connector()
        data = db.get_comprehensive_node_data(node_id, hours_back)

        node_metrics = data['node_metrics']
        link_metrics = data['link_metrics']

        if node_metrics.empty and link_metrics.empty:
            logger.warning(f"No data available for {node_id}")
            return pd.DataFrame()

        # Start with link metrics (QKD data) as primary
        if not link_metrics.empty:
            # Average across all links for this node
            forecast_df = link_metrics.groupby('timestamp').agg({
                'qkd_qber': 'mean',
                'qkd_key_rate': 'mean',
                'qkd_visibility': 'mean',
                'secure_key_rate': 'mean'
            }).reset_index()

            # Add laser power if it exists
            if 'qkd_laser_power' in link_metrics.columns:
                laser_agg = link_metrics.groupby('timestamp')['qkd_laser_power'].mean().reset_index()
                forecast_df = pd.merge(forecast_df, laser_agg, on='timestamp', how='left')
            else:
                forecast_df['qkd_laser_power'] = 10.0  # Default

            # Rename for compatibility with existing forecasting code
            forecast_df = forecast_df.rename(columns={
                'timestamp': 'DateTime',
                'qkd_key_rate': 'qkdKeyRate',
                'qkd_qber': 'qkdQber',
                'qkd_visibility': 'qkdVisibility',
                'qkd_laser_power': 'qkdLaserPower'
            })

        else:
            # Fallback to node metrics only
            forecast_df = node_metrics[['timestamp']].copy()
            forecast_df = forecast_df.rename(columns={'timestamp': 'DateTime'})
            # Add dummy QKD metrics
            forecast_df['qkdKeyRate'] = 1000
            forecast_df['qkdQber'] = 0.025
            forecast_df['qkdVisibility'] = 0.97

        # Add node metrics if available
        if not node_metrics.empty:
            # Check which columns actually exist
            agg_dict = {}
            if 'temperature' in node_metrics.columns:
                agg_dict['temperature'] = 'mean'
            if 'cpu_load' in node_metrics.columns:
                agg_dict['cpu_load'] = 'mean'
            if 'memory_usage' in node_metrics.columns:
                agg_dict['memory_usage'] = 'mean'
            if 'aggregated_key_rate' in node_metrics.columns:
                agg_dict['aggregated_key_rate'] = 'mean'

            if agg_dict:
                node_summary = node_metrics.groupby('timestamp').agg(agg_dict).reset_index()
                node_summary = node_summary.rename(columns={'timestamp': 'DateTime'})

                # Merge with link data
                if not link_metrics.empty:
                    forecast_df = pd.merge(forecast_df, node_summary, on='DateTime', how='outer')
                else:
                    forecast_df = node_summary
                    # Add QKD columns if not present
                    if 'qkdKeyRate' not in forecast_df.columns:
                        forecast_df['qkdKeyRate'] = forecast_df.get('aggregated_key_rate', 1000)
                        forecast_df['qkdQber'] = 0.025
                        forecast_df['qkdVisibility'] = 0.97
            elif link_metrics.empty:
                # No aggregatable columns and no link metrics
                forecast_df = pd.DataFrame()
                logger.warning(f"No aggregatable data found for {node_id}")
                return forecast_df

        # Add Node column for compatibility
        forecast_df['Node'] = node_id

        # Sort by time
        forecast_df = forecast_df.sort_values('DateTime').reset_index(drop=True)

        # Forward fill any NaN values
        forecast_df = forecast_df.fillna(method='ffill').fillna(method='bfill')

        logger.info(f"Prepared forecast data for {node_id}: {len(forecast_df)} records")
        return forecast_df

    except Exception as e:
        logger.error(f"Failed to get forecast data for {node_id}: {e}")
        return pd.DataFrame()