"""
Populate ML Database with Synthetic QKD Data
============================================
Generates realistic quantum key distribution metrics for training
fault detection models.
"""

import os
import sys
import pymysql
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

class SyntheticDataGenerator:
    """Generate realistic QKD network data with various operational patterns"""

    def __init__(self, start_date=None, days=30):
        """
        Initialize generator

        Args:
            start_date: Starting date for data generation (default: 30 days ago)
            days: Number of days to generate
        """
        self.end_date = datetime.now()
        self.start_date = start_date or (self.end_date - timedelta(days=days))
        self.days = days

        # ML Database connection parameters
        self.host = os.getenv('ML_DB_HOST', 'localhost')
        self.port = int(os.getenv('ML_DB_PORT', '3307'))
        self.database = os.getenv('ML_DB_NAME', 'qkd_ml')
        self.user = os.getenv('ML_DB_USER', 'root')
        self.password = os.getenv('ML_DB_PASSWORD', '')

        # Define nodes and links
        self.nodes = ['QKD_001', 'QKD_002', 'QKD_003']
        self.links = [
            ('LINK_QKD_001_002', 'QKD_001', 'QKD_002', 10.5),
            ('LINK_QKD_002_003', 'QKD_002', 'QKD_003', 15.2),
            ('LINK_QKD_001_003', 'QKD_001', 'QKD_003', 8.7)
        ]

    def get_connection(self):
        """Get database connection"""
        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database
        )

    def clear_existing_data(self):
        """Clear existing metrics data (keeps nodes and links)"""
        print("\nClearing existing metrics data...")
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Clear in order due to foreign keys
            cursor.execute("DELETE FROM fault_predictions")
            cursor.execute("DELETE FROM alerts")
            cursor.execute("DELETE FROM link_metrics")
            cursor.execute("DELETE FROM node_metrics")
            conn.commit()
            print("[OK] Existing metrics cleared")
        except Exception as e:
            print(f"[FAIL] Error clearing data: {e}")
            conn.rollback()
        finally:
            conn.close()

    def generate_node_metrics(self, node_name, timestamp, condition='normal'):
        """
        Generate node metrics based on operational condition

        Args:
            node_name: Node identifier
            timestamp: Timestamp for the metrics
            condition: 'normal', 'warning', 'critical', or 'degrading'
        """
        # Base values vary by node
        node_base = {
            'QKD_001': {'cpu': 45, 'mem': 60, 'temp': 38, 'key_rate': 1400},
            'QKD_002': {'cpu': 40, 'mem': 55, 'temp': 36, 'key_rate': 1350},
            'QKD_003': {'cpu': 50, 'mem': 65, 'temp': 40, 'key_rate': 1200}
        }

        base = node_base[node_name]

        # Add time-based variations (day/night, weekday/weekend)
        hour = timestamp.hour
        day_of_week = timestamp.weekday()

        # Night time reduction (22:00 - 06:00)
        if hour >= 22 or hour < 6:
            activity_factor = 0.7
        # Peak hours (09:00 - 17:00)
        elif 9 <= hour <= 17:
            activity_factor = 1.2
        else:
            activity_factor = 1.0

        # Weekend reduction
        if day_of_week >= 5:  # Saturday = 5, Sunday = 6
            activity_factor *= 0.8

        # Generate based on condition
        if condition == 'normal':
            cpu = base['cpu'] * activity_factor + np.random.normal(0, 5)
            memory = base['mem'] * activity_factor + np.random.normal(0, 3)
            temperature = base['temp'] + np.random.normal(0, 2)
            key_gen_rate = base['key_rate'] * activity_factor + np.random.normal(0, 50)
            error_count = 0
            status = 'Healthy'

        elif condition == 'warning':
            cpu = base['cpu'] * activity_factor * 1.5 + np.random.normal(0, 8)
            memory = base['mem'] * activity_factor * 1.3 + np.random.normal(0, 5)
            temperature = base['temp'] + 5 + np.random.normal(0, 3)
            key_gen_rate = base['key_rate'] * activity_factor * 0.85 + np.random.normal(0, 70)
            error_count = np.random.randint(1, 5)
            status = 'Warning'

        elif condition == 'critical':
            cpu = 85 + np.random.normal(0, 5)
            memory = 92 + np.random.normal(0, 3)
            temperature = 48 + np.random.normal(0, 2)
            key_gen_rate = base['key_rate'] * 0.5 + np.random.normal(0, 100)
            error_count = np.random.randint(5, 20)
            status = 'Critical'

        else:  # degrading
            # Gradual degradation over time
            degradation = (timestamp - self.start_date).total_seconds() / (self.days * 86400)
            cpu = base['cpu'] * (1 + degradation * 0.5) + np.random.normal(0, 5)
            memory = base['mem'] * (1 + degradation * 0.3) + np.random.normal(0, 3)
            temperature = base['temp'] * (1 + degradation * 0.2) + np.random.normal(0, 2)
            key_gen_rate = base['key_rate'] * (1 - degradation * 0.3) + np.random.normal(0, 50)
            error_count = int(degradation * 10)
            status = 'Warning' if degradation < 0.7 else 'Critical'

        # Ensure values are within realistic bounds
        cpu = np.clip(cpu, 10, 100)
        memory = np.clip(memory, 20, 100)
        temperature = np.clip(temperature, 20, 60)
        key_gen_rate = np.clip(key_gen_rate, 100, 2000)

        # Calculate aggregated key rate (will be updated with link data)
        aggregated_rate = key_gen_rate * np.random.uniform(1.8, 2.2)

        return {
            'node_name': node_name,
            'timestamp': timestamp,
            'cpu_usage': round(cpu, 2),
            'memory_usage': round(memory, 2),
            'temperature': round(temperature, 2),
            'power_consumption': round(100 + cpu * 0.5 + np.random.normal(0, 5), 2),
            'key_generation_rate': round(key_gen_rate, 2),
            'aggregated_key_rate': round(aggregated_rate, 2),
            'raw_key_rate': round(key_gen_rate * 12, 2),
            'packet_loss': round(np.random.uniform(0, 2), 3),
            'latency': round(np.random.uniform(1, 10), 2),
            'throughput': round(np.random.uniform(800, 1000), 2),
            'operational_status': status,
            'error_count': error_count,
            'warning_count': error_count // 2
        }

    def generate_link_metrics(self, link_info, timestamp, condition='normal'):
        """
        Generate link metrics based on operational condition

        Args:
            link_info: Tuple of (link_name, source, dest, distance)
            timestamp: Timestamp for the metrics
            condition: 'normal', 'warning', 'critical', or 'degrading'
        """
        link_name, source, dest, distance = link_info

        # Distance affects base metrics
        distance_factor = 1 + (distance - 10) * 0.02

        # Time-based variations
        hour = timestamp.hour
        environmental_noise = 1.0

        # Higher noise during day
        if 8 <= hour <= 18:
            environmental_noise = 1.15

        # Generate based on condition
        if condition == 'normal':
            qber = 0.015 + distance * 0.001 * environmental_noise + np.random.uniform(0, 0.01)
            visibility = 0.95 - distance * 0.002 + np.random.normal(0, 0.02)
            key_rate = 1400 / distance_factor + np.random.normal(0, 50)
            laser_power = 10 + np.random.normal(0, 0.5)
            photon_loss = 15 * distance_factor + np.random.normal(0, 3)
            quality = 'Excellent' if qber < 0.02 else 'Good'
            alert = 'Normal'

        elif condition == 'warning':
            qber = 0.035 + np.random.uniform(0, 0.008)
            visibility = 0.88 + np.random.normal(0, 0.03)
            key_rate = 950 + np.random.normal(0, 80)
            laser_power = 9 + np.random.normal(0, 0.8)
            photon_loss = 30 + np.random.normal(0, 5)
            quality = 'Fair'
            alert = 'Warning'

        elif condition == 'critical':
            qber = 0.045 + np.random.uniform(0, 0.015)
            visibility = 0.82 + np.random.normal(0, 0.03)
            key_rate = 700 + np.random.normal(0, 100)
            laser_power = 8 + np.random.normal(0, 1)
            photon_loss = 45 + np.random.normal(0, 8)
            quality = 'Poor'
            alert = 'Critical'

        else:  # degrading
            degradation = (timestamp - self.start_date).total_seconds() / (self.days * 86400)
            qber = 0.015 + degradation * 0.03 + np.random.uniform(0, 0.01)
            visibility = 0.95 - degradation * 0.15 + np.random.normal(0, 0.02)
            key_rate = 1400 * (1 - degradation * 0.5) + np.random.normal(0, 50)
            laser_power = 10 * (1 - degradation * 0.2) + np.random.normal(0, 0.5)
            photon_loss = 15 + degradation * 30 + np.random.normal(0, 3)
            quality = 'Good' if degradation < 0.3 else ('Fair' if degradation < 0.7 else 'Poor')
            alert = 'Normal' if degradation < 0.3 else ('Warning' if degradation < 0.7 else 'Critical')

        # Ensure values are within bounds
        qber = np.clip(qber, 0.001, 0.1)
        visibility = np.clip(visibility, 0.7, 0.99)
        key_rate = np.clip(key_rate, 100, 2000)
        laser_power = np.clip(laser_power, 5, 15)
        photon_loss = np.clip(photon_loss, 5, 60)

        return {
            'link_name': link_name,
            'timestamp': timestamp,
            'qber': round(qber, 6),
            'visibility': round(visibility, 4),
            'key_rate': round(key_rate, 2),
            'laser_power': round(laser_power, 2),
            'attenuation': round(distance * 0.8 + np.random.normal(0, 1), 2),
            'photon_loss': round(photon_loss, 2),
            'detector_efficiency': round(0.1 + np.random.uniform(-0.02, 0.02), 3),
            'dark_count_rate': round(45 + np.random.normal(0, 10), 2),
            'raw_key_rate': round(key_rate * 15, 2),
            'sifted_key_rate': round(key_rate * 7, 2),
            'secure_key_rate': round(key_rate * 0.9, 2),
            'error_correction_efficiency': round(1.2 + np.random.uniform(-0.1, 0.1), 3),
            'privacy_amplification_ratio': round(0.85 + np.random.uniform(-0.05, 0.05), 3),
            'link_quality': quality,
            'signal_to_noise_ratio': round(20 - qber * 100, 2),
            'bit_error_count': int(qber * 1000000),
            'alert_status': alert
        }

    def generate_time_series(self):
        """Generate complete time series data"""
        print(f"\nGenerating {self.days} days of data...")
        print(f"Start: {self.start_date}")
        print(f"End: {self.end_date}")

        # Generate timestamps (every 15 minutes)
        timestamps = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='15min'
        )

        total_records = len(timestamps)
        print(f"Total timestamps: {total_records}")

        node_metrics_data = []
        link_metrics_data = []

        # Define condition periods
        for i, ts in enumerate(timestamps):
            if i % 100 == 0:
                print(f"  Processing {i}/{total_records} ({i*100/total_records:.1f}%)...")

            # Determine condition based on various patterns
            rand = np.random.random()

            # 85% normal, 10% warning, 5% critical
            if rand < 0.85:
                condition = 'normal'
            elif rand < 0.95:
                condition = 'warning'
            else:
                condition = 'critical'

            # Add some degradation periods
            if self.days > 7:
                # Last 3 days show gradual degradation for QKD_003
                if ts > (self.end_date - timedelta(days=3)) and 'QKD_003' in self.nodes:
                    if np.random.random() < 0.3:
                        condition = 'degrading'

            # Generate metrics for each node
            for node in self.nodes:
                # Node-specific conditions
                node_condition = condition
                if node == 'QKD_003' and condition == 'normal' and np.random.random() < 0.1:
                    node_condition = 'warning'  # QKD_003 has more issues

                node_metrics = self.generate_node_metrics(node, ts, node_condition)
                node_metrics_data.append(node_metrics)

            # Generate metrics for each link
            for link_info in self.links:
                # Links involving QKD_003 have more issues
                link_condition = condition
                if 'QKD_003' in link_info[0] and condition == 'normal' and np.random.random() < 0.15:
                    link_condition = 'warning'

                link_metrics = self.generate_link_metrics(link_info, ts, link_condition)
                link_metrics_data.append(link_metrics)

        return node_metrics_data, link_metrics_data

    def insert_data(self, node_metrics_data, link_metrics_data):
        """Insert generated data into database"""
        print("\nInserting data into database...")
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Insert node metrics
            print(f"  Inserting {len(node_metrics_data)} node metrics records...")
            for i, metrics in enumerate(node_metrics_data):
                if i % 1000 == 0:
                    print(f"    {i}/{len(node_metrics_data)}...")

                cursor.execute("""
                    INSERT INTO node_metrics
                    (node_name, timestamp, cpu_usage, memory_usage, temperature,
                     power_consumption, key_generation_rate, aggregated_key_rate,
                     raw_key_rate, packet_loss, latency, throughput,
                     operational_status, error_count, warning_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, tuple(metrics.values()))

            # Insert link metrics
            print(f"  Inserting {len(link_metrics_data)} link metrics records...")
            for i, metrics in enumerate(link_metrics_data):
                if i % 1000 == 0:
                    print(f"    {i}/{len(link_metrics_data)}...")

                cursor.execute("""
                    INSERT INTO link_metrics
                    (link_name, timestamp, qber, visibility, key_rate,
                     laser_power, attenuation, photon_loss, detector_efficiency,
                     dark_count_rate, raw_key_rate, sifted_key_rate, secure_key_rate,
                     error_correction_efficiency, privacy_amplification_ratio,
                     link_quality, signal_to_noise_ratio, bit_error_count, alert_status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, tuple(metrics.values()))

            # Update aggregated key rates (skip if procedure doesn't exist)
            # print("  Updating aggregated key rates...")
            # for node in self.nodes:
            #     cursor.execute(f"CALL update_aggregated_key_rate('{node}')")

            conn.commit()
            print("[OK] Data inserted successfully")

        except Exception as e:
            print(f"[FAIL] Error inserting data: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def generate_sample_alerts(self):
        """Generate some sample alerts based on the data"""
        print("\nGenerating sample alerts...")
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Find high QBER events
            cursor.execute("""
                INSERT INTO alerts (entity_type, entity_name, timestamp, severity,
                                  alert_type, metric_name, message, current_value,
                                  threshold_value, threshold_type)
                SELECT 'link', link_name, timestamp, 'Critical',
                       'High QBER', 'qber', 'QBER exceeded critical threshold',
                       qber, 0.04, 'max'
                FROM link_metrics
                WHERE qber > 0.04
                ORDER BY timestamp DESC
                LIMIT 5
            """)

            # Find high temperature events
            cursor.execute("""
                INSERT INTO alerts (entity_type, entity_name, timestamp, severity,
                                  alert_type, metric_name, message, current_value,
                                  threshold_value, threshold_type)
                SELECT 'node', node_name, timestamp, 'Warning',
                       'High Temperature', 'temperature', 'Node temperature above threshold',
                       temperature, 45, 'max'
                FROM node_metrics
                WHERE temperature > 45
                ORDER BY timestamp DESC
                LIMIT 5
            """)

            conn.commit()
            print("[OK] Sample alerts generated")

        except Exception as e:
            print(f"[FAIL] Error generating alerts: {e}")
            conn.rollback()
        finally:
            conn.close()

def main():
    """Main execution"""
    print("\n" + "="*60)
    print(" SYNTHETIC DATA GENERATOR FOR ML DATABASE")
    print("="*60)

    # Get parameters
    days = 30  # Generate 30 days of data

    generator = SyntheticDataGenerator(days=days)

    # Clear existing data
    generator.clear_existing_data()

    # Generate time series data
    node_metrics, link_metrics = generator.generate_time_series()

    print(f"\nGenerated:")
    print(f"  * {len(node_metrics)} node metrics records")
    print(f"  * {len(link_metrics)} link metrics records")

    # Insert into database
    generator.insert_data(node_metrics, link_metrics)

    # Generate sample alerts
    generator.generate_sample_alerts()

    print("\n" + "="*60)
    print(" DATA GENERATION COMPLETE")
    print("="*60)
    print("\n[SUCCESS] ML database populated with synthetic data")
    print("\nNext steps:")
    print("1. Run test_ml_connection.py to verify data")
    print("2. Run train_fault_models_improved.py to train models")

if __name__ == "__main__":
    main()