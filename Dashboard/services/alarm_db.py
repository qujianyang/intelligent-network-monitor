"""
Standalone Database Connector for Alarm Management
==================================================
Provides direct MySQL access for alarm/fault data without Flask dependencies.
Compatible with AI agent execution outside Flask application context.
"""

import mysql.connector
import logging
import os
from dotenv import load_dotenv
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

load_dotenv()
logger = logging.getLogger(__name__)

class AlarmDBConnector:
    """Standalone MySQL connector for alarm/fault data"""

    def __init__(self):
        """Initialize using same DB config as GUI"""
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = int(os.getenv('DB_PORT', '3306'))
        self.database = os.getenv('DB_NAME', 'qkd')
        self.user = os.getenv('DB_USER', 'root')
        self.password = os.getenv('DB_PASSWORD', '')

        logger.info(f"Initialized alarm DB connector: {self.host}:{self.port}/{self.database}")

    def _get_connection(self):
        """Create new MySQL connection"""
        return mysql.connector.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database
        )

    @contextmanager
    def get_cursor(self, dictionary=True):
        """Context manager for database operations"""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=dictionary)
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            cursor.close()
            conn.close()

    def get_alarm_list(self) -> List[Dict[str, Any]]:
        """
        Get all faults/alarms with enriched data.

        Replicates the logic from GUI/services/fault_service.py:get_alarm_list()
        but works without Flask application context.

        Returns:
            List of fault dictionaries with enriched component and user names
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT * FROM faults;")
                faults = cursor.fetchall()

                # Enrich each fault with human-readable names
                for fault in faults:
                    # Get component name (Link or Node)
                    if fault['component_type'] == 'Link':
                        cursor.execute("SELECT link_id FROM links WHERE id = %s;", (fault['component_id'],))
                        link = cursor.fetchone()
                        fault['component'] = link['link_id'] if link else 'Unknown'
                    else:
                        cursor.execute("SELECT node_id FROM nodes WHERE id = %s;", (fault['component_id'],))
                        node = cursor.fetchone()
                        fault['component'] = node['node_id'] if node else 'Unknown'

                    # Get acknowledged_by user name
                    if fault['acknowledged_by'] is not None:
                        cursor.execute("SELECT name FROM users WHERE uuid = %s;", (fault['acknowledged_by'],))
                        user = cursor.fetchone()
                        fault['acknowledged_by'] = user['name'] if user else "Not acknowledged"
                    else:
                        fault['acknowledged_by'] = "Not acknowledged"

                    # Get resolved_by user name
                    if fault['resolved_by'] is not None:
                        cursor.execute("SELECT name FROM users WHERE uuid = %s;", (fault['resolved_by'],))
                        user = cursor.fetchone()
                        fault['resolved_by'] = user['name'] if user else "Not resolved"
                    else:
                        fault['resolved_by'] = "Not resolved"

                    # Get assigned_to user name
                    if fault['assigned_to'] is not None:
                        cursor.execute("SELECT name FROM users WHERE uuid = %s;", (fault['assigned_to'],))
                        user = cursor.fetchone()
                        fault['assigned_to_uuid'] = fault['assigned_to']
                        fault['assigned_to'] = user['name'] if user else "Not assigned"
                    else:
                        fault['assigned_to'] = "Not assigned"

                logger.info(f"Retrieved {len(faults)} alarms from database")
                return faults

        except Exception as e:
            logger.error(f"Failed to get alarm list: {e}", exc_info=True)
            return []


# Global connector instance
_alarm_connector = None

def get_alarm_connector() -> AlarmDBConnector:
    """Get global alarm database connector instance"""
    global _alarm_connector
    if _alarm_connector is None:
        _alarm_connector = AlarmDBConnector()
        logger.info("Initialized global alarm database connector")
    return _alarm_connector

def get_alarm_list_standalone() -> List[Dict[str, Any]]:
    """
    Get alarm list without Flask context (for AI agent).

    This is the main function to use from Dashboard services.
    Works independently of Flask application context.

    Returns:
        List of fault dictionaries with all enriched data
    """
    connector = get_alarm_connector()
    return connector.get_alarm_list()

def acknowledge_fault_standalone(fault_id: str, acknowledged_by_uuid: str,
                                 assigned_to_uuid: str, notes: str) -> Dict[str, Any]:
    """
    Acknowledge a fault without Flask context (for AI agent).

    Validates permissions and updates fault status following same logic as GUI.

    Args:
        fault_id: The fault ID to acknowledge (string like 'FLT_005')
        acknowledged_by_uuid: UUID of user performing acknowledgement
        assigned_to_uuid: UUID of user to assign the fault to
        notes: Notes about the acknowledgement

    Returns:
        Dict with 'success' (bool) and 'message' or 'error' (str)
    """
    try:
        connector = get_alarm_connector()

        with connector.get_cursor() as cursor:
            # 1. Check if fault exists and is not already Open
            cursor.execute("SELECT id, fault_id, status FROM faults WHERE fault_id = %s;", (fault_id,))
            fault = cursor.fetchone()

            if not fault:
                logger.warning(f"Fault {fault_id} not found")
                return {"success": False, "error": f"Fault {fault_id} does not exist"}

            if fault['status'] != 'Open':
                logger.warning(f"Fault {fault_id} status is {fault['status']}, must be Open to acknowledge")
                return {"success": False, "error": f"You can only acknowledge an open fault. Current status: {fault['status']}"}

            # 2. Validate acknowledger has fault.acknowledge permission
            cursor.execute("""
                SELECT users.name, users.uuid
                FROM users
                JOIN roles ON users.role_id = roles.id
                JOIN roles_permissions ON roles.id = roles_permissions.role_id
                JOIN permissions ON roles_permissions.permission_id = permissions.id
                WHERE permissions.permission_name = 'fault.acknowledge'
                AND users.uuid = %s
                AND users.status IN ('Online', 'Offline', 'Locked')
                AND roles.role_name != 'Super Admin'
            """, (acknowledged_by_uuid,))

            acknowledger = cursor.fetchone()
            if not acknowledger:
                logger.warning(f"User {acknowledged_by_uuid} does not have fault.acknowledge permission")
                return {"success": False, "error": "You do not have permission to acknowledge faults"}

            # 3. Validate assignee has fault.resolve permission
            cursor.execute("""
                SELECT users.name, users.uuid
                FROM users
                JOIN roles ON users.role_id = roles.id
                JOIN roles_permissions ON roles.id = roles_permissions.role_id
                JOIN permissions ON roles_permissions.permission_id = permissions.id
                WHERE permissions.permission_name = 'fault.resolve'
                AND users.uuid = %s
                AND users.status IN ('Online', 'Offline', 'Locked')
                AND roles.role_name != 'Super Admin'
            """, (assigned_to_uuid,))

            assignee = cursor.fetchone()
            if not assignee:
                logger.warning(f"User {assigned_to_uuid} does not have fault.resolve permission")
                return {"success": False, "error": "The user you are trying to assign does not have permission to resolve faults"}

            # 4. Update fault record
            cursor.execute("""
                UPDATE faults
                SET status = 'Ongoing',
                    acknowledged_at = NOW(),
                    acknowledged_by = %s,
                    assigned_to = %s
                WHERE fault_id = %s
            """, (acknowledged_by_uuid, assigned_to_uuid, fault_id))

            # 5. Log the action (using correct column order from fixed fault_log function)
            cursor.execute("""
                INSERT INTO fault_log (timestamp, action_type, fault_id, performed_by, outcome, notes)
                VALUES (NOW(), %s, %s, %s, %s, %s)
            """, ('Acknowledged', fault['id'], acknowledged_by_uuid, 'Success', notes))

            logger.info(f"Fault {fault_id} acknowledged by {acknowledger['name']} and assigned to {assignee['name']}")
            return {
                "success": True,
                "message": f"Fault {fault_id} acknowledged successfully and assigned to {assignee['name']}"
            }

    except Exception as e:
        logger.error(f"Failed to acknowledge fault {fault_id}: {e}", exc_info=True)
        return {"success": False, "error": f"Database error: {str(e)}"}
