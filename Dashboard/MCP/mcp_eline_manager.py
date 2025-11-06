"""
E-Line Circuit Manager
=======================
Manages E-Line circuit creation, activation, and monitoring
"""

import json
import time
import logging
import os
from typing import Dict, Optional, Tuple
from datetime import datetime

# Handle imports for different execution contexts
try:
    from Dashboard.MCP.mcp_api import MCPClient
except ImportError:
    from mcp_api import MCPClient

logger = logging.getLogger(__name__)


class ELineManager:
    """Manages E-Line circuit creation workflow"""

    def __init__(self,
                 base_url: str = "https://10.1.1.3",
                 username: str = "admin",
                 password: str = "adminpw",
                 tenant: str = "master"):
        """Initialize E-Line manager with MCP credentials"""
        self.base_url = base_url
        self.username = username
        self.password = password
        self.tenant = tenant
        self.product_id = "42e5b5a4-589f-11f0-a3b5-5d8caab1d313"  # L2ServiceIntentFacade

    def get_next_service_name(self) -> str:
        """
        Generate the next available service name (test1, test2, test3, etc.)

        Returns:
            Next available service name
        """
        try:
            # Create client and get all services
            client = MCPClient(self.base_url, self.username, self.password, self.tenant)

            if not client.login():
                logger.error("Failed to login to MCP")
                return "test1"  # Default if can't connect

            # Query existing services
            endpoint = "/bpocore/market/api/v1/resources"
            params = {"pageSize": "200"}
            result = client.api_call(endpoint, method="GET", params=params)

            if not result or result.get('error'):
                logger.warning("Could not query existing services")
                return "test1"

            # Find all test services
            test_numbers = []
            items = result.get('items', [])

            for item in items:
                props = item.get('properties', {})
                name = props.get('name', '')

                # Check if name matches pattern testX
                if name.startswith('test'):
                    try:
                        # Extract number after 'test'
                        num_str = name[4:]
                        if num_str.isdigit():
                            test_numbers.append(int(num_str))
                    except:
                        pass

            client.close()

            # Find next available number
            if test_numbers:
                next_num = max(test_numbers) + 1
            else:
                next_num = 1

            return f"test{next_num}"

        except Exception as e:
            logger.error(f"Error generating service name: {e}")
            # Return timestamp-based name as fallback
            return f"test_{int(time.time()) % 10000}"

    def create_eline_service(self,
                           endpoint_a: str,
                           port_a: str,
                           endpoint_z: str,
                           port_z: str,
                           service_name: Optional[str] = None,
                           customer: str = "STE LAB ELAN") -> Tuple[bool, Dict]:
        """
        Create an E-Line EPL service.

        Args:
            endpoint_a: Source node (e.g., 'C01-5164-01')
            port_a: Source port (e.g., '3')
            endpoint_z: Destination node (e.g., 'C02-5164-01')
            port_z: Destination port (e.g., '4')
            service_name: Optional service name (auto-generated if None)
            customer: Customer name

        Returns:
            Tuple of (success, result_dict)
        """
        # Generate service name if not provided
        if not service_name:
            service_name = self.get_next_service_name()
            logger.info(f"Auto-generated service name: {service_name}")

        # Create MCP client
        client = MCPClient(self.base_url, self.username, self.password, self.tenant)

        if not client.login():
            return False, {"error": "Failed to login to MCP"}

        try:
            # Build service request payload
            service_request = {
                "productId": self.product_id,
                "label": f"{service_name}_label",
                "desiredOrchState": "requested",
                "discovered": False,
                "properties": {
                    # Basic Service Properties
                    "name": service_name,
                    "customerName": customer,
                    "serviceType": "EPL",
                    "structure": "P2P",
                    "type": "FDFR",

                    # OAM Configuration (802.1ag)
                    "oamEnabled": True,
                    "ccmPriority": 5,
                    "ccmInterval": "10sec",
                    "maName": f"{service_name}_MA",

                    # Route Metadata
                    "routeMeta": {
                        "originator": "BP2"
                    },

                    # Transport settings
                    "linearOnly": False,
                    "interactive_mode": "false",

                    # Endpoints
                    "endpoints": [
                        {
                            "settings": {
                                "node": endpoint_a,
                                "role": "A_UNI",
                                "oamEnabled": True,
                                "ccmTransmitEnabled": True,
                                "dmmEnabled": True,
                                "dmmInterval": "10sec",
                                "dmmPriority": 5,
                                "slmEnabled": True,
                                "slmInterval": "10sec",
                                "slmPriority": 1,
                                "details": [
                                    {
                                        "flowSettings": [
                                            {
                                                "filter": {},  # EPL has no VLAN filter
                                                "location": {
                                                    "port": port_a
                                                },
                                                "controlFrameTunneling": "disabled",
                                                "ingressCosSetting": {
                                                    "ingressCosPolicy": "L2PcpCos"
                                                }
                                            }
                                        ]
                                    }
                                ]
                            }
                        },
                        {
                            "settings": {
                                "node": endpoint_z,
                                "role": "Z_UNI",
                                "oamEnabled": True,
                                "ccmTransmitEnabled": True,
                                "dmmEnabled": True,
                                "dmmInterval": "10sec",
                                "dmmPriority": 5,
                                "slmEnabled": False,  # SLM only on A_UNI
                                "details": [
                                    {
                                        "flowSettings": [
                                            {
                                                "filter": {},  # EPL has no VLAN filter
                                                "location": {
                                                    "port": port_z
                                                },
                                                "controlFrameTunneling": "disabled",
                                                "ingressCosSetting": {
                                                    "ingressCosPolicy": "L2PcpCos"
                                                }
                                            }
                                        ]
                                    }
                                ]
                            }
                        }
                    ]
                }
            }

            # Create the service
            logger.info(f"Creating E-Line service '{service_name}'...")
            endpoint = "/bpocore/market/api/v1/resources"
            result = client.api_call(
                endpoint=endpoint,
                method="POST",
                data=service_request
            )

            if result and not result.get('error'):
                service_id = result.get('id')
                logger.info(f"Service created with ID: {service_id}")

                return True, {
                    "success": True,
                    "service_id": service_id,
                    "service_name": service_name,
                    "state": result.get('orchState', 'unknown')
                }
            else:
                error_msg = "Unknown error"
                if result and result.get('details'):
                    if isinstance(result['details'], dict):
                        failure_info = result['details'].get('failureInfo', {})
                        error_msg = failure_info.get('reason', 'Unknown reason')
                    else:
                        error_msg = str(result['details'])

                logger.error(f"Service creation failed: {error_msg}")
                return False, {
                    "success": False,
                    "error": error_msg
                }

        except Exception as e:
            logger.error(f"Exception during service creation: {e}")
            return False, {
                "success": False,
                "error": str(e)
            }
        finally:
            client.close()

    def activate_service(self, service_id: str, max_wait: int = 60) -> Tuple[bool, str]:
        """
        Activate a service and wait for it to become active.

        Args:
            service_id: Service ID to activate
            max_wait: Maximum seconds to wait for activation

        Returns:
            Tuple of (success, status_message)
        """
        client = MCPClient(self.base_url, self.username, self.password, self.tenant)

        if not client.login():
            return False, "Failed to login to MCP"

        try:
            # Update desired state to active
            endpoint = f"/bpocore/market/api/v1/resources/{service_id}"
            update_payload = {"desiredOrchState": "active"}

            logger.info(f"Activating service {service_id}...")
            update_result = client.api_call(
                endpoint=endpoint,
                method="PATCH",
                data=update_payload
            )

            if not update_result or update_result.get('error'):
                return False, "Failed to send activation request"

            # Monitor activation progress
            start_time = time.time()
            last_state = None

            while time.time() - start_time < max_wait:
                result = client.api_call(endpoint, method="GET")

                if result and not result.get('error'):
                    orch_state = result.get('orchState', 'unknown')

                    if orch_state != last_state:
                        logger.info(f"Service state: {orch_state}")
                        last_state = orch_state

                    if orch_state == 'active':
                        return True, "Service activated successfully"

                    elif orch_state == 'failed':
                        reason = result.get('reason', 'Unknown reason')
                        return False, f"Service activation failed: {reason}"

                time.sleep(2)

            return False, f"Service still in '{last_state}' state after {max_wait} seconds"

        except Exception as e:
            logger.error(f"Exception during activation: {e}")
            return False, str(e)
        finally:
            client.close()

    def create_and_activate_circuit(self,
                                   endpoint_a: str,
                                   port_a: str,
                                   endpoint_z: str,
                                   port_z: str,
                                   service_name: Optional[str] = None,
                                   customer: str = "STE LAB ELAN") -> Dict:
        """
        Complete workflow: Create and activate an E-Line circuit.

        Returns:
            Dict with status and details
        """
        # Step 1: Create the service
        success, create_result = self.create_eline_service(
            endpoint_a, port_a, endpoint_z, port_z, service_name, customer
        )

        if not success:
            return {
                "success": False,
                "step": "creation",
                "error": create_result.get('error', 'Creation failed')
            }

        service_id = create_result.get('service_id')
        service_name = create_result.get('service_name')

        # Step 2: Activate the service
        logger.info("Waiting 3 seconds before activation...")
        time.sleep(3)  # Give MCP time to process creation

        activate_success, activate_msg = self.activate_service(service_id, max_wait=120)

        return {
            "success": activate_success,
            "service_id": service_id,
            "service_name": service_name,
            "endpoints": f"{endpoint_a}:{port_a} ↔ {endpoint_z}:{port_z}",
            "status": activate_msg,
            "step": "completed" if activate_success else "activation",
            "error": None if activate_success else activate_msg
        }


# Test the manager
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\nE-Line Manager Test")
    print("=" * 50)

    manager = ELineManager()

    # Test 1: Get next service name
    print("\n1. Testing service name generation...")
    name = manager.get_next_service_name()
    print(f"   Next available name: {name}")

    # Test 2: Create and activate a circuit
    print("\n2. Testing circuit creation and activation...")
    result = manager.create_and_activate_circuit(
        endpoint_a="C01-5164-01",
        port_a="3",
        endpoint_z="C02-5164-01",
        port_z="4"
    )

    if result['success']:
        print(f"   ✓ Circuit created successfully!")
        print(f"     Service: {result['service_name']}")
        print(f"     ID: {result['service_id']}")
        print(f"     Endpoints: {result['endpoints']}")
        print(f"     Status: {result['status']}")
    else:
        print(f"   ✗ Circuit creation failed at {result['step']}")
        print(f"     Error: {result['error']}")