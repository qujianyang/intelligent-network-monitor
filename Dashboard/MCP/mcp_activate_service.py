"""
Activate E-Line Service
========================
Change service state from requested to active
"""

import json
import logging
from mcp_api import MCPClient

logger = logging.getLogger(__name__)


def activate_service(client: MCPClient, service_id: str):
    """Attempt to activate a service"""

    # First, get current service details
    endpoint = f"/bpocore/market/api/v1/resources/{service_id}"
    result = client.api_call(endpoint, method="GET")

    if not result or result.get('error'):
        print(f"[FAILED] Could not retrieve service")
        return False

    print(f"\nCurrent State: {result.get('orchState')}")

    # Try to update the desired state to active
    update_payload = {
        "desiredOrchState": "active"
    }

    print(f"\nAttempting to activate service...")
    update_result = client.api_call(
        endpoint=endpoint,
        method="PATCH",
        data=update_payload
    )

    if update_result and not update_result.get('error'):
        print(f"[OK] Activation request sent")
        new_state = update_result.get('orchState', 'unknown')
        print(f"New State: {new_state}")
        return True
    else:
        print(f"[FAILED] Could not activate service")
        if update_result:
            print(f"  Error: {update_result.get('details')}")
        return False


def calculate_routes(client: MCPClient, service_id: str):
    """Try to calculate routes for the service"""

    # Try route calculation endpoint
    route_endpoint = f"/bpocore/market/api/v1/resources/{service_id}/routes"

    print(f"\nAttempting to calculate routes...")
    route_result = client.api_call(
        endpoint=route_endpoint,
        method="POST",
        data={}
    )

    if route_result and not route_result.get('error'):
        print(f"[OK] Route calculation initiated")
        return True
    else:
        print(f"[INFO] Route calculation not available via API")
        if route_result and route_result.get('details'):
            print(f"  Details: {route_result.get('details')}")
        return False


def main():
    """Main execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\nMCP Service Activation")
    print("="*50)

    service_id = "a86d6a83-b0b4-11f0-ab20-0d83d91152c5"

    # Initialize and login
    client = MCPClient(
        base_url="https://10.1.1.3",
        username="admin",
        password="adminpw"
    )

    if not client.login():
        print("Login failed!")
        return

    print(f"[OK] Logged in successfully")
    print(f"\nService ID: {service_id}")

    # Try to calculate routes first
    calculate_routes(client, service_id)

    # Try to activate the service
    activate_service(client, service_id)

    client.close()


if __name__ == "__main__":
    main()