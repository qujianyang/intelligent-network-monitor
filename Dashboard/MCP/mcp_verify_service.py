"""
Verify E-Line Service Status
=============================
Check the status of the created E-Line service
"""

import json
import logging
import sys
from mcp_api import MCPClient

logger = logging.getLogger(__name__)


def check_service_status(client: MCPClient, service_id: str):
    """Check the status of a specific service"""

    endpoint = f"/bpocore/market/api/v1/resources/{service_id}"

    print(f"\nQuerying service: {service_id}")
    result = client.api_call(endpoint, method="GET")

    if result and not result.get('error'):
        print(f"[OK] Service found")

        # Extract key information
        orch_state = result.get('orchState', 'unknown')
        desired_state = result.get('desiredOrchState', 'unknown')
        label = result.get('label', 'N/A')

        props = result.get('properties', {})
        service_name = props.get('name', 'N/A')
        service_type = props.get('serviceType', 'N/A')

        # Display status
        print(f"\n  Service Details:")
        print(f"    Name: {service_name}")
        print(f"    Label: {label}")
        print(f"    Type: {service_type}")
        print(f"    Orchestration State: {orch_state}")
        print(f"    Desired State: {desired_state}")

        # Check endpoints
        endpoints = props.get('endpoints', [])
        if endpoints:
            print(f"\n  Endpoints ({len(endpoints)}):")
            for i, ep in enumerate(endpoints, 1):
                settings = ep.get('settings', {})
                node = settings.get('node', 'N/A')
                role = settings.get('role', 'N/A')
                print(f"    {i}. {node} ({role})")

        # Check if service is active
        if orch_state == 'active':
            print(f"\n[SUCCESS] Service is ACTIVE!")
        elif orch_state == 'requested':
            print(f"\n[PENDING] Service is being provisioned...")
        else:
            print(f"\n[STATUS] Service orchestration state: {orch_state}")

        # Save full response
        with open('service_status.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n  Full status saved to: service_status.json")

        return result
    elif result and result.get('error'):
        print(f"[FAILED] Error querying service")
        print(f"  Status: {result.get('status')}")
        print(f"  Details: {result.get('details')}")
    else:
        print(f"[FAILED] No response")

    return None


def main():
    """Main execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\nMCP E-Line Service Verification")
    print("="*50)

    # Get service ID from command line or use the one we just created
    if len(sys.argv) > 1:
        service_id = sys.argv[1]
    else:
        # Use the service ID from the last creation
        try:
            with open('eline_response.json', 'r') as f:
                data = json.load(f)
                service_id = data.get('id')
                if not service_id:
                    print("No service ID found in eline_response.json")
                    return
        except FileNotFoundError:
            print("Please provide service ID as argument or ensure eline_response.json exists")
            return

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

    # Check the service status
    check_service_status(client, service_id)

    client.close()


if __name__ == "__main__":
    main()