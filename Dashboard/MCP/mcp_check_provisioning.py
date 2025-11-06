"""
Check Service Provisioning Status
==================================
Monitor the provisioning progress of the service
"""

import json
import time
import logging
from mcp_api import MCPClient

logger = logging.getLogger(__name__)


def monitor_service(client: MCPClient, service_id: str, max_wait: int = 60):
    """Monitor service provisioning status"""

    endpoint = f"/bpocore/market/api/v1/resources/{service_id}"
    start_time = time.time()
    last_state = None

    print(f"\nMonitoring service: {service_id}")
    print("Waiting for provisioning to complete...")
    print("-" * 50)

    while time.time() - start_time < max_wait:
        result = client.api_call(endpoint, method="GET")

        if result and not result.get('error'):
            orch_state = result.get('orchState', 'unknown')
            update_state = result.get('updateState', 'unknown')
            reason = result.get('reason', '')
            update_reason = result.get('updateReason', '')

            if orch_state != last_state:
                elapsed = int(time.time() - start_time)
                print(f"[{elapsed}s] State: {orch_state}")

                if update_state and update_state != 'unknown':
                    print(f"       Update State: {update_state}")
                if reason:
                    print(f"       Reason: {reason}")
                if update_reason:
                    print(f"       Update Reason: {update_reason}")

                last_state = orch_state

            # Check if provisioning is complete
            if orch_state == 'active':
                print("\n" + "="*50)
                print("[SUCCESS] Service is now ACTIVE!")
                print("="*50)

                # Get final details
                props = result.get('properties', {})
                print(f"\nService Details:")
                print(f"  Name: {props.get('name')}")
                print(f"  Type: {props.get('serviceType')}")
                print(f"  Customer: {props.get('customerName')}")

                # Check endpoints
                endpoints = props.get('endpoints', [])
                if endpoints:
                    print(f"\nEndpoints:")
                    for i, ep in enumerate(endpoints, 1):
                        settings = ep.get('settings', {})
                        node = settings.get('node', 'N/A')
                        role = settings.get('role', 'N/A')
                        details = settings.get('details', [{}])[0]
                        flow = details.get('flowSettings', [{}])[0]
                        port = flow.get('location', {}).get('port', 'N/A')
                        print(f"  {i}. {node} Port {port} ({role})")

                return True

            elif orch_state == 'failed':
                print("\n[FAILED] Service provisioning failed!")
                print(f"  Reason: {reason or update_reason or 'Unknown'}")
                return False

        else:
            print(f"[ERROR] Failed to query service status")
            return False

        # Wait before next check
        time.sleep(2)

    print(f"\n[TIMEOUT] Service still in '{last_state}' state after {max_wait} seconds")
    return False


def main():
    """Main execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\nMCP Service Provisioning Monitor")
    print("="*50)

    # Use the service ID we created
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

    # Monitor the service
    monitor_service(client, service_id, max_wait=120)  # Wait up to 2 minutes

    client.close()


if __name__ == "__main__":
    main()