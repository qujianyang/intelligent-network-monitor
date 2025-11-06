"""
List All Services in MCP
=========================
Query and display all services to find our created service
"""

import json
import logging
from mcp_api import MCPClient

logger = logging.getLogger(__name__)


def list_all_services(client: MCPClient):
    """List all services from MCP"""

    endpoint = "/bpocore/market/api/v1/resources"

    # Query with larger page size
    params = {
        "pageSize": "100"
    }

    print(f"\nQuerying all services...")
    result = client.api_call(endpoint, method="GET", params=params)

    if result and not result.get('error'):
        items = result.get('items', [])
        print(f"[OK] Found {len(items)} services\n")

        # Group services by type
        services_by_type = {}

        for item in items:
            props = item.get('properties', {})
            service_type = props.get('serviceType', 'Unknown')
            service_name = props.get('name', item.get('label', 'N/A'))
            service_id = item.get('id', 'N/A')
            orch_state = item.get('orchState', 'unknown')
            resource_type = item.get('resourceTypeId', 'unknown')

            if service_type not in services_by_type:
                services_by_type[service_type] = []

            services_by_type[service_type].append({
                'name': service_name,
                'id': service_id,
                'state': orch_state,
                'resourceType': resource_type
            })

        # Display services grouped by type
        for svc_type, services in services_by_type.items():
            print(f"\n{svc_type} Services ({len(services)}):")
            print("-" * 50)
            for svc in services[:10]:  # Show first 10 of each type
                print(f"  Name: {svc['name']}")
                print(f"    ID: {svc['id']}")
                print(f"    State: {svc['state']}")
                print(f"    Resource Type: {svc['resourceType']}")
                print()

        # Look specifically for our test2 service
        print("\n" + "="*50)
        print("Searching for 'test2' service...")
        found = False
        for item in items:
            props = item.get('properties', {})
            if props.get('name') == 'test2' or item.get('label') == 'testing':
                found = True
                print(f"\n[FOUND] test2 service:")
                print(f"  ID: {item.get('id')}")
                print(f"  Label: {item.get('label')}")
                print(f"  Service Type: {props.get('serviceType')}")
                print(f"  Resource Type: {item.get('resourceTypeId')}")
                print(f"  Orch State: {item.get('orchState')}")
                print(f"  Properties: {json.dumps(props, indent=4)}")

        if not found:
            print("  'test2' service NOT found in the response")

        # Save full response
        with open('all_services.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nFull response saved to: all_services.json")

        return result
    elif result and result.get('error'):
        print(f"[FAILED] Error querying services")
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

    print("\nMCP Service List")
    print("="*50)

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

    # List all services
    list_all_services(client)

    client.close()


if __name__ == "__main__":
    main()