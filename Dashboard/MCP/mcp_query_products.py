"""
Query MCP for available products and service types
===================================================
Helps identify the correct product ID for E-Line services
"""

import json
import logging
from mcp_api import MCPClient

logger = logging.getLogger(__name__)


def query_products(client: MCPClient):
    """Query available products from MCP"""

    # Try different endpoints to find products
    endpoints = [
        "/bpocore/market/api/v1/products",
        "/bpocore/market/api/v1/productCatalogue",
        "/bpocore/market/api/v1/catalog/products",
        "/bpocore/api/v1/products",
        "/tron/api/v1/products"
    ]

    for endpoint in endpoints:
        print(f"\nTrying endpoint: {endpoint}")
        result = client.api_call(endpoint, method="GET")

        if result and not result.get('error'):
            print(f"[OK] Found products at {endpoint}")
            return result, endpoint
        elif result and result.get('error'):
            print(f"  Error {result.get('status')}: {result.get('details')}")
        else:
            print(f"  No response")

    return None, None


def query_services(client: MCPClient):
    """Query existing services to understand the structure"""

    endpoint = "/bpocore/market/api/v1/resources"
    print(f"\nQuerying existing services at: {endpoint}")

    result = client.api_call(endpoint, method="GET")

    if result and not result.get('error'):
        print(f"[OK] Found services")
        return result
    elif result and result.get('error'):
        print(f"  Error {result.get('status')}: {result.get('details')}")
    else:
        print(f"  No response")

    return None


def main():
    """Main execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\nMCP Product Discovery")
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

    print(f"[OK] Logged in successfully\n")

    # Query products
    print("Searching for available products...")
    products, endpoint = query_products(client)

    if products:
        print(f"\nProducts found:")

        # Save products
        with open('mcp_products.json', 'w') as f:
            json.dump(products, f, indent=2)
        print(f"  Products saved to: mcp_products.json")

        # Try to extract L2 service products
        if isinstance(products, list):
            l2_products = [p for p in products if 'L2' in str(p) or 'Ethernet' in str(p) or 'EPL' in str(p)]
            if l2_products:
                print(f"\n  Found {len(l2_products)} potential L2 service products")
                for p in l2_products[:5]:  # Show first 5
                    print(f"    - {p.get('id', 'N/A')}: {p.get('name', 'N/A')}")
        elif isinstance(products, dict) and 'data' in products:
            # Handle different response structure
            data = products['data']
            if isinstance(data, list):
                for item in data[:10]:  # Show first 10
                    if isinstance(item, dict):
                        print(f"  - ID: {item.get('id', 'N/A')}")
                        attrs = item.get('attributes', {})
                        print(f"    Name: {attrs.get('name', 'N/A')}")
                        print(f"    Type: {attrs.get('type', item.get('type', 'N/A'))}")
    else:
        print("No products found through standard endpoints")

    # Query existing services
    print("\n" + "="*50)
    print("Querying existing services...")
    services = query_services(client)

    if services:
        with open('mcp_services.json', 'w') as f:
            json.dump(services, f, indent=2)
        print(f"  Services saved to: mcp_services.json")

        # Try to find EPL services
        if isinstance(services, dict) and 'data' in services:
            data = services['data']
            if isinstance(data, list):
                epl_services = []
                for item in data[:20]:  # Check first 20
                    if isinstance(item, dict):
                        attrs = item.get('attributes', {})
                        props = attrs.get('properties', {})
                        if props.get('serviceType') == 'EPL' or 'EPL' in str(attrs):
                            epl_services.append(item)
                            print(f"\n  Found EPL service:")
                            print(f"    ID: {item.get('id')}")
                            print(f"    Product ID: {attrs.get('productId', 'N/A')}")
                            print(f"    Name: {props.get('name', 'N/A')}")
                            print(f"    Service Type: {props.get('serviceType', 'N/A')}")

                if epl_services:
                    print(f"\n  Total EPL services found: {len(epl_services)}")
                    # Extract unique product IDs
                    product_ids = set()
                    for svc in epl_services:
                        pid = svc.get('attributes', {}).get('productId')
                        if pid:
                            product_ids.add(pid)

                    if product_ids:
                        print(f"\n  Unique Product IDs used by EPL services:")
                        for pid in product_ids:
                            print(f"    - {pid}")
    else:
        print("No services found")

    client.close()


if __name__ == "__main__":
    main()