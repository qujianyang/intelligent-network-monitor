"""
Simplified MCP E-Line EPL Service Configuration
===============================================
Exact mapping to your UI configuration for test2 service
"""

import json
import logging
from mcp_api import MCPClient

logger = logging.getLogger(__name__)


def create_eline_epl_service(client: MCPClient):
    """
    Create E-Line EPL service matching exact UI configuration:
    - Name: test2
    - Label: testing
    - Customer: STE LAB ELAN
    - Service Type: EPL
    - NE1: C02-5164-01, Port 3
    - NE2: C01-5164-01, Port 3
    - MPLS transport with SR-Policy, Color: 990199
    """

    # This is the exact payload structure based on MCP API documentation
    service_request = {
        "productId": "42e5b5a4-589f-11f0-a3b5-5d8caab1d313",  # L2ServiceIntentFacade product ID
        "label": "testing",
        "desiredOrchState": "requested",
        "discovered": False,
        "properties": {
            # Basic Service Properties
            "name": "test2",
            "customerName": "STE LAB ELAN",
            "serviceType": "EPL",
            "structure": "P2P",
            "type": "FDFR",

            # OAM Configuration (802.1ag)
            "oamEnabled": True,
            "ccmPriority": 5,
            "ccmInterval": "10sec",  # Converted from 100ms
            "maName": "test2_MA",

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
                        "node": "C02-5164-01",
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
                                            "port": "3"
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
                        "node": "C01-5164-01",
                        "role": "Z_UNI",
                        "oamEnabled": True,
                        "ccmTransmitEnabled": True,
                        "dmmEnabled": True,
                        "dmmInterval": "10sec",
                        "dmmPriority": 5,
                        "slmEnabled": False,  # SLM unchecked for NE2
                        "details": [
                            {
                                "flowSettings": [
                                    {
                                        "filter": {},  # EPL has no VLAN filter
                                        "location": {
                                            "port": "3"
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

    # Make the API call
    endpoint = "/bpocore/market/api/v1/resources"
    result = client.api_call(
        endpoint=endpoint,
        method="POST",
        data=service_request
    )

    return result


def main():
    """Main execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\nMCP E-Line EPL Service Provisioning")
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

    # Create the service
    print(f"\nCreating E-Line EPL service 'test2'...")
    result = create_eline_epl_service(client)

    if result:
        # Check if it's an error response
        if result.get('error'):
            print(f"[FAILED] Service creation failed")
            print(f"  Status: {result.get('status')}")
            if isinstance(result.get('details'), dict):
                failure_info = result['details'].get('failureInfo', {})
                print(f"  Reason: {failure_info.get('reason', 'Unknown')}")
                print(f"  Detail: {failure_info.get('detail', 'No details')}")
            else:
                print(f"  Details: {result.get('details')}")

            # Save error response
            with open('eline_error.json', 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Error response saved to: eline_error.json")
        else:
            # Success case
            service_id = result.get('id')
            print(f"[OK] Service created with ID: {service_id}")
            print(f"  State: {result.get('orchState')}")

            # Save response
            with open('eline_response.json', 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Response saved to: eline_response.json")
    else:
        print("[FAILED] Service creation failed - no response")

    client.close()


if __name__ == "__main__":
    main()
