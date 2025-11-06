"""
Ciena Blue Planet MCP API Integration - Simple Token Management
================================================================
Focused on token generation and authenticated API calls only.
"""

import requests
import json
import logging
from typing import Optional, Dict, Any
import warnings

# Suppress SSL warnings for self-signed certificates
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

logger = logging.getLogger(__name__)


class MCPClient:
    """Simple MCP client focused on token management and API calls."""

    def __init__(self,
                 base_url: str = "https://10.1.1.3",
                 username: str = "admin",
                 password: str = "adminpw",
                 tenant: str = "master"):
        """Initialize MCP client with credentials."""
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.password = password
        self.tenant = tenant
        self.token = None

        # Session for connection pooling
        self.session = requests.Session()
        self.session.verify = False  # Disable SSL verification for self-signed certs

        logger.info(f"MCP Client initialized for {self.base_url}")

    def login(self) -> bool:
        """
        Login to MCP and get token.

        Returns:
            bool: True if login successful and token obtained
        """
        try:
            # Login endpoint
            login_url = f"{self.base_url}/tron/api/v1/tokens"

            # Form data for login (matching your curl command)
            login_data = {
                'authType': 'password',
                'username': self.username,
                'password': self.password,
                'tenant': self.tenant
            }

            logger.info(f"Attempting login to {login_url}")

            # Make POST request with form data
            response = self.session.post(
                login_url,
                data=login_data,  # Form data, not JSON
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=30
            )

            # Check response
            if response.status_code in [200, 201]:
                result = response.json()

                # Check if login was successful
                if result.get('isSuccessful'):
                    # Extract and store token
                    self.token = result.get('token')

                    if self.token:
                        logger.info(f"Login successful! Token obtained: {self.token[:10]}...")

                        # Update session headers with token - use both formats for compatibility
                        # MCP alarm API uses 'Authorization: token {token}'
                        # MCP service API might use 'X-Auth-Token: {token}'
                        self.session.headers['Authorization'] = f'token {self.token}'
                        self.session.headers['X-Auth-Token'] = self.token
                        return True
                    else:
                        logger.error("Login response missing token")
                        return False
                else:
                    logger.error(f"Login failed: {result}")
                    return False
            else:
                logger.error(f"Login failed with status {response.status_code}: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    def api_call(self,
                 endpoint: str,
                 method: str = "GET",
                 params: Optional[Dict] = None,
                 data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make an authenticated API call to MCP.

        Args:
            endpoint: API endpoint (e.g., "/nsa/api/v1/alarms/alarmRecordsCounts")
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: Request body for POST/PUT

        Returns:
            Dict: Response JSON or None if failed
        """
        # Check if we have a token
        if not self.token:
            logger.warning("No token available, attempting login...")
            if not self.login():
                logger.error("Failed to obtain token")
                return None

        try:
            # Build full URL
            url = f"{self.base_url}{endpoint}"

            logger.info(f"Making {method} request to {url}")

            # Make request with token in header
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,  # Send as JSON if data provided
                timeout=30
            )

            # Check if token expired (401)
            if response.status_code == 401:
                logger.info("Token expired, attempting re-login...")
                self.token = None
                # Clear old headers
                self.session.headers.pop('Authorization', None)
                self.session.headers.pop('X-Auth-Token', None)
                if self.login():
                    # Retry request with new token
                    response = self.session.request(
                        method=method,
                        url=url,
                        params=params,
                        json=data,
                        timeout=30
                    )
                else:
                    return None

            # Check response - accept 200 (OK), 201 (Created), 202 (Accepted)
            if response.status_code in [200, 201, 202]:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    # Some successful responses might not have JSON body
                    logger.warning(f"Response has no JSON body. Status: {response.status_code}")
                    return {"status": response.status_code, "message": "Success"}
            else:
                error_msg = f"API call failed: {response.status_code}"
                try:
                    error_data = response.json()
                    logger.error(f"{error_msg} - {json.dumps(error_data, indent=2)}")
                    return {"error": True, "status": response.status_code, "details": error_data}
                except:
                    logger.error(f"{error_msg} - {response.text}")
                    return {"error": True, "status": response.status_code, "details": response.text}

        except Exception as e:
            logger.error(f"API call error: {e}")
            return None

    def close(self):
        """Close the session and clean up resources."""
        self.session.close()
        logger.info("MCP Client session closed")


# Simple test - now tests just the authentication
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n=== MCP Client Test ===\n")

    # Create client
    client = MCPClient()

    # Test login
    print("1. Testing login...")
    if client.login():
        print(f"   Success! Token: {client.token[:20]}...")

        # Test a simple API call using the generic api_call method
        print("\n2. Testing generic API call...")
        result = client.api_call("/nsa/api/v1/alarms/alarmRecordsCounts")
        if result:
            print(f"   Success! Response received")
            print(f"   Data: {json.dumps(result, indent=2)}")
    else:
        print("   Login failed!")

    # Clean up
    client.close()