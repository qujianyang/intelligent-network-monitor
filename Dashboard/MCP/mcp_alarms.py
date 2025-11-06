"""
Simple MCP Alarm Query Module
==============================
Uses MCPClient for authentication and queries MCP alarms.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re

# Handle imports for both module use and direct execution
try:
    from Dashboard.MCP.mcp_api import MCPClient
except ImportError:
    from mcp_api import MCPClient


def query_alarms(
    client: Optional[MCPClient] = None,
    alarm_id: Optional[str] = None,
    state: Optional[List[str]] = None,
    severity: Optional[List[str]] = None,
    device_name: Optional[str] = None,
    device_type: Optional[List[str]] = None,
    ip_address: Optional[str] = None,
    service_affecting: Optional[bool] = None,
    hours_ago: Optional[int] = None,
    sort: Optional[str] = None,
    page_size: int = 20
) -> Dict:
    """
    Query MCP alarms using MCPClient.

    Args:
        client: MCPClient instance (creates default if None)
        alarm_id: Specific alarm ID to retrieve
        state: List of states like ["ACTIVE"]
        severity: List of severities like ["CRITICAL", "MAJOR"]
        device_name: Filter by device name
        device_type: List of device types like ["6500", "WAVESERVER AI", "5164", "8110", "NMS"]
        ip_address: Filter by IP address
        service_affecting: True for SERVICE_AFFECTING, False for NON_SERVICE_AFFECTING
        hours_ago: Number of hours to look back from now
        sort: Sort parameter like "-last-raise-time"
        page_size: Number of results (default 20)

    Returns:
        Dict: Raw JSON response from MCP or error dict
    """
    # Create client if not provided
    if client is None:
        client = MCPClient()

    # Build query parameters as list of tuples for multiple values
    params = []

    if alarm_id:
        params.append(("filter[alarmId][]", str(alarm_id)))

    if state:
        for s in state:
            params.append(("filter[state][]", s))

    if severity:
        for sev in severity:
            params.append(("filter[severity][]", sev))

    if device_name:
        params.append(("filter[deviceName][]", device_name))

    if device_type:
        # Ensure it's a list for consistency
        if isinstance(device_type, str):
            device_type = [device_type]
        for dt in device_type:
            params.append(("filter[deviceType][]", dt))

    if ip_address:
        params.append(("filter[ipAddress][]", ip_address))

    if service_affecting is not None:
        # Convert boolean to API values
        value = "SERVICE_AFFECTING" if service_affecting else "NON_SERVICE_AFFECTING"
        params.append(("filter[serviceAffecting][]", value))

    if hours_ago:
        # Calculate the time from now
        time_from = datetime.utcnow() - timedelta(hours=hours_ago)
        # Format as required: yyyy-MM-ddTHH:mm:ss.SSS
        time_str = time_from.strftime('%Y-%m-%dT%H:%M:%S.000')
        # Note: lastRaisedTimeFrom is a scalar parameter (no [])
        params.append(("filter[lastRaisedTimeFrom]", time_str))

    if sort:
        params.append(("sort", sort))

    params.append(("pageSize", str(page_size)))

    # Use MCPClient's api_call method
    endpoint = "/nsa/api/v2_0/alarms/filter/filteredAlarms"
    result = client.api_call(endpoint, method="GET", params=params)

    if result:
        return result
    else:
        return {"error": "Failed to query alarms"}


def format_single_alarm(alarm: Dict) -> str:
    """
    Format a single alarm with detailed information.

    Args:
        alarm: Single alarm dictionary

    Returns:
        str: Detailed formatted alarm information
    """
    attr = alarm.get("attributes", {})

    # Extract all relevant fields
    alarm_id = attr.get("alarm-id", "Unknown")
    device = attr.get("device-name", "Unknown")
    ip = attr.get("ip-address", "Unknown")
    mac = attr.get("mac-address", "Unknown")
    severity = attr.get("condition-severity", "Unknown")
    state = attr.get("state", "Unknown")
    condition = attr.get("native-condition-type", "Unknown")
    qualifier = attr.get("native-condition-type-qualifier", "")
    service_affecting = attr.get("service-affecting", "Unknown")
    ack_state = attr.get("acknowledge-state", "Unknown")
    additional_text = attr.get("additional-text", "None")
    resource = attr.get("resource", "Unknown")

    # Format timestamps
    first_raise = attr.get("first-raise-time", "")
    last_raise = attr.get("last-raise-time", "")
    occurrences = attr.get("number-of-occurrences", 0)

    def format_time(time_str):
        if time_str:
            try:
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                return dt.strftime("%b %d, %Y %H:%M:%S")
            except:
                return time_str[:19]
        return "Unknown"

    # Build detailed output
    lines = []
    lines.append(f"Alarm ID: {alarm_id}")
    lines.append(f"Device: {device} ({ip})")
    lines.append(f"MAC Address: {mac}")
    lines.append(f"Resource: {resource}")
    lines.append(f"Severity: {severity}")
    lines.append(f"Status: {state}")
    lines.append(f"Condition: {condition}")
    if qualifier:
        lines.append(f"Description: {qualifier}")
    lines.append(f"Service Affecting: {service_affecting.replace('_', ' ').title()}")
    lines.append(f"Acknowledgement: {ack_state.replace('_', ' ').title()}")
    lines.append(f"First Raised: {format_time(first_raise)}")
    lines.append(f"Last Raised: {format_time(last_raise)}")
    lines.append(f"Occurrences: {occurrences}")
    if additional_text and additional_text != "None":
        lines.append(f"Additional Info: {additional_text}")

    return "\n".join(lines)


def format_alarms_to_text(alarm_data: Dict, requested_alarm_id: Optional[str] = None) -> str:
    """
    Convert alarm JSON to human-readable text for chatbot.

    Args:
        alarm_data: JSON response from query_alarms()
        requested_alarm_id: If provided, format as single alarm detail

    Returns:
        str: Formatted text description of alarms
    """
    if "error" in alarm_data:
        return f"Error querying alarms: {alarm_data['error']}"

    alarms = alarm_data.get("data", [])
    meta = alarm_data.get("meta", {})

    # Get totals
    query_total = meta.get("query-total", 0)

    if query_total == 0:
        if requested_alarm_id:
            return f"No alarm found with ID: {requested_alarm_id}"
        return "No alarms found matching your criteria."

    # If single alarm was requested and found, show detailed view
    if requested_alarm_id and len(alarms) == 1:
        return format_single_alarm(alarms[0])

    # Count by severity
    severity_counts = meta.get("query-aggregations", {}).get("severity", {})

    # Build summary text with consistent format
    lines = []
    lines.append(f"Found {query_total} alarm(s)")

    # Always show severity breakdown if available
    if severity_counts:
        severity_text = []
        for sev in ["CRITICAL", "MAJOR", "MINOR", "WARNING", "INFO"]:
            count = severity_counts.get(sev, 0)
            if count > 0:
                severity_text.append(f"{count} {sev.lower()}")
        if severity_text:
            lines.append(f"Severity: {', '.join(severity_text)}")

    # Top affected devices
    device_counts = meta.get("query-aggregations", {}).get("deviceName", {})
    if device_counts:
        top_devices = sorted(device_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        lines.append(f"Most affected devices: {', '.join([f'{d[0]} ({d[1]} alarms)' for d in top_devices])}")

    # Always show recent alarms in the same format
    if alarms:
        lines.append("")  # Empty line for readability
        lines.append("Recent alarms:")
        for alarm in alarms[:5]:
            attr = alarm.get("attributes", {})
            device = attr.get("device-name", "Unknown")
            severity = attr.get("condition-severity", "Unknown")
            condition = attr.get("native-condition-type-qualifier", attr.get("native-condition-type", "Unknown"))

            # Format time consistently
            raise_time = attr.get("last-raise-time", "")
            if raise_time:
                try:
                    dt = datetime.fromisoformat(raise_time.replace('Z', '+00:00'))
                    time_str = dt.strftime("%b %d %H:%M")
                except:
                    time_str = raise_time[:10]
            else:
                time_str = "Unknown time"

            lines.append(f"• {severity}: {condition} on {device} at {time_str}")

    return "\n".join(lines)


def parse_alarm_query(user_input: str) -> Dict:
    """
    Parse natural language input to extract alarm query parameters.

    Args:
        user_input: Natural language query from user

    Returns:
        Dict with extracted parameters (alarm_id, severity, state, device_name, etc.)
    """
    params = {}
    input_lower = user_input.lower()

    # Extract alarm ID - multiple patterns to catch various formats
    alarm_id_patterns = [
        r'alarm\s+id\s+([+-]?\d+)',           # "alarm id 12345"
        r'alarm\s+#?\s*([+-]?\d+)',           # "alarm 12345" or "alarm #12345"
        r'id\s+([+-]?\d+)',                   # "id 12345"
        r'#([+-]?\d+)',                       # "#12345"
        r'retrieve\s+alarm\s+([+-]?\d+)',     # "retrieve alarm 12345"
        r'get\s+alarm\s+([+-]?\d+)',          # "get alarm 12345"
        r'show\s+alarm\s+([+-]?\d+)',         # "show alarm 12345"
        r'alarm\s+with\s+id\s+([+-]?\d+)',    # "alarm with id 12345"
    ]

    for pattern in alarm_id_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            params['alarm_id'] = match.group(1)
            break

    # Extract severity - check for keywords
    severities = {
        'critical': ['CRITICAL'],
        'major': ['MAJOR'],
        'minor': ['MINOR'],
        'warning': ['WARNING'],
        'info': ['INFO']
    }

    severity_list = []
    for keyword, value in severities.items():
        if keyword in input_lower:
            severity_list.extend(value)

    if severity_list:
        params['severity'] = severity_list

    # Extract state
    if 'active' in input_lower:
        params['state'] = ['ACTIVE']
    elif 'cleared' in input_lower:
        params['state'] = ['CLEARED']
    elif 'superseded' in input_lower:
        params['state'] = ['SUPERSEDED']

    # Extract device name - pattern for Ciena device naming convention
    device_patterns = [
        r'(C\d{2}-[\w]+-\d{2})',           # C01-S32-01, C02-WSAI-01, etc.
        r'(CA-[\w]+-\d{2})',               # CA-S8-01, CA-5164-02, etc.
        r'device\s+([\w-]+)',              # "device XXX"
        r'for\s+device\s+([\w-]+)',        # "for device XXX"
        r'on\s+device\s+([\w-]+)',         # "on device XXX"
        r'from\s+device\s+([\w-]+)',       # "from device XXX"
    ]

    for pattern in device_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            params['device_name'] = match.group(1)
            break

    # Extract device type - common network equipment types
    device_types = []
    # Check for specific device type keywords
    if '6500' in user_input:
        device_types.append('6500')
    if '5164' in user_input:
        device_types.append('5164')
    if '8110' in user_input:
        device_types.append('8110')
    if 'nms' in input_lower:
        device_types.append('NMS')
    if 'waveserver' in input_lower or 'wave server' in input_lower or 'wsai' in input_lower:
        device_types.append('WAVESERVER AI')

    if device_types:
        params['device_type'] = device_types

    # Extract IP address
    ip_pattern = r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'
    ip_match = re.search(ip_pattern, user_input)
    if ip_match:
        params['ip_address'] = ip_match.group(1)

    # Extract service affecting status
    if 'service' in input_lower:
        if 'non' in input_lower or 'not' in input_lower or 'non-service' in input_lower:
            params['service_affecting'] = False
        elif 'affecting' in input_lower or 'impacting' in input_lower:
            params['service_affecting'] = True

    # Extract time period (hours ago)
    time_patterns = [
        r'last\s+(\d+)\s+hour',            # "last 3 hours"
        r'past\s+(\d+)\s+hour',            # "past 24 hours"
        r'(\d+)\s+hours?\s+ago',           # "3 hours ago"
        r'from\s+last\s+(\d+)\s+hour',     # "from last 2 hours"
        r'within\s+(\d+)\s+hour',          # "within 24 hours"
    ]

    for pattern in time_patterns:
        match = re.search(pattern, input_lower)
        if match:
            params['hours_ago'] = int(match.group(1))
            break

    # Special time keywords
    if 'hours_ago' not in params:
        if 'today' in input_lower:
            params['hours_ago'] = 24
        elif 'yesterday' in input_lower:
            params['hours_ago'] = 48
        elif 'this week' in input_lower:
            params['hours_ago'] = 168  # 7 days

    # Check for sorting preferences
    if 'latest' in input_lower or 'recent' in input_lower or 'newest' in input_lower:
        params['sort'] = '-last-raise-time'
    elif 'oldest' in input_lower:
        params['sort'] = 'last-raise-time'

    # Extract count/limit
    count_patterns = [
        r'top\s+(\d+)',                    # "top 10"
        r'first\s+(\d+)',                  # "first 5"
        r'last\s+(\d+)',                   # "last 20"
        r'(\d+)\s+alarms?',                # "10 alarms" or "1 alarm"
        r'limit\s+(\d+)',                  # "limit 50"
    ]

    for pattern in count_patterns:
        match = re.search(pattern, input_lower)
        if match:
            params['page_size'] = int(match.group(1))
            break

    # Default state to ACTIVE if not specified and not querying specific alarm
    if 'state' not in params and 'alarm_id' not in params:
        params['state'] = ['ACTIVE']

    return params


def interactive_test():
    """Interactive test mode for alarm queries."""
    print("\n" + "=" * 60)
    print("MCP ALARM QUERY - INTERACTIVE TEST MODE")
    print("=" * 60)
    print("\nType natural language queries to test the system.")
    print("Examples:")
    print("  - help me retrieve alarm with id 411473945349786091")
    print("  - show critical alarms")
    print("  - get alarms for device C01-S32-01")
    print("  - active major alarms from 10.1.1.31")
    print("  - show me the latest 5 alarms")
    print("\nNew features:")
    print("  - service affecting alarms from 6500 devices")
    print("  - non-service affecting minor alarms")
    print("  - critical alarms from last 3 hours")
    print("  - waveserver alarms from today")
    print("  - show 5164 and 6500 device alarms")
    print("\nType 'quit' or 'exit' to stop.")
    print("-" * 60)

    # Create client
    client = MCPClient(
        base_url="https://10.1.1.3",
        username="admin",
        password="adminpw"
    )

    while True:
        # Get user input
        print("\n" + "=" * 40)
        user_input = input("Enter query: ").strip()

        # Check for exit
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break

        if not user_input:
            print("Please enter a query.")
            continue

        try:
            # Parse the query
            print("\n Parsing query...")
            params = parse_alarm_query(user_input)
            print(f"Extracted parameters: {params}")

            # Execute the query
            print("\n Querying MCP...")
            result = query_alarms(client, **params)

            # Format and display results
            if "error" not in result:
                # Pass alarm_id for detailed formatting if querying single alarm
                text = format_alarms_to_text(result, params.get('alarm_id'))
                print("\n Results:")
                print("-" * 40)
                print(text)
            else:
                print(f"\n Error: {result['error']}")

        except Exception as e:
            print(f"\n Error occurred: {e}")

    # Clean up
    client.close()
    print("\nSession closed.")


# Quick test if run directly
if __name__ == "__main__":
    import sys

    # Check for interactive mode flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-i', '--interactive']:
        interactive_test()
    else:
        # Run automated tests
        print("MCP Alarm Query - Automated Tests")
        print("-" * 50)
        print("\nFor interactive mode, run: python mcp_alarms.py -i")
        print("-" * 50)

        # Create client with credentials
        client = MCPClient(
            base_url="https://10.1.1.3",
            username="admin",
            password="adminpw"
        )

        # Test parser
        print("\nTest: Natural Language Parser")
        test_queries = [
            "help me retrieve alarm with id 411473945349786091",
            "show critical alarms",
            "get alarms for device C01-S32-01",
            "service affecting alarms from 6500 devices",
            "non-service affecting alarms from last 2 hours",
            "waveserver ai alarms from today",
        ]

        for query in test_queries:
            params = parse_alarm_query(query)
            print(f"Query: '{query}'")
            print(f"Parsed: {params}\n")

        print("-" * 50)

        # Test actual query
        print("\nTest: Query specific alarm")
        test_alarm_id = "411473945349786091"
        result = query_alarms(client, alarm_id=test_alarm_id)

        if "error" not in result:
            text = format_alarms_to_text(result, requested_alarm_id=test_alarm_id)
            print(text[:500] + "..." if len(text) > 500 else text)
        else:
            print(result["error"])

        # Clean up
        client.close()