"""
E-Line Circuit Request Parser
==============================
Parses natural language requests to extract circuit parameters
"""

import re
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_eline_request(query: str) -> Dict[str, any]:
    """
    Parse natural language E-Line circuit creation request.

    Args:
        query: Natural language request like:
            - "create circuit between C01-5164-01 port 3 and C02-5164-01 port 4"
            - "provision eline from C01-5164-01:3 to C02-5164-01:4"
            - "setup connection C01-5164-01 to C02-5164-01"
            - "create circuit C01-5164-01 port 2 to C02-5164-01 port 5"

    Returns:
        Dict with parsed parameters:
        {
            'endpoint_a': 'C01-5164-01',
            'port_a': '3',
            'endpoint_z': 'C02-5164-01',
            'port_z': '4',
            'service_name': None,  # Will be auto-generated
            'customer': 'STE LAB ELAN'  # Default
        }
    """
    # Convert to uppercase for node matching
    query_upper = query.upper()
    query_lower = query.lower()

    # Initialize result
    result = {
        'endpoint_a': None,
        'port_a': '3',  # Default port
        'endpoint_z': None,
        'port_z': '3',  # Default port
        'service_name': None,
        'customer': 'STE LAB ELAN'
    }

    # Pattern 1: "C01-5164-01 port 3 to/and C02-5164-01 port 4"
    pattern1 = r'([C]\d{2}-\d{4}-\d{2})\s*(?:PORT\s*)?(\d+)?\s*(?:TO|AND)\s*([C]\d{2}-\d{4}-\d{2})\s*(?:PORT\s*)?(\d+)?'
    match1 = re.search(pattern1, query_upper)

    if match1:
        result['endpoint_a'] = match1.group(1)
        if match1.group(2):
            result['port_a'] = match1.group(2)
        result['endpoint_z'] = match1.group(3)
        if match1.group(4):
            result['port_z'] = match1.group(4)
        logger.info(f"Parsed using pattern 1: {result}")
        return result

    # Pattern 2: "C01-5164-01:3 to/and C02-5164-01:4" (colon notation)
    pattern2 = r'([C]\d{2}-\d{4}-\d{2}):(\d+)\s*(?:TO|AND)\s*([C]\d{2}-\d{4}-\d{2}):(\d+)'
    match2 = re.search(pattern2, query_upper)

    if match2:
        result['endpoint_a'] = match2.group(1)
        result['port_a'] = match2.group(2)
        result['endpoint_z'] = match2.group(3)
        result['port_z'] = match2.group(4)
        logger.info(f"Parsed using pattern 2 (colon): {result}")
        return result

    # Pattern 3: "between C01-5164-01 and C02-5164-01" (no ports specified)
    pattern3 = r'(?:BETWEEN\s+)?([C]\d{2}-\d{4}-\d{2})\s+(?:AND|TO)\s+([C]\d{2}-\d{4}-\d{2})'
    match3 = re.search(pattern3, query_upper)

    if match3:
        result['endpoint_a'] = match3.group(1)
        result['endpoint_z'] = match3.group(2)
        logger.info(f"Parsed using pattern 3 (no ports): {result}")
        return result

    # Pattern 4: Extract any two node patterns found in the text
    node_pattern = r'[C]\d{2}-\d{4}-\d{2}'
    nodes = re.findall(node_pattern, query_upper)

    if len(nodes) >= 2:
        result['endpoint_a'] = nodes[0]
        result['endpoint_z'] = nodes[1]

        # Try to find associated ports
        for i, node in enumerate(nodes[:2]):
            # Look for port after node name
            port_after = re.search(f'{node}\\s*(?:PORT\\s*)?(\\d+)', query_upper)
            # Look for port before node name
            port_before = re.search(f'PORT\\s*(\\d+)\\s*{node}', query_upper)
            # Look for colon notation
            port_colon = re.search(f'{node}:(\\d+)', query_upper)

            if port_colon:
                port = port_colon.group(1)
            elif port_after:
                port = port_after.group(1)
            elif port_before:
                port = port_before.group(1)
            else:
                port = None

            if port:
                if i == 0:
                    result['port_a'] = port
                else:
                    result['port_z'] = port

        logger.info(f"Parsed using pattern 4 (node extraction): {result}")
        return result

    # Pattern 5: Look for variations with 'from' and 'to'
    pattern5 = r'FROM\s+([C]\d{2}-\d{4}-\d{2})\s*(?:PORT\s*)?(\d+)?\s*TO\s+([C]\d{2}-\d{4}-\d{2})\s*(?:PORT\s*)?(\d+)?'
    match5 = re.search(pattern5, query_upper)

    if match5:
        result['endpoint_a'] = match5.group(1)
        if match5.group(2):
            result['port_a'] = match5.group(2)
        result['endpoint_z'] = match5.group(3)
        if match5.group(4):
            result['port_z'] = match5.group(4)
        logger.info(f"Parsed using pattern 5 (from/to): {result}")
        return result

    # If no nodes found, return error
    if not result['endpoint_a'] or not result['endpoint_z']:
        logger.warning(f"Could not parse endpoints from query: {query}")
        raise ValueError(f"Could not identify endpoints in request. Please specify two nodes like 'C01-5164-01' and 'C02-5164-01'")

    return result


def validate_parsed_request(params: Dict[str, any]) -> Tuple[bool, Optional[str]]:
    """
    Validate parsed E-Line request parameters.

    Args:
        params: Parsed parameters dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required endpoints
    if not params.get('endpoint_a'):
        return False, "Missing source endpoint (endpoint_a)"

    if not params.get('endpoint_z'):
        return False, "Missing destination endpoint (endpoint_z)"

    # Validate endpoint format
    node_pattern = r'^C\d{2}-\d{4}-\d{2}$'

    if not re.match(node_pattern, params['endpoint_a']):
        return False, f"Invalid endpoint format: {params['endpoint_a']}. Expected format: CXX-XXXX-XX"

    if not re.match(node_pattern, params['endpoint_z']):
        return False, f"Invalid endpoint format: {params['endpoint_z']}. Expected format: CXX-XXXX-XX"

    # Check endpoints are different
    if params['endpoint_a'] == params['endpoint_z']:
        return False, "Source and destination endpoints cannot be the same"

    # Validate ports (should be numeric strings)
    try:
        port_a = int(params.get('port_a', '3'))
        port_z = int(params.get('port_z', '3'))

        if port_a < 1 or port_a > 48:  # Typical port range
            return False, f"Port {port_a} out of range (1-48)"

        if port_z < 1 or port_z > 48:
            return False, f"Port {port_z} out of range (1-48)"

        # Convert back to strings
        params['port_a'] = str(port_a)
        params['port_z'] = str(port_z)

    except ValueError:
        return False, "Ports must be numeric values"

    return True, None


# Test the parser
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_queries = [
        "create circuit between C01-5164-01 port 3 and C02-5164-01 port 4",
        "provision eline from C01-5164-01:3 to C02-5164-01:4",
        "setup connection C01-5164-01 to C02-5164-01",
        "create circuit C01-5164-01 port 2 to C02-5164-01 port 5",
        "connect C02-5164-01:1 and C01-5164-01:2",
        "establish link between C01-5164-01 and C02-5164-01",
        "C01-5164-01 to C02-5164-01",
        "provision circuit from node C01-5164-01 port 6 to node C02-5164-01 port 7"
    ]

    print("Testing E-Line Request Parser")
    print("=" * 50)

    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            parsed = parse_eline_request(query)
            is_valid, error = validate_parsed_request(parsed)

            if is_valid:
                print(f"  [OK] Parsed successfully:")
                print(f"    Endpoint A: {parsed['endpoint_a']} port {parsed['port_a']}")
                print(f"    Endpoint Z: {parsed['endpoint_z']} port {parsed['port_z']}")
            else:
                print(f"  [FAILED] Validation failed: {error}")

        except ValueError as e:
            print(f"  [ERROR] Parse error: {e}")