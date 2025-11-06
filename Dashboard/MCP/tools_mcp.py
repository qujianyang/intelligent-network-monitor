"""
MCP Tools for LangChain Agent - Full Implementation
===================================================
Integrates comprehensive MCP alarm querying with natural language processing.
"""

import logging
from typing import List, Dict
from langchain.tools import Tool

# Handle imports for different execution contexts
try:
    # Production imports (when used as part of Dashboard)
    from Dashboard.MCP.mcp_api import MCPClient
    from Dashboard.MCP.mcp_alarms import (
        parse_alarm_query,
        query_alarms,
        format_alarms_to_text
    )
    from Dashboard.MCP.mcp_eline_parser import parse_eline_request, validate_parsed_request
    from Dashboard.MCP.mcp_eline_manager import ELineManager
except ImportError:
    # Direct imports for testing
    from mcp_api import MCPClient
    from mcp_alarms import (
        parse_alarm_query,
        query_alarms,
        format_alarms_to_text
    )
    from mcp_eline_parser import parse_eline_request, validate_parsed_request
    from mcp_eline_manager import ELineManager

logger = logging.getLogger(__name__)


def query_mcp_alarms(query: str) -> str:
    """
    Query MCP/Blue Planet for network alarm information using natural language.

    Args:
        query: Natural language query about alarms

    Returns:
        str: Human-readable formatted alarm information

    Examples:
        - "retrieve alarm with id 411473945349786091"
        - "show critical alarms"
        - "service affecting 6500 alarms from last 2 hours"
        - "get alarms for device C01-S32-01"
        - "non-service affecting minor alarms from today"
    """
    try:
        logger.info(f"Processing MCP query: {query}")

        # Parse the natural language query
        params = parse_alarm_query(query)
        logger.debug(f"Parsed parameters: {params}")

        # Create MCP client
        client = MCPClient(
            base_url="https://10.1.1.3",
            username="admin",
            password="adminpw",
            tenant="master"
        )

        # Execute the query with parsed parameters
        result = query_alarms(client, **params)

        # Check for errors in the result
        if "error" in result:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"MCP query error: {error_msg}")

            # Provide helpful error messages
            if "401" in str(error_msg):
                return "Authentication failed. Please check MCP credentials."
            elif "timeout" in str(error_msg).lower():
                return "Connection timeout. Please check if MCP is accessible at 10.1.1.3."
            else:
                return f"Unable to query alarms: {error_msg}"

        # Format the response based on the query type
        # Pass alarm_id to get detailed view for single alarm queries
        formatted_text = format_alarms_to_text(
            result,
            requested_alarm_id=params.get('alarm_id')
        )

        # Clean up
        client.close()

        return formatted_text

    except ImportError as e:
        logger.error(f"Import error: {e}")
        return "Error: MCP alarm modules not properly installed. Please check the installation."

    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return "Cannot connect to MCP. Please check network connectivity and VPN if required."

    except Exception as e:
        logger.error(f"Unexpected error in MCP query: {e}")

        # Provide helpful fallback message
        return (
            f"I encountered an error querying MCP alarms: {str(e)}\n\n"
            "Try queries like:\n"
            "- 'show active alarms'\n"
            "- 'alarm id 12345'\n"
            "- 'critical alarms from C01-S32-01'\n"
            "- 'service affecting alarms from last 2 hours'"
        )


def create_eline_circuit(query: str) -> str:
    """
    Create an E-Line EPL circuit between two nodes using natural language.

    Args:
        query: Natural language request like:
            - "create circuit between C01-5164-01 port 3 and C02-5164-01 port 4"
            - "provision eline from C01-5164-01:3 to C02-5164-01:4"
            - "connect C01-5164-01 to C02-5164-01"

    Returns:
        str: Human-readable status message
    """
    try:
        logger.info(f"Processing E-Line creation request: {query}")

        # Step 1: Parse the request
        try:
            parsed_params = parse_eline_request(query)
            is_valid, error_msg = validate_parsed_request(parsed_params)

            if not is_valid:
                return f"❌ Invalid request: {error_msg}\n\nPlease specify endpoints like: 'create circuit between C01-5164-01 port 3 and C02-5164-01 port 4'"

        except ValueError as e:
            logger.error(f"Parse error: {e}")
            return (
                f"❌ Could not understand the request: {str(e)}\n\n"
                "Examples of valid requests:\n"
                "• create circuit between C01-5164-01 port 3 and C02-5164-01 port 4\n"
                "• provision eline from C01-5164-01:3 to C02-5164-01:4\n"
                "• connect C01-5164-01 to C02-5164-01"
            )

        # Step 2: Create and activate the circuit
        manager = ELineManager()

        logger.info(f"Creating circuit: {parsed_params['endpoint_a']}:{parsed_params['port_a']} ↔ {parsed_params['endpoint_z']}:{parsed_params['port_z']}")

        result = manager.create_and_activate_circuit(
            endpoint_a=parsed_params['endpoint_a'],
            port_a=parsed_params['port_a'],
            endpoint_z=parsed_params['endpoint_z'],
            port_z=parsed_params['port_z'],
            service_name=parsed_params.get('service_name'),
            customer=parsed_params.get('customer', 'STE LAB ELAN')
        )

        # Step 3: Format the response
        if result['success']:
            response = (
                f"✅ **E-Line Circuit Created Successfully!**\n\n"
                f"📋 **Service Details:**\n"
                f"• Name: {result['service_name']}\n"
                f"• ID: {result['service_id']}\n"
                f"• Endpoints: {result['endpoints']}\n"
                f"• Status: Active\n\n"
                f"🔗 **Connection:**\n"
                f"• {parsed_params['endpoint_a']} (Port {parsed_params['port_a']}) ──────► "
                f"{parsed_params['endpoint_z']} (Port {parsed_params['port_z']})\n\n"
                f"✨ The circuit is now operational and ready for use."
            )
        else:
            response = (
                f"❌ **Circuit Creation Failed**\n\n"
                f"• Step: {result['step']}\n"
                f"• Error: {result.get('error', 'Unknown error')}\n\n"
            )

            # Add troubleshooting tips
            if "Invalid product" in str(result.get('error', '')):
                response += "💡 **Tip:** The product ID may be incorrect. Please verify MCP configuration.\n"
            elif "requested" in str(result.get('error', '')):
                response += "💡 **Tip:** The service is created but stuck in provisioning. Manual intervention may be required in the MCP GUI.\n"
            elif "login" in str(result.get('error', '').lower()):
                response += "💡 **Tip:** Cannot connect to MCP. Please check credentials and network connectivity.\n"

        return response

    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return "❌ Cannot connect to MCP. Please check network connectivity and VPN if required."

    except Exception as e:
        logger.error(f"Unexpected error in E-Line creation: {e}")
        return (
            f"❌ An error occurred while creating the circuit: {str(e)}\n\n"
            "Please check the logs for more details."
        )


def create_mcp_tools() -> List[Tool]:
    """
    Create MCP tools for LangChain agent.

    Returns:
        List containing MCP tools for alarms and E-Line circuit management
    """

    # Comprehensive alarm query tool - handles all alarm queries including statistics
    alarm_query_tool = Tool(
        name="mcp_alarms",
        func=query_mcp_alarms,
        description=(
            "Query MCP/Blue Planet for network alarms using natural language. "
            "Returns formatted alarm information. "
            "IMPORTANT: Return the tool output EXACTLY as provided - do not reformat or rewrite the response. "
            "\n\nCapabilities:"
            "\n• Alarm Statistics: 'how many alarms?', 'alarm count', 'show alarm stats', 'active alarm summary'"
            "\n• Specific Alarms: 'show alarm 12345', 'get alarm details for ID 411473945349786091'"
            "\n• Filtered Queries: 'critical alarms', 'alarms from last 2 hours', 'service affecting 6500 alarms'"
            "\n• Device Queries: 'alarms from C01-S32-01', 'show device alarms'"
            "\n\nFilters available: severity (critical/major/minor), state (active/cleared), "
            "device name, device type (6500/waveserver/5164/8110), IP address, service impact, time periods."
        )
    )

    # E-Line circuit creation tool - creates and activates E-Line EPL services
    eline_creation_tool = Tool(
        name="create_eline_circuit",
        func=create_eline_circuit,
        description=(
            "Create and activate an E-Line EPL (Ethernet Private Line) circuit between two network nodes. "
            "Automatically generates service names (test1, test2, test3...) and activates the circuit. "
            "IMPORTANT: Return the tool output EXACTLY as provided - includes status and circuit details. "
            "\n\nExamples of valid requests:"
            "\n• 'create circuit between C01-5164-01 port 3 and C02-5164-01 port 4'"
            "\n• 'provision eline from C01-5164-01:3 to C02-5164-01:4'"
            "\n• 'connect C01-5164-01 to C02-5164-01' (uses default port 3)"
            "\n• 'establish link between C02-5164-01 port 2 and C01-5164-01 port 5'"
            "\n\nRequired: Two endpoints (nodes) in format CXX-XXXX-XX"
            "\nOptional: Port numbers (defaults to port 3 if not specified)"
            "\n\nThe tool will create the circuit, activate it, and return the service ID and status."
        )
    )

    return [alarm_query_tool, eline_creation_tool]



if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_integration()