"""
Enhanced Tool Framework for Fault Detection and Anomaly Monitoring
====================================================================
Extends the agent's capabilities with advanced fault detection tools.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from langchain.tools import Tool

# Import fault detection services
from Dashboard.services.fault_improved import detect_fault
from Dashboard.services.fault_ml_pure import (
    analyze_historical_anomalies
)
from Dashboard.services.fault_db_mysql import (
    fetch_latest_metrics_mysql,
    get_available_nodes_mysql
)

logger = logging.getLogger(__name__)

# ============== Tool Functions ==============

def get_current_node_metrics(node_id: str) -> str:
    """Fetch and format current metrics for a specific node."""
    try:
        # Clean the node_id from any LangChain formatting (both single and double quotes)
        node_id = node_id.strip().strip('"').strip("'").replace('node_id=', '').strip('"').strip("'")

        metrics = fetch_latest_metrics_mysql(node_id)
        if not metrics:
            return f"No metrics data available for node {node_id}"

        # Format metrics nicely
        output = f"Current Metrics for {node_id}:\n"
        output += f"• QBER: {metrics.get('qkdQber', 0):.3f} ({metrics.get('qkdQber', 0)*100:.1f}%)\n"
        output += f"• Key Rate: {metrics.get('qkdKeyRate', 0):.0f} keys/sec\n"
        output += f"• Visibility: {metrics.get('qkdVisibility', 0):.3f}\n"
        output += f"• Laser Power: {metrics.get('qkdLaserPower', 0):.2f} mW\n"
        output += f"• CPU Load: {metrics.get('neCpuLoad', 0):.0f}%\n"
        output += f"• Memory Usage: {metrics.get('neMemUsage', 0):.0f}%\n"
        output += f"• Temperature: {metrics.get('neTemperature', 0):.0f}°C\n"

        if 'connectedLinks' in metrics:
            output += f"• Connected Links: {metrics['connectedLinks']}\n"

        return output
    except Exception as e:
        logger.error(f"Error fetching metrics for {node_id}: {e}")
        return f"Error fetching metrics: {str(e)}"

def detect_node_anomaly(node_id: str, include_trends: bool = False, trend_hours: int = 3) -> str:
    """Detect anomalies and determine severity for a specific node."""
    try:
        # Clean the node_id from any LangChain formatting (both single and double quotes)
        node_id = node_id.strip().strip('"').strip("'").replace('node_id=', '').strip('"').strip("'")

        result = detect_fault(node_id, include_trends=include_trends, trend_hours=trend_hours)

        if result.get("status") == "error":
            return f"Error: {result['message']}"

        # Format the result
        output = f"Anomaly Analysis for {node_id}:\n"
        output += f"• Status: {result['status'].upper()}\n"

        if result['status'] == 'fault':
            output += f"• Severity: {result['severity']}\n"
            output += f"• Anomaly Score: {result['anomaly_score']:.3f}\n"

            if result.get('threshold_violations'):
                output += "\nThreshold Violations:\n"
                for v in result['threshold_violations']:
                    output += f"  - {v['metric']}: {v['value']:.3f} {v['type']} {v['threshold']}\n"

            # Add fault localization
            violations = result.get('threshold_violations', [])
            link_issues = [v for v in violations if v['metric'] in ['qkdQber', 'qkdVisibility', 'qkdLaserPower']]
            node_issues = [v for v in violations if v['metric'] in ['neCpuLoad', 'neMemUsage', 'neTemperature']]

            if link_issues and not node_issues:
                output += "\nFault Type: LINK/OPTICAL ISSUE\n"
            elif node_issues and not link_issues:
                output += "\nFault Type: NODE/HARDWARE ISSUE\n"
            elif link_issues and node_issues:
                output += "\nFault Type: MIXED (Both link and node issues)\n"

            if result.get('recommendations'):
                output += "\nRecommendations:\n"
                for i, rec in enumerate(result['recommendations'][:3], 1):
                    output += f"  {i}. {rec}\n"
        else:
            output += f"• Anomaly Score: {result['anomaly_score']:.3f}\n"
            output += "• No faults detected - System operating normally\n"

        # Add trends if included
        if include_trends and 'trends' in result:
            trends = result['trends']
            if trends.get('status') != 'insufficient_data':
                output += f"\nTrend Analysis ({trends['hours_analyzed']}h):\n"
                output += f"• Overall Trend: {trends['overall'].upper()}\n"
                for metric, data in trends.get('metrics', {}).items():
                    if data['direction'] != 'stable':
                        output += f"  - {metric}: {data['direction']}\n"

        return output
    except Exception as e:
        logger.error(f"Error detecting anomaly for {node_id}: {e}")
        return f"Error detecting anomaly: {str(e)}"

def scan_network_for_anomalies(include_trends: bool = False) -> str:
    """Scan all nodes in the network for anomalies."""
    try:
        nodes = get_available_nodes_mysql()
        if not nodes:
            return "No nodes available in the network"

        output = "Network-Wide Anomaly Scan Results:\n"
        output += "=" * 40 + "\n\n"

        summary = {'normal': 0, 'low': 0, 'medium': 0, 'high': 0}
        critical_nodes = []

        for node_id in nodes:
            result = detect_fault(node_id, include_trends=include_trends)

            if result.get("status") == "error":
                output += f"{node_id}: Error - {result['message']}\n"
                continue

            status = result['status']
            severity = result.get('severity', 'NONE')
            score = result['anomaly_score']

            output += f"{node_id}:\n"
            output += f"  • Status: {status.upper()}\n"
            output += f"  • Severity: {severity}\n"
            output += f"  • Anomaly Score: {score:.3f}\n"

            if status == 'fault':
                if severity == 'HIGH':
                    summary['high'] += 1
                    critical_nodes.append(node_id)
                elif severity == 'MEDIUM':
                    summary['medium'] += 1
                else:
                    summary['low'] += 1

                # Add main issue
                if result.get('threshold_violations'):
                    main_violation = result['threshold_violations'][0]
                    output += f"  • Main Issue: {main_violation['metric']} {main_violation['type']} threshold\n"
            else:
                summary['normal'] += 1

            output += "\n"

        # Add summary
        output += "Summary:\n"
        output += f"• Total Nodes: {len(nodes)}\n"
        output += f"• Normal: {summary['normal']}\n"
        output += f"• Low Severity: {summary['low']}\n"
        output += f"• Medium Severity: {summary['medium']}\n"
        output += f"• High Severity: {summary['high']}\n"

        if critical_nodes:
            output += f"\n CRITICAL: Immediate attention required for: {', '.join(critical_nodes)}\n"

        return output
    except Exception as e:
        logger.error(f"Error scanning network: {e}")
        return f"Error scanning network: {str(e)}"

def analyze_historical_anomalies_tool(node_id: str, hours_back: int = 24) -> str:
    """Analyze historical anomaly patterns for a node."""
    try:
        # Clean the node_id from any LangChain formatting (both single and double quotes)
        node_id = node_id.strip().strip('"').strip("'").replace('node_id=', '').strip('"').strip("'")

        result = analyze_historical_anomalies(node_id, hours_back)

        if "error" in result:
            return f"Error: {result['error']}"

        output = f"{hours_back}-Hour Anomaly History for {node_id}:\n\n"
        output += f"• Total Data Points: {result['total_points']}\n"
        output += f"• Anomalies Detected: {result['anomalies_detected']}\n"
        output += f"• Anomaly Rate: {result['anomaly_rate']:.2f}%\n"

        scores = result['anomaly_scores']
        output += f"\nAnomaly Score Statistics:\n"
        output += f"• Average: {scores['mean']:.3f}\n"
        output += f"• Std Dev: {scores['std']:.3f}\n"
        output += f"• Min: {scores['min']:.3f}\n"
        output += f"• Max: {scores['max']:.3f}\n"

        if result['anomaly_rate'] > 10:
            output += f"\n High anomaly rate detected. System may need maintenance.\n"
        elif result['anomaly_rate'] < 5:
            output += f"\n✓ Low anomaly rate. System appears stable.\n"

        # Show sample anomalies
        if result.get('anomalies'):
            output += f"\nRecent Anomalies (showing first 3):\n"
            for i, anomaly in enumerate(result['anomalies'][:3], 1):
                output += f"  {i}. Score: {anomaly['anomaly_score']:.3f} at {anomaly.get('timestamp', 'N/A')}\n"

        return output
    except Exception as e:
        logger.error(f"Error analyzing historical anomalies: {e}")
        return f"Error analyzing historical anomalies: {str(e)}"


# ============== Create Tools ==============

def create_fault_detection_tools() -> List[Tool]:
    """Create a comprehensive set of fault detection tools for the agent."""

    tools = [
        # 1. Current Metrics Tool
        Tool(
            name="get_node_metrics",
            func=get_current_node_metrics,
            description="Get current real-time metrics for a specific QKD node. Use this when users ask about current values, latest measurements, or want to check specific metric values like QBER, temperature, CPU load, etc."
        ),

        # 2. Anomaly Detection Tool
        Tool(
            name="detect_anomaly",
            func=lambda input_str: detect_node_anomaly(
                node_id=input_str.split(',')[0].strip() if ',' in input_str else input_str,
                include_trends='trend' in input_str.lower()
            ),
            description="Detect anomalies and determine fault severity for a specific node. Returns anomaly score, severity level (HIGH/MEDIUM/LOW/NONE), and recommendations. Use when users ask about faults, anomalies, or problems with a specific node."
        ),

        # 3. Network Scan Tool
        Tool(
            name="scan_network",
            func=lambda _: scan_network_for_anomalies(),
            description="Scan all nodes in the network for anomalies and provide a summary. Use when users ask about network-wide status, want to check all nodes, or need a system-wide health check."
        ),

        # 4. Historical Analysis Tool
        Tool(
            name="analyze_history",
            func=lambda input_str: analyze_historical_anomalies_tool(
                node_id=input_str.split(',')[0].strip(),
                hours_back=int(input_str.split(',')[1].strip()) if ',' in input_str else 24
            ),
            description="Analyze historical anomaly patterns for a node over specified hours. Use when users ask about past anomalies, trends over time, or anomaly rates."
        )
    ]

    return tools

# For backwards compatibility
def create_enhanced_qkd_tools() -> List[Tool]:
    """Create enhanced tools including fault detection capabilities."""
    # Import original tools
    from Dashboard.services.tools import create_qkd_tools

    # Get original tools
    original_tools = create_qkd_tools()

    # Add fault detection tools
    fault_tools = create_fault_detection_tools()

    # Add MCP tools
    try:
        from Dashboard.MCP.tools_mcp import create_mcp_tools
        mcp_tools = create_mcp_tools()
        logger.info(f"Loaded {len(mcp_tools)} MCP tools")
    except ImportError as e:
        mcp_tools = []
        logger.warning(f"MCP tools not available: {e}")

    # Combine all tools
    all_tools = original_tools + fault_tools + mcp_tools

    return all_tools