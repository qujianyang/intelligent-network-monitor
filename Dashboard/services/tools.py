"""
Unified Tool Framework for Agentic AI
=====================================
Wraps all QKD services as LangChain/LlamaIndex compatible tools.
"""

import json
import logging
import sys
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

# LangChain imports
from langchain.tools import Tool

# Import all existing services
from Dashboard.services.fault_improved import detect_fault

from Dashboard.services.qkd_assistant_multi_index import get_qkd_answer_multi_index
from Dashboard.services.forecasting import get_node_forecast_api

# Import standalone alarm database connector (no Flask context needed)
from Dashboard.services.alarm_db import get_alarm_list_standalone, acknowledge_fault_standalone, get_alarm_connector

# Note: Removed file-based retriever imports - now using MySQL unified database

logger = logging.getLogger(__name__)

# Note: All tools now use simple string input for better LLM compatibility
# Structured schemas were removed as agents struggled with JSON formatting





# Tool wrapper functions with error handling
def safe_qkd_knowledge_query(query: str, top_k: int = 5) -> str:
    """Safely query the QKD knowledge base using unified MySQL vector store."""
    try:
        query = query.strip().strip("'\"")
        response, confidence = get_qkd_answer_multi_index(query, top_k=top_k)
        return f"{response}\n\n[Confidence: {confidence:.2f}]"
    except Exception as e:
        logger.error(f"QKD knowledge query error: {e}")
        return f"Error querying knowledge base: {str(e)}"

def safe_qkd_knowledge_query_simple(input_str: str) -> str:
    """Plain-text wrapper for QKD knowledge queries.

    Accepts a free-text question and routes to the same safe query function with default top_k.
    This improves agent usability by avoiding JSON args for quick lookups.
    """
    try:
        return safe_qkd_knowledge_query(input_str, top_k=5)
    except Exception as e:
        logger.error(f"QKD knowledge (simple) error: {e}")
        return f"Error querying knowledge base: {str(e)}"

def safe_node_forecast(node_id: str, days_ahead: int = 7) -> str:
    """Safely forecast node metrics."""
    try:
        # Ensure proper input validation
        if not isinstance(node_id, str):
            return f"Error: Invalid node_id format. Expected string, got {type(node_id)}"
        
        node_id = node_id.strip().strip("'\"")
        if not node_id:
            return "Error: Node ID cannot be empty"
        
        # Validate days_ahead
        if not isinstance(days_ahead, int) or days_ahead < 1 or days_ahead > 30:
            return "Error: days_ahead must be an integer between 1 and 30"
        
        result = get_node_forecast_api(node_id, days_ahead, include_metadata=True)
        
        # Check for errors first
        if 'error' in result:
            return f"Forecast error for {node_id}: {result['error']}"
        
        # Format the result with actual data structure
        output = f"Forecast for {node_id} ({days_ahead} days):\n"
        output += f"- Current Average Rate: {result.get('current_avg_rate', 'N/A')} kbps\n"
        output += f"- Predicted Average Rate: {result.get('predicted_avg_rate', 'N/A')} kbps\n"
        output += f"- Forecast Period: {result.get('forecast_period', f'{days_ahead} days')}\n"
        
        # Add trend analysis if available
        if 'trend_analysis' in result:
            output += f"- Trend: {result['trend_analysis']}\n"
        
        if 'risk_level' in result:
            output += f"- Risk Level: {result['risk_level']}\n"
            
        # Show first few predictions
        if 'predictions' in result and result['predictions']['timestamps']:
            output += f"\nFirst 3 predictions:\n"
            timestamps = result['predictions']['timestamps'][:3]
            values = result['predictions']['predicted_values'][:3]
            for i, (timestamp, value) in enumerate(zip(timestamps, values)):
                output += f"  {timestamp}: {value:.2f} kbps\n"
        
        return output
    except Exception as e:
        logger.error(f"Forecast error for {node_id}: {e}")
        return f"Error forecasting node {node_id}: {str(e)}. Please check that the node ID is valid (e.g., 'QKD_001', 'QKD_002', 'QKD_003') and that sufficient historical data exists for forecasting."


def safe_comprehensive_forecast(node_id: str, days_ahead: int = 7) -> str:
    """Enhanced safe wrapper for comprehensive multi-metric forecasting."""
    try:
        # Validate inputs
        if not isinstance(node_id, str) or not node_id.strip():
            return "Error: Node ID must be a non-empty string"
        
        if not isinstance(days_ahead, int) or days_ahead <= 0 or days_ahead > 30:
            return "Error: Days ahead must be an integer between 1 and 30"
        
        node_id = node_id.strip().strip("'\"")
        logger.info(f"Generating comprehensive forecast for {node_id}, {days_ahead} days ahead")
        
        # Try comprehensive forecasting first
        try:
            from Dashboard.services.forecasting import get_comprehensive_forecast_api
            forecast_result = get_comprehensive_forecast_api(node_id, days_ahead)
            
            # Handle errors - fallback to simple forecast
            if "error" in forecast_result:
                logger.warning(f"Comprehensive forecast failed for {node_id}: {forecast_result['error']}")
                logger.info(f"Falling back to simple forecast for {node_id}")
                return safe_node_forecast(node_id, days_ahead)
            
            # Extract comprehensive information
            forecasts = forecast_result.get("forecasts", {})
            insights = forecast_result.get("insights", {})
            data_source = forecast_result.get("data_source", "unknown")
            metrics_count = forecast_result.get("metrics_forecasted", 0)
            
            # Build comprehensive response
            response_parts = [
                f" Comprehensive {days_ahead}-day forecast for {node_id} ({metrics_count} metrics analyzed)",
                f" Data source: {data_source}",
                ""
            ]
            
            # Add forecasts for each metric
            metric_order = ['qkdKeyRate', 'qkdQber', 'qkdVisibility', 'temperature', 'cpu_load', 'memory_usage']
            
            for metric_key in metric_order:
                if metric_key in forecasts:
                    forecast_data = forecasts[metric_key]
                    if "error" not in forecast_data:
                        name = forecast_data['name']
                        current = forecast_data['current_avg']
                        predicted = forecast_data['predicted_avg']
                        change_pct = forecast_data['change_pct']
                        
                        # Format based on metric type
                        if metric_key == 'qkdKeyRate':
                            response_parts.append(f" {name}: {current:.0f} → {predicted:.0f} bps ({change_pct:+.1f}%)")
                        elif metric_key in ['qkdQber']:
                            response_parts.append(f" {name}: {current:.4f} → {predicted:.4f} ({change_pct:+.1f}%)")
                        elif metric_key in ['qkdVisibility']:
                            response_parts.append(f"{name}: {current:.3f} → {predicted:.3f} ({change_pct:+.1f}%)")
                        elif metric_key == 'temperature':
                            response_parts.append(f"{name}: {current:.1f} → {predicted:.1f}°C ({change_pct:+.1f}%)")
                        elif metric_key in ['cpu_load', 'memory_usage']:
                            response_parts.append(f"{name}: {current:.1f} → {predicted:.1f}% ({change_pct:+.1f}%)")
            
            # Add insights section
            summary = insights.get("summary", [])
            warnings = insights.get("warnings", [])
            recommendations = insights.get("recommendations", [])
            risk_level = insights.get("risk_assessment", "low")
            
            if summary or warnings or recommendations:
                response_parts.append("")
                response_parts.append(" ANALYSIS:")
                
                for item in summary[:2]:  # Limit to 2 summary items
                    response_parts.append(f"✓ {item}")
                
                for warning in warnings[:3]:  # Limit to 3 warnings
                    response_parts.append(f" {warning}")
                
                if recommendations:
                    response_parts.append("")
                    response_parts.append("RECOMMENDATIONS:")
                    for rec in recommendations[:2]:  # Limit to 2 recommendations
                        response_parts.append(f"• {rec}")
            
            # Add risk assessment
            if risk_level in ["medium", "high", "critical"]:
                response_parts.append("")
                response_parts.append(f" RISK ASSESSMENT: {risk_level.upper()}")
            
            final_response = "\n".join(response_parts)
            logger.info(f"Comprehensive forecast completed for {node_id}")
            return final_response
            
        except ImportError:
            logger.warning("Comprehensive forecasting not available, using simple forecast")
            return safe_node_forecast(node_id, days_ahead)
        
    except Exception as e:
        logger.error(f"Comprehensive forecast failed for {node_id}: {e}")
        return safe_node_forecast(node_id, days_ahead)





# Removed safe_network_topology function as it was causing issues

def safe_get_alarm_list(input_str: str = None) -> str:
    """Get list of all alarms from the QKD system.

    Retrieves current alarm status including fault ID, status, severity,
    affected component, detection time, and assignment information.
    No input required - returns all current alarms.
    """
    try:
        # Use standalone DB connector (no Flask context needed)
        faults = get_alarm_list_standalone()

        if not faults:
            return "No alarms found in the system. All components are operating normally."

        # Format alarms for AI readability
        output = f"=== ALARM LIST ({len(faults)} total) ===\n\n"

        # Show first 10 alarms in detail
        for idx, fault in enumerate(faults[:10], 1):
            output += f"[{idx}] Alarm ID: {fault['fault_id']}\n"
            output += f"    Status: {fault['status']}\n"
            output += f"    Severity: {fault['severity']}\n"
            output += f"    Component: {fault['component']}\n"
            output += f"    Detected: {fault['detected_at']}\n"
            output += f"    Description: {fault['description']}\n"

            if fault['acknowledged_by'] != "Not acknowledged":
                output += f"    Acknowledged by: {fault['acknowledged_by']}\n"
                if fault['acknowledged_at']:
                    output += f"    Acknowledged at: {fault['acknowledged_at']}\n"

            if fault['assigned_to'] != "Not assigned":
                output += f"    Assigned to: {fault['assigned_to']}\n"

            if fault['resolved_by'] != "Not resolved":
                output += f"    Resolved by: {fault['resolved_by']}\n"
                if fault['resolved_at']:
                    output += f"    Resolved at: {fault['resolved_at']}\n"

            output += "\n"

        # Show count of remaining alarms if more than 10
        if len(faults) > 10:
            output += f"... and {len(faults) - 10} more alarms (showing first 10)\n\n"

        # Summary statistics
        open_count = sum(1 for f in faults if f['status'] == 'Open')
        ongoing_count = sum(1 for f in faults if f['status'] == 'Ongoing')
        resolved_count = sum(1 for f in faults if f['status'] == 'Resolved')
        critical_count = sum(1 for f in faults if f['severity'] == 'Critical')
        warning_count = sum(1 for f in faults if f['severity'] == 'Warning')
        info_count = sum(1 for f in faults if f['severity'] == 'Info')

        output += "=== SUMMARY STATISTICS ===\n"
        output += f"By Status: Open={open_count}, Ongoing={ongoing_count}, Resolved={resolved_count}\n"
        output += f"By Severity: Critical={critical_count}, Warning={warning_count}, Info={info_count}\n"

        # Highlight critical alarms that need attention
        unresolved_critical = [f for f in faults if f['severity'] == 'Critical' and f['status'] != 'Resolved']
        if unresolved_critical:
            output += f"\n  ATTENTION: {len(unresolved_critical)} CRITICAL alarms require immediate action!\n"

        return output

    except Exception as e:
        logger.error(f"Error getting alarm list: {e}", exc_info=True)
        return f"Error retrieving alarm list: {str(e)}. Please check database connection and ensure fault_service is properly configured."

def safe_acknowledge_alarm(input_str: str) -> str:
    """
    Acknowledge an alarm and assign it to a user.

    Input format: "fault_id,acknowledger_name,assignee_name,notes"
    Example: "FLT_005,Kenneth Tan,Frank Taylor,Investigating network connectivity issue"

    - fault_id: The alarm/fault ID to acknowledge (string like FLT_005) (required)
    - acknowledger_name: Name of user performing the acknowledgement (required)
    - assignee_name: Name of user to assign the fault to (required)
    - notes: Notes about the acknowledgement (optional, use empty string if none)

    Permission requirements:
    - Acknowledger must have 'fault.acknowledge' permission
    - Assignee must have 'fault.resolve' permission
    """
    try:
        # Parse comma-separated input
        input_str = input_str.strip()

        # Remove surrounding quotes if LLM added them (e.g., 'FLT_002,...' or "FLT_002,...")
        if (input_str.startswith("'") and input_str.endswith("'")) or \
           (input_str.startswith('"') and input_str.endswith('"')):
            input_str = input_str[1:-1]

        parts = [p.strip() for p in input_str.split(',', 3)]

        if len(parts) < 3:
            return (" Error: Invalid input format.\n"
                   "Expected: fault_id,acknowledger_name,assignee_name,notes\n"
                   "Example: FLT_005,Kenneth Tan,Frank Taylor,Investigating network issue")

        # Clean each part of any remaining quotes
        fault_id = parts[0].strip().strip("'\"")
        acknowledger_name = parts[1].strip().strip("'\"")
        assignee_name = parts[2].strip().strip("'\"")
        notes = parts[3].strip().strip("'\"") if len(parts) > 3 else ""

        # Validate fault_id format (should start with FLT_)
        if not fault_id.startswith('FLT_'):
            return f" Error: Fault ID should be in format FLT_XXX, got: {fault_id}"

        # Look up user UUIDs from names
        connector = get_alarm_connector()

        with connector.get_cursor() as cursor:
            # Get acknowledger UUID
            cursor.execute("SELECT uuid, name FROM users WHERE name = %s;", (acknowledger_name,))
            acknowledger = cursor.fetchone()

            if not acknowledger:
                return f" Error: User '{acknowledger_name}' not found in the system"

            # Get assignee UUID
            cursor.execute("SELECT uuid, name FROM users WHERE name = %s;", (assignee_name,))
            assignee = cursor.fetchone()

            if not assignee:
                return f" Error: User '{assignee_name}' not found in the system"

        # Call the standalone acknowledge function
        result = acknowledge_fault_standalone(
            fault_id=fault_id,
            acknowledged_by_uuid=acknowledger['uuid'],
            assigned_to_uuid=assignee['uuid'],
            notes=notes
        )

        # Format response
        if result['success']:
            output = f" {result['message']}\n\n"
            output += f"Details:\n"
            output += f"  - Fault ID: {fault_id}\n"
            output += f"  - Acknowledged by: {acknowledger_name}\n"
            output += f"  - Assigned to: {assignee_name}\n"
            output += f"  - Status changed to: Ongoing\n"
            if notes:
                output += f"  - Notes: {notes}\n"
            return output
        else:
            return f" Error: {result['error']}"

    except Exception as e:
        logger.error(f"Error acknowledging alarm: {e}", exc_info=True)
        return f" Error acknowledging alarm: {str(e)}"

def safe_index_stats(input=None) -> str:
    """Enhanced knowledge base discovery tool using MySQL unified database."""
    try:
        from Dashboard.services.unified_db_manager import UnifiedDBManager

        db_manager = UnifiedDBManager()
        documents = db_manager.list_documents(include_deleted=False)

        if not documents:
            return (
                "=== KNOWLEDGE BASE DISCOVERY ===\n"
                "Status: empty\n"
                "Total documents: 0\n"
                "Total chunks: 0\n\n"
                "--- USAGE GUIDANCE ---\n"
                "- The knowledge base is currently empty\n"
                "- Documents need to be uploaded and indexed\n"
                "- Contact administrator to add documentation\n"
            )

        # Calculate statistics
        total_docs = len(documents)
        total_chunks = sum(doc.get('chunk_count', 0) for doc in documents)

        output = "=== KNOWLEDGE BASE DISCOVERY ===\n"
        output += f"Status: ready\n"
        output += f"Total documents: {total_docs}\n"
        output += f"Total chunks: {total_chunks:,}\n"
        output += f"Storage: MySQL unified database\n"

        output += f"\n--- AVAILABLE DOCUMENTATION ---\n"
        for doc in documents:
            display_name = doc.get('display_name') or doc.get('filename', 'Unknown')
            chunk_count = doc.get('chunk_count', 0)
            output += f"  * {display_name} ({chunk_count} chunks)\n"

        output += f"\n--- USAGE GUIDANCE ---\n"
        output += f"- For specific procedures: Use 'qkd_knowledge_base' with your question\n"
        output += f"- All documents are searchable through unified vector search\n"
        output += f"- System uses hybrid BM25 + semantic search with reranking\n"

        return output

    except Exception as e:
        logger.error(f"Index stats error: {e}")
        return f"Error getting index statistics: {str(e)}. The knowledge base may not be properly initialized."

# Tool creation function
def create_qkd_tools() -> List[Tool]:
    """Create all QKD tools for the agent."""
    
    tools = []
    
    # 1. QKD Knowledge Base Tool (RAG) - Single simple version
    qkd_knowledge_tool = Tool(
        func=safe_qkd_knowledge_query_simple,
        name="qkd_knowledge_base",
        description=(
            "Search ALL technical documentation and knowledge base. "
            "Use this when user asks: 'what should I do', 'recommended action', 'how to fix', mentions error codes/alarm names, "
            "or needs documented procedures from manuals, guides, specifications. "
            "CRITICAL: Use the EXACT terms the user mentioned as input (don't change 'xcvr-rx-los' to 'installation procedure'). "
            "Input: The specific term/error/alarm the user asked about."
        )
    )
    tools.append(qkd_knowledge_tool)

    # 2. Forecasting Tool - Using simple Tool to avoid parameter parsing issues
    def forecast_wrapper(input_str: str) -> str:
        """Enhanced wrapper for comprehensive multi-metric forecasting"""
        try:
            # Parse the input - expect "NodeX" or "NodeX,7" or "NodeX 7"
            input_str = input_str.strip()
            
            # Try different parsing approaches
            if ',' in input_str:
                parts = input_str.split(',')
                node_id = parts[0].strip().strip("'\"")
                days_ahead = int(parts[1].strip().strip("'\"")) if len(parts) > 1 else 7
            elif ' ' in input_str:
                parts = input_str.split()
                node_id = parts[0].strip().strip("'\"")
                days_ahead = int(parts[1].strip().strip("'\"")) if len(parts) > 1 else 7
            else:
                # Just node ID provided
                node_id = input_str.strip().strip("'\"")
                days_ahead = 7
            
            # Use enhanced comprehensive forecasting
            return safe_comprehensive_forecast(node_id, days_ahead)
            
        except Exception as e:
            logger.error(f"Enhanced forecast wrapper error: {e}")
            return f"Error parsing forecast parameters. Use format: QKD_001,7 or QKD_001 7. Error: {str(e)}"
    
    forecast_tool = Tool(
        func=forecast_wrapper,
        name="forecast_node_metrics", 
        description=(
            "Generate forecasts for QKD node metrics. "
            "Input format: 'QKD_001,7' or 'QKD_001 7' where first is node ID and second is days (1-30). "
            "If only node ID provided, defaults to 7 days. Examples: 'QKD_002,7' or 'QKD_003 14'"
        )
    )
    tools.append(forecast_tool)
    
    # 4. Knowledge Base Discovery Tool (Future-ready for multi-KB scenarios)
    index_stats_tool = Tool(
        func=safe_index_stats,
        name="get_knowledge_base_stats",
        description=(
            "KNOWLEDGE BASE DISCOVERY: Get information about available knowledge bases and their contents. "
            "Use this FIRST when user asks about documentation to understand what sources are available. "
            "Shows document types, coverage areas, and chunk counts. "
            "After reviewing stats, use 'qkd_knowledge_base' to search specific content. "
            "Future-ready for multiple specialized knowledge bases (maintenance, troubleshooting, vendor manuals)."
        )
    )
    tools.append(index_stats_tool)

    # 5. Alarm Management Tool - Read-only alarm listing
    alarm_list_tool = Tool(
        func=safe_get_alarm_list,
        name="get_alarm_list",
        description=(
            "Get the current list of all alarms and faults in the QKD system. "
            "Shows alarm ID, status (Open/Ongoing/Resolved), severity (Critical/Warning/Info), "
            "affected component (node or link), detection time, description, and assignment details. "
            "Use this to view active issues, check alarm status, investigate system problems, or answer "
            "questions about current faults. Also provides summary statistics by status and severity. "
            "No input required - returns all current alarms from the database."
        )
    )
    tools.append(alarm_list_tool)

    # 6. Alarm Acknowledgement Tool - Acknowledge and assign alarms
    acknowledge_alarm_tool = Tool(
        func=safe_acknowledge_alarm,
        name="acknowledge_alarm",
        description=(
            "Acknowledge an alarm and assign it to a user for resolution. "
            "Input format: 'fault_id,acknowledger_name,assignee_name,notes' "
            "Example: 'FLT_005,Kenneth Tan,Frank Taylor,Investigating network connectivity issue' "
            "Note: fault_id must be the string identifier (like FLT_005, FLT_016) shown in get_alarm_list, NOT a number. "
            "Requirements: Acknowledger must have 'fault.acknowledge' permission, "
            "assignee must have 'fault.resolve' permission. The fault must be in 'Open' status. "
            "Updates alarm status to 'Ongoing' and records the action in the fault log. "
            "Use this when a user wants to take ownership of an alarm or assign it to someone for investigation."
        )
    )
    tools.append(acknowledge_alarm_tool)

    logger.info(f"Created {len(tools)} QKD tools for agent")
    return tools

