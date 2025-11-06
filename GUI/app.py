from flask import Flask, render_template, redirect, url_for, request, jsonify, flash, session, g
from flask_wtf.csrf import CSRFProtect, generate_csrf, CSRFError
from werkzeug.utils import secure_filename
import mysql.connector
from dotenv import load_dotenv
import os
import random
from collections import defaultdict, Counter
from itertools import groupby
from pathlib import Path
from auth.routes import auth  # Import the auth blueprint
import services.fault_service as fs
import services.policy_service as ps
import services.others as others
from flask_config import Config
from services.decorator import permission_required
from db import close_db, db_transaction
from flask_mail import Mail, Message
import logging
import sys
import time
import json
import requests
from datetime import datetime, timedelta
from flask import send_from_directory, Response


app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
app.config.from_object(Config)

# Session cookie configuration for cross-origin access
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Allow cookies on cross-site requests
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True if using HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Security: prevent JS access
app.config['WTF_CSRF_TIME_LIMIT'] = None  # Disable CSRF token expiration
app.config['WTF_CSRF_CHECK_DEFAULT'] = False  # Disable CSRF for development

csrf = CSRFProtect(app)
mail = Mail(app)



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# RAG Assistant (Updated to use unified MySQL vector store)
# Import your RAG assistant
try:
    from Dashboard.services.qkd_assistant_multi_index import get_qkd_answer_multi_index
    # Also import the fallback system
    from Dashboard.services.qkd_assistant import get_qkd_answer
    RAG_AVAILABLE = True
    print("[SUCCESS] RAG system (unified MySQL) loaded successfully!")
except ImportError as e:
    print(f"[ERROR] Could not import unified RAG system: {e}")
    RAG_AVAILABLE = False

# Import additional services for AI features
try:
    # Import the new fault detection modules
    from Dashboard.services.fault_improved import detect_fault
    # Only import what's actually used
    from Dashboard.services.fault_db_mysql import fetch_latest_metrics_mysql, get_available_nodes_mysql

    # Forecasting removed - not used by chatbot API
    SERVICES_AVAILABLE = True
    print("[SUCCESS] Additional AI services loaded successfully!")
except ImportError as e:
    print(f"[ERROR] Could not import additional services: {e}")
    SERVICES_AVAILABLE = False

# Import agent for agentic AI support
try:
    from Dashboard.services.agent import get_agent, QKDAgent
    AGENT_AVAILABLE = True
    print("[SUCCESS] Agent system loaded successfully!")
except ImportError as e:
    print(f"[ERROR] Could not import agent system: {e}")
    AGENT_AVAILABLE = False



# Import Visual Search Enhancement (simplified vision-based query enhancement)
try:
    from Dashboard.services.visual_search import process_visual_query
    VISUAL_SEARCH_AVAILABLE = True
    print("[SUCCESS] Visual search processor loaded successfully!")
except ImportError as e:
    print(f"[WARNING] Could not import visual search processor: {e}")
    VISUAL_SEARCH_AVAILABLE = False


# Import RAG System for contextual responses based on OCR text
# Old RAG modules removed - using qkd_assistant_multi_index instead
RAG_SYSTEM_AVAILABLE = RAG_AVAILABLE


# Serve SOP PDFs from local folder for citation links
@app.route('/docs/<path:filename>')
def serve_docs(filename):
    try:
        from config import SOP_DOCUMENTS_PATH
    except Exception:
        # Default to data/sop_documents relative to repo if config missing
        SOP_DOCUMENTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sop_documents')
    # Security: only serve files from this directory
    return send_from_directory(SOP_DOCUMENTS_PATH, filename, as_attachment=False)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔧 ENHANCED: Response post-processing for consistent formatting
def post_process_llm_response(response: str, mode: str = "chat") -> str:
    """
    ULTRA-AGGRESSIVE post-processing for consistent formatting.
    Converts any wall of text into proper structured format.
    """
    if not response or not response.strip():
        return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
    
    response = response.strip()
    
    # STEP 1: Check if already well-formatted
    has_structure = any(pattern in response for pattern in [
        "\n1.", "\n2.", "**STEPS:**", "**KEY POINTS:**", "**ANSWER:**"
    ])
    
    # STEP 2: Only apply gentle HTML formatting, preserve content structure
    # Remove aggressive conversion to maintain RAG content quality
    logger.info("[EDIT] Frontend: Applying HTML formatting while preserving content structure")
    
    # STEP 3: Apply general formatting improvements
    response = _fix_common_formatting_issues(response)
    
    return response

# Removed unused formatting functions:
# - _convert_wall_of_text_frontend()
# - _convert_to_steps_format_frontend()
# - _convert_to_points_format_frontend()
# These were disabled to maintain RAG content quality

def _fix_common_formatting_issues(text: str) -> str:
    """Fix common LLM formatting inconsistencies."""
    import re
    
    # Fix bullet points
    text = re.sub(r'^(\s*)[-*]\s*', r'\1• ', text, flags=re.MULTILINE)
    
    # Fix numbered lists with inconsistent spacing
    text = re.sub(r'^(\s*)(\d+)\.\s+', r'\1\2. ', text, flags=re.MULTILINE)
    
    # Ensure code blocks are properly formatted
    text = re.sub(r'```(\w+)?\n?(.*?)\n?```', r'```\1\n\2\n```', text, flags=re.DOTALL)
    
    # Fix headers with inconsistent spacing
    text = re.sub(r'^(#{1,3})\s*(.+)', r'\1 \2', text, flags=re.MULTILINE)
    
    # Remove excessive blank lines
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    return text.strip()

# Direct processing functions (no HTTP calls to api.py needed!)
def process_sop_mode(user_input: str, document_filter: str = "all"):
    """Process query with SOP/RAG mode using unified MySQL vector store"""
    try:
        from Dashboard.services.qkd_assistant_multi_index import get_qkd_answer_multi_index

        # Pass document filter to the MySQL RAG system
        filter_param = document_filter if document_filter != "all" else None
        logger.info(f"[RAG] SOP mode query with filter: {filter_param}")

        response, confidence = get_qkd_answer_multi_index(
            question=user_input,
            document_filter=filter_param,
            top_k=5
        )

        return response, confidence
    except Exception as e:
        logger.error(f"SOP mode error: {e}")
        return f"Error processing SOP query: {str(e)}", 0.1

def process_chat_mode(user_input: str):
    """Process query with pure LLM (no RAG, no tools)"""
    try:
        # Direct LLM call without RAG or tools
        from Dashboard.services.llm_client import generate_answer, health_check

        # Check if Ollama is available first
        if not health_check():
            return "Ollama server is not available. Please start Ollama with 'ollama serve' and ensure the phi3:3.8b-mini-instruct-4k-fp16 model is installed.", 0.1

        # Generate response using Ollama with conversational prompt
        response = generate_answer(
            prompt=f"You are a helpful AI assistant. Please provide a natural, conversational response to the following: {user_input}",
            temperature=0.7,
            top_p=0.9
        )

        if response:
            return response, 0.8  # Moderate confidence for pure LLM
        else:
            return "I apologize, but I couldn't generate a response. Please try again.", 0.1

    except Exception as e:
        logger.error(f"Chat mode error: {e}")
        return f"Chat mode error: {str(e)}. Make sure Ollama is running.", 0.1

def process_agent_mode(user_input: str):
    """Process query through agent with tools and reasoning"""
    try:
        if not AGENT_AVAILABLE:
            logger.warning("Agent not available, falling back to SOP mode")
            return process_sop_mode(user_input)

        # Get or create agent instance
        agent = get_agent()

        # Process query with full agent capabilities
        response, metadata = agent.process_query_with_investigation(user_input)

        # Extract confidence and reasoning from metadata
        confidence = 0.9 if not metadata.get("error") else 0.3

        # Build reasoning information for UI
        reasoning = {
            "steps": metadata.get("reasoning_steps", 0),
            "tools_used": metadata.get("tools_used", []),
            "processing_time": metadata.get("processing_time", 0)
        }

        return response, confidence, reasoning

    except Exception as e:
        logger.error(f"Agent mode error: {e}, falling back to SOP mode")
        return process_sop_mode(user_input)

@app.teardown_appcontext
def teardown_db(error):
    close_db(error)

# Register the blueprint
app.register_blueprint(auth)

@app.before_request
def require_login():
    exempt_routes = [
        'auth.login',
        'static',
        # Document operations
        'list_documents',
        'list_documents_detailed',
        'upload_pdf',
        'upload_status',
        'delete_document',
        'rebuild_indexes',
        # Chatbot operations - CRITICAL!
        'chatbot_api',
        'chatbot_stream',
        'get_stream_progress',
        # System status
        'ai_status',
        'get_features'
    ]  
    if request.endpoint in exempt_routes:
        return

    token = request.cookies.get('session_token')
    if not token:
        flash('Please log in.', 'warning')
        return redirect(url_for('auth.login'))

    with db_transaction() as cursor:
        # Get user from active session
        cursor.execute("""
            SELECT s.user_id, u.name
            FROM user_sessions s
            JOIN users u ON s.user_id = u.uuid
            WHERE s.session_token = %s AND s.expires_at > NOW()
        """, (token,))
        user = cursor.fetchone()

        if not user:
            flash('Session expired or invalid. Please log in again.', 'warning')
            return redirect(url_for('auth.login'))

        # Fetch permissions based on role
        cursor.execute("""
            SELECT p.permission_name
            FROM users u
            JOIN roles_permissions rp ON u.role_id = rp.role_id
            JOIN permissions p ON rp.permission_id = p.id
            WHERE u.uuid = %s
        """, (user['user_id'],))
        permissions = [row['permission_name'] for row in cursor.fetchall()]

        g.user_id = user['user_id']
        g.permissions = permissions

@app.context_processor
def inject_permissions():
    return dict(permissions=getattr(g, 'permissions', []))

@app.after_request
def apply_security_headers(response):
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://code.jquery.com "
        "https://cdn.datatables.net https://unpkg.com https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdn.datatables.net"
        "https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
        "font-src 'self' https://cdn.jsdelivr.net https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
        "object-src 'none'; "
        "img-src 'self' data:; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self';"
    )
    response.headers['Strict-Transport-Security'] = 'max-age=63072000; includeSubDomains; preload'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response

@app.context_processor
def inject_csrf():
    return dict(csrf_token=generate_csrf)

@app.after_request
def add_no_cache_headers(response):
    if request.cookies.get('session_token'):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# Return JSON for CSRF errors so frontend can handle gracefully
@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    return jsonify({
        "success": False,
        "error": f"CSRF error: {e.description}"
    }), 400

@app.route("/")
def home_redirect():
    # Check if user is logged in
    session_token = request.cookies.get('session_token')
    if not session_token:
        return redirect(url_for('login'))
    
    # Check user permissions
    PERMISSION_TO_ROUTE = {
    "fault.view_overview": "fault_overview",
    "path.view_overview": "path_overview",
    "alert.view_overview": "security_overview",
    "performance.view_overview": "performance_overview",
    "user.view_overview": "user_overview",
    "path.view_network": "network_topology",
    "policy.view": "policy_view",
    "fault.view_details": "alarm_list",
    "alert.view_unresolved": "alert_list",
    "path.sesssion.view_table": "kmb_session",
    "audit.view_policies": "policy_audit",
    "audit.view_alerts": "alert_audit_log",
    "audit.view_faults": "alarm_audit_log",
    "performance.view_individual": "individual_performance_metrics"
    }

    user_permissions = g.permissions


    # Find first route they can access
    for perm, route_name in PERMISSION_TO_ROUTE.items():
        if perm in user_permissions:
            return redirect(url_for(route_name))

    # If no permission matched, show error or logout
    return redirect(url_for('unauthorized'))

@app.route("/network-topology")
@permission_required('path.view_network')
def network_topology():
    try:
        with db_transaction(commit=True) as cursor:

            cursor.execute("SELECT * FROM nodes")
            nodes = cursor.fetchall()

            # Build Cytoscape-compatible elements list
            node_map = {}
            elements = []
            for node in nodes:
                node_id = node['node_id']
                node_map[node_id] = {
                    "node_id": node_id,
                    "status": node['status'],
                    "type": node['type'],
                    "last_updated": node['last_updated'],
                    "key_pool_remaining": node['key_pool_remaining'],
                    "aggregated_key_rate": node['aggregated_key_rate'],
                    "avg_photon_loss": node['avg_photon_loss'],
                    "consumption_rate": node['consumption_rate'],
                    "site": node['site'],
                    "state": node['state']
                }
                elements.append({
                    "data": {"id": node_id},
                    "classes": f"{node['status']} {node['type']} {node['state']}", 
                })
        
                cursor.execute("""
                    SELECT l.*, src.node_id AS source_node_id, dst.node_id AS destination_node_id
                    FROM links l
                    INNER JOIN nodes src ON l.source_node = src.id
                    INNER JOIN nodes dst ON l.destination_node = dst.id;
                """)
                links = cursor.fetchall()
                link_map = {}
                for link in links:
                    elements.append({
                        "data": {
                            "id": link["link_id"],
                            "source": link["source_node_id"],
                            "target": link["destination_node_id"]
                        },
                        "classes": f"{link['status']} {link['state']}", 
                    })
                    link_map[link["link_id"]] = {
                        "link_id": link["link_id"],
                        "link_type": link["link_type"],
                        "status": link["status"],
                        "qber": link["qber"],
                        "visibility": link["visibility"],
                        "key_rate": link["key_rate"],
                        "photon_loss": link["photon_loss"],
                        "last_updated": link["last_updated"],
                        "state": link["state"]
                }

            
            # Total number of devices (do not count devices that are deleted or temporary)
            # total_devices = len(node_map)
            total_devices = sum(1 for device in node_map.values() if device["state"] == "Created")

            # Count number of online devices
            online_devices = sum(1 for device in node_map.values() if (device["status"] == "Online" and device["state"] == "Created") )

            # Count number of temporary devices
            temporary_devices = sum(1 for device in node_map.values() if device["state"] == "Temporary")
    
    except Exception as e:
        # logging.error(f"Error fetching network topology data: {e}", exc_info=True)
        flash('An error occurred while fetching network topology data. Please try again later.', 'danger')
        return render_template("network-topology.html", node_map={}, link_map={}, elements=[], total_devices=0, online_devices=0, temporary_devices=0)
    
    return render_template("network-topology.html",
                           node_map=node_map, link_map=link_map,
                           elements=elements, total_devices = total_devices, online_devices = online_devices, temporary_devices = temporary_devices)

# create a new node
@app.route('/create-node', methods=['POST'])
@permission_required('path.node.create')
def create_node():
    node_type = request.form['nodeType']
    node_status = request.form['nodeStatus']
    node_site = request.form['nodeSite']

    node_id = f'TMP_{random.randint(0, 999):03}'
    node_state = 'Temporary'

    if node_type == 'QKD':
        aggregated_key_rate = request.form['aggregatedKeyRate']
        avg_photon_loss = request.form['avgPhotonLoss']
        consumption_rate = request.form['consumptionRate']
        key_pool_remaining = request.form['keyPoolRemaining']
    else:
        aggregated_key_rate = None
        avg_photon_loss = None
        consumption_rate = None
        key_pool_remaining = None   
    
    with db_transaction(commit=True) as cursor:
        try:
            # Insert into your database
            cursor.execute("""
                INSERT INTO nodes (node_id, type, status, key_pool_remaining, aggregated_key_rate, avg_photon_loss, consumption_rate, site, state, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (node_id, node_type, node_status, key_pool_remaining, aggregated_key_rate, avg_photon_loss, consumption_rate, node_site, node_state))
            print(f"Node {node_id} created successfully.")

            return redirect(url_for('network_topology'))
        
        except Exception as e:
            # logging.error(f"Error creating node: {e}", exc_info=True)
            flash('An error occurred while creating the node. Please try again later.', 'danger')

    return redirect(url_for('network_topology'))

# update an existing node
@app.route('/update-node', methods=['POST'])
@permission_required('path.node.edit')
def update_node():
    node_id = request.form['nodeId']
    node_type = request.form['nodeTypeChange']
    node_status = request.form['nodeStatusChange']
    node_site = request.form['nodeSiteChange']

    if node_type == 'QKD':
        aggregated_key_rate = request.form['aggKeyRateChange']
        avg_photon_loss = request.form['avgPhotonLossChange']
        consumption_rate = request.form['consumptionRateChange']
        key_pool_remaining = request.form['keyPoolRemainingChange']
    else:
        aggregated_key_rate = None
        avg_photon_loss = None
        consumption_rate = None
        key_pool_remaining = None   
    with db_transaction(commit=True) as cursor:
        try:
            # Update in your database
            cursor.execute("""
                UPDATE nodes 
                SET type = %s, status = %s, key_pool_remaining = %s, aggregated_key_rate = %s, 
                    avg_photon_loss = %s, consumption_rate = %s, site = %s, last_updated = NOW()
                WHERE node_id = %s
            """, (node_type, node_status, key_pool_remaining, aggregated_key_rate, avg_photon_loss, consumption_rate, node_site, node_id))
            print(f"Node {node_id} updated successfully.")

            return redirect(url_for('network_topology'))
        
        except Exception as e:
            # logging.error(f"Error updating node {node_id}: {e}", exc_info=True)
            flash('An error occurred while updating the node. Please try again later.', 'danger')
    return redirect(url_for('network_topology'))

# delete a node
@app.route('/delete-node', methods=['POST'])
@permission_required('path.node.delete')
def delete_node():
    node_id = request.form['nodeId']
    with db_transaction(commit=True) as cursor:
        # check if the node_id starts with "TMP" to ensure it's a temporary node
        if node_id.startswith("TMP"):
            try:
                # Delete from your database
                cursor.execute("DELETE FROM nodes WHERE node_id = %s", (node_id,))
                return redirect(url_for('network_topology'))
            
            except Exception as e:
                # logging.error(f"Error deleting node {node_id}: {e}", exc_info=True)
                flash('An error occurred while deleting the node. Please try again later.', 'danger')
    
    return redirect(url_for('network_topology'))
        
# create a new link
@app.route('/create-link', methods=['POST'])
@permission_required('path.link.create')
def create_link():
    source_node = request.form['sourceNode']
    destination_node = request.form['destinationNode']
    link_status = request.form['linkStatus']
    link_id = f'TMP_{random.randint(0, 999):03}'
    link_state = 'Temporary'
    with db_transaction(commit=True) as cursor:
        if source_node != destination_node:
            try:
                # Check if the source and target nodes exist
                # Lookup source ID
                cursor.execute("SELECT id, type FROM nodes WHERE node_id = %s", (source_node,))
                source_row = cursor.fetchone()
                if not source_row:
                    flash("Invalid source node", "danger")
                    return redirect(url_for('network_topology'))
                source_id = source_row['id']
                source_type = source_row['type']

                # Lookup target ID
                cursor.execute("SELECT id, type FROM nodes WHERE node_id = %s", (destination_node,))
                destination_row = cursor.fetchone()
                if not destination_row:
                    flash("Invalid destination node", "danger")
                    return redirect(url_for('network_topology'))
                destination_id = destination_row['id']
                destination_type = destination_row['type']

                # Check if the link already exists
                cursor.execute("""
                    SELECT * FROM links
                    WHERE (source_node = %s AND destination_node = %s)
                    OR (source_node = %s AND destination_node = %s)
                """, (source_id, destination_id, destination_id, source_id))
                existing_link = cursor.fetchone()
                if existing_link:
                    flash("Link already exists", "danger")
                    return redirect(url_for('network_topology'))
                
                # check the type of the link
                if source_type == 'QKD' and destination_type == 'QKD':
                    link_type = 'Quantum'
                else:
                    link_type = 'Classical'

                # Insert into your database
                cursor.execute("""
                    INSERT INTO links (source_node, destination_node, link_type, status, link_id, state, last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                """, (source_id, destination_id, link_type, link_status, link_id, link_state))
            
            except Exception as e:
                # logging.error(f"Error creating link: {e}", exc_info=True)
                flash('An error occurred while creating the link. Please try again later.', 'danger')
    return redirect(url_for('network_topology'))

# update an existing link
@app.route('/update-link', methods=['POST'])
@permission_required('path.link.edit')
def update_link():
    link_id = request.form['linkId']
    link_status = request.form['linkStatusChange']

    # Only quantum links have these properties
    qber = request.form.get('qberChange')
    visibility = request.form.get('visibilityChange')
    key_rate = request.form.get('keyRateChange')
    photon_loss = request.form.get('photonLossChange')
    with db_transaction(commit=True) as cursor:
        try:
            # Update in your database
            cursor.execute("""
                UPDATE links 
                SET status = %s,  last_updated = NOW(), qber = %s, visibility = %s, key_rate = %s, photon_loss = %s
                WHERE link_id = %s
            """, (link_status, link_id, qber, visibility, key_rate, photon_loss))
            print(f"Link {link_id} updated successfully.")
        
        except Exception as e:
            # logging.error(f"Error updating link {link_id}: {e}", exc_info=True)
            flash('An error occurred while updating the link. Please try again later.', 'danger')
    
    return redirect(url_for('network_topology'))

# delete a link
@app.route('/delete-link', methods=['POST'])
@permission_required('path.link.delete')
def delete_link():
    link_id = request.form['linkId']
    with db_transaction(commit=True) as cursor:
        # check if the link_id starts with "TMP" to ensure it's a temporary link
        if link_id.startswith("TMP"):
            try:
                # Delete from your database
                cursor.execute("DELETE FROM links WHERE link_id = %s", (link_id,))
                flash(f"Link {link_id} deleted successfully.")            
            except Exception as e:
                # logging.error(f"Error deleting link {link_id}: {e}", exc_info=True)
                flash('An error occurred while deleting the link. Please try again later.', 'danger')
    return redirect(url_for('network_topology'))

# @app.route("/session-audit")
# def session_audit():
#     return render_template("session-audit.html")

# fault management
@app.route("/fault-overview")
@permission_required('fault.view_overview')
def fault_overview():
    try:
        with db_transaction(commit=True) as cursor:
            # get the number of active faults
            active_faults = fs.get_active_faults_stats(cursor)
            # get the average time to resolve and acknowledge faults
            average_time_to_resolve, average_time_to_acknowledge = fs.get_average_times(cursor)
            # count the number of faults and their dates for the last 5 days
            num_of_faults, dates = fs.get_faults_last_5_days(cursor)
            # count the number of resolved faults
            num_resolved_faults = fs.get_resolved_faults_last_5_days(cursor)
            # get the affected components
            component_array, component_count_array = fs.get_affected_components(cursor)
            
            return render_template("fault-overview.html", active_faults=active_faults, average_time_to_resolve=average_time_to_resolve, dates=dates, num_of_faults=num_of_faults, num_resolved_faults = num_resolved_faults, average_time_to_acknowledge = average_time_to_acknowledge, component_array = component_array, component_count_array = component_count_array)
    except Exception as e:
        # logging.error(f"Error fetching fault overview data: {e}", exc_info=True)
        flash('An error occurred while fetching fault overview data. Please try again later.', 'danger')
        return render_template("fault-overview.html", active_faults={}, average_time_to_resolve=0, dates=[], num_of_faults={}, num_resolved_faults = 0, average_time_to_acknowledge = 0, component_array = [], component_count_array = [])
@app.route("/alarm-list")
@permission_required('fault.view_details')
def alarm_list():
    try:
        with db_transaction(commit=True) as cursor:
            faults = fs.get_alarm_list(cursor)
            # get users who have the permission to resolve the fault
            users = fs.get_users_with_permission(cursor, 'fault.resolve')
            return render_template("alarm-list.html", faults=faults, users=users)
    except Exception as e:
        logging.error(f"Error fetching alarm list: {e}", exc_info=True)
        flash('An error occurred while fetching the alarm list. Please try again later.', 'danger')
        return render_template("alarm-list.html", faults=[])

@app.route("/acknowledge-alarm", methods=['POST'])
@permission_required('fault.acknowledge')
def acknowledge_alarm():
    try:
        fault_id = request.form['faultId']
        acknowledgement_notes = request.form['acknowledgementNotes']
        acknowledged = request.form.get('acknowledged')
        assigned_to = request.form.get('assignedTo')
        currentuser = g.user_id

        with db_transaction(commit=True) as cursor:
            cursor.execute("""SELECT id, status FROM faults WHERE fault_id = %s;""", (fault_id,))
            fault = cursor.fetchone()

            if acknowledged != 'yes':
                flash("You must acknowledge the fault by ticking the checkbox.", "danger")
                fs.fault_log(cursor, fault['id'], currentuser, 'Acknowledged', 'Failed', "Failed to acknowledge the fault.")
                return redirect(url_for('alarm_list'))
            elif acknowledgement_notes == None:
                flash("You must provide acknowledgement notes.", "danger")
                fs.fault_log(cursor, fault['id'], currentuser, 'Acknowledged', 'Failed', "Failed to provide acknowledgment notes")
                return redirect(url_for('alarm_list'))
            elif len(acknowledgement_notes) > 255:
                flash("Acknowledgement notes must be less than 255 characters.", "danger")
                fs.fault_log(cursor, fault['id'], currentuser, 'Acknowledged', 'Failed', "Acknowledgement notes must be less than 255 characters.")
                return redirect(url_for('alarm_list'))
            elif fault is None:
                flash("Fault not found.", "danger")
                fs.fault_log(cursor, None, currentuser, 'Acknowledged', 'Failed', f"Attempt to acknowledge a non-existent fault {fault_id}.")
                return redirect(url_for('alarm_list'))
            elif fault['status'] != 'Open':
                flash("You can only acknowledge an open fault.", "danger")
                fs.fault_log(cursor, fault['id'], currentuser, 'Acknowledged', 'Failed', "Attempt to acknowledge an unopened fault.")
                return redirect(url_for('alarm_list'))
            
            cursor.execute("""SELECT name FROM users WHERE uuid = %s;""", (assigned_to,))
            user = cursor.fetchone()
            
            # check whether the assigned_to has the permission to resolve
            users = fs.get_users_with_permission(cursor, "fault.resolve")
            is_authorised = any(user['uuid'] == currentuser for user in users)
            if not is_authorised:
                flash("The user you are trying to assign do not have permission to resolve this fault.", "danger")
                fs.fault_log(cursor, fault['id'], currentuser, 'Acknowledged', 'Failed', f"{user['name']} does not have permission to resolve this fault.")
                return redirect(url_for('alarm_list'))

            # need to change when rbac is created
            cursor.execute("""UPDATE faults SET status = 'Ongoing', acknowledgement_notes = %s, acknowledged_by = %s, assigned_to = %s, acknowledged_at = NOW() WHERE fault_id = %s;""", (acknowledgement_notes, currentuser, assigned_to, fault_id))
            # insert into the fault log
            fs.fault_log(cursor, fault['id'], currentuser, 'Acknowledged', 'Success', f"Acknowledged the fault {fault_id} and assigned to {user['name']} successfully.")
            flash(f"Alarm {fault_id} acknowledged successfully.", 'success')
    except Exception as e:
        # logging.error(f"Error acknowledging alarm {fault_id}: {e}", exc_info=True)
        flash('An error occurred while acknowledging the alarm. Please try again later.', 'danger')
        
    return redirect(url_for('alarm_list'))

@app.route("/resolve-alarm", methods=['POST'])
@permission_required('fault.resolve')
def resolve_alarm():
    try:
        fault_id = request.form['faultId']
        remediation = request.form['remediation']
        clearing_condition = request.form['clearingCondition']
        acknowledged = request.form.get('acknowledged')
        currentuser = g.user_id
        with db_transaction(commit=True) as cursor:
            cursor.execute("""SELECT id, assigned_to, status, alert_id FROM faults WHERE fault_id = %s;""", (fault_id,))
            fault = cursor.fetchone()

            if acknowledged != 'yes':
                flash("You must acknowledge the fault by ticking the checkbox.", "danger")
                fs.fault_log(cursor, fault['id'], currentuser, 'Resolved', 'Failed', "Failed to acknowledge the fault.")
                return redirect(url_for('alarm_list'))
            elif clearing_condition == None:
                flash("You must provide clearning condition.", "danger")
                fs.fault_log(cursor, fault['id'], currentuser, 'Resolved', 'Failed', "Failed to provide clearing condition")
                return redirect(url_for('alarm_list'))
            elif remediation == None:
                flash("You must provide remediation.", "danger")
                fs.fault_log(cursor, fault['id'], currentuser, 'Resolved', 'Failed', "Failed to provide remediation")
                return redirect(url_for('alarm_list'))
            elif len(remediation) > 255:
                flash("Remediation must be less than 255 characters.", "danger")
                fs.fault_log(cursor, fault['id'], currentuser, 'Resolved', 'Failed', "Remediation must be less than 255 characters.")
                return redirect(url_for('alarm_list'))
            elif len(clearing_condition) > 255:
                flash("Clearing condition must be less than 255 characters.", "danger")
                fs.fault_log(cursor, fault['id'], currentuser, 'Resolved', 'Failed', "Clearing condition must be less than 255 characters.")
                return redirect(url_for('alarm_list'))
            elif fault is None:
                flash("Fault not found.", "danger")
                fs.fault_log(cursor, None, currentuser, 'Resolved', 'Failed', f"Attempt to resolve a non-existent fault {fault_id}.")
                return redirect(url_for('alarm_list'))
            elif fault['status'] != 'Ongoing':
                flash("You can only resolve an ongoing fault.", "danger")
                fs.fault_log(cursor, fault['id'], currentuser, 'Resolved', 'Failed', "Attempt to resolve a fault that is not ongoing.")
                return redirect(url_for('alarm_list'))
            elif fault['assigned_to'] != currentuser:
                flash("You are not assigned to resolve this fault.", "danger")
                fs.fault_log(cursor, fault['id'], currentuser, 'Resolved', 'Failed', "Unauthorised attempt to resolve a fault not assigned to the user.")
                return redirect(url_for('alarm_list'))
    
            cursor.execute("""UPDATE faults SET status = "Resolved", remediation=%s, clearing_condition = %s, resolved_by = %s, resolved_at = NOW() WHERE fault_id = %s;""", (remediation, clearing_condition, currentuser, fault_id))
            # insert into the fault log
            fs.fault_log(cursor, fault['id'], currentuser, 'Resolved', 'Success', f"Resolved the fault {fault_id} successfully.")
            
            if fault['alert_id'] is not None:
                # if it is, delete it from the alert table
                cursor.execute("""UPDATE alerts SET status = "Resolved", resolved_at = NOW() WHERE id = %s;""", (fault['alert_id'],))
                fs.alert_log(cursor, fault['alert_id'], currentuser, 'Resolved', 'Success', f'Alert resolved successfully due to resolution of fault {fault_id}.')
            flash(f"Alarm {fault_id} resolved successfully.", 'success')
    except Exception as e:
        # logging.error(f"Error resolving alarm {fault_id}: {e}", exc_info=True)
        flash('An error occurred while resolving the alarm. Please try again later.', 'danger')
    
    return redirect(url_for('alarm_list'))

# performance management
@app.route("/performance-overview")
@permission_required('performance.view_overview')
def performance_overview():
    try:
        with db_transaction(commit=True) as cursor:
            cursor.execute("""SELECT qber, visibility, key_rate FROM qkd.links WHERE link_type = "Quantum";""")
            quantum_metrics = cursor.fetchall()
            number_of_links = 0
            total_qber = 0
            total_visibility = 0
            total_keyrate = 0
            for metric in quantum_metrics:
                number_of_links += 1
                total_qber += metric['qber']
                total_visibility += metric['visibility']
                total_keyrate += metric['key_rate']
            
            average_qber = round(total_qber / number_of_links,2)
            average_visibility = round(total_visibility / number_of_links,2)
            average_keyrate = round(total_keyrate / number_of_links,2)

            cursor.execute("""SELECT key_pool_remaining FROM nodes WHERE type = "QKD";""")
            key_pool = cursor.fetchall()
            key_pool_remaining = 0
            number_of_nodes = 0
            for metric in key_pool:
                number_of_nodes += 1
                key_pool_remaining += metric['key_pool_remaining']

            # get key summaries CURDATE()
            cursor.execute("""SELECT SUM(keys_generated) AS total_keys_generated,
            SUM(keys_consumed) AS total_keys_consumed,
            SUM(keys_expired) AS total_keys_expired, SUM(avg_retention_time * keys_generated) AS weighted_sum FROM keys_summaries WHERE DATE(timestamp) = '2025-06-17';""")
            key_summaries = cursor.fetchone()
            key_summaries['key_efficiency'] = round((key_summaries['total_keys_consumed'] / key_summaries['total_keys_generated']) * 100,2) # in percentage
            if key_summaries['total_keys_generated'] > 0:
                weighted_avg_retention_time = round(key_summaries['weighted_sum'] / float(key_summaries['total_keys_generated']),2)
            else:
                weighted_avg_retention_time = 0
            
            # get keys generated and keys consumed per day
            cursor.execute("""
            SELECT 
                date_series.date AS dates, 
                SUM(keys_summaries.keys_generated) AS keys_generated, 
                SUM(keys_summaries.keys_consumed) AS keys_consumed
            FROM  
                (SELECT CURDATE() - INTERVAL a DAY AS date 
                FROM (SELECT 0 AS a UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4) AS days) AS date_series
            LEFT JOIN keys_summaries 
                ON DATE(keys_summaries.timestamp) = date_series.date
            GROUP BY date_series.date
            ORDER BY date_series.date ASC;
            """)
            keys_per_day = cursor.fetchall()    
            dates = [row['dates'].strftime('%d-%m-%Y')for row in keys_per_day]
            keys_generated = [row['keys_generated'] if row['keys_generated'] is not None else 0 for row in keys_per_day]
            keys_consumed = [row['keys_consumed'] if row['keys_consumed'] is not None else 0 for row in keys_per_day]

            return render_template("performance-overview.html", average_qber=average_qber, average_visibility=average_visibility, average_keyrate=average_keyrate, key_summaries=key_summaries, key_pool_remaining=key_pool_remaining, number_of_nodes=number_of_nodes, weighted_avg_retention_time=weighted_avg_retention_time, dates=dates, keys_generated=keys_generated, keys_consumed=keys_consumed)
    
    except Exception as e:
        # logging.error(f"Error fetching performance overview data: {e}", exc_info=True)
        flash('An error occurred while fetching performance overview data. Please try again later.', 'danger')
        return render_template("performance-overview.html", average_qber=0, average_visibility=0, average_keyrate=0, key_summaries={}, key_pool_remaining=0, number_of_nodes=0, weighted_avg_retention_time=0, dates=[], keys_generated=[], keys_consumed=[])

@app.route("/individual-performance-metrics")
@permission_required('performance.view_individual')
def individual_performance_metrics():
    try:
        with db_transaction(commit=True) as cursor:
            cursor.execute("""SELECT link_id FROM qkd.links WHERE link_type = "Quantum";""")
            links = cursor.fetchall()
            cursor.execute("""SELECT node_id FROM qkd.nodes WHERE type = "QKD";""")
            nodes = cursor.fetchall()
            return render_template("individual-performance-metrics.html", links=links, nodes=nodes)
    except Exception as e:
        # logging.error(f"Error fetching individual performance metrics: {e}", exc_info=True)
        flash('An error occurred while fetching individual performance metrics. Please try again later.', 'danger')
        return render_template("individual-performance-metrics.html", links=[], nodes=[])

@app.route("/qkd-metrics")
def qkd_metrics():
    return render_template("qkd-metrics.html")

# @app.route("/key-performance")
# def key_performance():
#     return render_template("key-performance.html")

@app.route("/path-overview")
@permission_required('path.view_overview')
def path_overview():
    try:
        with db_transaction(commit=True) as cursor:
            # identify the top 5 degraded links
            cursor.execute("""SELECT l.link_id, l.qber, l.visibility, l.key_rate, l.photon_loss, l.last_updated,
            src.node_id AS source_node_id, dst.node_id AS destination_node_id,  COALESCE(session_counts.active_session_count, 0) AS active_session_count
            FROM links l
            JOIN nodes src ON l.source_node = src.id
            JOIN nodes dst ON l.destination_node = dst.id
            LEFT JOIN (SELECT sp.link_id, COUNT(DISTINCT s.id) AS active_session_count
            FROM sessions s
            JOIN session_path sp ON s.id = sp.session_id
            WHERE s.status = 'Active'
            GROUP BY  sp.link_id) session_counts ON l.id = session_counts.link_id
            WHERE l.link_type = 'Quantum'; """)
            degraded_links = cursor.fetchall()
            total_healthy = 0
            total_warning = 0
            total_critical = 0
            for degraded_link in degraded_links:
                degraded_link['instability_score'], degraded_link['qlqi'] = calculate_instability_score(degraded_link['qber'], degraded_link['visibility'], degraded_link['key_rate'], degraded_link['photon_loss'])
                if degraded_link['qlqi'] < 40:
                    degraded_link['severity'] = 'Critical'
                    degraded_link['severity_color'] = 'danger'
                elif degraded_link['qlqi'] < 70:
                    degraded_link['severity'] = 'Warning'
                    degraded_link['severity_color'] = 'warning'
                else:
                    degraded_link['severity'] = 'Healthy'
                    degraded_link['severity_color'] = 'success'
                
                if degraded_link['active_session_count'] > 0:
                    if degraded_link['instability_score'] <30:
                        total_healthy += 1
                    elif degraded_link['instability_score'] < 60:
                        total_warning += 1
                    else:
                        total_critical += 1
            
            # live path status
            live_path_status =  [
            ['success', 'Healthy', total_healthy],
            ['warning', 'Warning', total_warning],
            ['danger', 'Critical',total_critical]
            ]

            if total_critical == 0 and total_warning > 0:
                link_health = [f"{total_warning} Warning", "warning"]
            elif total_critical > 0:
                link_health = [f"{total_critical} Critical", "danger"]
            elif total_critical == 0 and total_warning == 0 and total_healthy > 0:
                link_health = [f"All Links Healthy", "success"]
            else:
                link_health = ["No Links Found", "info"]

            # sort the top 5 degraded links by instability score
            degraded_links.sort(key=lambda x: x['instability_score'], reverse=True)
            top_5_degraded_links = degraded_links[:5]

            # identify the number of blocked sessions
            cursor.execute("""SELECT COUNT(*) AS blocked_sessions FROM sessions WHERE status = 'Failed'; """)
            blocked_sessions = cursor.fetchone()['blocked_sessions']

            # identify the top 3 most commonly used paths and path status
            cursor.execute("""SELECT s.session_id, sp.step_order, l.source_node, l.destination_node, l.status, l.link_id
                            FROM session_path sp
                            JOIN links l ON sp.link_id = l.id
                            JOIN sessions s ON sp.session_id = s.id
                            WHERE s.status = 'Active'
                            ORDER BY s.session_id, sp.step_order ASC ; """)
            results = cursor.fetchall()
            session_paths = defaultdict(list)
            for row in results:
                session_id = row['session_id']
                source_node = row['source_node']
                destination_node = row['destination_node']
                session_paths[session_id].append((source_node, destination_node))
            
            segment_counter = Counter()
            for session_id, links in session_paths.items():
                # Track segments we've already counted in this session
                counted_segments = set()

                nodes = []
                current_node = links[0][0]
                nodes.append(current_node)
                for link in links:
                    if link[0] == current_node:
                        next_node = link[1]
                    else:
                        next_node = link[0]
                    
                    nodes.append(next_node)
                    current_node = next_node
                
                for i in range(len(nodes) - 2):
                    segment = (nodes[i], nodes[i+1], nodes[i+2])
                    # make the bidirectional segments equivalent
                    segment_sorted = tuple(sorted(segment))
                    
                    # Count the segment only if it hasn't been counted in this session
                    if segment_sorted not in counted_segments:
                        segment_counter[segment_sorted] += 1
                        counted_segments.add(segment_sorted)

            top_3_segments = segment_counter.most_common(3)
            unique_ids = set(i for group, _ in top_3_segments for i in group)

            # Lookup node_id for each id
            placeholders = ','.join(['%s'] * len(unique_ids))
            query = f"SELECT id, node_id FROM nodes WHERE id IN ({placeholders})"
            cursor.execute(query, list(unique_ids))
            results = cursor.fetchall()
            id_to_node_id = {row['id']: row['node_id'] for row in results}

            # Map segments
            converted_data_top_3 = []
            for ids, value in top_3_segments:
                try:
                    node_ids = [id_to_node_id[i] for i in ids]
                    converted_data_top_3.append((node_ids, value))
                except KeyError as e:
                    print(f"Missing node_id for internal id {e}")

        return render_template("path-overview.html", link_health=link_health, top_5_degraded_links = top_5_degraded_links, blocked_sessions = blocked_sessions, converted_data_top_3 = converted_data_top_3, live_path_status = live_path_status)
    except Exception as e:
        # logging.error(f"Error fetching path overview data: {e}", exc_info=True)
        flash('An error occurred while fetching path overview data. Please try again later.', 'danger')
        return render_template("path-overview.html", link_health=["No Links Found", "info"], top_5_degraded_links=[], blocked_sessions=0, converted_data_top_3=[], live_path_status=[])
    
def calculate_instability_score(qber, visibility, key_rate, photon_loss):
        normalised_qber = qber / 11 if qber < 11 else  1 # Assuming that 11% is critical
        normalised_visibility = 1 - (visibility / 100)
        normalised_photon_loss = photon_loss / 100
        normalised_key_rate = 1 - (key_rate / 10) if key_rate < 10 else 0  # Assuming 10 kbps is healthy
        instability_score = ((0.4 * normalised_qber) + (0.3 * normalised_visibility) + (0.1 * normalised_key_rate) + (0.2 * normalised_photon_loss)) * 100
        qlqi = (0.4 * (1-normalised_qber) + 0.3 * (1-normalised_visibility) + 0.1 * (1-normalised_key_rate) + 0.2 * (1-normalised_photon_loss)) * 100 # 0 - 100
        return round(instability_score, 2), round(qlqi, 2) # round off to 2dp

@app.route("/kmb-session")
@permission_required('path.session.view_table')
def kmb_session():
    try:
        search_query = request.args.get('search', '', type=str)
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        export_format = request.args.get('export', None)  

        start_time_str = request.args.get('start_time')
        end_time_str = request.args.get('end_time')

        start_time = None
        end_time = None

        if start_time_str and end_time_str:
            try:
                start_time = datetime.fromisoformat(start_time_str)
                end_time = datetime.fromisoformat(end_time_str)
            except ValueError:
                flash("Invalid time range provided", "danger")
                return redirect(url_for('kmb_session'))
            
        with db_transaction() as cursor:
            query = """SELECT  s.*, 
            source.node_id AS source_node_id, 
            destination.node_id AS destination_node_id
            FROM sessions s
            JOIN nodes source ON s.source_node = source.id
            JOIN nodes destination ON s.destination_node = destination.id WHERE 1=1"""
            
            conditions = []
            params = []

            if search_query:
                conditions.append("""
                    (s.status LIKE %s OR source.node_id LIKE %s OR destination.node_id LIKE %s OR s.session_id LIKE %s)
                """)
                search_term = f'%{search_query}%'
                params.extend([search_term] * 4)

            if start_time and end_time:
                conditions.append("s.start_time BETWEEN %s AND %s")
                params.extend([start_time, end_time])

            if conditions:
                query += " AND " + " AND ".join(conditions)

            query += " ORDER BY s.id DESC"

            cursor.execute(query, tuple(params))

            sessions = cursor.fetchall()
            for session in sessions:
                session_id = session['session_id']
                source_node_id = session['source_node_id']
                path = get_session_nodes(session_id, source_node_id)
                session['path'] = path
            
            if export_format:
                filtered_audits = [{k: v for k, v in row.items() if k != 'id'}
                                    for row in sessions]
                if export_format == 'excel':
                    return others.generate_excel_response("KMB Sessions Log", filtered_audits, "sessionLog")
                elif export_format == 'pdf':
                    return others.generate_pdf_response("KMB Sessions Log", filtered_audits, "sessionLog")
                else:
                    flash("Unsupported export format", "danger")
                    return redirect(url_for('user_audit'))
            
            total = len(sessions)
            start = (page - 1) * per_page
            end = start + per_page
            pagniated_sessions = sessions[start:end]

            total_pages = (total + per_page - 1) // per_page

        return render_template("kmb-session.html", sessions=pagniated_sessions, current_page=page,
                total_pages=total_pages, per_page=per_page, search_query=search_query)
    except Exception as e:
        # logging.error(f"Error fetching KMB session data: {e}", exc_info=True)
        flash('An error occurred while fetching KMB session data. Please try again later.', 'danger')
        return render_template("kmb-session.html", sessions=[], current_page=page,
                total_pages=1, per_page=per_page, search_query=search_query)

def get_session_nodes(session_id, source_node_id):
    with db_transaction() as cursor:
        # Fetch all links in the session path ordered by sequence
        cursor.execute("""
            SELECT sp.step_order, ns.node_id AS source_node_id, nd.node_id AS destination_node_id
            FROM session_path sp
            JOIN links ln ON sp.link_id = ln.id
            JOIN nodes ns ON ln.source_node = ns.id
            JOIN nodes nd ON ln.destination_node = nd.id
            JOIN sessions s ON sp.session_id = s.id
            WHERE s.session_id = %s
            ORDER BY sp.step_order ASC;
        """, (session_id,))

        links = cursor.fetchall()

        if not links:
            return []

        nodes = []

        # Start with the source_node of the first link
        current_node = source_node_id
        nodes.append(current_node)

        for link in links:
            if link['source_node_id'] == current_node:
                next_node = link['destination_node_id']
            else:
                next_node = link['source_node_id']  # Bidirectional fallback

            nodes.append(next_node)
            current_node = next_node  # Move forward

        return nodes


@app.route("/policy-view")
@permission_required('policy.view')
def policy_view():
    try:
        with db_transaction() as cursor:
            cursor.execute("""SELECT * FROM policies WHERE is_deleted = FALSE;""")
            policies = cursor.fetchall()
            for policy in policies:
                if policy['created_by'] is not None:
                    cursor.execute("""SELECT name FROM users WHERE uuid = %s;""", (policy['created_by'],))
                    user = cursor.fetchone()
                    policy['created_by_name'] = user['name']
                if policy['modified_by'] is not None:
                    cursor.execute("""SELECT name FROM users WHERE uuid = %s;""", (policy['modified_by'],))
                    user = cursor.fetchone()
                    policy['modified_by_name'] = user['name']
            
            policy_ids = [policy['id'] for policy in policies]

            # for the rules, we need to fetch them based on the policy_ids
            format_strings = ','.join(['%s'] * len(policy_ids))
            cursor.execute(f"""SELECT * FROM policy_rules WHERE policy_id IN ({format_strings})""", policy_ids)
            rules = cursor.fetchall()

            # for the targets
            cursor.execute(f"""SELECT * FROM policy_target WHERE policy_id IN ({format_strings})""", policy_ids)
            targets = cursor.fetchall()

            # 4. Group rules and targets by policy_id
            from collections import defaultdict
            rule_map = defaultdict(list)
            target_map = defaultdict(list)

            for rule in rules:
                rule_map[rule['policy_id']].append(rule)

            for target in targets:
                target_map[target['policy_id']].append(target)

            # 5. Inject rules and targets into policies
            for policy in policies:
                policy['rules'] = rule_map[policy['id']]
                policy['targets'] = target_map[policy['id']]
            
            policy_types = {"Weightage": "Weightage", "Threshold": "Threshold", "Security": "Security"}
            policy_scope = {"Global": "Global", "Node": "Node", "Link": "Link"}

            # get the qkd links and nodes
            cursor.execute("""SELECT id, link_id FROM links WHERE link_type = "Quantum";""")
            qkd_links = {row['id']: row['link_id'] for row in cursor.fetchall()}
            
            cursor.execute("""SELECT id, node_id FROM nodes WHERE type = "QKD";""")
            qkd_nodes = {row['id']: row['node_id'] for row in cursor.fetchall()}

            for target in targets:
                if target['target_type'] == 'Node':
                    target['target_name'] = qkd_nodes.get(int(target['target_id']), 'Unknown')
                elif target['target_type'] == 'Link':
                    target['target_name'] = qkd_links.get(int(target['target_id']), 'Unknown')
                else:
                    target['target_name'] = "Unknown"

        return render_template("policy-view.html", policies=policies, policy_types=policy_types, policy_scope=policy_scope, qkd_links=qkd_links, qkd_nodes=qkd_nodes)
    
    except Exception as e:
        # logging.error(f"Error fetching policy data: {e}", exc_info=True)
        flash('An error occurred while fetching policy data. Please try again later.', 'danger')
        return render_template("policy-view.html", policies=[], policy_types={}, policy_scope={}, qkd_links={}, qkd_nodes={})

@app.route("/policy-create", methods=['POST'])
@permission_required('policy.create')
def policy_create():
    try:
        policy_name = request.form['policyname']
        policy_type = request.form.get('policytype')
        policy_description = request.form['policydescription']
        created_by = modified_by = g.user_id
        status = "Enabled"
        with db_transaction(commit=True) as cursor:
            if policy_name is None or policy_type is None or policy_description is None:
                flash("All fields are required.", "danger")
                ps.policy_audit(cursor, None, "Created", g.user_id, "Failed", "Attempt to create a policy with missing fields.")
                return redirect(url_for('policy_view'))
            
            # check if the policy name already exists
            existing_policy = ps.get_id_from_policy_name(cursor, policy_name)
            if existing_policy:
                flash("Policy with this name already exists.", "danger")
                ps.policy_audit(cursor, None, "Created", g.user_id, "Failed", "Attempt to create a policy with a name that already exists.")
                return redirect(url_for('policy_view'))
            
            # check the policy type
            if policy_type not in ["Weightage", "Threshold", "Security"]:
                flash("Invalid policy type.", "danger")
                ps.policy_audit(cursor, None, "Created", g.user_id, "Failed", "Attempt to create a policy with an invalid policy type.")
                return redirect(url_for('policy_view'))


            if policy_type == "Security":
                max_login = request.form['loginAttempts']
                scope = "Global"
                purpose = "login"
                notes = request.form['rulenotes']
                if max_login is None or notes is None:
                    flash("All fields are required.", "danger")
                    ps.policy_audit(cursor, None, "Created", g.user_id, "Failed", "Attempt to create a security policy with missing fields.")
                    return redirect(url_for('policy_view'))
                else:
                    ps.create_policy(cursor, policy_name, policy_type, scope, status, created_by, modified_by, 3,policy_description, purpose)
                    policy_id = ps.get_id_from_policy_name(cursor, policy_name)
                    ps.create_policy_rules(cursor, policy_id['id'], "login attempts", "<=", max_login, notes, None, None, None, None)
                             
            elif policy_type == "Weightage":
                scope = "Global"
                purpose = request.form.get('weightageType')
                ps.create_policy(cursor, policy_name, policy_type, scope, status, created_by, modified_by, 3, policy_description, purpose)
                policy_id = ps.get_id_from_policy_name(cursor, policy_name)
                
                # get the rules
                try:
                    qber = float(request.form['qberWeight'])
                    visibility = float(request.form['visibilityWeight'])
                    key_rate = float(request.form['keyRateWeight'])
                    photon_loss = float(request.form['photonLossWeight'])
                except ValueError:
                    flash("Invalid weightage values. Have to be a float value", "danger")
                    ps.policy_audit(cursor, None, "Created", g.user_id, "Failed", "Attempt to create a weightage policy with invalid weightage values.")
                    return redirect(url_for('policy_view'))

                if qber is None or visibility is None or key_rate is None or photon_loss is None:
                    flash("All fields are required.", "danger")
                    ps.policy_audit(cursor, None, "Created", g.user_id, "Failed", "Attempt to create a weightage policy with missing fields.")
                    return redirect(url_for('policy_view'))
                
                elif ps.submit_weights(qber, visibility, key_rate, photon_loss) == False:
                    flash("Weights must add up to 1.0", "danger")
                    ps.policy_audit(cursor, None, "Created", g.user_id, "Failed", "Attempt to create a weightage policy with invalid weightage values.")
                    return redirect(url_for('policy_view'))

                ps.create_policy_rules(cursor, policy_id['id'], "qber", "=", qber, f"QBER Weightage for {purpose}", None, None, None, None)
                ps.create_policy_rules(cursor, policy_id['id'], "visibility", "=", visibility, f"Visibility Weightage for {purpose}", None, None, None, None)
                ps.create_policy_rules(cursor, policy_id['id'], "key rate", "=", key_rate, f"Key Rate Weightage for {purpose}", None, None, None, None)
                ps.create_policy_rules(cursor, policy_id['id'], "photon loss", "=", photon_loss, f"Photon Loss Weightage for {purpose}", None, None, None, None)

            elif policy_type == "Threshold":
                scope = request.form.get('policyscope')
                purpose = metric = request.form.get('metricType')
                if scope not in ["Global", "Node", "Link"]:
                    ps.policy_audit(cursor, None, "Created", g.user_id, "Failed", "Attempt to create a threshold policy with an invalid scope.")
                    flash("Invalid policy scope.", "danger")
                    return redirect(url_for('policy_view'))
                
                if scope == "Global":
                    priority = 2
                else:
                    priority = 3
                ps.create_policy(cursor, policy_name, policy_type, scope, status, created_by, modified_by, priority, policy_description, purpose)

                policy_id = ps.get_id_from_policy_name(cursor, policy_name)
                if metric == "qber" or metric == "photon loss":
                    condition = ">"
                else:
                    condition = "<"

                clear_condition_warning = f"{metric} {condition} {request.form['clear_condition_warning']} for 30 seconds"
                clear_condition_critical = f"{metric} {condition} {request.form['clear_condition_critical']} for 30 seconds"

                # warning
                ps.create_policy_rules(policy_id['id'], metric, condition, request.form['threshold_value_warning'], request.form['rule_notes_warning'], "warning", request.form['duration_window_warning'], request.form["raise_delay_warning"], clear_condition_warning)
                # critical
                ps.create_policy_rules(policy_id['id'], metric, condition, request.form['threshold_value_critical'], request.form['rule_notes_critical'], "critical", request.form['duration_window_critical'], request.form["raise_delay_critical"], clear_condition_critical)

                if scope == "Node":
                    nodes = request.form.getlist('targetNodes')
                    for node in nodes:
                        # get the node id from the node_id
                        cursor.execute("""SELECT id FROM nodes WHERE node_id = %s AND type = "QKD";""", (node,))
                        result = cursor.fetchone()
                        node_id = result[0]
                        if node_id is None:
                            ps.policy_audit(cursor, None, "Created", g.user_id, "Failed", f"Attempt to create a threshold policy with an invalid node id {node}.")                        
                        else:
                            cursor.execute("""INSERT INTO policy_target (policy_id, target_type, target_id) VALUES (%s, %s, %s);""", (policy_id['id'], "Node", node_id))
                
                elif scope == "Link":
                    links = request.form.getlist('targetLinks')
                    for link in links:
                        # get the link id from the link_id
                        cursor.execute("""SELECT id FROM links WHERE link_id = %s AND link_type = "Quantum";""", (link,))
                        result = cursor.fetchone()
                        link_id = result[0]
                        if link_id is None:
                            ps.policy_audit(cursor, None, "Created", g.user_id, "Failed", f"Attempt to create a threshold policy with an invalid link id {link}.")
                        else:
                            cursor.execute("""INSERT INTO policy_target (policy_id, target_type, target_id) VALUES (%s, %s, %s);""", (policy_id['id'], "Link", link_id))
                else:
                    flash("Invalid policy scope.", "danger")
                    ps.policy_audit(cursor, None, "Created", g.user_id, "Failed", "Attempt to create a threshold policy with an invalid scope.")
                    return redirect(url_for('policy_view'))
            # policy audit
            ps.policy_audit(cursor, policy_id['id'], "Created", g.user_id, "Success", f"Policy '{policy_name}' created successfully.")
            flash(f"Policy '{policy_name}' created successfully.", 'success')
    except Exception as e:
        # logging.error(f"Error creating policy: {e}", exc_info=True)
        flash('An error occurred while creating the policy. Please try again later.', 'danger')
    return redirect(url_for('policy_view'))

@app.route("/policy-edit", methods=['POST'])
@permission_required('policy.edit')
def policy_edit():
    try:
        policy_id = request.form.get('policy_id')
        policy_name = request.form['edit-policyname']
        policy_description = request.form['edit-policydescription']
        policy_type = request.form.get('edit-policytype-value')
        modified_by = g.user_id
    
        with db_transaction(commit=True) as cursor:
            check = ps.check_policy_status(cursor, policy_id)
            if check is None or check['is_deleted'] == True:
                ps.policy_audit(cursor, None, "Edited", g.user_id, "Failed", f"Attempt to update a non-existent policy {policy_id}.")
                flash("Policy does not exist.", "danger")
                return redirect(url_for('policy_view'))
            elif policy_name is None or policy_description is None or policy_type is None:
                ps.policy_audit(cursor, policy_id, "Edited", g.user_id, "Failed", "Attempt to update a policy with invalid values.")
                flash("Invalid policy name, description or type.", "danger")
                return redirect(url_for('policy_view'))
            
            cursor.execute("SELECT * FROM policies WHERE policy_name = %s AND id != %s;", (policy_name,policy_id,)) 
            result = cursor.fetchone()
            if result is not None:
                ps.policy_audit(cursor, policy_id, "Edited", g.user_id, "Failed", "Attempt to update a policy with a name that already exists.")
                flash("Policy name already exists.", "danger")
                return redirect(url_for('policy_view'))
            
            
            # update the policy
            cursor.execute("""UPDATE policies SET policy_name = %s,  modified_at = NOW(), modified_by = %s, description = %s WHERE id = %s;""", (policy_name, modified_by, policy_description, policy_id))

            # check if the policy type is security
            if policy_type == "Security":
                max_login = request.form['edit-loginAttempts']
                notes = request.form['edit-rulenotes']
                if max_login is None or notes is None:
                    ps.policy_audit(cursor, policy_id, "Edited", g.user_id, "Failed", "Attempt to update a policy with invalid values.")
                    flash("Invalid login attempts or notes.", "danger")
                    return redirect(url_for('policy_view'))
                cursor.execute("""UPDATE policy_rules SET value = %s, notes=%s WHERE policy_id = %s;""", (max_login, notes, policy_id))
            elif policy_type == "Weightage":
                # get the rules
                try:
                    qber = float(request.form['edit-qberWeight'])
                    visibility = float(request.form['edit-visibilityWeight'])
                    key_rate = float(request.form['edit-keyRateWeight'])
                    photon_loss = float(request.form['edit-photonLossWeight'])
                except ValueError:
                    ps.policy_audit(cursor, policy_id, "Edited", g.user_id, "Failed", "Attempt to update a policy with invalid values.")
                    flash("Invalid weightages.", "danger")
                    return redirect(url_for('policy_view'))
                
                if qber is None or visibility is None or key_rate is None or photon_loss is None:
                    ps.policy_audit(cursor, policy_id, "Edited", g.user_id, "Failed", "Attempt to update a policy with invalid values.")
                    flash("Invalid weightages.", "danger")
                    return redirect(url_for('policy_view'))
                elif ps.submit_weights(cursor, policy_id, qber, visibility, key_rate, photon_loss) == False:
                    ps.policy_audit(cursor, policy_id, "Edited", g.user_id, "Failed", "Attempt to update a policy with invalid values.")
                    flash("Invalid weightages.", "danger")
                    return redirect(url_for('policy_view'))

                cursor.execute("""UPDATE policy_rules SET value = %s WHERE policy_id = %s AND metric = 'qber';""", (qber, policy_id))
                cursor.execute("""UPDATE policy_rules SET value = %s WHERE policy_id = %s AND metric = 'visibility';""", (visibility, policy_id))
                cursor.execute("""UPDATE policy_rules SET value = %s WHERE policy_id = %s AND metric = 'key rate';""", (key_rate, policy_id))
                cursor.execute("""UPDATE policy_rules SET value = %s WHERE policy_id = %s AND metric = 'photon loss';""", (photon_loss, policy_id))
            elif policy_type == "Threshold":
                scope = request.form.get('edit-policyscope-value')
                metric = request.form.get('edit-metric-value')
                # get the condition based on the metric
                if metric in ["qber", "photon loss"]:
                    condition = ">"
                else:
                    condition = "<"
                # update the warning and critical rules
                cursor.execute("""UPDATE policy_rules SET value = %s,  duration_window = %s, raise_delay = %s, clear_condition = %s, notes = %s WHERE policy_id = %s AND severity = 'warning';""", 
                            (request.form['edit_threshold_value_warning'], request.form['edit_duration_window_warning'], request.form["edit_raise_delay_warning"], f"{metric} {condition} {request.form['edit_clear_condition_warning']} for 30 seconds", request.form['edit_rule_notes_warning'], policy_id))        
                cursor.execute("""UPDATE policy_rules SET value = %s,  duration_window = %s, raise_delay = %s, clear_condition = %s, notes = %s WHERE policy_id = %s AND severity = 'critical';""", 
                            (request.form['edit_threshold_value_critical'], request.form['edit_duration_window_critical'], request.form["edit_raise_delay_critical"], f"{metric} {condition} {request.form['edit_clear_condition_critical']} for 30 seconds", request.form['edit_rule_notes_critical'], policy_id))        

                if scope == "Node":
                    # get the nodes
                    nodes = request.form.getlist('edit-targetNodes')
                    # delete the existing nodes
                    cursor.execute("""DELETE FROM policy_target WHERE policy_id = %s AND target_type = 'Node';""", (policy_id,))
                    # insert the new nodes
                    for node in nodes:
                        cursor.execute("""SELECT id FROM nodes WHERE node_id = %s AND type = 'QKD';""", (node,))
                        result = cursor.fetchone()
                        node_id = result['id']
                        if result is not None:
                            cursor.execute("""INSERT INTO policy_target (policy_id, target_type, target_id) VALUES (%s, %s, %s);""", (policy_id, "Node", node_id))
                        else:
                            ps.policy_audit(cursor, policy_id, "Edited", g.user_id, "Failed", f"Attempt to edit a threshold policy with an invalid node id {node}.")

                elif scope == "Link": 
                    # get the links
                    links = request.form.getlist('edit-targetLinks')
                    # delete the existing links
                    cursor.execute("""DELETE FROM policy_target WHERE policy_id = %s AND target_type = 'Link';""", (policy_id,))
                    # insert the new links
                    for link in links:
                        print(f"Link: {link}")
                        cursor.execute("""SELECT id FROM links WHERE link_id = %s AND link_type = 'Quantum';""", (link,))
                        result = cursor.fetchone()
                        link_id = result['id']
                        if result is not None:
                            cursor.execute("""INSERT INTO policy_target (policy_id, target_type, target_id) VALUES (%s, %s, %s);""", (policy_id, "Link", link_id))
                        else:
                            ps.policy_audit(cursor, policy_id, "Edited", g.user_id, "Failed", f"Attempt to edit a threshold policy with an invalid link id {link}.")
            

            # update the policy audit
            ps.policy_audit(cursor, policy_id, "Edited", modified_by, "Success", f"Policy {policy_name} edited successfully.")
            flash(f"Policy '{policy_name}' edited successfully.", 'success')
        
    except Exception as e:
        # logging.error(f"Error editing policy: {e}", exc_info=True)
        flash('An error occurred while editing the policy. Please try again later.', 'danger')
    return redirect(url_for('policy_view'))

@app.route("/policy-deactivate", methods=['POST'])
@permission_required('policy.deactivate')
def policy_deactivate():
    try:
        policy_id = request.form.get('policyId')
        with db_transaction(commit=True) as cursor:
            # check if the policy is not global
            result = ps.check_policy_status(cursor, policy_id)
            if result['priority'] == 1:
                ps.policy_audit(cursor, policy_id, "Deactivated", g.user_id, "Failed", "Attempt to deactivate a default policy.")
                flash("You cannot deactivate a global policy.", "danger")
                return redirect(url_for('policy_view'))
            elif result['is_deleted'] == True or result is None:
                ps.policy_audit(cursor, None, "Deactivated", g.user_id, "Failed", f"Attempt to deactivate a non-existent policy {policy_id}.")
                flash("Policy not found.", "danger")
                return redirect(url_for('policy_view'))
            elif result['status'] == "Disabled":
                ps.policy_audit(cursor, policy_id, "Deactivated", g.user_id, "Failed", "Attempt to deactivate an already deactivated policy.")
                flash("Policy is already disabled.", "danger")
                return redirect(url_for('policy_view'))
            else:
                # update the policy status to disabled
                ps.update_status(cursor, "Disabled", g.user_id, policy_id)
                # update the policy audit
                ps.policy_audit(cursor, policy_id, "Deactivated", g.user_id, "Success", f"Policy {result['policy_name']} deactivated successfully.")
                flash(f'Policy {result["policy_name"]} deactivated successfully.', 'success')
    except Exception as e:
        # logging.error(f"Error deactivating policy: {e}", exc_info=True)
        flash('An error occurred while deactivating the policy. Please try again later.', 'danger')
    return redirect(url_for('policy_view'))

@app.route("/policy-activate", methods=['POST'])
@permission_required('policy.activate')
def policy_activate():
    try:
        policy_id = request.form.get('policyId')
        with db_transaction(commit=True) as cursor:
            # check if the policy is not global
            result = ps.check_policy_status(cursor, policy_id)
            if result['priority'] == 1:
                flash("You cannot activate a default policy.", "danger")
                ps.policy_audit(cursor, policy_id, "Activated", g.user_id, "Failed", "Attempt to activate a default policy.")
                return redirect(url_for('policy_view'))
            
            elif result is None or result['is_deleted'] == True:
                flash("Policy not found.", "danger")
                ps.policy_audit(cursor, None, "Activated", g.user_id, "Failed", f"Attempt to activate a non-existent policy {policy_id}.")
                return redirect(url_for('policy_view'))
            
            elif result['status'] == "Enabled":
                flash("Policy is already active.", "danger")
                ps.policy_audit(cursor, policy_id, "Activated", g.user_id, "Failed", "Attempt to activate an already active policy.")
                return redirect(url_for('policy_view'))         
            else:
                # update the policy status to enabled
                ps.update_status(cursor, "Enabled", g.user_id, policy_id)
                # update the policy audit
                ps.policy_audit(cursor, policy_id, "Activated", g.user_id, "Success", f"Policy {result['policy_name']} activated successfully.")
                flash(f'Policy {result["policy_name"]} activated successfully.', 'success')
    except Exception as e:
        # logging.error(f"Error activating policy: {e}", exc_info=True)
        flash('An error occurred while activating the policy. Please try again later.', 'danger')
    return redirect(url_for('policy_view'))

@app.route("/policy-delete", methods=['POST'])
@permission_required('policy.delete')
def policy_delete():
    try:
        policy_id = request.form.get('policyId')
        with db_transaction(commit=True) as cursor:
            # check if the policy is not global
            result = ps.check_policy_status(cursor, policy_id)
            if result['priority'] == 1:
                flash("You cannot delete a global policy.", "danger")
                return redirect(url_for('policy_view'))
            elif result['is_deleted'] == True or result is None:
                flash("Policy not found.", "danger")
                ps.policy_audit(cursor, None, "Deleted", g.user_id, "Failed", f"Attempt to delete a non-existent policy {policy_id}.")
                return redirect(url_for('policy_view'))
            else:
                # update the policy is_deleted to true
                cursor.execute("""UPDATE policies SET is_deleted = TRUE modified_at = NOW(), modified_by = %s WHERE id = %s;""", (g.user_id, policy_id,))
                # update the policy audit
                ps.policy_audit(cursor, policy_id, "Deleted", g.user_id, "Success", f"Policy {result['policy_name']} deleted successfully.")
                flash(f'Policy {result["policy_name"]} deleted successfully.', 'success')
    except Exception as e:
        # logging.error(f"Error deleting policy: {e}", exc_info=True)
        flash('An error occurred while deleting the policy. Please try again later.', 'danger')
    return redirect(url_for('policy_view'))

@app.route("/policy-audit")
@permission_required('audit.view_policies')
def policy_audit():
    try:
        search_query = request.args.get('search', '', type=str)
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        export_format = request.args.get('export', None)  

        start_time_str = request.args.get('start_time')
        end_time_str = request.args.get('end_time')

        start_time = None
        end_time = None

        if start_time_str and end_time_str:
            try:
                start_time = datetime.fromisoformat(start_time_str)
                end_time = datetime.fromisoformat(end_time_str)
            except ValueError:
                flash("Invalid time range provided", "danger")
                return redirect(url_for('policy_audit'))
            
        with db_transaction() as cursor:
            query="""SELECT pa.*, p.policy_name, u.name AS changed_by_name
            FROM policy_audit pa
            LEFT JOIN policies p ON pa.policy_id = p.id
            LEFT JOIN users u ON pa.changed_by = u.uuid
            WHERE 1=1"""
            
            conditions = []
            params = []

            if search_query:
                conditions.append("""
                    (pa.action_type LIKE %s OR u.name LIKE %s OR pa.notes LIKE %s OR pa.outcome LIKE %s)
                """)
                search_term = f'%{search_query}%'
                params.extend([search_term] * 4)

            if start_time and end_time:
                conditions.append("pa.changed_at BETWEEN %s AND %s")
                params.extend([start_time, end_time])

            if conditions:
                query += " AND " + " AND ".join(conditions)

            query += " ORDER BY pa.changed_at DESC"

            cursor.execute(query, tuple(params))
            policy_audits = cursor.fetchall()
            for policy_audit in policy_audits:
                if policy_audit['policy_name'] is None:
                    policy_audit['policy_name'] = "Unknown Policy"

            if export_format:
                filtered_audits = [{k: v for k, v in row.items() if k != 'id' and k!= 'policy_id' and k != 'changed_by'}
                                    for row in policy_audits]
                if export_format == 'excel':
                    return others.generate_excel_response("Policy Audit Log", filtered_audits, "policyLog")
                elif export_format == 'pdf':
                    return others.generate_pdf_response("Policy Audit Log", filtered_audits, "policyLog")
                else:
                    flash("Unsupported export format", "danger")
                    return redirect(url_for('policy_audit'))
            
            total = len(policy_audits)
            start = (page - 1) * per_page
            end = start + per_page
            paginated_audits = policy_audits[start:end]

            total_pages = (total + per_page - 1) // per_page

        return render_template("policy-audit-log.html", policy_audits=paginated_audits, current_page=page, per_page=per_page, total_pages=total_pages, search_query=search_query)
    except Exception as e:
        # logging.error(f"Error fetching policy audit data: {e}", exc_info=True)
        print("error is here ", e)
        flash('An error occurred while fetching policy audit data. Please try again later.', 'danger')
        return render_template("policy-audit-log.html", policy_audits=[], current_page=1, per_page=10, total_pages=1, search_query="")

@app.route("/topology-tracker")
@permission_required('audit.view_topology')
def topology_tracker():
    try:
        search_query = request.args.get('search', '', type=str)
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        export_format = request.args.get('export', None)  

        start_time_str = request.args.get('start_time')
        end_time_str = request.args.get('end_time')

        start_time = None
        end_time = None

        if start_time_str and end_time_str:
            try:
                start_time = datetime.fromisoformat(start_time_str)
                end_time = datetime.fromisoformat(end_time_str)
            except ValueError:
                flash("Invalid time range provided", "danger")
                return redirect(url_for('topology_tracker'))
            
        with db_transaction() as cursor:
            query = """SELECT topology_change.*, users.name AS performed_by_name
            FROM topology_change
            LEFT JOIN users ON topology_change.performed_by = users.uuid WHERE 1=1"""
            
            conditions = []
            params = []

            if search_query:
                conditions.append("""
                    (topology_change.change_type LIKE %s OR users.name LIKE %s OR topology_change.notes LIKE %s)
                """)
                search_term = f'%{search_query}%'
                params.extend([search_term] * 3)

            if start_time and end_time:
                conditions.append("topology_change.changed_at BETWEEN %s AND %s")
                params.extend([start_time, end_time])

            if conditions:
                query += " AND " + " AND ".join(conditions)

            query += " ORDER BY topology_change.changed_at DESC"

            cursor.execute(query, tuple(params))
        
            topology_changes = cursor.fetchall()
            for topology_change in topology_changes:
                if topology_change['component_type'] == 'Node':
                    cursor.execute("""SELECT node_id FROM nodes WHERE id = %s;""", (topology_change['component_id'],))
                    topology_change['component'] = cursor.fetchone()['node_id']
                elif topology_change['component_type'] == 'Link':
                    cursor.execute("""SELECT link_id FROM links WHERE id = %s;""", (topology_change['component_id'],))
                    topology_change['component'] = cursor.fetchone()['link_id']
                else:
                    topology_change['component'] = "Unknown Component"
            
            if export_format:
                filtered_audits = [{k: v for k, v in row.items() if k != 'id' and k != 'performed_by' and k != 'component_id' and k != 'component_type'}
                                    for row in topology_changes]
                if export_format == 'excel':
                    return others.generate_excel_response("Topology Audit Log", filtered_audits, "topologyLog")
                elif export_format == 'pdf':
                    return others.generate_pdf_response("Topology Audit Log", filtered_audits, "topologyLog")
                else:
                    flash("Unsupported export format", "danger")
                    return redirect(url_for('user_audit'))
            
            total = len(topology_changes)
            start = (page - 1) * per_page
            end = start + per_page
            paginated_audits = topology_changes[start:end]

            total_pages = (total + per_page - 1) // per_page

        return render_template("topology-tracker.html", topology_changes=paginated_audits, current_page=page, per_page=per_page, total_pages=total_pages, search_query=search_query)
    except Exception as e:
        print("error is here ", e)
        # logging.error(f"Error fetching topology tracker data: {e}", exc_info=True)
        flash('An error occurred while fetching topology tracker data. Please try again later.', 'danger')
        return render_template("topology-tracker.html", topology_changes=[], current_page=1, per_page=10, total_pages=1, search_query="")

@app.route("/security-overview")
@permission_required('alert.view_overview')
def security_overview():
    try:
        # count the number of links / nodes that have metrics crossing the threshold
        with db_transaction() as cursor:
            cursor.execute("""SELECT qber, visibility, key_rate, photon_loss FROM qkd.links WHERE link_type = "Quantum";""")
            quantum_metrics = cursor.fetchall()
            qber = 0
            visibility = 0
            secret_key_rate = 0
            photon_loss = 0
            for metric in quantum_metrics:
                if metric['qber'] > 8.0:
                    qber += 1
                if metric['visibility'] < 85.0:
                    visibility += 1
                if metric['photon_loss'] > 20.0:
                    photon_loss += 1
                if metric['key_rate'] < 15.0:
                    secret_key_rate += 1

            cursor.execute("""SELECT key_pool_remaining FROM nodes WHERE type = "QKD";""")
            key_pool = cursor.fetchall()
            key_pool_remaining = 0
            for metric in key_pool:
                if metric['key_pool_remaining'] < 3000:
                    key_pool_remaining += 1

            cursor.execute("""SELECT * FROM alerts WHERE status != 'Resolved';""")
            alerts = cursor.fetchall()
            number_of_alerts = len(alerts)
            number_of_critical_alerts = len([alert for alert in alerts if alert['severity'] == 'Critical'])
            number_of_warning_alerts = len([alert for alert in alerts if alert['severity'] == 'Warning'])

            cursor.execute(""" 
            SELECT date_series.date AS alert_day, 
            COUNT(alerts.detected_at) AS total_alerts,
            SUM(CASE WHEN alerts.type = 'Low Visibility' THEN 1 ELSE 0 END) AS low_visibility_count,
            SUM(CASE WHEN alerts.type = 'Low Key Pool' THEN 1 ELSE 0 END) AS low_key_pool_count,
            SUM(CASE WHEN alerts.type = 'High Photon Loss' THEN 1 ELSE 0 END) AS high_photon_loss_count,
            SUM(CASE WHEN alerts.type = 'QBER Spike' THEN 1 ELSE 0 END) AS high_qber_count
            FROM  
                (SELECT CURDATE() - INTERVAL a DAY AS date 
                FROM (SELECT 0 AS a UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4) AS days) AS date_series
            LEFT JOIN qkd.alerts 
                ON DATE(alerts.detected_at) = date_series.date
                AND alerts.detected_at >= CURDATE() - INTERVAL 4 DAY
                AND alerts.detected_at < CURDATE() + INTERVAL 1 DAY
            GROUP BY date_series.date
            ORDER BY date_series.date ASC;
            """)
            alerts_by_day = cursor.fetchall()
            dates = []
            total_alerts_data = []
            low_visibility_data = []
            low_key_pool_data = []
            high_photon_loss_data = []
            high_qber_data = []

            # Process the results to fill in the data
            for alert in alerts_by_day:
                dates.append(alert['alert_day'].strftime('%d-%m-%Y'))  # Format the date to string for the label
                total_alerts_data.append(int(alert['total_alerts']))
                low_visibility_data.append(int(alert['low_visibility_count']))
                low_key_pool_data.append(int(alert['low_key_pool_count']))
                high_photon_loss_data.append(int(alert['high_photon_loss_count']))
                high_qber_data.append(int(alert['high_qber_count']))

            return render_template("security-overview.html", qber=qber, visibility=visibility, secret_key_rate=secret_key_rate, photon_loss=photon_loss, key_pool_remaining=key_pool_remaining
                            , number_of_alerts=number_of_alerts, number_of_critical_alerts=number_of_critical_alerts, number_of_warning_alerts=number_of_warning_alerts,dates=dates, 
                            total_alerts_data=total_alerts_data, low_visibility_data=low_visibility_data, low_key_pool_data=low_key_pool_data, high_photon_loss_data=high_photon_loss_data, high_qber_data=high_qber_data)
    except Exception as e:
        # logging.error(f"Error fetching security overview data: {e}", exc_info=True)
        flash('An error occurred while fetching security overview data. Please try again later.', 'danger')
        return render_template("security-overview.html", qber=0, visibility=0, secret_key_rate=0, photon_loss=0, key_pool_remaining=0
                            , number_of_alerts=0, number_of_critical_alerts=0, number_of_warning_alerts=0, dates=[], 
                            total_alerts_data=[], low_visibility_data=[], low_key_pool_data=[], high_photon_loss_data=[], high_qber_data=[])

@app.route("/alert-list")
@permission_required('alert.view_unresolved')
def alert_list():
    try:
        with db_transaction() as cursor:
            cursor.execute("""SELECT * FROM alerts WHERE status != 'Resolved' ORDER BY detected_at DESC;""")
            alerts = cursor.fetchall()
            for alert in alerts:
                if alert['component_type'] == 'Link':
                    cursor.execute("""SELECT link_id FROM qkd.links WHERE id = %s;""", (alert['component_id'],))
                    alert['component'] = cursor.fetchone()['link_id']
                else:
                    cursor.execute("""SELECT node_id FROM qkd.nodes WHERE id = %s;""", (alert['component_id'],))
                    alert['component'] = cursor.fetchone()['node_id']

            return render_template("alert-list.html", alerts=alerts)
    except Exception as e:
        # logging.error(f"Error fetching alert list data: {e}", exc_info=True)
        flash('An error occurred while fetching alert list data. Please try again later.', 'danger')
        return render_template("alert-list.html", alerts=[])

@app.route("/escalate", methods=['POST'])
@permission_required('alert.escalate')
def escalate():
    try:
        alert_id = request.form['alertId']
        with db_transaction(commit=True) as cursor:
            # get the information from the alerts table first
            cursor.execute("""SELECT * FROM alerts WHERE alert_id = %s;""", (alert_id,))
            alert = cursor.fetchone()
            if alert is None:
                fs.alert_log(cursor, None, g.user_id, 'Escalated', 'Failed', f'Attempt to escalate an non-existent alert {alert_id}.')
                flash('This alert does not exist.', 'warning')
                return redirect(url_for('alert_list'))
            elif alert['status'] == 'Escalated':
                fs.alert_log(cursor, alert_id, g.user_id, 'Escalated', 'Failed', 'Attempt to escalate an alert that has been escalated already.')
                flash('This alert has already been escalated.', 'warning')
                return redirect(url_for('alert_list'))
            elif alert['status'] == 'Resolved':
                fs.alert_log(cursor, alert_id, g.user_id, 'Escalated', 'Failed', 'Attempt to escalate a resolved alert.')
                flash('This alert has already been resolved.', 'warning')
                return redirect(url_for('alert_list'))
            elif alert['status'] == 'Info':
                fs.alert_log(cursor, alert_id, g.user_id, 'Escalated', 'Failed', 'Attempt to escalate an info alert.')
                flash('This alert is an info alert.', 'warning')
                return redirect(url_for('alert_list'))
            else:
            # get the first sentence of the whole description
                alert['description'] = alert['description'].split('.')[0]
                # insert into the fault table 
                cursor.execute("""INSERT INTO faults (fault_id, component_type, component_id, severity, detected_at, status, description, alert_id, raising_condition)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (alert_id, alert['component_type'], alert['component_id'], alert['severity'], alert['detected_at'], 'Open', alert['description'], alert['id'], 'Escalated from Alert'))
                # update the fault id
                cursor.execute("""UPDATE faults
                                SET fault_id = CONCAT('FLT_', LPAD(LAST_INSERT_ID(), 3, '0'))
                                WHERE id = LAST_INSERT_ID();""")
                # then insert it into the alerts table
                # escalated_by has to be changed when user management is set up
                cursor.execute("""UPDATE alerts SET status = 'Escalated', performed_by = %s, escalated_at = NOW() WHERE alert_id = %s;""", (g.user_id,alert_id,))
                # insert into the alert log
                fs.alert_log(cursor, alert['id'], g.user_id, 'Escalated', 'Success', 'Alert has been escalated to a fault successfully.')
                flash(f"Alert {alert_id} escalated successfully.", 'success')   
    except Exception as e:
        # logging.error(f"Error escalating alert: {e}", exc_info=True)
        flash('An error occurred while escalating the alert. Please try again later.', 'danger')
    
    return redirect(url_for('alert_list'))

@app.route("/alert-audit-log")
@permission_required('audit.view_alerts')
def alert_audit_log():
    try:
        search_query = request.args.get('search', '', type=str)
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        export_format = request.args.get('export', None)  

        start_time_str = request.args.get('start_time')
        end_time_str = request.args.get('end_time')

        start_time = None
        end_time = None

        if start_time_str and end_time_str:
            try:
                start_time = datetime.fromisoformat(start_time_str)
                end_time = datetime.fromisoformat(end_time_str)
            except ValueError:
                flash("Invalid time range provided", "danger")
                return redirect(url_for('alert_audit_log'))
            
        with db_transaction() as cursor:
            query = """ SELECT al.*, a.alert_id AS alert_code, u.name AS performed_by_name, a.component_type as component_type, a.component_id as component_id
            FROM alert_log al
            LEFT JOIN alerts a ON al.alert_id = a.id
            JOIN users u ON al.performed_by = u.uuid WHERE 1=1
            """

            conditions = []
            params = []

            if search_query:
                conditions.append("""
                    (al.action_type LIKE %s OR u.name LIKE %s OR al.notes LIKE %s OR al.outcome LIKE %s)
                """)
                search_term = f'%{search_query}%'
                params.extend([search_term] * 4)

            if start_time and end_time:
                conditions.append("al.timestamp BETWEEN %s AND %s")
                params.extend([start_time, end_time])

            if conditions:
                query += " AND " + " AND ".join(conditions)

            query += " ORDER BY al.timestamp DESC"

            cursor.execute(query, tuple(params))

            alert_logs = cursor.fetchall()
            for alert_log in alert_logs:
                if alert_log['component_type'] == 'Link':
                    cursor.execute("""SELECT link_id FROM links WHERE id = %s;""", (alert_log['component_id'],))
                    alert_log['component'] = cursor.fetchone()['link_id']
                else:
                    cursor.execute("""SELECT node_id FROM nodes WHERE id = %s;""", (alert_log['component_id'],))
                    alert_log['component'] = cursor.fetchone()['node_id']
                if alert_log['alert_code'] is None:
                    alert_log['alert_code'] = "Unknown Alert"
            
            if export_format:
                filtered_audits = [{k: v for k, v in row.items() if k != 'id' and k != 'user_id'}
                                    for row in alert_logs]
                if export_format == 'excel':
                    return others.generate_excel_response("Alert Audit Log", filtered_audits, "alertLog")
                elif export_format == 'pdf':
                    return others.generate_pdf_response("Alert Audit Log", filtered_audits, "alertLog")
                else:
                    flash("Unsupported export format", "danger")
                    return redirect(url_for('alert_audit_log'))
            
            total = len(alert_logs)
            start = (page - 1) * per_page
            end = start + per_page
            paginated_audits = alert_logs[start:end]

            total_pages = (total + per_page - 1) // per_page
        return render_template("alert-audit-log.html", alert_logs=paginated_audits, current_page=page, per_page=per_page, total_pages=total_pages, search_query=search_query)
    except Exception as e:
        # logging.error(f"Error fetching alert audit log data: {e}", exc_info=True)
        flash('An error occurred while fetching alert audit log data. Please try again later.', 'danger')
        return render_template("alert-audit-log.html", alert_logs=[], page=1, per_page=10, total_pages=1, search_query="")

@app.route("/alarm-audit-log")
@permission_required('audit.view_faults')
def alarm_audit_log():
    try:
        search_query = request.args.get('search', '', type=str)
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        export_format = request.args.get('export', None)  

        start_time_str = request.args.get('start_time')
        end_time_str = request.args.get('end_time')

        start_time = None
        end_time = None

        if start_time_str and end_time_str:
            try:
                start_time = datetime.fromisoformat(start_time_str)
                end_time = datetime.fromisoformat(end_time_str)
            except ValueError:
                flash("Invalid time range provided", "danger")
                return redirect(url_for('alarm_audit_log'))
            
        with db_transaction() as cursor:
            query = """ SELECT fl.*, f.fault_id AS fault_code, u.name AS performed_by_name, f.component_type as component_type, f.component_id as component_id
            FROM fault_log fl
            LEFT JOIN faults f ON fl.fault_id = f.id
            JOIN users u ON fl.performed_by = u.uuid WHERE 1=1
            """
            conditions = []
            params = []

            if search_query:
                conditions.append("""
                    (f.fault_id LIKE %s OR u.name LIKE %s OR fl.notes LIKE %s OR fl.outcome LIKE %s)
                """)
                search_term = f'%{search_query}%'
                params.extend([search_term] * 4)

            if start_time and end_time:
                conditions.append("fl.timestamp BETWEEN %s AND %s")
                params.extend([start_time, end_time])

            if conditions:
                query += " AND " + " AND ".join(conditions)

            query += " ORDER BY fl.timestamp DESC"
            cursor.execute(query, tuple(params))
            
            fault_logs = cursor.fetchall()
            for fault_log in fault_logs:
                if fault_log['component_type'] == 'Link':
                    cursor.execute("""SELECT link_id FROM links WHERE id = %s;""", (fault_log['component_id'],))
                    fault_log['component'] = cursor.fetchone()['link_id']
                else:
                    cursor.execute("""SELECT node_id FROM nodes WHERE id = %s;""", (fault_log['component_id'],))
                    fault_log['component'] = cursor.fetchone()['node_id']
                if fault_log['fault_code'] is None:
                    fault_log['fault_code'] = 'Unknown Fault'
            
            if export_format:
                filtered_fault_logs = [{k: v for k, v in row.items() if k != 'id' and k != 'user_id'}
                                    for row in fault_logs]
                if export_format == 'excel':
                    return others.generate_excel_response('Alarm Audit Log', filtered_fault_logs, 'alarm-audit-log')
                elif export_format == 'pdf':
                    return others.generate_pdf_response('Alarm Audit Log', filtered_fault_logs, 'alarm-audit-log')
                else:
                    flash('Invalid export format', 'danger')
                    return render_template("alarm-audit-log.html", fault_logs=fault_logs)
            
            total = len(fault_logs)
            start = (page - 1) * per_page
            end = start + per_page
            paginated_audits = fault_logs[start:end]

            total_pages = (total + per_page - 1) // per_page
            return render_template("alarm-audit-log.html", fault_logs=paginated_audits, current_page=page, total_pages=total_pages, per_page=per_page, search_query=search_query)
    except Exception as e:
        # logging.error(f"Error fetching alarm audit log data: {e}", exc_info=True)
        print("Error fetching alarm audit log data:", e)
        flash('An error occurred while fetching alarm audit log data. Please try again later.', 'danger')
        return render_template("alarm-audit-log.html", fault_logs=[], current_page=1, total_pages=1, per_page=10, search_query='')

@app.route("/user-overview")
@permission_required('user.view_overview')
def user_overview():
    try:
        with db_transaction() as cursor:

            # user statistics
            cursor.execute("SELECT COUNT(name) as total_users," 
            "COUNT(CASE WHEN status = 'Locked' THEN 1 END) as locked_count," 
            "COUNT(CASE WHEN status = 'Deactivated' THEN 1 END) as deactivated_count," 
            "COUNT(CASE WHEN last_login < NOW() - INTERVAL 30 DAY THEN 1 END) as inactive_30_days_count," 
            "COUNT(CASE WHEN DATE(last_login) = CURDATE() THEN 1 END) as logged_in_today_count FROM users")
            row = cursor.fetchone()
            total_users = row['total_users']
            locked_count = row['locked_count']
            deactivated_count = row['deactivated_count']
            inactive_30_days_count = row['inactive_30_days_count']
            logged_in_today_count = row['logged_in_today_count']

            # role-based statistics
            cursor.execute("SELECT roles.role_name, COUNT(users.name) as user_count, " 
            "SUM(CASE WHEN users.last_login >= CURDATE() - INTERVAL 7 DAY THEN 1 ELSE 0 END) as active_count," 
            "SUM(CASE WHEN (users.last_login IS NULL OR users.last_login < CURDATE() - INTERVAL 7 DAY) THEN 1 ELSE 0 END) AS inactive_count"
            " FROM users JOIN roles ON users.role_id = roles.id WHERE roles.role_name NOT IN ('Super Admin', 'System Automation') GROUP BY roles.id")
            role_stats = cursor.fetchall()
            role_types = [row['role_name'] for row in role_stats]
            user_counts = [row['user_count'] for row in role_stats]
            active_counts = [row['active_count'] for row in role_stats]
            inactive_counts = [row['inactive_count'] for row in role_stats]

            # count the number of permissions per role
            cursor.execute("""SELECT roles.role_name, COUNT(roles_permissions.permission_id) AS permission_count
            FROM roles
            JOIN roles_permissions ON roles.id = roles_permissions.role_id
            JOIN permissions ON roles_permissions.permission_id = permissions.id
            WHERE roles.role_name NOT IN ('Super Admin', 'System Automation')
            GROUP BY roles.id, roles.role_name""")
            permission_stats = cursor.fetchall()
            permission_counts = [row['permission_count'] for row in permission_stats]

            return render_template("user-overview.html", total_users=total_users, locked_count=locked_count,
                            deactivated_count=deactivated_count, inactive_30_days_count=inactive_30_days_count, logged_in_today_count=logged_in_today_count, 
                            active_counts=active_counts, inactive_counts=inactive_counts, role_types=role_types, user_counts=user_counts, permission_counts=permission_counts)  
    except Exception as e:
        # logging.error(f"Error fetching user overview data: {e}", exc_info=True)
        flash('An error occurred while fetching user overview data. Please try again later.', 'danger')
        return render_template("user-overview.html", total_users=0, locked_count=0,
                            deactivated_count=0, inactive_30_days_count=0, logged_in_today_count=0, 
                            active_counts=[], inactive_counts=[], role_types=[], user_counts=[], permission_counts=[])

@app.route("/user")
@permission_required('user.view_user')
def user():
    try:
        with db_transaction() as cursor:
            cursor.execute("SELECT users.*, roles.role_name FROM users JOIN roles ON users.role_id = roles.id WHERE roles.role_name NOT IN ('Super Admin', 'System Automation')")
            users = cursor.fetchall()
            for user in users:
                if user['role_assignment_approval_id'] is not None:
                    cursor.execute("""
                        SELECT 
                            u.id,
                            u.role_assignment_approval_id,
                            ra.role_id,
                            r.role_name, ra.requested_by 
                        FROM users u
                        LEFT JOIN role_assignment_approval ra ON u.role_assignment_approval_id = ra.id
                        LEFT JOIN roles r ON ra.role_id = r.id
                        WHERE ra.id = %s and ra.status = "Pending Approval"
                    """, (user['role_assignment_approval_id'],))
                    role_assignment = cursor.fetchone()
                    if role_assignment:
                        if role_assignment['requested_by'] == g.user_id:
                            user['approval'] = False
                        else:
                            user['approval'] = True
                        user['role_requested'] = role_assignment['role_name']
                        user['role_requested_id'] = role_assignment['role_id']
                        user['role_assignment_approval_id'] = role_assignment['role_assignment_approval_id']
                else:
                    user['approval'] = False
                    user['role_requested'] = None
                    user['role_requested_id'] = None
                    user['role_assignment_approval_id'] = None

            # cannot assign admin role to a user (unless is a super admin)
            cursor.execute("SELECT role_name FROM roles WHERE role_name NOT IN ('Super Admin', 'System Automation', 'Administrator') AND status = 'Approved' ORDER BY role_name;")
            roles = cursor.fetchall()

        return render_template('user.html', users=users, roles=roles)
    except Exception as e:
        # logging.error(f"Error fetching user data: {e}", exc_info=True)
        flash('An error occurred while fetching user data. Please try again later.', 'danger')
        return render_template('user.html', users=[], roles=[])

@app.route("/user-role")
@permission_required('role.view')
def user_role():
    try:
        with db_transaction() as cursor:
            cursor.execute("""
            SELECT 
                r.role_name, r.description, r.status, r.default, r.id,
                COALESCE(p.permission_count, 0) AS permission_count,
                COALESCE(u.user_count, 0) AS user_count
            FROM 
                roles r
            LEFT JOIN (
                SELECT 
                    rp.role_id,
                    COUNT(rp.permission_id) AS permission_count
                FROM 
                    roles_permissions rp
                GROUP BY 
                    rp.role_id
            ) p ON r.id = p.role_id
            LEFT JOIN (
                SELECT 
                    u.role_id,
                    COUNT(u.name) AS user_count
                FROM 
                    users u
                GROUP BY 
                    u.role_id
            ) u ON r.id = u.role_id
            WHERE r.role_name NOT IN ('Super Admin', 'System Automation')
            ORDER BY r.role_name;
            """)
            roles = cursor.fetchall()

            # list all permissions that are not sensitive
            cursor.execute("SELECT permission_name, description FROM permissions WHERE is_sensitive = 0")
            non_sensitive_permissions = cursor.fetchall()

            # list all permissions based on the role
            cursor.execute("""SELECT 
            r.role_name, p.description AS permission_description
            FROM roles r
            JOIN roles_permissions rp ON r.id = rp.role_id
            JOIN permissions p ON rp.permission_id = p.id
            WHERE r.role_name NOT IN ('Super Admin', 'System Automation')
            ORDER BY r.role_name, p.description;
            """)
            role_permissions = cursor.fetchall()
            for role in roles:
                role['permissions'] = [perm['permission_description'] for perm in role_permissions if perm['role_name'] == role['role_name']]    
            
            return render_template('user-role.html', roles=roles, non_sensitive_permissions=non_sensitive_permissions)
    except Exception as e:
        # logging.error(f"Error fetching user role data: {e}", exc_info=True)
        flash('An error occurred while fetching user role data. Please try again later.', 'danger')
        return render_template('user-role.html', roles=[], permissions=[])

@app.route("/user-audit")
@permission_required('audit.view_user_actions')
def user_audit():
    try:
        search_query = request.args.get('search', '', type=str)
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        export_format = request.args.get('export', None)  

        start_time_str = request.args.get('start_time')
        end_time_str = request.args.get('end_time')

        start_time = None
        end_time = None

        if start_time_str and end_time_str:
            try:
                start_time = datetime.fromisoformat(start_time_str)
                end_time = datetime.fromisoformat(end_time_str)
            except ValueError:
                flash("Invalid time range provided", "danger")
                return redirect(url_for('user_audit'))
            
        with db_transaction(commit=True) as cursor:
            query = """
                SELECT ua.*, u.name AS performed_by_name
                FROM user_audit ua
                JOIN users u ON ua.user_id = u.uuid
                WHERE 1=1
            """
            conditions = []
            params = []

            if search_query:
                conditions.append("""
                    (ua.action_type LIKE %s OR u.name LIKE %s OR ua.notes LIKE %s OR ua.outcome LIKE %s)
                """)
                search_term = f'%{search_query}%'
                params.extend([search_term] * 4)

            if start_time and end_time:
                conditions.append("ua.timestamp BETWEEN %s AND %s")
                params.extend([start_time, end_time])

            if conditions:
                query += " AND " + " AND ".join(conditions)

            query += " ORDER BY ua.timestamp DESC"

            cursor.execute(query, tuple(params))
            
            user_audits = cursor.fetchall()

            if export_format:
                filtered_audits = [{k: v for k, v in row.items() if k != 'id' and k != 'user_id'}
                                    for row in user_audits]
                if export_format == 'excel':
                    return others.generate_excel_response("User Audit Log", filtered_audits, "userLog")
                elif export_format == 'pdf':
                    return others.generate_pdf_response("User Audit Log", filtered_audits, "userLog")
                else:
                    flash("Unsupported export format", "danger")
                    return redirect(url_for('user_audit'))
            
            total = len(user_audits)
            start = (page - 1) * per_page
            end = start + per_page
            paginated_audits = user_audits[start:end]

            total_pages = (total + per_page - 1) // per_page

            return render_template(
                "user-audit-log.html",
                user_audits=paginated_audits,
                current_page=page,
                total_pages=total_pages, per_page=per_page, search_query=search_query
            )

    except Exception as e:
        # logging.error(f"Error fetching user audit data: {e}", exc_info=True)
        print(f"error is {e}")
        flash('An error occurred while fetching user audit data. Please try again later.', 'danger')
        return render_template("user-audit-log.html", user_audits=[], current_page=1, total_pages=1, per_page=10, search_query="")

# 🤖 ENHANCED CHATBOT API ENDPOINT - Now supports agentic AI
@csrf.exempt  # Exempt from CSRF protection for API access
@app.route("/api/chatbot", methods=['POST'])
def chatbot_api():
    try:
        # Check if this is an image upload request
        if 'image' in request.files:
            return handle_enhanced_image_request()
        
        # Handle text-based requests with mode support
        data = request.get_json()
        user_message = data.get('message', '').strip()
        mode = data.get('mode', 'chat')  # Get mode from request
        use_agent = data.get('use_agent', mode == 'agent')
        force_rag = data.get('force_rag', mode == 'sop')
        document_filter = data.get('document_filter', 'all')  # Get document filter
        
        # 🔒 SAFETY: Validate mode parameter
        valid_modes = ['chat', 'sop', 'agent']
        if mode not in valid_modes:
            logger.warning(f"[WARNING] Invalid mode '{mode}' received, defaulting to 'chat'")
            mode = 'chat'
        
        # 🔍 DEBUG: Log what we received from frontend
        logger.info(f"[TARGET] GUI Bridge received: mode='{mode}', message='{user_message[:50]}...', use_agent={use_agent}, force_rag={force_rag}, document_filter='{document_filter}'")
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400
        
        # Direct processing - no HTTP call needed!
        logger.info(f"[OUT] Processing directly: mode={mode}, message={user_message[:50]}...")

        # Process based on mode using direct functions
        if mode == 'sop' or force_rag:
            response, confidence = process_sop_mode(user_message, document_filter)
            ai_response = {
                'status': 'success',
                'response': response,
                'confidence': confidence,
                'architecture': 'sop_mode',
                'mode': mode
            }
        elif mode == 'agent' and use_agent:
            result = process_agent_mode(user_message)
            if len(result) == 3:  # Agent mode with reasoning
                response, confidence, reasoning = result
                ai_response = {
                    'status': 'success',
                    'response': response,
                    'confidence': confidence,
                    'architecture': 'agent_mode',
                    'mode': mode,
                    'reasoning': reasoning
                }
            else:  # Fallback to SOP
                response, confidence = result
                ai_response = {
                    'status': 'success',
                    'response': response,
                    'confidence': confidence,
                    'architecture': 'sop_mode',
                    'mode': mode
                }
        else:  # Chat mode
            response, confidence = process_chat_mode(user_message)
            ai_response = {
                'status': 'success',
                'response': response,
                'confidence': confidence,
                'architecture': 'chat_mode',
                'mode': mode
            }

        if ai_response and ai_response.get('status') == 'success':
            # 🔍 DEBUG: Log what we got back from API
            logger.info(f"[IN] GUI Bridge received from API: status={ai_response.get('status')}, architecture={ai_response.get('architecture')}, mode={ai_response.get('mode')}")
            
            # 🔧 ENHANCED: Post-process response for consistent formatting
            raw_response = ai_response.get('response', '')
            processed_response = post_process_llm_response(raw_response, mode)
            
            # 🔍 DEBUG: Log successful response
            response_architecture = ai_response.get('architecture', 'unknown')
            response_mode = ai_response.get('mode', mode)
            logger.info(f"[SUCCESS] GUI Bridge returning: architecture='{response_architecture}', mode='{response_mode}'")
            
            return jsonify({
                'success': True,
                'response': processed_response,
                'confidence': ai_response.get('confidence', 0.0),
                'architecture': response_architecture,
                'mode': response_mode,
                'reasoning': ai_response.get('reasoning', {}),
                'citations': ai_response.get('citations', []),
                'timestamp': datetime.now().isoformat(),
                'formatting': {
                    'post_processed': raw_response != processed_response,
                    'original_length': len(raw_response),
                    'processed_length': len(processed_response)
                }
            })
        
        # Fallback to local RAG system
        if RAG_AVAILABLE:
            # Use the unified MySQL RAG system with document filtering
            logger.info(f"[RAG] Using unified MySQL RAG system with filter: {document_filter}")
            try:
                from Dashboard.services.qkd_assistant_multi_index import get_qkd_answer_multi_index
                answer, confidence = get_qkd_answer_multi_index(
                    question=user_message,
                    document_filter=document_filter if document_filter != 'all' else None,
                    top_k=5
                )
                citations = []  # TODO: Extract citations from response if available
            except Exception as e:
                logger.error(f"MySQL RAG system failed: {e}")
                # Final fallback to old system
                answer, confidence = get_qkd_answer(user_message, top_k=5)
                citations = []
            # 🔧 ENHANCED: Post-process RAG responses too
            processed_answer = post_process_llm_response(answer, mode)
            
            return jsonify({
                'success': True,
                'response': processed_answer,
                'confidence': confidence,
                'architecture': 'rag_first',
                'mode': mode,
                'citations': citations,
                'reasoning': {},
                'timestamp': datetime.now().isoformat(),
                'formatting': {
                    'post_processed': answer != processed_answer,
                    'original_length': len(answer),
                    'processed_length': len(processed_answer)
                }
            })
        else:
            return jsonify({
                'success': True,
                'response': f"AI systems not available. You asked: '{user_message}'. Please ensure the AI API server is running (python api.py) and all dependencies are installed.",
                'confidence': 0.1,
                'architecture': 'fallback',
                'reasoning': {},
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        print(f"Error in chatbot API: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500
@csrf.exempt
@app.route("/api/chatbot-stream", methods=['GET', 'POST'])
def chatbot_stream():
    """Streaming chatbot endpoint with real-time status updates using Server-Sent Events."""
    try:
        # Handle both GET (EventSource) and POST requests
        if request.method == 'GET':
            user_message = request.args.get('message', '').strip()
            mode = request.args.get('mode', 'chat')
            document_filter = request.args.get('document_filter', 'all')
        else:
            data = request.get_json()
            user_message = data.get('message', '').strip()
            mode = data.get('mode', 'chat')
            document_filter = data.get('document_filter', 'all')
        
        if not user_message:
            return jsonify({"error": "Missing message"}), 400
        
        # Import streaming response system
        from Dashboard.services.streaming_response import StreamingResponse
        
        # Create streaming response
        streaming_response = StreamingResponse(user_message, mode, document_filter=document_filter)
        
        # Return Server-Sent Events response
        return Response(
            streaming_response.stream_response(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Cache-Control',
                'X-Accel-Buffering': 'no'  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming endpoint error: {e}")
        # Return SSE-formatted error
        error_response = f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        return Response(
            error_response,
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )
@csrf.exempt
@app.route("/api/stream-progress/<session_id>", methods=['GET'])
def get_stream_progress(session_id):
    """Get current progress for a streaming session (polling fallback)."""
    try:
        from Dashboard.services.streaming_response import StreamingResponse
        
        progress = StreamingResponse.get_session_progress(session_id)
        if progress is None:
            return jsonify({"error": "Session not found"}), 404
        
        return jsonify({
            "success": True,
            "progress": progress
        })
        
    except Exception as e:
        logger.error(f"Progress endpoint error: {e}")
        return jsonify({"error": str(e)}), 500
@csrf.exempt
@app.route("/api/features", methods=['GET'])
def get_features():
    """Get available features and capabilities."""
    try:
        from Dashboard.services.streaming_response import get_active_sessions
        
        return jsonify({
            "success": True,
            "features": {
                "streaming_enabled": True,
                "sse_supported": True,
                "fallback_available": True,
                "modes_supported": ["chat", "sop", "agent"],
                "max_streaming_timeout": 60,
                "active_sessions": len(get_active_sessions())
            }
        })
    except Exception as e:
        logger.error(f"Features endpoint error: {e}")
        return jsonify({
            "success": False,
            "features": {
                "streaming_enabled": False,
                "sse_supported": False,
                "fallback_available": True,
                "modes_supported": ["chat", "sop", "agent"],
                "error": str(e)
            }
        })
@csrf.exempt
def handle_enhanced_image_request():
    """Handle image upload and processing using visual search enhancement (simplified)"""
    try:
        if not VISUAL_SEARCH_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Visual search processor not available. Please ensure LLaVA is running.'
            }), 500

        # Get the uploaded image and user question
        image_file = request.files['image']
        user_message = request.form.get('message', '').strip()

        if not image_file:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400

        logger.info(f"Processing visual query: {image_file.filename}")

        # Process through simplified visual search system
        result = process_visual_query(image_file, user_message)

        if not result.get('success', False):
            return jsonify({
                'success': False,
                'error': result.get('error', 'Visual search processing failed')
            }), 500

        # Extract vision analysis for frontend
        vision_analysis = result.get('vision_analysis', {})
        manufacturer = vision_analysis.get('manufacturer', '')
        model = vision_analysis.get('model', '')
        equipment_type = vision_analysis.get('equipment_type', '')
        status = vision_analysis.get('status', '')

        # Build device label for frontend display
        device_label = ""
        if manufacturer and model:
            device_label = f"{manufacturer} {model}"
        elif manufacturer or model:
            device_label = manufacturer or model
        elif equipment_type:
            device_label = equipment_type.title()

        has_device_label = bool(device_label)

        # Return simplified response matching frontend expectations
        return jsonify({
            'success': True,
            'response': result.get('response', ''),
            'confidence': result.get('confidence', 0.7),
            'architecture': 'visual_search_enhanced',
            'workflow_results': {
                'device_label': device_label,
                'manufacturer': manufacturer,
                'model': model,
                'equipment_type': equipment_type,
                'status': status,
                'has_device_label': has_device_label,
                'detection_confidence': f"{result.get('confidence', 0.7) * 100:.1f}%",
                'enhanced_query': result.get('enhanced_query', '')
            },
            'timestamp': result.get('timestamp', datetime.now().isoformat())
        })

    except Exception as e:
        logger.error(f"Visual search processing error: {e}")
        return jsonify({
            'success': False,
            'error': f'Enhanced workflow processing failed: {str(e)}'
        }), 500



# ============================================
# STREAMING HELPER FUNCTIONS FOR CHAT AND AGENT MODES
# ============================================

def process_pure_llm_with_streaming(user_input: str, progress_callback=None):
    """Process query with pure LLM with streaming progress support"""
    try:
        if progress_callback:
            progress_callback("llm_init", 10, "Initializing conversational AI...")

        # Direct LLM call without RAG or tools
        from Dashboard.services.llm_client import generate_answer, health_check

        if progress_callback:
            progress_callback("health_check", 20, "Checking AI model availability...")

        # Check if Ollama is available first
        if not health_check():
            return "Ollama server is not available. Please start Ollama with 'ollama serve' and ensure the llama3:8b-instruct-q4_0 model is installed.", 0.1

        if progress_callback:
            progress_callback("generating", 40, "Generating conversational response...")

        # Generate response using Ollama
        response = generate_answer(
            prompt=f"You are a helpful AI assistant. Respond naturally and conversationally to: {user_input}",
            temperature=0.7,  # More creative responses for chat
            top_p=0.9
        )

        if progress_callback:
            progress_callback("finalizing", 90, "Finalizing response...")

        if not response or len(response.strip()) == 0:
            return "I received an empty response from the language model. Please try again.", 0.3

        return response, 0.8  # Standard confidence for pure LLM

    except Exception as e:
        logger.error(f"Pure LLM processing failed: {e}")
        return "I apologize, but I'm having trouble processing your request right now.", 0.3


def process_agent_with_streaming(user_input: str, progress_callback=None):
    """Process query through agent with streaming progress support."""
    try:
        if progress_callback:
            progress_callback("agent_init", 10, "Initializing AI agent...")

        # Get the agent
        from Dashboard.services.agent import get_agent
        agent = get_agent()
        if not agent:
            return "Agent system is not available. Please check the system configuration.", 0.1, {}

        if progress_callback:
            progress_callback("planning", 30, "Planning investigation strategy...")
            progress_callback("context_init", 35, " Phase 1: Initializing investigation context...")

        # Process query through agent
        logger.info(f"Processing query through agent: {user_input[:50]}...")

        if progress_callback:
            progress_callback("executing", 60, "Executing investigation plan...")

        # Use Investigation Context tracking
        response, metadata = agent.process_query_with_investigation(user_input)

        if progress_callback:
            progress_callback("analyzing", 85, "Analyzing results and generating report...")

        # Extract confidence based on reasoning success
        confidence = 0.9 if not metadata.get("error") else 0.3

        # Minimal metadata for frontend (tools list)
        tools_used = metadata.get("tools_used") or []
        return response, confidence, {"tools_used": tools_used}

    except Exception as e:
        logger.error(f"Agent processing failed: {e}")
        return "I apologize, but the agent system encountered an error. Please try again.", 0.3, {}


@csrf.exempt# 🔍 AI SYSTEM STATUS ENDPOINT
@app.route("/api/ai-status", methods=['GET'])
def ai_status():
    """Get AI system status and capabilities"""
    try:
        # Check local services status (no API server needed anymore)
        from Dashboard.services.llm_client import health_check
        ollama_available = health_check()

        status = {
            'ai_api_available': False,  # No longer using API server
            'ai_api_details': None,
            'ollama_available': ollama_available,
            'local_rag_available': RAG_AVAILABLE,
            'local_services_available': SERVICES_AVAILABLE,
            'agent_available': AGENT_AVAILABLE,
            # 'ai_topology_available': AI_TOPO_AVAILABLE,
            'capabilities': [],
            'architecture': 'unknown',
            'timestamp': datetime.now().isoformat()
        }
        
        # Determine architecture and capabilities
        if AGENT_AVAILABLE:
            status['architecture'] = 'agent_mode'
            status['capabilities'] = ['agent', 'tools', 'reasoning', 'rag', 'llm']
        elif RAG_AVAILABLE:
            status['architecture'] = 'rag_first'
            status['capabilities'] = ['basic_rag', 'qa_system']
        else:
            status['architecture'] = 'fallback'
            status['capabilities'] = ['mock_responses']
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"AI status error: {e}")
        return jsonify({
            'ai_api_available': False,
            'local_rag_available': RAG_AVAILABLE,
            'local_services_available': SERVICES_AVAILABLE,
            'agent_available': AGENT_AVAILABLE,
            # 'ai_topology_available': AI_TOPO_AVAILABLE,
            'capabilities': ['error_handling'],
            'architecture': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# 📄 PDF UPLOAD ENDPOINTS
@csrf.exempt
@app.route("/api/upload_pdf", methods=['POST'])
def upload_pdf():
    """Upload PDF for incremental indexing"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify(success=False, error="No file provided"), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify(success=False, error="No file selected"), 400
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            return jsonify(success=False, error="Only PDF files are allowed"), 400
        
        # Get optional vendor parameter
        vendor = request.form.get('vendor') or request.args.get('vendor')
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify(success=False, error="Invalid filename"), 400
        
        # Ensure upload directory exists - use project root path
        project_root = Path(__file__).parent.parent  # Go up from GUI/ to project root
        upload_dir = project_root / "data" / "sop_documents"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = upload_dir / filename
        
        # Check if file already exists
        if file_path.exists():
            return jsonify(success=False, error=f"File '{filename}' already exists"), 409
        
        file.save(str(file_path))
        
        # Start MySQL-based PDF processing
        try:
            import time
            from Dashboard.services.mysql_pdf_processor import MySQLPDFProcessor
            
            # Process PDF directly to MySQL
            processor = MySQLPDFProcessor()
            display_name = filename.replace('.pdf', '').replace('_', ' ')
            result = processor.process_pdf(str(file_path), display_name)
            
            if result['success']:
                # Create job_id for compatibility
                job_id = f"mysql_{result['document_id']}_{int(time.time())}"
                
                return jsonify(
                    success=True,
                    job_id=job_id,
                    document_id=result['document_id'],
                    filename=filename,
                    chunk_count=result['chunk_count'],
                    message=f"File uploaded and processed successfully! Created {result['chunk_count']} chunks in MySQL.",
                    status="completed"
                )
            else:
                # Clean up file if processing failed
                try:
                    file_path.unlink()
                except:
                    pass
                return jsonify(success=False, error=f"Processing failed: {result['error']}"), 500
                
        except Exception as e:
            # Clean up file if processing failed
            try:
                file_path.unlink()
            except:
                pass
            return jsonify(success=False, error=f"MySQL processing failed: {str(e)}"), 500
        
    except Exception as e:
        logger.error(f"PDF upload error: {e}", exc_info=True)
        return jsonify(success=False, error=f"Upload failed: {str(e)}"), 500

@csrf.exempt
@app.route("/api/upload_status/<job_id>", methods=['GET'])
def upload_status(job_id):
    """Get status of PDF processing job (MySQL-based processing is synchronous)"""
    try:
        # For MySQL-based processing, jobs are completed immediately
        if job_id.startswith("mysql_"):
            # Extract document_id from job_id format: mysql_{doc_id}_{timestamp}
            parts = job_id.split("_")
            if len(parts) >= 2:
                doc_id = parts[1]
                return jsonify(
                    success=True,
                    job_id=job_id,
                    status="completed",
                    message="Document processed and stored in MySQL",
                    document_id=doc_id,
                    progress=100
                )
            else:
                return jsonify(success=False, error="Invalid job ID format"), 400
        else:
            return jsonify(success=False, error="Job not found"), 404
        
    except Exception as e:
        logger.error(f"Status check error: {e}", exc_info=True)
        return jsonify(success=False, error=f"Status check failed: {str(e)}"), 500

@csrf.exempt
@app.route("/api/documents", methods=['GET'])
def list_documents():
    """List all indexed documents from MySQL database"""
    try:
        from Dashboard.services.unified_db_manager import UnifiedDBManager
        from sqlalchemy import text
        
        db_manager = UnifiedDBManager()
        
        # Get documents from MySQL database
        with db_manager.engine.connect() as conn:
            results = conn.execute(text("""
                SELECT filename, display_name, chunk_count, status, created_at, metadata
                FROM documents 
                WHERE status = 'active' 
                ORDER BY created_at DESC
            """)).fetchall()
            
            documents = []
            total_chunks = 0
            
            for row in results:
                # Parse metadata if available
                metadata = {}
                if row.metadata:
                    try:
                        metadata = json.loads(row.metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                
                doc_info = {
                    'filename': row.filename,
                    'display_name': row.display_name or row.filename.replace('.pdf', '').replace('_', ' '),
                    'chunk_count': row.chunk_count or 0,
                    'upload_date': row.created_at.isoformat() if row.created_at else None,
                    'status': row.status,
                    'file_size_mb': metadata.get('file_size_mb', 0)
                }
                
                documents.append(doc_info)
                total_chunks += doc_info['chunk_count']
        
        # Create stats similar to old format
        stats = {
            'total_documents': len(documents),
            'total_chunks': total_chunks,
            'storage_backend': 'mysql'
        }
        
        logger.info(f"[DOC] Listed {len(documents)} documents from MySQL database")
        
        return jsonify(
            success=True,
            documents=documents,
            stats=stats,
            total_count=len(documents)
        )
        
    except Exception as e:
        logger.error(f"Document list error: {e}", exc_info=True)
        return jsonify(success=False, error=f"Failed to list documents: {str(e)}"), 500

@csrf.exempt
@app.route("/api/delete_document", methods=['POST'])
def delete_document():
    """Delete a document from the database (soft delete)"""
    try:
        data = request.get_json()
        document_id = data.get('document_id')
        filename = data.get('filename')

        if not document_id and not filename:
            return jsonify(success=False, error="Either document_id or filename must be provided"), 400

        from Dashboard.services.unified_db_manager import UnifiedDBManager

        db_manager = UnifiedDBManager()
        result = db_manager.delete_document(document_id=document_id, filename=filename)

        if result['success']:
            logger.info(f"[DELETE] Document deleted: {result['document']['filename']}")
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as e:
        logger.error(f"Document deletion error: {e}", exc_info=True)
        return jsonify(success=False, error=f"Failed to delete document: {str(e)}"), 500

@csrf.exempt
@app.route("/api/rebuild_indexes", methods=['POST'])
def rebuild_indexes():
    """Rebuild search indexes (BM25 and FAISS) after document changes"""
    try:
        from Dashboard.services.unified_db_manager import UnifiedDBManager
        from Dashboard.services.simple_bm25_manager import SimpleBM25Manager
        from Dashboard.services.faiss_gpu_manager import FAISSManager
        import os
        from pathlib import Path

        # Get absolute paths for indexes
        project_root = Path(__file__).parent.parent
        bm25_path = project_root / "data" / "simple_bm25_index.pkl"
        faiss_path = project_root / "data" / "faiss_gpu_index.bin"
        faiss_meta = project_root / "data" / "faiss_metadata.pkl"

        db_manager = UnifiedDBManager()
        results = {"success": True, "indexes": {}}

        # Rebuild BM25 index
        try:
            logger.info("[INDEX] Starting BM25 index rebuild...")
            bm25_manager = SimpleBM25Manager(db_manager, index_path=str(bm25_path))
            num_chunks = bm25_manager.rebuild_index()
            results["indexes"]["bm25"] = {
                "status": "success",
                "chunks_indexed": num_chunks
            }
            logger.info(f"[INDEX] BM25 index rebuilt with {num_chunks} chunks")
        except Exception as e:
            logger.error(f"BM25 rebuild failed: {e}")
            results["indexes"]["bm25"] = {
                "status": "failed",
                "error": str(e)
            }
            results["success"] = False

        # Rebuild FAISS index
        try:
            logger.info("[INDEX] Starting FAISS index rebuild...")
            faiss_manager = FAISSManager(
                db_manager,
                index_path=str(faiss_path),
                metadata_path=str(faiss_meta)
            )
            faiss_result = faiss_manager.rebuild_index()
            results["indexes"]["faiss"] = faiss_result
            logger.info(f"[INDEX] FAISS index rebuilt: {faiss_result}")
        except Exception as e:
            logger.error(f"FAISS rebuild failed: {e}")
            results["indexes"]["faiss"] = {
                "status": "failed",
                "error": str(e)
            }
            results["success"] = False

        return jsonify(results)

    except Exception as e:
        logger.error(f"Index rebuild error: {e}", exc_info=True)
        return jsonify(success=False, error=f"Failed to rebuild indexes: {str(e)}"), 500

@csrf.exempt
@app.route("/api/documents_detailed", methods=['GET'])
def list_documents_detailed():
    """List all documents with additional details including ID for deletion"""
    try:
        from Dashboard.services.unified_db_manager import UnifiedDBManager

        db_manager = UnifiedDBManager()
        documents = db_manager.list_documents(include_deleted=False)

        # Add formatted file size to each document
        for doc in documents:
            if doc.get('metadata') and doc['metadata'].get('file_size'):
                file_size = doc['metadata']['file_size']
                doc['file_size_mb'] = round(file_size / (1024 * 1024), 2)
            else:
                doc['file_size_mb'] = 0

        return jsonify(
            success=True,
            documents=documents,
            total_count=len(documents)
        )

    except Exception as e:
        logger.error(f"Document list error: {e}", exc_info=True)
        return jsonify(success=False, error=f"Failed to list documents: {str(e)}"), 500

# Fault detection API endpoints removed - functionality available through agent tools

if __name__ == "__main__":
    import sys
    import platform
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run the QKD Dashboard Flask application')
    parser.add_argument('--no-reload', action='store_true', 
                        help='Disable auto-reload (recommended on Windows)')
    parser.add_argument('--no-debug', action='store_true', 
                        help='Disable debug mode')
    parser.add_argument('--port', type=int, default=5000, 
                        help='Port to run the server on (default: 5000)')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()

    # RAG System: Now uses MySQL-based storage with lazy initialization
    # UnifiedDBManager + BM25 + FAISS GPU loads on first query (no startup delay)
    # Old file-based system (retriever.py) is deprecated and no longer used

    # Determine debug and reloader settings
    debug = not args.no_debug
    use_reloader = not args.no_reload
    
    # Auto-disable reloader on Windows if not explicitly set
    if platform.system() == 'Windows' and not args.no_reload:
        print("\n[WARNING] Running on Windows - auto-reload is problematic due to socket issues")
        print("[TIP] To force enable reload: Remove the Windows check below")
        print("[TIP] To disable this warning: Run with --no-reload flag")
        use_reloader = False
    
    # Run the application
    print(f"\n[STARTING] Flask server on http://{args.host}:{args.port}")
    print(f"   Debug mode: {'ON' if debug else 'OFF'}")
    print(f"   Auto-reload: {'ON' if use_reloader else 'OFF'}")
    print(f"   Platform: {platform.system()}\n")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=debug,
        use_reloader=use_reloader
    )
    
    #app.run(ssl_context='adhoc')  # Enables HTTPS locally
