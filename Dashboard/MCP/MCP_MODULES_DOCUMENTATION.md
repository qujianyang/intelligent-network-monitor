# MCP E-Line Service Provisioning Modules Documentation

## Overview
This suite of Python modules provides a complete API-based solution for creating, managing, and monitoring E-Line EPL (Ethernet Private Line) services on Ciena Blue Planet MCP (Management, Control and Planning) platform.

---

## Core Modules

### 1. **mcp_api.py** - Base API Client
**Purpose**: Foundation module that handles all authentication and API communication with MCP.

**Key Features**:
- Token-based authentication using form data
- Automatic token refresh on expiry
- Session management with connection pooling
- Support for both `Authorization` and `X-Auth-Token` headers
- Enhanced error reporting with detailed failure information
- SSL certificate handling for self-signed certificates

**What It Does**:
- Logs into MCP and obtains JWT token
- Manages authenticated API calls (GET, POST, PATCH, DELETE)
- Handles HTTP status codes (200, 201, 202, 400, 401, 404, etc.)
- Automatically retries on token expiration

---

### 2. **mcp_eline_simple.py** - E-Line Service Creator
**Purpose**: Creates E-Line EPL services with full configuration including OAM settings.

**Key Features**:
- Creates point-to-point Ethernet Private Line services
- Configures 802.1ag and Y.1731 OAM protocols
- Sets up endpoints with specific ports and roles (A_UNI/Z_UNI)
- Supports DMM (Delay Measurement) and SLM (Synthetic Loss Measurement)

**What It Does**:
- Constructs the complete service payload with all required properties
- Submits service creation request to MCP
- Saves response with service ID for tracking
- Handles validation errors and schema compliance

---

### 3. **mcp_query_products.py** - Product Discovery Tool
**Purpose**: Discovers available products and service types in your MCP instance.

**Key Features**:
- Queries multiple endpoints to find products
- Identifies L2 service products and their IDs
- Lists existing services to understand structure
- Extracts product IDs used by specific service types

**What It Does**:
- Finds the correct product ID for your MCP deployment
- Shows all available service types and products
- Helps diagnose why certain product IDs don't work
- Saves product catalog for reference

---

### 4. **mcp_verify_service.py** - Service Status Checker
**Purpose**: Checks the current status and details of a specific service.

**Key Features**:
- Retrieves complete service configuration
- Shows orchestration state (requested/active/failed)
- Displays endpoint details and OAM settings
- Identifies provisioning issues

**What It Does**:
- Queries specific service by ID
- Shows current provisioning state
- Lists all endpoints and their configuration
- Saves full service details to JSON file

---

### 5. **mcp_list_all_services.py** - Service Inventory Tool
**Purpose**: Lists all services in MCP and groups them by type.

**Key Features**:
- Retrieves all services with pagination support
- Groups services by type (EPL, EPLAN, EVC, etc.)
- Searches for specific services by name
- Shows service states and resource types

**What It Does**:
- Provides complete inventory of all services
- Helps locate specific services among many
- Shows service distribution by type
- Identifies services stuck in provisioning

---

### 6. **mcp_check_provisioning.py** - Provisioning Monitor
**Purpose**: Continuously monitors service provisioning progress until completion.

**Key Features**:
- Real-time status monitoring with timestamps
- Configurable timeout periods
- Shows state transitions as they happen
- Displays provisioning failure reasons

**What It Does**:
- Polls service status every 2 seconds
- Tracks state changes (requested → active)
- Reports when service becomes active
- Times out if provisioning takes too long

---

### 7. **mcp_activate_service.py** - Service Activator
**Purpose**: Triggers service activation and attempts route calculation.

**Key Features**:
- Changes service state from requested to active
- Attempts automatic route calculation
- Updates desired orchestration state
- Handles activation errors

**What It Does**:
- Sends PATCH request to update service state
- Triggers MCP to complete provisioning
- Bypasses manual route calculation requirement
- Initiates automatic path computation

---

### 8. **read_docx.py** - DOCX Configuration Reader (Utility)
**Purpose**: Extracts text from Word documents containing configuration details.

**Key Features**:
- Reads DOCX files without external dependencies
- Extracts all text content preserving structure
- Saves output to text file for easy reading

**What It Does**:
- Unzips DOCX file (which is actually a ZIP archive)
- Parses XML to extract text content
- Useful for reading configuration documentation

---

## How to Use - Complete Workflow

### Prerequisites
```bash
# Install required packages
pip install requests

# Ensure you have access to MCP at https://10.1.1.3
# Default credentials: admin/adminpw
```

### Step 1: Test Authentication
```bash
# Verify MCP connectivity and credentials
python mcp_api.py

# Expected output:
# === MCP Client Test ===
# 1. Testing login...
#    Success! Token: cad7ae0c3a37808ed862...
```

### Step 2: Discover Available Products
```bash
# Find the correct product ID for your MCP instance
python mcp_query_products.py

# This will save:
# - mcp_products.json (all available products)
# - mcp_services.json (existing services for reference)
```

### Step 3: Create E-Line Service
```bash
# Create your E-Line EPL service
python mcp_eline_simple.py

# Expected output:
# MCP E-Line EPL Service Provisioning
# [OK] Logged in successfully
# Creating E-Line EPL service 'test2'...
# [OK] Service created with ID: a86d6a83-b0b4-11f0-ab20-0d83d91152c5

# This saves: eline_response.json with service details
```

### Step 4: Monitor Provisioning
```bash
# Watch the service provisioning progress
python mcp_check_provisioning.py

# This will show real-time status updates:
# [0s] State: requested
# [30s] State: active (if successful)
```

### Step 5: Activate if Stuck
```bash
# If service stays in "requested" state, activate it
python mcp_activate_service.py

# This triggers MCP to complete provisioning
# Wait 2-3 minutes for activation to complete
```

### Step 6: Verify Service Status
```bash
# Check final service status
python mcp_verify_service.py

# Expected output:
# Service Details:
#   Name: test2
#   Type: EPL
#   Orchestration State: active
# [SUCCESS] Service is ACTIVE!
```

### Step 7: List All Services
```bash
# See all services including your new one
python mcp_list_all_services.py

# This groups services by type and shows complete inventory
```

---

## Common Issues and Solutions

### Issue 1: Invalid Product ID
**Error**: "Invalid product: 58cabe48-3594-485c-8135-62ba563afe94"
**Solution**: Run `mcp_query_products.py` to find correct product ID for your MCP instance

### Issue 2: Service Stuck in "Requested" State
**Cause**: Service needs route calculation and activation
**Solution**: Run `mcp_activate_service.py` to trigger automatic activation

### Issue 3: Schema Validation Errors
**Error**: "properties which are not allowed by the schema"
**Solution**: Remove unsupported properties like `lastTransition`, `routingConstraints`

### Issue 4: Service Not Visible in GUI
**Causes**:
- Wrong view (check Planning tab, not just Network tab)
- Service still provisioning (wait for "active" state)
- Filter hiding service type (check filter settings)

---

## Service Configuration Details

### Default E-Line EPL Configuration
```yaml
Service Name: test2
Service Type: EPL (Ethernet Private Line)
Customer: STE LAB ELAN
Structure: P2P (Point-to-Point)

Endpoints:
  A_UNI: C02-5164-01, Port 3
  Z_UNI: C01-5164-01, Port 3

OAM Settings:
  802.1ag: Enabled
  CCM Interval: 10 seconds
  CCM Priority: 5
  Y.1731 DMM: Enabled
  Y.1731 SLM: Enabled (A_UNI only)
```

---

## Important Notes

1. **Product IDs are deployment-specific** - The product ID that works in one MCP instance may not work in another.

2. **Two-phase provisioning** - Services are created in "requested" state and must be activated to become operational.

3. **GUI vs API terminology** - GUI may show "EVC" for all Ethernet services, but the underlying type is correctly set as EPL.

4. **Activation delay** - After triggering activation, it may take 2-3 minutes for the service to fully provision.

5. **Route calculation** - EPL services require network paths to be calculated. The API can trigger automatic calculation.

---

## Module Dependencies

```
mcp_api.py (base - required by all others)
    ├── mcp_eline_simple.py
    ├── mcp_query_products.py
    ├── mcp_verify_service.py
    ├── mcp_list_all_services.py
    ├── mcp_check_provisioning.py
    └── mcp_activate_service.py
```

---

## Files Generated During Operation

- **eline_response.json** - Service creation response with ID
- **eline_error.json** - Error details if creation fails
- **service_status.json** - Current service configuration
- **all_services.json** - Complete service inventory
- **mcp_products.json** - Available products catalog
- **mcp_services.json** - Existing services reference

---

## Support and Troubleshooting

For debugging, check the log output which shows:
- Exact API endpoints being called
- Request/response details
- Token management operations
- Error messages from MCP

Each module includes detailed logging at INFO level for troubleshooting.

---

*Last Updated: October 24, 2024*
*MCP Version: Blue Planet MCP*
*API Version: /bpocore/market/api/v1*