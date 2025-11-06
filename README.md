# Intelligent Network Monitor for QKD Systems

An AI-powered network management platform for Quantum Key Distribution (QKD) infrastructure, featuring intelligent fault detection, predictive analytics, and automated troubleshooting capabilities.

## Overview

This system provides comprehensive monitoring, management, and AI-driven insights for QKD networks. It combines traditional network monitoring with advanced machine learning techniques to deliver proactive fault detection, performance forecasting, and intelligent troubleshooting through a conversational AI interface.

## Key Features

### Core Capabilities
- **Real-time Network Monitoring**: Live tracking of QKD metrics including QBER, key rates, visibility, and system health
- **AI-Powered Fault Detection**: Isolation Forest-based anomaly detection for proactive issue identification
- **Predictive Analytics**: Prophet-based forecasting for key rate and performance metrics
- **Intelligent Chatbot Assistant**: RAG-powered conversational interface with voice recognition support
- **Network Topology Visualization**: Interactive network topology and path management
- **Alarm Management**: Comprehensive alarm tracking, categorization, and audit logging

### Advanced AI Features
- **RAG System**: Retrieval-Augmented Generation for accurate troubleshooting guidance from technical documentation
- **Hybrid Search**: Combines semantic search (embeddings) with keyword-based BM25 for optimal retrieval
- **Agentic AI**: Multi-tool agent system for complex network operations
- **Visual Query Processing**: Enhanced query understanding through visual context analysis
- **Local LLM Integration**: Privacy-first design using Ollama for offline AI capabilities

### MCP Integration
- **Ciena Blue Planet MCP**: Full API integration for E-Line service provisioning
- **Service Management**: Create, monitor, and manage EPL (Ethernet Private Line) services
- **Automated Provisioning**: Streamlined service activation and route calculation

### Security & Access Control
- **Role-Based Access Control (RBAC)**: Granular permissions for users and roles
- **User Management**: Complete user lifecycle management with audit trails
- **Policy Management**: Network policy configuration and enforcement
- **Audit Logging**: Comprehensive logging of all user actions and system events

## Architecture

```
intelligent-network-monitor/
├── GUI/                          # Flask web application
│   ├── templates/                # HTML templates
│   ├── static/                   # CSS, JS, assets
│   ├── auth/                     # Authentication module
│   └── services/                 # Business logic services
├── Dashboard/                    # AI/ML backend services
│   ├── services/                 # ML services (fault detection, forecasting, RAG)
│   ├── MCP/                      # Ciena Blue Planet integration
│   ├── models/                   # Trained ML models and embeddings
│   └── scripts/                  # Training and maintenance scripts
├── data/                         # Application data and SOP documents
├── docker/                       # Docker configuration files
├── config.py                     # Centralized configuration
├── requirements.txt              # Python dependencies
└── docker-compose.dev.yml        # Docker Compose for development
```

## Technology Stack

### Backend
- **Python 3.10+**: Core programming language
- **Flask**: Web framework for GUI and API
- **MySQL 8.0**: Primary database for network data and ML features
- **Ollama**: Local LLM inference engine (Qwen2 7B model)

### Machine Learning & AI
- **scikit-learn**: Isolation Forest for anomaly detection
- **Prophet**: Time series forecasting
- **Sentence Transformers**: Text embeddings (all-mpnet-base-v2)
- **FAISS**: Vector similarity search
- **BM25**: Keyword-based search
- **Hugging Face Transformers**: NLP tasks

### Frontend
- **HTML/CSS/JavaScript**: Web interface
- **Bootstrap**: UI framework
- **Voice Recognition**: Web Speech API integration

### DevOps
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **Git**: Version control

## Prerequisites

- **Python**: 3.10 or higher
- **Docker & Docker Compose**: For containerized deployment
- **MySQL**: 8.0 or higher (if running without Docker)
- **Ollama**: For local LLM inference
- **GPU (Optional)**: NVIDIA GPU with CUDA support for improved performance

## Installation

### Option 1: Docker Deployment (Recommended)

1. **Clone the repository**
```bash
git clone <repository-url>
cd intelligent-network-monitor
```

2. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your settings
```

3. **Start all services**
```bash
docker-compose -f docker-compose.dev.yml up -d
```

4. **Access the application**
- GUI: http://localhost:5000
- Ollama API: http://localhost:11434
- MySQL: localhost:3307

### Option 2: Manual Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd intelligent-network-monitor
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up MySQL database**
```bash
# Create databases
mysql -u root -p
CREATE DATABASE qkd;
CREATE DATABASE qkd_ml;
```

5. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your database credentials and settings
```

6. **Install and configure Ollama**
```bash
# Install Ollama from https://ollama.ai
ollama pull qwen2:7b-instruct-q4_0
```

7. **Initialize ML models**
```bash
python Dashboard/scripts/populate_ml_database.py
python Dashboard/scripts/train_fault_models_improved.py
python Dashboard/scripts/train_prophet_models.py
python Dashboard/scripts/rebuild_faiss_index.py
python Dashboard/scripts/rebuild_simple_bm25.py
```

8. **Run the application**
```bash
python GUI/app.py
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Environment
QKD_ENV=development          # development, staging, production

# Database
DB_HOST=localhost
DB_PORT=3307
DB_NAME=qkd
DB_USER=root
DB_PASSWORD=your_password

# LLM Configuration
LOCAL_LLM_API=http://localhost:11434
LOCAL_LLM_MODEL=qwen2:7b-instruct-q4_0

# MCP Integration (Optional)
MCP_BASE_URL=https://10.1.1.3
MCP_USERNAME=admin
MCP_PASSWORD=adminpw

# Security
SECRET_KEY=your_secret_key_here

# Features
ENABLE_CHATBOT=true
ENABLE_FAULT_DETECTION=true
ENABLE_FORECASTING=true
ENABLE_RAG_SYSTEM=true
```

### QKD Thresholds

Configure operational thresholds in `config.py`:

```python
PRODUCTION_QKD_THRESHOLDS = {
    "qkdQber": ("max", 0.05),           # Quantum Bit Error Rate
    "qkdKeyRate": ("min", 1000),        # Key rate (bits/second)
    "qkdVisibility": ("min", 0.90),     # Visibility
    "qkdLaserPower": ("min", 0.8),      # Laser power
    "neCpuLoad": ("max", 70),           # CPU load (%)
    "neMemUsage": ("max", 85),          # Memory usage (%)
    "neTemperature": ("max", 65)        # Temperature (°C)
}
```

## Usage

### Accessing the GUI

1. Navigate to http://localhost:5000
2. Log in with default credentials (configure in database)
3. Explore the dashboard features:
   - **Network Topology**: View network structure and connections
   - **Performance Overview**: Monitor real-time metrics
   - **Alarm Management**: Track and respond to alarms
   - **Fault Detection**: View detected anomalies
   - **Chatbot**: Ask questions and get troubleshooting guidance

### Using the Chatbot

The AI chatbot supports multiple query types:

**General Questions**
```
"How do I install a QKD device?"
"What is QBER and why is it important?"
```

**Fault Detection**
```
"Check for faults on QKD_001"
"Are there any anomalies in the network?"
```

**Performance Queries**
```
"Show me the key rate for QKD_002"
"What's the current QBER?"
```

**Forecasting**
```
"Predict key rate for the next 7 days"
"Forecast temperature trends"
```

**Voice Commands**
- Click the microphone icon
- Speak your query clearly
- View the transcribed text and response

### MCP Service Provisioning

Create E-Line EPL services programmatically:

```python
from Dashboard.MCP.mcp_eline_simple import create_eline_service

# Create service
service_id = create_eline_service(
    service_name="test-service",
    a_uni_port="C02-5164-01:3",
    z_uni_port="C01-5164-01:3"
)
```

Monitor service provisioning:

```bash
python Dashboard/MCP/mcp_check_provisioning.py
```

## Training ML Models

### Fault Detection Models

Train Isolation Forest models for anomaly detection:

```bash
python Dashboard/scripts/train_fault_models_improved.py
```

### Forecasting Models

Train Prophet models for time series prediction:

```bash
python Dashboard/scripts/train_prophet_models.py
```

### RAG System

Rebuild search indices after adding new documentation:

```bash
# Rebuild FAISS index
python Dashboard/scripts/rebuild_faiss_index.py

# Rebuild BM25 index
python Dashboard/scripts/rebuild_simple_bm25.py
```

## API Endpoints

### Chatbot API
```
POST /chat
Body: {
    "message": "Your query here",
    "mode": "chat",  // chat, sop, or agent
    "use_agent": false
}
```

### Fault Detection API
```
GET /api/detect-fault?node=QKD_001
Response: {
    "is_anomaly": true/false,
    "anomaly_score": -0.234,
    "severity": "high",
    "details": {...}
}
```

### Forecasting API
```
GET /api/forecast?node=QKD_001&metric=qkdKeyRate&days=7
Response: {
    "predictions": [...],
    "confidence_intervals": {...}
}
```

## Monitoring & Maintenance

### Health Checks

Monitor service health:
```bash
# Check MySQL
docker exec qkd-mysql-dev mysqladmin ping

# Check Ollama
curl http://localhost:11434/api/tags

# Check Flask app
curl http://localhost:5000/health
```

### Database Maintenance

```bash
# Backup database
docker exec qkd-mysql-dev mysqldump -u root -p qkd > backup.sql

# View logs
docker logs qkd-gui-dev
docker logs qkd-ollama-dev
```

### Performance Tuning

- **GPU Acceleration**: Enable GPU for Ollama and FAISS (configure in `.env`)
- **Connection Pooling**: Adjust `DB_POOL_SIZE` in configuration
- **Cache Settings**: Configure Redis for improved response times (optional)

## Project Structure Details

### Dashboard Services

- **fault_improved.py**: Enhanced fault detection with ML
- **forecasting.py**: Time series forecasting engine
- **qkd_assistant_multi_index.py**: RAG system with hybrid search
- **agent.py**: Agentic AI for complex operations
- **llm_client.py**: Ollama LLM client wrapper
- **parallel_hybrid_search.py**: Optimized search implementation

### GUI Services

- **fault_service.py**: Fault management business logic
- **user_service.py**: User and authentication services
- **policy_service.py**: Network policy management
- **decorator.py**: RBAC decorators

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black Dashboard/ GUI/
```

## Troubleshooting

### Common Issues

**Ollama Connection Error**
```
Solution: Ensure Ollama is running and accessible
curl http://localhost:11434/api/tags
```

**Database Connection Error**
```
Solution: Verify MySQL credentials in .env
Check if MySQL container is running: docker ps
```

**RAG System Returns Empty Results**
```
Solution: Rebuild indices
python Dashboard/scripts/rebuild_faiss_index.py
python Dashboard/scripts/rebuild_simple_bm25.py
```

**Model Not Found**
```
Solution: Train models first
python Dashboard/scripts/train_fault_models_improved.py
python Dashboard/scripts/train_prophet_models.py
```

## License

[Specify your license here - MIT, Apache 2.0, etc.]

## Acknowledgments

- **Ciena**: Blue Planet MCP integration
- **Ollama**: Local LLM inference
- **Sentence Transformers**: Embedding models
- **Prophet**: Time series forecasting

## Contact & Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the development team

---

**Version**: 1.0.0
**Last Updated**: 2025
**Status**: Active Development
