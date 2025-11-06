import os
from typing import Dict, Any, List

# ═════════════════════════════════════════════════════════════════════════════
# QKD AI NETWORK MANAGEMENT SYSTEM - COMPREHENSIVE CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
# This configuration file centralizes all system parameters for:
# - Fault Detection (IsolationForest)
# - Key-Rate Forecasting (Prophet)
# - Schema Mapping & Vendor Integration
# - RAG-based Troubleshooting
# - Intent Classification
# - API & Caching
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 1. ENVIRONMENT & DEPLOYMENT SETUP
# ─────────────────────────────────────────────────────────────────────────────
ENV = os.getenv("QKD_ENV", "development")  # "development", "staging", "production"
DEBUG_MODE = (ENV == "development")
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# 2. DIRECTORY STRUCTURE & PATHS
# ─────────────────────────────────────────────────────────────────────────────
# Core directories
MODELS_DIR = os.path.join(ROOT_DIR, "Dashboard", "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
TESTS_DIR = os.path.join(ROOT_DIR, "tests")

# Ensure directories exist
for directory in [MODELS_DIR, DATA_DIR, LOGS_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# BACKWARD COMPATIBILITY: Keep existing variable names
INTENT_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "intent_classifier.pkl")
FAULT_MODEL_PATH = os.path.join(MODELS_DIR, "fault_detector.pkl")
EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "embeddings")
RAG_INDEX_PATH = os.path.join(MODELS_DIR, "index")
SOP_DOCUMENTS_PATH = os.path.join(DATA_DIR, "sop_documents")

# ─────────────────────────────────────────────────────────────────────────────
# 3. LOCAL LLM & AI MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD COMPATIBILITY: Keep existing variable names
OFFLINE_MODE = True
LOCAL_LLM_API = os.getenv("LOCAL_LLM_API", "http://localhost:11434")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2:7b-instruct-q4_0")  # Upgraded for better instruction following
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# Enhanced LLM configuration
LLM_CONFIG = {
    "offline_mode": OFFLINE_MODE,
    "api_url": LOCAL_LLM_API,
    "model_name": LOCAL_LLM_MODEL,
    "timeout_seconds": 120,
    "max_retries": 3,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "fallback_enabled": True,
    # 🔧 ENHANCED: Improved prompt engineering for better instruction following
    "enhanced_prompting": True,
    "structured_output_prompts": True,
    "format_enforcement": True,
    # Token optimization for multi-tool workflows
    "context_window": 8192,     # Increased from default 4096 for multi-alarm searches
    "top_p": 0.9,               # Nucleus sampling parameter
    "top_k": 40,                # Top-k sampling parameter
    "temperature_by_mode": {
        "chat": 0.7,    # More creative for general chat
        "sop": 0.2,     # Very focused for procedures
        "agent": 0.3    # Balanced for reasoning
    }
}

# Embedding model for RAG
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: ADVANCED PDF PROCESSING & QUERY EXPANSION CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Advanced PDF Processing Configuration
PDF_PROCESSING_CONFIG = {
    "enable_table_extraction": True,
    "table_extraction_methods": ["camelot", "tabula", "pdfplumber"],  # Priority order
    "table_extraction_accuracy_threshold": 0.7,
    "preserve_table_structure": True,
    "generate_table_summaries": True,
    "chunk_tables_separately": True,
    "enhanced_metadata_extraction": True,
    "structure_aware_chunking": True
}

# Query Expansion Configuration
QUERY_EXPANSION_CONFIG = {
    "enable_query_expansion": True,
    "max_expansion_terms": 5,
    "expansion_weight": 0.7,
    "enable_acronym_expansion": True,
    "enable_synonym_mapping": True,
    "enable_context_awareness": True,
    "enable_domain_knowledge": True,
    "enable_technical_normalization": True,
    "query_suggestion_count": 3
}

# Hybrid Search Configuration
HYBRID_SEARCH_CONFIG = {
    "enable_hybrid_search": True,
    "semantic_weight": 0.8,  # Weight for semantic search (0.0-1.0) - INCREASED to combat BM25 dominance
    "keyword_weight": 0.2,   # Weight for keyword search (should sum to 1.0) - REDUCED
    "enable_bm25": True,
    "bm25_k1": 1.2,         # BM25 term frequency saturation parameter
    "bm25_b": 0.75,         # BM25 length normalization parameter
    "rerank_results": True,
    "min_hybrid_score": 0.1,
    # BM25 Score Normalization (addresses score dominance problem)
    "use_statistical_normalization": True,  # Use min-max scaling instead of fixed division
    "installation_content_boost": 1.8,      # Boost factor for installation-related content
    "table_content_boost": 1.5,             # Boost factor for table content (reduced from 2.0)
    # Domain-Aware Reranking Configuration (fixes reranker bias toward introductory content)
    "domain_aware_reranking": True,          # Enable domain-aware reranking adjustments
    "disable_rerank_for_installation": False, # FIXED: Re-enable reranking for installation/procedure queries
    "disable_rerank_for_procedures": False,   # FIXED: Re-enable reranking for general procedure queries  
    "technical_content_rerank_boost": 1.2,  # Boost technical content when reranking is enabled
    "general_content_rerank_penalty": 0.9   # Reduce score for general/introductory content
}

# Enhanced Chunking Configuration
ENHANCED_CHUNKING_CONFIG = {
    "chunk_size": 1024,  # INCREASED from 512 to better capture complete procedures
    "chunk_overlap": 128,  # INCREASED proportionally to maintain context
    "min_chunk_size": 100,
    "preserve_headers": True,
    "preserve_sequential_procedures": True,  # NEW: Keep numbered steps together
    "procedure_chunk_boost": 1.5,  # NEW: Larger chunks for procedural content
    "max_procedure_chunk_size": 1536,  # NEW: Allow larger chunks for complete procedures
}

# Metadata-Aware RAG Configuration
METADATA_RAG_CONFIG = {
    "enable_structured_json_loading": False,
    "json_data_file": "qkd_manual_chunks.json",
    "json_data_path": os.path.join(RAG_INDEX_PATH, "qkd_manual_chunks.json"),
    "enforce_citation_format": True,
    "require_metadata_in_context": True,
    "citation_format": "[Source: {source}, Page: {page}]",
    "enable_section_awareness": True,
    "fallback_response": "I am sorry, the provided documentation does not contain information on this topic."
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. QKD FAULT DETECTION CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD COMPATIBILITY: Keep existing variable names and values
PRODUCTION_QKD_THRESHOLDS = {
    "qkdQber": ("max", 0.05),
    "qkdKeyRate": ("min", 1000),
    "qkdVisibility": ("min", 0.90),
    "qkdLaserPower": ("min", 0.8),
    "neCpuLoad": ("max", 70),
    "neMemUsage": ("max", 85),
    "neTemperature": ("max", 65)
}

# Enhanced QKD thresholds (alias for backward compatibility)
QKD_THRESHOLDS = PRODUCTION_QKD_THRESHOLDS

# BACKWARD COMPATIBILITY: Keep existing variable names
CRITICAL_METRICS = ["qkdQber", "qkdKeyRate", "qkdVisibility"]
SEVERITY_THRESHOLDS = {
    "critical_score": -0.4,
    "high_score": -0.15,
    "critical_failures": 2,
    "high_failures": 1,
    "medium_failures": 2
}

# BACKWARD COMPATIBILITY: Keep existing variable names
MODEL_CONTAMINATION = 0.03
MODEL_RANDOM_STATE = 42
MINIMUM_TRAINING_SAMPLES = 100

# ─────────────────────────────────────────────────────────────────────────────
# 5. KEY-RATE FORECASTING CONFIGURATION (Prophet)
# ─────────────────────────────────────────────────────────────────────────────
# Forecasting parameters
FORECASTING_CONFIG = {
    "default_horizon_days": 7,
    "default_horizon_hours": 7 * 24,  # 168 hours
    "min_training_points": 48,
    "confidence_interval": 0.8,
    "data_source": "database",  # "database", "csv" or "snmp"
    "csv_file": "qkd_fault_dataset_500.csv",
    "enable_caching": True,
    "cache_ttl_hours": 1
}

# Prophet model hyperparameters
PROPHET_PARAMS = {
    "daily_seasonality": True,
    "weekly_seasonality": True,
    "yearly_seasonality": False,
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "holidays_prior_scale": 10.0,
    "mcmc_samples": 0,
    "uncertainty_samples": 1000,
    "interval_width": 0.8
}

# Key-rate thresholds
KEY_RATE_THRESHOLDS = {
    "critical_minimum": 300,
    "warning_minimum": 600,
    "optimal_minimum": 1000
}

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION COMPLETE
# ═════════════════════════════════════════════════════════════════════════════ 