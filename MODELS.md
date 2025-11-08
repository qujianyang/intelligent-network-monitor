# Machine Learning Models

This document describes how to download and manage ML models for the intelligent-network application.

## Quick Start

```bash
# Install dependencies
pip install -r requirements-clean.txt

# Download models
python scripts/download_models.py
```

## Why Models Are Not in Git

Machine learning models are **not included in this repository** for the following reasons:

1. **Size**: Models total ~850 MB, making the repository slow to clone
2. **Best Practice**: Models should be treated as build artifacts, not source code
3. **Reproducibility**: Models can be downloaded from official sources (HuggingFace)
4. **Flexibility**: Allows using different model versions without changing git history

## Models Used

### 1. Sentence Transformers Embedding Model
- **Name**: `sentence-transformers/all-mpnet-base-v2`
- **Size**: ~420 MB
- **Purpose**: Text embeddings for semantic search in SOP documents
- **Source**: HuggingFace
- **Download**: Automatically downloaded by `scripts/download_models.py`

### 2. Prophet Forecasting Models
- **Count**: 18 models (6 metrics × 3 devices)
- **Size**: ~1-5 MB each
- **Purpose**: Time-series forecasting for QKD metrics
- **Generation**: Trained automatically on first run using historical database data
- **Metrics Predicted**:
  - `qkdVisibility` - Visibility of QKD link
  - `qkdKeyRate` - Cryptographic key generation rate
  - `qkdQber` - Quantum Bit Error Rate
  - `temperature` - Device temperature
  - `cpu_load` - CPU usage
  - `memory_usage` - Memory consumption

### 3. Fault Detection Models
- **Count**: 3 models (one per QKD device)
- **Size**: ~1-2 MB each
- **Purpose**: Anomaly detection for QKD devices (QKD_001, QKD_002, QKD_003)
- **Generation**: Trained automatically on first run using historical database data

### 4. Search Indexes
- **BM25 Index**: Traditional text search index
- **FAISS Index**: Vector similarity search index for semantic search
- **Size**: ~1-2 MB combined
- **Generation**: Built automatically from SOP documents on first run

## Setup Methods

### Method 1: Automatic (Docker)
If you're using Docker, models are downloaded automatically during container build:

```bash
docker-compose -f docker-compose.dev.yml up --build
```

The Dockerfile includes a model download step that runs `scripts/download_models.py`.

### Method 2: Manual (Local Development)
For local development without Docker:

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements-clean.txt

# 3. Download models
python scripts/download_models.py

# 4. Ensure database contains historical data
# Prophet and fault detection models require historical data to train

# 5. Run the application
python GUI/app.py
```

## Directory Structure

```
Dashboard/models/
├── README.md                           # Detailed model information
├── embeddings/                         # Sentence transformers cache
│   └── models--sentence-transformers--all-mpnet-base-v2/
├── prophet_QKD_001_qkdVisibility.pkl  # Generated at runtime
├── prophet_QKD_002_qkdVisibility.pkl  # Generated at runtime
├── prophet_QKD_003_qkdVisibility.pkl  # Generated at runtime
├── ... (15 more prophet models)
├── fault_detector_QKD_001.pkl         # Generated at runtime
├── fault_detector_QKD_002.pkl         # Generated at runtime
├── fault_detector_QKD_003.pkl         # Generated at runtime
├── index/
│   └── bm25_index.pkl                 # Generated at runtime
└── reranker_cache/
    └── cache.db                        # Generated at runtime

data/
├── faiss_gpu_index.bin                 # Generated at runtime
└── faiss_metadata.pkl                  # Generated at runtime
```

## Model Lifecycle

1. **Embedding Model**: Downloaded once, cached, reused indefinitely
2. **Prophet Models**: Trained on app startup if not present, periodically retrained
3. **Fault Detectors**: Trained on app startup if not present
4. **Search Indexes**: Built from SOP documents, rebuilt when documents change
5. **Cache DB**: Created on first query, grows over time

## Troubleshooting

### "ModuleNotFoundError: No module named 'sentence_transformers'"
Install dependencies: `pip install -r requirements-clean.txt`

### "Model download failed"
- Check internet connection
- Ensure you can access huggingface.co
- Try manual download: `python scripts/download_models.py`

### "Cannot train Prophet models - no historical data"
- Ensure your database is running
- Verify historical QKD data exists in the database
- Check database connection settings in `.env`

### Models taking too long to download
The embedding model is ~420 MB. On slow connections, this can take several minutes. The download happens once and is cached.

### Docker build fails at model download
The Dockerfile includes fallback handling. If model download fails during build, models will be downloaded at runtime when the application starts.

## Updating Models

### Update Embedding Model
```bash
# Delete cached model
rm -rf Dashboard/models/embeddings/

# Re-run download script
python scripts/download_models.py
```

### Retrain Prophet Models
```bash
# Delete existing models
rm Dashboard/models/prophet_*.pkl

# Restart application - models will retrain automatically
python GUI/app.py
```

### Rebuild Search Indexes
```bash
# Delete existing indexes
rm Dashboard/models/index/bm25_index.pkl
rm data/faiss_gpu_index.bin
rm data/faiss_metadata.pkl

# Restart application - indexes will rebuild automatically
python GUI/app.py
```

## GPU Support

- **FAISS Index**: Will use GPU if CUDA is available, otherwise falls back to CPU
- **Embeddings**: Can leverage GPU acceleration if PyTorch with CUDA is installed
- **Prophet Models**: CPU-only (no GPU acceleration needed)

## For More Information

- See `Dashboard/models/README.md` for detailed technical documentation
- See `scripts/download_models.py` for the download implementation
- Check `.dockerignore` and `.gitignore` for ignored model files
