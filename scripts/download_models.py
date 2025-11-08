#!/usr/bin/env python3
"""
Model Download Script
====================
This script downloads and initializes all required ML models for the intelligent-network application.

Models downloaded:
- sentence-transformers/all-mpnet-base-v2 (Embedding model for semantic search)

Models generated at runtime:
- Prophet forecasting models (trained on historical data)
- Fault detection models (trained on historical data)
- BM25 indexes (generated from documents)
- FAISS indexes (generated from embeddings)
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import Dashboard modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def download_embedding_model():
    """Download the sentence-transformers embedding model."""
    print("=" * 60)
    print("Downloading Sentence Transformers Model")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer

        model_name = "sentence-transformers/all-mpnet-base-v2"
        cache_dir = Path(__file__).parent.parent / "Dashboard" / "models" / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Model: {model_name}")
        print(f"Cache directory: {cache_dir}")
        print("\nDownloading... (this may take a few minutes)")

        model = SentenceTransformer(model_name, cache_folder=str(cache_dir))

        print(f"\n✓ Model downloaded successfully to {cache_dir}")
        print(f"  Model max sequence length: {model.max_seq_length}")

        return True

    except ImportError:
        print("ERROR: sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"ERROR downloading model: {e}")
        return False


def initialize_model_directories():
    """Create necessary model directories."""
    print("\n" + "=" * 60)
    print("Initializing Model Directories")
    print("=" * 60)

    base_dir = Path(__file__).parent.parent / "Dashboard" / "models"

    directories = [
        base_dir / "embeddings",
        base_dir / "trained_models",
        base_dir / "index",
        base_dir / "reranker_cache",
        Path(__file__).parent.parent / "data",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created/verified: {directory}")

    return True


def create_placeholder_models_info():
    """Create a README explaining model generation."""
    print("\n" + "=" * 60)
    print("Creating Model Information File")
    print("=" * 60)

    readme_path = Path(__file__).parent.parent / "Dashboard" / "models" / "README.md"

    content = """# ML Models Directory

This directory contains machine learning models used by the intelligent-network application.

## Model Types

### 1. Embedding Models
- **Location**: `embeddings/`
- **Model**: sentence-transformers/all-mpnet-base-v2
- **Purpose**: Text embeddings for semantic search
- **Download**: Run `python scripts/download_models.py`

### 2. Prophet Forecasting Models
- **Location**: `prophet_*.pkl`
- **Purpose**: Time-series forecasting for QKD metrics
- **Generation**: Automatically trained on first run with historical data
- **Metrics**:
  - qkdVisibility
  - qkdKeyRate
  - qkdQber
  - temperature
  - cpu_load
  - memory_usage

### 3. Fault Detection Models
- **Location**: `fault_detector_*.pkl`
- **Purpose**: Anomaly detection for QKD devices
- **Generation**: Automatically trained on first run with historical data
- **Devices**: QKD_001, QKD_002, QKD_003

### 4. Search Indexes
- **BM25 Index**: `index/bm25_index.pkl`
- **FAISS Index**: `../data/faiss_gpu_index.bin`
- **Purpose**: Fast document retrieval
- **Generation**: Built from SOP documents on first run

### 5. Cache Databases
- **Location**: `reranker_cache/cache.db`
- **Purpose**: Cache reranker results for faster responses
- **Generation**: Created automatically at runtime

## Setup

### Initial Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download models: `python scripts/download_models.py`
3. Run application: Models will be trained on first run

### Manual Model Training
If you need to regenerate the Prophet or fault detection models:
1. Ensure historical data is available in the database
2. Delete existing `.pkl` files
3. Restart the application - models will retrain automatically

## Model Sizes (Approximate)

- Embedding model: ~420 MB
- Prophet models: ~1-5 MB each (18 total)
- Fault detectors: ~1-2 MB each (3 total)
- BM25 index: ~150 KB
- FAISS index: ~1 MB

## Notes

- All `.pkl` files are generated at runtime and should not be committed to git
- The embedding model is downloaded once and cached locally
- Model training requires historical data in the database
- FAISS index may use GPU if available, falls back to CPU
"""

    readme_path.write_text(content)
    print(f"✓ Created: {readme_path}")

    return True


def main():
    """Main function to download and initialize all models."""
    print("\n" + "=" * 60)
    print("INTELLIGENT NETWORK - MODEL SETUP")
    print("=" * 60)
    print()

    success = True

    # Step 1: Initialize directories
    if not initialize_model_directories():
        success = False

    # Step 2: Download embedding model
    if not download_embedding_model():
        success = False

    # Step 3: Create model info
    if not create_placeholder_models_info():
        success = False

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✓ MODEL SETUP COMPLETE")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Ensure your database contains historical data")
        print("2. Start the application - remaining models will train automatically")
        print("3. Check Dashboard/models/ for generated model files")
        return 0
    else:
        print("✗ MODEL SETUP FAILED")
        print("=" * 60)
        print("\nPlease fix the errors above and try again.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
