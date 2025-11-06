"""
MySQL PDF Processor for Unified Vector Storage
==============================================
Processes PDF files and stores chunks + embeddings directly in MySQL
"""

import os
import time
import logging
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from Dashboard.services.unified_db_manager import UnifiedDBManager

# PDF processing imports - use fast processor first, fallback to advanced
try:
    from Dashboard.services.fast_pdf_processor import create_fast_pdf_chunks
    FAST_PDF_AVAILABLE = True
except ImportError:
    FAST_PDF_AVAILABLE = False
    logger.warning("Fast PDF processor not available")

# Fallback to advanced processor
try:
    from Dashboard.services.pdf_processor import create_enhanced_pdf_chunks
    ADVANCED_PDF_AVAILABLE = True
except ImportError:
    ADVANCED_PDF_AVAILABLE = False

# Fallback to basic LlamaIndex if advanced processor unavailable
try:
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.readers.file import PDFReader
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False

# Embedding model
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class MySQLPDFProcessor:
    """
    Processes PDFs and stores directly in MySQL via UnifiedDBManager
    """
    
    def __init__(self):
        self.db_manager = UnifiedDBManager()
        self.embedding_model = None

        # Initialize PDF processor type with priority: Fast > Advanced > Basic
        if FAST_PDF_AVAILABLE:
            self.processor_type = "fast"
            logger.info("Using FAST PDF processor (10-20x speed improvement)")
        elif ADVANCED_PDF_AVAILABLE:
            self.processor_type = "advanced"
            logger.info("Using advanced PDF processor with table extraction")
        else:
            self.processor_type = "basic"
            # Fallback to basic chunking
            if LLAMAINDEX_AVAILABLE:
                self.node_parser = SentenceSplitter(
                    chunk_size=512,
                    chunk_overlap=50,
                    separator=" "
                )
            logger.warning("Fast and advanced processors unavailable, using basic processing")
        
    def _initialize_embedding_model(self):
        """Initialize embedding model on first use"""
        if self.embedding_model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers not available")
            logger.info("Loading sentence-transformers model...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            logger.info("Embedding model loaded successfully")
    
    def process_pdf(self, pdf_path: str, display_name: str = None) -> Dict[str, Any]:
        """
        Process PDF and store in MySQL
        
        Args:
            pdf_path: Path to PDF file
            display_name: Display name for the document
            
        Returns:
            Dictionary with processing results
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError("llama-index not available for PDF processing")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        filename = pdf_path.name
        if display_name is None:
            display_name = filename.replace('.pdf', '').replace('_', ' ')
        
        logger.info(f"Processing PDF: {filename}")
        
        try:
            if FAST_PDF_AVAILABLE and self.processor_type == "fast":
                # 1. Use FAST PDF processor (10-20x faster)
                logger.info("Using FAST PDF processing for speed...")
                start_time = time.time()
                chunks_data = create_fast_pdf_chunks(str(pdf_path), chunk_size=512, chunk_overlap=64)

                # Extract text chunks from fast processor
                all_chunks = []
                for chunk_info in chunks_data:
                    if isinstance(chunk_info, dict):
                        chunk_text = chunk_info.get('content', chunk_info.get('text', str(chunk_info)))
                    else:
                        chunk_text = str(chunk_info)
                    all_chunks.append(chunk_text)

                elapsed = time.time() - start_time
                logger.info(f"FAST processing created {len(all_chunks)} chunks in {elapsed:.2f}s")

            elif ADVANCED_PDF_AVAILABLE and self.processor_type == "advanced":
                # 2. Fallback to advanced processor with table extraction
                logger.info("Using advanced PDF processing with table extraction...")
                chunks_data = create_enhanced_pdf_chunks(str(pdf_path), chunk_size=512, overlap=64)

                # Extract text chunks from enhanced structure
                all_chunks = []
                for chunk_info in chunks_data:
                    if isinstance(chunk_info, dict):
                        chunk_text = chunk_info.get('content', chunk_info.get('text', str(chunk_info)))
                    else:
                        chunk_text = str(chunk_info)
                    all_chunks.append(chunk_text)

                logger.info(f"Advanced processing created {len(all_chunks)} chunks with table extraction")
                
            else:
                # 3. Final fallback to basic processing
                if not LLAMAINDEX_AVAILABLE:
                    raise ImportError("Neither advanced nor basic PDF processing available")
                
                logger.info("Using basic PDF processing...")
                reader = PDFReader()
                documents = reader.load_data(str(pdf_path))
                logger.info(f"Loaded {len(documents)} pages from PDF")
                
                # Parse into chunks
                all_chunks = []
                for doc in documents:
                    nodes = self.node_parser.get_nodes_from_documents([doc])
                    for node in nodes:
                        all_chunks.append(node.text)
                
                logger.info(f"Basic processing created {len(all_chunks)} chunks")
            
            # 3. Generate embeddings
            self._initialize_embedding_model()
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(all_chunks).tolist()
            logger.info(f"Generated {len(embeddings)} embeddings with {len(embeddings[0])} dimensions")
            
            # 4. Store in MySQL
            doc_id = self.db_manager.store_document(
                filename=filename,
                display_name=display_name,
                file_path=str(pdf_path),
                chunks=all_chunks,
                embeddings=embeddings,
                metadata={
                    'processed_at': datetime.now().isoformat(),
                    'chunk_count': len(all_chunks),
                    'file_size': pdf_path.stat().st_size,
                    'processor': f'mysql_pdf_processor_{self.processor_type}'
                }
            )
            
            logger.info(f"Successfully stored document in MySQL with ID: {doc_id}")
            
            return {
                'success': True,
                'document_id': doc_id,
                'filename': filename,
                'display_name': display_name,
                'chunk_count': len(all_chunks),
                'embedding_dimensions': len(embeddings[0]),
                'processing_time': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Failed to process PDF {filename}: {e}")
            return {
                'success': False,
                'error': str(e),
                'filename': filename
            }
