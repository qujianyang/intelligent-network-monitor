"""
FAISS Manager for high-performance vector search
Uses IVFFlat index optimized for CPU performance
"""

import os
import logging
import time
import numpy as np
import faiss
import pickle
from typing import List, Tuple, Optional
from sqlalchemy import text

logger = logging.getLogger(__name__)

class FAISSManager:
    """
    CPU-optimized FAISS with IVFFlat index.
    Manual rebuild control for production stability.
    """

    def __init__(self, db_manager, index_path: str = "data/faiss_index.bin",
                 metadata_path: str = "data/faiss_metadata.pkl",
                 dimension: int = 768):
        """
        Initialize FAISS manager.

        Args:
            db_manager: Database manager for loading embeddings
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load metadata (chunk IDs)
            dimension: Embedding dimension (768 for all-mpnet-base-v2)
        """
        self.db_manager = db_manager
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = dimension
        self.index = None
        self.chunk_ids = []

        # Try to load existing index
        self.load_index()


    def rebuild_index(self) -> dict:
        """
        Rebuild FAISS index from MySQL embeddings.
        Manual trigger only - no auto-increment.

        Returns:
            Dictionary with rebuild statistics
        """
        logger.info("Starting FAISS index rebuild...")
        start_time = time.time()

        try:
            # Load all embeddings from MySQL
            logger.info("Loading embeddings from MySQL...")
            embeddings, chunk_ids = self._load_all_embeddings()

            if len(embeddings) == 0:
                logger.warning("No embeddings found in database")
                return {
                    'status': 'error',
                    'message': 'No embeddings found'
                }

            num_vectors = len(embeddings)
            logger.info(f"Loaded {num_vectors} embeddings")

            # Convert to numpy array
            embeddings_np = np.array(embeddings, dtype='float32')

            # Build IVFFlat index
            logger.info("Building IVFFlat index...")
            self.index = self._build_ivf_index(embeddings_np)

            # Store chunk IDs
            self.chunk_ids = chunk_ids

            # Save index to disk
            self.save_index()

            duration = time.time() - start_time

            logger.info(f"FAISS index rebuilt successfully!")
            logger.info(f"  Vectors indexed: {num_vectors}")
            logger.info(f"  Time taken: {duration:.1f}s")
            logger.info(f"  Vectors/second: {num_vectors/duration:.0f}")

            return {
                'status': 'success',
                'vectors_indexed': num_vectors,
                'duration_seconds': duration,
                'vectors_per_second': num_vectors / duration,
                'index_type': 'IVFFlat CPU'
            }

        except Exception as e:
            logger.error(f"FAISS rebuild failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _load_all_embeddings(self) -> Tuple[List, List]:
        """
        Load all embeddings from MySQL in batches.

        Returns:
            Tuple of (embeddings, chunk_ids)
        """
        embeddings = []
        chunk_ids = []
        batch_size = 5000
        offset = 0

        with self.db_manager.engine.connect() as conn:
            while True:
                query = """
                    SELECT dc.id, dc.embedding
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.status = 'active'
                      AND dc.embedding IS NOT NULL
                    ORDER BY dc.id
                    LIMIT :limit OFFSET :offset
                """

                batch = conn.execute(
                    text(query),
                    {"limit": batch_size, "offset": offset}
                ).fetchall()

                if not batch:
                    break

                # Process batch
                import json
                for row in batch:
                    try:
                        embedding = json.loads(row.embedding)
                        embeddings.append(embedding)
                        chunk_ids.append(row.id)
                    except Exception as e:
                        logger.warning(f"Failed to load embedding for chunk {row.id}: {e}")
                        continue

                offset += batch_size

                if offset % 10000 == 0:
                    logger.info(f"Loaded {len(embeddings)} embeddings...")

        return embeddings, chunk_ids

    def _build_ivf_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build IVFFlat index with optimal parameters.

        Args:
            embeddings: Numpy array of embeddings (n_samples, dimension)

        Returns:
            Trained FAISS index
        """
        num_vectors = len(embeddings)

        # Calculate optimal number of clusters
        # Rule of thumb: sqrt(n) clusters
        nlist = min(int(np.sqrt(num_vectors)), 1000)
        logger.info(f"Using {nlist} clusters for {num_vectors} vectors")

        # Create quantizer (clustering index)
        quantizer = faiss.IndexFlatL2(self.dimension)

        # Create IVFFlat index
        index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)

        logger.info("Building CPU-optimized IVFFlat index")

        # Train the index
        logger.info("Training IVF index...")

        # Use subset for training (max 50k vectors)
        train_size = min(50000, num_vectors)
        train_data = embeddings[:train_size]

        index.train(train_data)
        logger.info(f"Index trained on {train_size} vectors")

        # Add all vectors
        logger.info("Adding vectors to index...")

        # Add in batches to avoid memory issues
        batch_size = 10000
        for i in range(0, num_vectors, batch_size):
            batch = embeddings[i:i+batch_size]
            index.add(batch)

            if (i + batch_size) % 50000 == 0:
                logger.info(f"Added {i + batch_size} vectors...")

        # Set search parameters
        index.nprobe = 32  # Check 32 clusters during search
        logger.info(f"Index built with nprobe={index.nprobe}")

        return index

    def search(self, query_embedding: List[float], k: int = 50) -> List[Tuple[int, float]]:
        """
        Search for nearest neighbors using FAISS.

        Args:
            query_embedding: Query vector
            k: Number of results to return

        Returns:
            List of (chunk_id, distance) tuples
        """
        if self.index is None:
            logger.warning("No FAISS index available, returning empty results")
            return []

        try:
            # Convert to numpy array
            query_np = np.array([query_embedding], dtype='float32')

            # Search
            distances, indices = self.index.search(query_np, k)

            # Convert to chunk IDs
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0 and idx < len(self.chunk_ids):
                    chunk_id = self.chunk_ids[idx]
                    # Convert L2 distance to similarity score (inverse)
                    similarity = 1.0 / (1.0 + dist)
                    results.append((chunk_id, float(similarity)))

            return results

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

    def save_index(self):
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        try:
            # Save index
            faiss.write_index(self.index, self.index_path)
            logger.info(f"FAISS index saved to {self.index_path}")

            # Save metadata (chunk IDs)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'chunk_ids': self.chunk_ids,
                    'dimension': self.dimension,
                    'num_vectors': len(self.chunk_ids),
                    'build_time': time.time()
                }, f)
            logger.info(f"Metadata saved to {self.metadata_path}")

        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def load_index(self) -> bool:
        """
        Load FAISS index from disk if exists.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            logger.info("No existing index found")
            return False

        try:
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            self.chunk_ids = metadata['chunk_ids']
            self.dimension = metadata['dimension']

            logger.info(f"Loading FAISS index with {len(self.chunk_ids)} vectors")

            # Load index
            self.index = faiss.read_index(self.index_path)
            logger.info("Index loaded successfully")

            # Set search parameters
            self.index.nprobe = 32

            logger.info(f"FAISS index loaded successfully")
            return True

        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            return False

    def get_index_stats(self) -> dict:
        """Get current index statistics."""
        if self.index is None:
            return {
                'status': 'not_built',
                'message': 'No index available. Run rebuild_index() first.'
            }

        return {
            'status': 'ready',
            'num_vectors': len(self.chunk_ids),
            'dimension': self.dimension,
            'index_type': 'IVFFlat',
            'nprobe': getattr(self.index, 'nprobe', 'N/A'),
            'memory_usage_mb': len(self.chunk_ids) * self.dimension * 4 / (1024 * 1024)
        }