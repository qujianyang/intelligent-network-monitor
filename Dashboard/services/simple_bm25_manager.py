"""
Simplified BM25 Manager with basic tokenization
Fast and simple implementation for 500k+ chunks
"""

import os
import pickle
import logging
import re
from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import text

logger = logging.getLogger(__name__)

class SimpleBM25Manager:
    """Simple BM25 with basic tokenization for fast searches."""

    def __init__(self, db_manager, index_path: str = "data/simple_bm25_index.pkl"):
        self.db_manager = db_manager
        self.index_path = index_path
        self.bm25_index = None
        self.chunk_ids = []
        self.load_index()

    def basic_tokenize(self, text: str) -> List[str]:
        """Simple tokenization - just lowercase and split on non-alphanumeric."""
        if not text:
            return []
        # Convert to lowercase
        text = text.lower()
        # Split on non-alphanumeric characters
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        # Filter out single characters except important ones
        return [t for t in tokens if len(t) > 1 or t in ['a', 'i']]

    def rebuild_index(self):
        """Rebuild BM25 index from MySQL chunks."""
        logger.info("Building simplified BM25 index...")

        # Load all chunks
        chunks = self._load_chunks()

        if not chunks:
            logger.warning("No chunks found in database")
            return 0

        # Simple tokenization
        tokenized_corpus = []
        self.chunk_ids = []

        for i, chunk in enumerate(chunks):
            tokens = self.basic_tokenize(chunk['text'])
            tokenized_corpus.append(tokens)
            self.chunk_ids.append(chunk['id'])

            if (i + 1) % 1000 == 0:
                logger.info(f"Tokenized {i + 1}/{len(chunks)} chunks...")

        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_corpus)

        # Save index
        self.save_index()

        logger.info(f"BM25 index built with {len(self.chunk_ids)} chunks")
        return len(self.chunk_ids)

    def _load_chunks(self) -> List[Dict]:
        """Load all chunks from database."""
        chunks = []
        batch_size = 10000
        offset = 0

        with self.db_manager.engine.connect() as conn:
            while True:
                query = """
                    SELECT dc.id, dc.content
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.status = 'active'
                    ORDER BY dc.id
                    LIMIT :limit OFFSET :offset
                """
                batch = conn.execute(
                    text(query),
                    {"limit": batch_size, "offset": offset}
                ).fetchall()

                if not batch:
                    break

                chunks.extend([{'id': row.id, 'text': row.content} for row in batch])
                offset += batch_size

                if offset % 50000 == 0:
                    logger.info(f"Loaded {len(chunks)} chunks from database...")

        return chunks

    def save_index(self):
        """Save index to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'bm25_index': self.bm25_index,
                'chunk_ids': self.chunk_ids
            }, f)
        logger.info(f"Saved BM25 index to {self.index_path}")

    def load_index(self) -> bool:
        """Load index from disk if exists."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'rb') as f:
                    data = pickle.load(f)
                self.bm25_index = data['bm25_index']
                self.chunk_ids = data['chunk_ids']
                logger.info(f"Loaded BM25 index with {len(self.chunk_ids)} chunks")
                return True
            except Exception as e:
                logger.warning(f"Failed to load BM25 index: {e}")
                return False
        return False

    def bm25_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Simple BM25 search - returns chunk IDs with scores.

        Args:
            query: Search query
            top_k: Number of results (default 50)

        Returns:
            List of (chunk_id, score) tuples
        """
        if not self.bm25_index:
            logger.warning("No BM25 index available, returning empty results")
            return []

        # Tokenize query
        query_tokens = self.basic_tokenize(query)

        if not query_tokens:
            logger.warning("No valid tokens in query")
            return []

        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)

        # Get top-k indices efficiently
        if len(scores) <= top_k:
            top_indices = np.argsort(scores)[::-1]
        else:
            # Partial sort for efficiency with large arrays
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])][::-1]

        # Return chunk IDs with scores
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk_id = self.chunk_ids[idx]
                score = float(scores[idx])
                results.append((chunk_id, score))

        return results