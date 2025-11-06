"""
Parallel Hybrid Search with BM25 + FAISS GPU + RRF Fusion
True parallel execution with no gating between methods
"""

import logging
import numpy as np
import json
from typing import List, Dict, Optional, Tuple
from sqlalchemy import text
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
import concurrent.futures
import time

logger = logging.getLogger(__name__)

class ParallelHybridSearch:
    """
    Parallel hybrid search with Reciprocal Rank Fusion (RRF).
    No gating - both BM25 and FAISS search independently.
    """

    def __init__(self, db_manager, bm25_manager, faiss_manager):
        """
        Initialize parallel hybrid search.

        Args:
            db_manager: Database manager
            bm25_manager: BM25 index manager
            faiss_manager: FAISS GPU index manager
        """
        self.db_manager = db_manager
        self.bm25_manager = bm25_manager
        self.faiss_manager = faiss_manager
        self.cross_encoder = None

    def _load_cross_encoder(self):
        """Lazy load cross-encoder model to save memory."""
        if self.cross_encoder is None:
            logger.info("Loading cross-encoder model for reranking...")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Cross-encoder model loaded")

    def search(self, query_text: str, query_embedding: List[float],
              top_k: int = 5, use_reranker: bool = True,
              document_filter: Optional[str] = None,
              bm25_top_k: int = 150, faiss_top_k: int = 50) -> List[Dict]:
        """
        Parallel hybrid search with RRF fusion.

        Steps:
        1. BM25 and FAISS search in parallel (no dependencies)
        2. RRF fusion of results
        3. Load chunk content
        4. Optional cross-encoder reranking
        5. Return top-k

        Args:
            query_text: Text query for BM25
            query_embedding: Embedding vector for FAISS
            top_k: Number of final results
            use_reranker: Whether to use cross-encoder
            document_filter: Optional document filter
            bm25_top_k: Number of BM25 candidates (default 150)
            faiss_top_k: Number of FAISS candidates (default 50)

        Returns:
            List of search results with scores
        """
        start_time = time.time()

        # Step 1: Parallel execution of BM25 and FAISS
        logger.info(f"Starting parallel search: BM25({bm25_top_k}) + FAISS({faiss_top_k})")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both searches simultaneously
            bm25_future = executor.submit(
                self._bm25_search,
                query_text,
                bm25_top_k
            )

            faiss_future = executor.submit(
                self._faiss_search,
                query_embedding,
                faiss_top_k
            )

            # Wait for both to complete
            bm25_results = bm25_future.result()
            faiss_results = faiss_future.result()

        search_time = time.time() - start_time
        logger.info(f"Parallel search completed in {search_time:.3f}s")

        # Log search statistics
        logger.info(f"BM25 found {len(bm25_results)} candidates")
        logger.info(f"FAISS found {len(faiss_results)} candidates")

        # Step 2: RRF Fusion
        merged_candidates = self._reciprocal_rank_fusion(
            bm25_results,
            faiss_results,
            k=60  # Standard RRF constant
        )

        logger.info(f"RRF fusion produced {len(merged_candidates)} unique candidates")

        if not merged_candidates:
            logger.warning("No candidates after RRF fusion")
            return []

        # Step 3: Load chunk content for merged candidates
        chunks = self._load_chunks_with_content(
            list(merged_candidates.keys()),
            document_filter
        )

        if not chunks:
            logger.warning("No chunks found after filtering")
            return []

        # Prepare results with RRF scores
        results = []
        for chunk in chunks:
            chunk_id = chunk['id']
            if chunk_id in merged_candidates:
                results.append({
                    'id': chunk_id,
                    'content': chunk['content'],
                    'rrf_score': merged_candidates[chunk_id]['rrf_score'],
                    'bm25_rank': merged_candidates[chunk_id].get('bm25_rank'),
                    'faiss_rank': merged_candidates[chunk_id].get('faiss_rank'),
                    'filename': chunk['filename'],
                    'display_name': chunk.get('display_name', chunk['filename']),
                    'chunk_index': chunk.get('chunk_index', 0)
                })

        # Sort by RRF score
        results.sort(key=lambda x: x['rrf_score'], reverse=True)

        # Step 4: Optional cross-encoder reranking
        if use_reranker and len(results) > 0:
            results = self._rerank_with_cross_encoder(query_text, results)

        # Step 5: Return top-k
        final_results = results[:top_k]

        total_time = time.time() - start_time
        logger.info(f"Total search time: {total_time:.3f}s, returning {len(final_results)} results")

        return final_results

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, int]]:
        """
        BM25 search branch.

        Returns:
            List of (chunk_id, rank) tuples
        """
        try:
            # Get BM25 results
            bm25_results = self.bm25_manager.bm25_search(query, top_k)

            # Convert to (chunk_id, rank) format
            results = []
            for rank, (chunk_id, score) in enumerate(bm25_results, 1):
                results.append((chunk_id, rank))

            logger.info(f"BM25 returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _faiss_search(self, query_embedding: List[float], top_k: int) -> List[Tuple[int, int]]:
        """
        FAISS GPU search branch.

        Returns:
            List of (chunk_id, rank) tuples
        """
        try:
            # Get FAISS results
            faiss_results = self.faiss_manager.search(query_embedding, top_k)

            # Convert to (chunk_id, rank) format
            results = []
            for rank, (chunk_id, similarity) in enumerate(faiss_results, 1):
                results.append((chunk_id, rank))

            logger.info(f"FAISS returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

    def _reciprocal_rank_fusion(self, bm25_results: List[Tuple[int, int]],
                               faiss_results: List[Tuple[int, int]],
                               k: int = 60) -> Dict[int, Dict]:
        """
        Reciprocal Rank Fusion (RRF) to merge results.

        Formula: RRF_score = sum(1 / (rank + k))

        Args:
            bm25_results: List of (chunk_id, rank) from BM25
            faiss_results: List of (chunk_id, rank) from FAISS
            k: RRF constant (default 60, standard value)

        Returns:
            Dictionary mapping chunk_id to fusion data
        """
        fusion_scores = {}

        # Process BM25 results
        for chunk_id, rank in bm25_results:
            if chunk_id not in fusion_scores:
                fusion_scores[chunk_id] = {
                    'rrf_score': 0,
                    'bm25_rank': None,
                    'faiss_rank': None
                }

            # Add BM25 contribution
            fusion_scores[chunk_id]['rrf_score'] += 1.0 / (rank + k)
            fusion_scores[chunk_id]['bm25_rank'] = rank

        # Process FAISS results
        for chunk_id, rank in faiss_results:
            if chunk_id not in fusion_scores:
                fusion_scores[chunk_id] = {
                    'rrf_score': 0,
                    'bm25_rank': None,
                    'faiss_rank': None
                }

            # Add FAISS contribution
            fusion_scores[chunk_id]['rrf_score'] += 1.0 / (rank + k)
            fusion_scores[chunk_id]['faiss_rank'] = rank

        # Calculate overlap statistics
        bm25_ids = set(chunk_id for chunk_id, _ in bm25_results)
        faiss_ids = set(chunk_id for chunk_id, _ in faiss_results)
        overlap = bm25_ids & faiss_ids

        logger.info(f"RRF fusion statistics:")
        logger.info(f"  BM25 unique: {len(bm25_ids - faiss_ids)}")
        logger.info(f"  FAISS unique: {len(faiss_ids - bm25_ids)}")
        logger.info(f"  Overlap: {len(overlap)}")
        logger.info(f"  Total unique: {len(fusion_scores)}")

        return fusion_scores

    def _load_chunks_with_content(self, chunk_ids: List[int],
                                 document_filter: Optional[str]) -> List[Dict]:
        """
        Load chunk content from MySQL.

        Args:
            chunk_ids: List of chunk IDs to load
            document_filter: Optional document filter

        Returns:
            List of chunk dictionaries
        """
        if not chunk_ids:
            return []

        with self.db_manager.engine.connect() as conn:
            # Build query
            if document_filter and document_filter != 'all':
                query = """
                    SELECT dc.id, dc.content,
                           d.filename, d.display_name,
                           dc.chunk_index, dc.metadata as chunk_metadata
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE dc.id IN :chunk_ids
                      AND d.filename = :doc_filter
                      AND d.status = 'active'
                """
                params = {
                    "chunk_ids": tuple(chunk_ids),
                    "doc_filter": document_filter
                }
            else:
                query = """
                    SELECT dc.id, dc.content,
                           d.filename, d.display_name,
                           dc.chunk_index, dc.metadata as chunk_metadata
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE dc.id IN :chunk_ids
                      AND d.status = 'active'
                """
                params = {"chunk_ids": tuple(chunk_ids)}

            results = conn.execute(text(query), params).fetchall()

        # Convert to dictionaries
        chunks = []
        for row in results:
            chunk = {
                'id': row.id,
                'content': row.content,
                'filename': row.filename,
                'display_name': row.display_name,
                'chunk_index': row.chunk_index
            }

            # Parse metadata if exists
            if row.chunk_metadata:
                try:
                    chunk['metadata'] = json.loads(row.chunk_metadata)
                except:
                    chunk['metadata'] = {}
            else:
                chunk['metadata'] = {}

            chunks.append(chunk)

        return chunks

    def _rerank_with_cross_encoder(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Rerank results using cross-encoder.

        Args:
            query: Query text
            results: List of results to rerank

        Returns:
            Reranked results
        """
        try:
            # Load cross-encoder if needed
            self._load_cross_encoder()

            # Prepare pairs for cross-encoder (limit to top 30)
            num_to_rerank = min(30, len(results))
            pairs = [[query, r['content']] for r in results[:num_to_rerank]]

            logger.info(f"Reranking top {num_to_rerank} results with cross-encoder")

            # Get cross-encoder scores
            ce_scores = self.cross_encoder.predict(pairs)

            # Update scores for reranked items
            for i, score in enumerate(ce_scores):
                results[i]['rerank_score'] = float(score)

                # Combine with RRF score (80% cross-encoder, 20% RRF)
                results[i]['final_score'] = (
                    0.8 * score +
                    0.2 * results[i]['rrf_score']
                )

            # Sort by final score
            results.sort(key=lambda x: x.get('final_score', x['rrf_score']), reverse=True)

            logger.info("Cross-encoder reranking completed")
            return results

        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}, using RRF scores")
            return results