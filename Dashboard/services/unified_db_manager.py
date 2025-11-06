from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

class UnifiedDBManager:
    """
    MySQL-based unified database manager for QKD system
    Handles: Vector storage, Prompt management, Investigation tracking
    """

    def __init__(self, connection_string: str = None):
        if connection_string is None:
            # Use your existing database connection info
            db_user = os.getenv('DB_USER', 'root')
            db_password = os.getenv('DB_PASSWORD', '')
            db_host = os.getenv('DB_HOST', '127.0.0.1')
            db_port = os.getenv('DB_PORT', '3307')
            db_name = os.getenv('DB_NAME', 'qkd')
            
            connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

        self.engine = create_engine(connection_string, pool_pre_ping=True, pool_recycle=300)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info("UnifiedDBManager initialized with MySQL backend")

    # VECTOR STORAGE METHODS (replaces multi_index_manager.py)
    def store_document(self, filename: str, display_name: str, file_path: str,
                      chunks: List[str], embeddings: List[List[float]],
                      metadata: Dict = None) -> int:
        """Store document and its chunks with embeddings"""
        with self.engine.begin() as conn:
            # Insert document - EXPLICIT status for clarity (KISS principle)
            result = conn.execute(text("""
                INSERT INTO documents (filename, display_name, file_path, chunk_count, status, metadata)
                VALUES (:filename, :display_name, :file_path, :chunk_count, :status, :metadata)
            """), {
                "filename": filename,
                "display_name": display_name,
                "file_path": file_path,
                "chunk_count": len(chunks),
                "status": "active",  # Explicit is better than implicit
                "metadata": json.dumps(metadata or {})
            })

            doc_id = result.lastrowid

            # Insert chunks with embeddings
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                conn.execute(text("""
                    INSERT INTO document_chunks (document_id, chunk_index, content, embedding, metadata)
                    VALUES (:doc_id, :idx, :content, :embedding, :metadata)
                """), {
                    "doc_id": doc_id,
                    "idx": i,
                    "content": chunk,
                    "embedding": json.dumps(embedding),  # Store as JSON array
                    "metadata": json.dumps({"chunk_length": len(chunk)})
                })

            logger.info(f"Stored document '{filename}' with {len(chunks)} chunks")
            return doc_id

    def semantic_search(self, query_embedding: List[float], top_k: int = 5,
                       document_filter: str = None) -> List[Dict]:
        """MySQL-based semantic search using cosine similarity"""
        with self.engine.connect() as conn:
            # Build query with optional document filter
            base_query = """
                SELECT dc.id, dc.content, dc.embedding, d.filename, d.display_name,
                       dc.chunk_index, dc.metadata as chunk_metadata
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE d.status = 'active'
            """

            params = {}
            if document_filter and document_filter != 'all':
                base_query += " AND d.filename = :doc_filter"
                params["doc_filter"] = document_filter

            results = conn.execute(text(base_query), params).fetchall()

            # Calculate cosine similarities
            scored_results = []
            query_embedding_np = np.array(query_embedding).reshape(1, -1)

            for row in results:
                try:
                    chunk_embedding = json.loads(row.embedding)
                    chunk_embedding_np = np.array(chunk_embedding).reshape(1, -1)
                    
                    similarity = cosine_similarity(query_embedding_np, chunk_embedding_np)[0][0]
                    
                    scored_results.append({
                        "id": row.id,
                        "content": row.content,
                        "similarity": float(similarity),
                        "filename": row.filename,
                        "display_name": row.display_name,
                        "chunk_index": row.chunk_index,
                        "metadata": json.loads(row.chunk_metadata or "{}")
                    })
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to process embedding for chunk {row.id}: {e}")
                    continue

            # Sort by similarity and return top_k
            scored_results.sort(key=lambda x: x["similarity"], reverse=True)
            return scored_results[:top_k]

    # HYBRID SEARCH METHODS
    def parallel_hybrid_search(self, query_text: str, query_embedding: List[float],
                              top_k: int = 5, use_reranker: bool = True,
                              document_filter: str = None) -> List[Dict]:
        """
        Parallel hybrid search with BM25 + FAISS GPU + RRF fusion.
        True parallel execution with no gating.

        Args:
            query_text: Text query for BM25
            query_embedding: Embedding vector for FAISS
            top_k: Number of results to return
            use_reranker: Whether to use cross-encoder reranking
            document_filter: Optional document filter

        Returns:
            List of search results with RRF scores
        """
        from Dashboard.services.simple_bm25_manager import SimpleBM25Manager
        from Dashboard.services.faiss_gpu_manager import FAISSManager
        from Dashboard.services.parallel_hybrid_search import ParallelHybridSearch

        # Get absolute paths for indexes
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        bm25_path = os.path.join(project_root, "data", "simple_bm25_index.pkl")
        faiss_path = os.path.join(project_root, "data", "faiss_gpu_index.bin")
        faiss_meta = os.path.join(project_root, "data", "faiss_metadata.pkl")

        # Initialize managers if not already done with proper paths
        if not hasattr(self, '_bm25_manager'):
            self._bm25_manager = SimpleBM25Manager(self, index_path=bm25_path)
            logger.info(f"Initialized SimpleBM25Manager with index at: {bm25_path}")

        if not hasattr(self, '_faiss_manager'):
            self._faiss_manager = FAISSManager(
                self,
                index_path=faiss_path,
                metadata_path=faiss_meta
            )
            logger.info(f"Initialized FAISSManager with index at: {faiss_path}")

        if not hasattr(self, '_parallel_search'):
            self._parallel_search = ParallelHybridSearch(
                self,
                self._bm25_manager,
                self._faiss_manager
            )
            logger.info("Initialized ParallelHybridSearch")

        # Perform parallel hybrid search
        return self._parallel_search.search(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            use_reranker=use_reranker,
            document_filter=document_filter
        )

    def delete_document(self, document_id: int = None, filename: str = None) -> Dict[str, Any]:
        """
        Soft delete a document by marking its status as 'deleted'

        Args:
            document_id: The ID of the document to delete
            filename: Alternative - delete by filename

        Returns:
            Dictionary with deletion status and details
        """
        if not document_id and not filename:
            return {
                "success": False,
                "error": "Either document_id or filename must be provided"
            }

        try:
            with self.engine.begin() as conn:
                # First, get document info before deletion
                if document_id:
                    query = text("""
                        SELECT id, filename, display_name, chunk_count, file_path
                        FROM documents
                        WHERE id = :doc_id AND status = 'active'
                    """)
                    params = {"doc_id": document_id}
                else:
                    query = text("""
                        SELECT id, filename, display_name, chunk_count, file_path
                        FROM documents
                        WHERE filename = :filename AND status = 'active'
                    """)
                    params = {"filename": filename}

                result = conn.execute(query, params).fetchone()

                if not result:
                    return {
                        "success": False,
                        "error": f"Document not found or already deleted"
                    }

                doc_info = {
                    "id": result.id,
                    "filename": result.filename,
                    "display_name": result.display_name,
                    "chunk_count": result.chunk_count,
                    "file_path": result.file_path
                }

                # Perform soft delete - SIMPLIFIED following KISS principle
                # Just set status to 'inactive' - that's all we need!
                update_query = text("""
                    UPDATE documents
                    SET status = 'inactive'
                    WHERE id = :doc_id
                """)

                conn.execute(update_query, {
                    "doc_id": doc_info["id"]
                })

                logger.info(f"Soft deleted document: {doc_info['filename']} (ID: {doc_info['id']})")

                return {
                    "success": True,
                    "message": f"Document '{doc_info['filename']}' deleted successfully",
                    "document": doc_info
                }

        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def list_documents(self, include_deleted: bool = False) -> List[Dict]:
        """
        List all documents in the database

        Args:
            include_deleted: Whether to include deleted documents

        Returns:
            List of document information dictionaries
        """
        try:
            with self.engine.connect() as conn:
                if include_deleted:
                    query = text("""
                        SELECT id, filename, display_name, file_path, chunk_count,
                               status, metadata, created_at
                        FROM documents
                        ORDER BY created_at DESC
                    """)
                else:
                    query = text("""
                        SELECT id, filename, display_name, file_path, chunk_count,
                               status, metadata, created_at
                        FROM documents
                        WHERE status = 'active'
                        ORDER BY created_at DESC
                    """)

                results = conn.execute(query).fetchall()

                documents = []
                for row in results:
                    metadata = {}
                    if row.metadata:
                        try:
                            metadata = json.loads(row.metadata)
                        except (json.JSONDecodeError, TypeError):
                            metadata = {}

                    documents.append({
                        "id": row.id,
                        "filename": row.filename,
                        "display_name": row.display_name,
                        "file_path": row.file_path,
                        "chunk_count": row.chunk_count,
                        "status": row.status,
                        "metadata": metadata,
                        "created_at": row.created_at.isoformat() if row.created_at else None
                    })

                return documents

        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    def test_connection(self):
        """Test if database connection works"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1")).fetchone()
                print("SUCCESS: Database connection successful!")
                return True
        except Exception as e:
            print(f"ERROR: Database connection failed: {e}")
            return False


# Test function
if __name__ == "__main__":
    db = UnifiedDBManager()
    db.test_connection()