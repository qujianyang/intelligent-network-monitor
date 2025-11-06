"""
Cross-Encoder Reranker Service
=============================
Lightweight, CPU-optimized reranking for improved RAG relevance.

Features:
- ms-marco-MiniLM-L-6-v2 model (CPU-optimized)
- Caching for repeated queries
- Lazy loading for performance
- Graceful fallback handling
- Performance monitoring
"""

import os
import time
import hashlib
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Core dependencies
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Caching
try:
    import diskcache as dc
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False

# Performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Configuration
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # CPU-optimized model
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "reranker_cache")
MAX_RERANK_CANDIDATES = 5  # Your optimization: only rerank top 5
CACHE_EXPIRY_HOURS = 24  # Cache results for 24 hours
BATCH_SIZE = 8  # Process in small batches for CPU efficiency

# Domain-aware reranking configuration
DOMAIN_AWARE_RERANKING = {
    "disable_for_installation": False,  # FIXED: Re-enable reranking for installation queries
    "disable_for_procedures": False,    # FIXED: Re-enable reranking for procedure queries
    "technical_content_boost": 1.2,   # Boost technical content when reranking is enabled
    "procedure_keywords": [
        "install", "installation", "mounting", "procedure", "steps", "instructions",
        "how to", "configure", "setup", "connect", "troubleshoot", "repair", "fix"
    ],
    "general_content_penalty": 0.9    # Reduce score for general/introductory content
}

# Import configuration from main config
try:
    from config import HYBRID_SEARCH_CONFIG
    # Update domain-aware settings from main config
    if HYBRID_SEARCH_CONFIG.get("domain_aware_reranking", False):
        DOMAIN_AWARE_RERANKING.update({
            "disable_for_installation": HYBRID_SEARCH_CONFIG.get("disable_rerank_for_installation", True),
            "disable_for_procedures": HYBRID_SEARCH_CONFIG.get("disable_rerank_for_procedures", True),
            "technical_content_boost": HYBRID_SEARCH_CONFIG.get("technical_content_rerank_boost", 1.2),
            "general_content_penalty": HYBRID_SEARCH_CONFIG.get("general_content_rerank_penalty", 0.9)
        })
        logger.info("Domain-aware reranking configuration loaded from main config")
except ImportError:
    logger.warning("Could not load main config - using default domain-aware reranking settings")


@dataclass
class RerankResult:
    """Result from reranking operation."""
    original_scores: List[float]
    reranked_scores: List[float]
    reranked_indices: List[int]
    processing_time: float
    cache_hit: bool
    model_used: str


class CrossEncoderReranker:
    """
    CPU-optimized cross-encoder reranker with caching and performance monitoring.
    """
    
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self.model = None
        self.cache = None
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "total_rerank_time": 0.0,
            "model_load_time": 0.0
        }
        self._lock = threading.Lock()
        self._initialized = False
        
        # Initialize cache if available
        if DISKCACHE_AVAILABLE:
            os.makedirs(CACHE_DIR, exist_ok=True)
            self.cache = dc.Cache(CACHE_DIR, size_limit=100_000_000)  # 100MB cache
            logger.info(f"✅ Reranker cache initialized at {CACHE_DIR}")
        else:
            logger.warning("WARNING: diskcache not available - reranking will not be cached")
    
    def _lazy_load_model(self) -> bool:
        """Lazy load the cross-encoder model when first needed."""
        if self._initialized:
            return True
        
        with self._lock:
            if self._initialized:
                return True
            
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.error("❌ sentence-transformers not available for reranking")
                return False
            
            try:
                start_time = time.perf_counter()
                logger.info(f"Loading cross-encoder model: {self.model_name}")
                
                self.model = CrossEncoder(self.model_name)
                
                load_time = time.perf_counter() - start_time
                self.stats["model_load_time"] = load_time
                self._initialized = True
                
                logger.info(f"Cross-encoder model loaded in {load_time:.2f}s")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to load cross-encoder model: {e}")
                return False
    
    def _create_cache_key(self, query: str, candidate_texts: List[str]) -> str:
        """Create a cache key for the query and candidates."""
        # Create a hash of query + candidate texts for caching
        content = query + "|".join(candidate_texts)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_scores(self, cache_key: str) -> Optional[List[float]]:
        """Get cached reranking scores if available."""
        if not self.cache:
            return None
        
        try:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                # Check if cache entry is still valid
                timestamp, scores = cached_data
                if time.time() - timestamp < CACHE_EXPIRY_HOURS * 3600:
                    return scores
                else:
                    # Remove expired entry
                    del self.cache[cache_key]
        except Exception as e:
            logger.warning(f"[WARNING] Cache read error: {e}")
        
        return None
    
    def _cache_scores(self, cache_key: str, scores: List[float]):
        """Cache reranking scores for future use."""
        if not self.cache:
            return
        
        try:
            self.cache[cache_key] = (time.time(), scores)
        except Exception as e:
            logger.warning(f"[WARNING] Cache write error: {e}")
    
    def _is_procedure_query(self, query: str) -> bool:
        """
        Detect if query is asking for procedures/installation instructions.
        
        Args:
            query: Search query to analyze
            
        Returns:
            bool: True if query appears to be asking for procedures
        """
        query_lower = query.lower()
        procedure_keywords = DOMAIN_AWARE_RERANKING.get("procedure_keywords", [])
        
        return any(keyword in query_lower for keyword in procedure_keywords)
    
    def _should_disable_reranking(self, query: str) -> bool:
        """
        Determine if reranking should be disabled for this query.
        
        Args:
            query: Search query to analyze
            
        Returns:
            bool: True if reranking should be disabled
        """
        if not DOMAIN_AWARE_RERANKING.get("disable_for_installation", False):
            return False
        
        # Check if this is a procedure/installation query
        if self._is_procedure_query(query):
            logger.debug(f"Disabling reranking for procedure query: {query[:50]}...")
            return True
        
        return False
    
    def _apply_content_scoring_adjustments(self, query: str, candidates: List[Any], 
                                         scores: List[float]) -> List[float]:
        """
        Apply domain-aware scoring adjustments to cross-encoder scores.
        
        Args:
            query: Search query
            candidates: List of candidate objects
            scores: Original cross-encoder scores
            
        Returns:
            List[float]: Adjusted scores
        """
        if not candidates or not scores:
            return scores
        
        adjusted_scores = scores.copy()
        technical_boost = DOMAIN_AWARE_RERANKING.get("technical_content_boost", 1.0)
        general_penalty = DOMAIN_AWARE_RERANKING.get("general_content_penalty", 1.0)
        
        for i, candidate in enumerate(candidates):
            if i >= len(adjusted_scores):
                break
            
            # Check if candidate has metadata (from LlamaIndex nodes)
            metadata = {}
            if hasattr(candidate, 'original_result') and hasattr(candidate.original_result, 'node'):
                metadata = candidate.original_result.node.metadata or {}
            elif hasattr(candidate, 'metadata'):
                metadata = candidate.metadata or {}
            
            # Get section header and content type
            section_header = metadata.get('section_header', '')
            chunk_type = metadata.get('chunk_type', 'text')
            
            # Apply technical content boost
            if any(term in section_header.lower() for term in ['install', 'procedure', 'mounting', 'configuration']):
                adjusted_scores[i] *= technical_boost
                logger.debug(f"Applied technical boost to: {section_header[:40]}...")
            
            # Apply penalty for general/introductory content  
            elif any(term in section_header.lower() for term in ['introduction', 'overview', 'general', 'about']):
                adjusted_scores[i] *= general_penalty
                logger.debug(f"Applied general content penalty to: {section_header[:40]}...")
        
        return adjusted_scores
    
    def rerank(self, query: str, candidates: List[Any], 
               max_candidates: int = MAX_RERANK_CANDIDATES) -> RerankResult:
        """
        Rerank candidates using cross-encoder model with domain-aware adjustments.
        
        Args:
            query: Search query
            candidates: List of candidate objects (with .text and .score attributes)
            max_candidates: Maximum number of candidates to rerank
            
        Returns:
            RerankResult with reranked scores and metadata
        """
        start_time = time.perf_counter()
        
        # DOMAIN-AWARE RERANKING: Check if reranking should be disabled
        if self._should_disable_reranking(query):
            # Return original order without reranking for installation/procedure queries
            candidates_to_rerank = candidates[:max_candidates]
            original_scores = [getattr(c, 'score', 0.0) for c in candidates_to_rerank]
            processing_time = time.perf_counter() - start_time
            
            logger.info(f"Reranking disabled for installation/procedure query: {query[:50]}...")
            
            return RerankResult(
                original_scores=original_scores,
                reranked_scores=original_scores,
                reranked_indices=list(range(len(candidates_to_rerank))),
                processing_time=processing_time,
                cache_hit=False,
                model_used="domain_aware_disabled"
            )
        
        # Limit candidates to max_candidates for efficiency
        candidates_to_rerank = candidates[:max_candidates]
        candidate_texts = [getattr(c, 'text', str(c)) for c in candidates_to_rerank]
        original_scores = [getattr(c, 'score', 0.0) for c in candidates_to_rerank]
        
        # Check cache first
        cache_key = self._create_cache_key(query, candidate_texts)
        cached_scores = self._get_cached_scores(cache_key)
        
        if cached_scores is not None:
            # Cache hit
            processing_time = time.perf_counter() - start_time
            self.stats["cache_hits"] += 1
            self.stats["total_queries"] += 1
            
            # Sort by cached scores
            scored_pairs = list(zip(cached_scores, range(len(candidates_to_rerank))))
            scored_pairs.sort(key=lambda x: x[0], reverse=True)
            reranked_indices = [idx for _, idx in scored_pairs]
            
            logger.debug(f"[CACHE_HIT] Reranker cache hit for query: {query[:50]}...")
            
            return RerankResult(
                original_scores=original_scores,
                reranked_scores=cached_scores,
                reranked_indices=reranked_indices,
                processing_time=processing_time,
                cache_hit=True,
                model_used=self.model_name
            )
        
        # Cache miss - need to compute scores
        if not self._lazy_load_model():
            # Fallback: return original order
            logger.warning("[WARNING] Reranker not available, returning original order")
            processing_time = time.perf_counter() - start_time
            return RerankResult(
                original_scores=original_scores,
                reranked_scores=original_scores,
                reranked_indices=list(range(len(candidates_to_rerank))),
                processing_time=processing_time,
                cache_hit=False,
                model_used="fallback"
            )
        
        try:
            # Create query-candidate pairs for cross-encoder
            pairs = [(query, text) for text in candidate_texts]
            
            # Get cross-encoder scores
            reranked_scores = self.model.predict(pairs).tolist()
            
            # Apply domain-aware scoring adjustments
            reranked_scores = self._apply_content_scoring_adjustments(query, candidates_to_rerank, reranked_scores)
            
            # Cache the results
            self._cache_scores(cache_key, reranked_scores)
            
            # Sort by reranked scores
            scored_pairs = list(zip(reranked_scores, range(len(candidates_to_rerank))))
            scored_pairs.sort(key=lambda x: x[0], reverse=True)
            reranked_indices = [idx for _, idx in scored_pairs]
            
            # Update stats
            processing_time = time.perf_counter() - start_time
            self.stats["total_queries"] += 1
            self.stats["total_rerank_time"] += processing_time
            
            logger.debug(f"🔄 Reranked {len(candidates_to_rerank)} candidates in {processing_time:.3f}s")
            
            return RerankResult(
                original_scores=original_scores,
                reranked_scores=reranked_scores,
                reranked_indices=reranked_indices,
                processing_time=processing_time,
                cache_hit=False,
                model_used=self.model_name
            )
            
        except Exception as e:
            logger.error(f"❌ Reranking failed: {e}")
            # Fallback: return original order
            processing_time = time.perf_counter() - start_time
            return RerankResult(
                original_scores=original_scores,
                reranked_scores=original_scores,
                reranked_indices=list(range(len(candidates_to_rerank))),
                processing_time=processing_time,
                cache_hit=False,
                model_used="fallback_error"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranker performance statistics."""
        cache_hit_rate = (self.stats["cache_hits"] / self.stats["total_queries"] 
                         if self.stats["total_queries"] > 0 else 0.0)
        
        avg_rerank_time = (self.stats["total_rerank_time"] / 
                          (self.stats["total_queries"] - self.stats["cache_hits"])
                          if (self.stats["total_queries"] - self.stats["cache_hits"]) > 0 else 0.0)
        
        stats = {
            "model_name": self.model_name,
            "initialized": self._initialized,
            "total_queries": self.stats["total_queries"],
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": cache_hit_rate,
            "model_load_time": self.stats["model_load_time"],
            "average_rerank_time": avg_rerank_time,
            "cache_available": self.cache is not None,
            "max_rerank_candidates": MAX_RERANK_CANDIDATES
        }
        
        # Add system info if available
        if PSUTIL_AVAILABLE:
            stats["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            stats["memory_percent"] = psutil.virtual_memory().percent
        
        return stats
    
    def clear_cache(self):
        """Clear the reranking cache."""
        if self.cache:
            self.cache.clear()
            logger.info("🧹 Reranker cache cleared")
    
    def warm_up(self, sample_query: str = "test query", sample_text: str = "test document"):
        """Warm up the model with a sample query to reduce first-query latency."""
        if not self._lazy_load_model():
            return False
        
        try:
            logger.info("🔥 Warming up reranker model...")
            start_time = time.perf_counter()
            
            # Simple warm-up prediction
            self.model.predict([(sample_query, sample_text)])
            
            warm_up_time = time.perf_counter() - start_time
            logger.info(f"✅ Reranker warmed up in {warm_up_time:.2f}s")
            return True
            
        except Exception as e:
            logger.warning(f"[WARNING] Reranker warm-up failed: {e}")
            return False




# Global reranker instance (singleton pattern)
_global_reranker = None
_reranker_lock = threading.Lock()


def get_reranker() -> CrossEncoderReranker:
    """Get the global reranker instance (singleton)."""
    global _global_reranker
    
    if _global_reranker is None:
        with _reranker_lock:
            if _global_reranker is None:
                _global_reranker = CrossEncoderReranker()
    
    return _global_reranker


def rerank_candidates(query: str, candidates: List[Any], 
                     max_candidates: int = MAX_RERANK_CANDIDATES) -> Tuple[List[Any], RerankResult]:
    """
    Convenience function to rerank candidates and return reordered list.
    
    Args:
        query: Search query
        candidates: List of candidate objects
        max_candidates: Maximum candidates to rerank
        
    Returns:
        Tuple of (reranked_candidates, rerank_result)
    """
    reranker = get_reranker()
    result = reranker.rerank(query, candidates, max_candidates)
    
    # Reorder candidates based on reranked indices
    candidates_to_rerank = candidates[:max_candidates]
    remaining_candidates = candidates[max_candidates:]
    
    reranked_candidates = [candidates_to_rerank[i] for i in result.reranked_indices]
    
    # Combine reranked candidates with remaining ones
    final_candidates = reranked_candidates + remaining_candidates
    
    return final_candidates, result


 