"""
QKD Assistant with Unified Database Architecture
===============================================
Enhanced version using MySQL-based vector storage for scalable, efficient retrieval.
"""

import logging
from typing import List, Tuple, Dict, Any
from Dashboard.services.unified_db_manager import UnifiedDBManager
from Dashboard.services.llm_client import generate_answer

logger = logging.getLogger(__name__)

# Same RAG prompt template as original
RAG_PROMPT_TEMPLATE = """You are a QKD technical documentation assistant. Use the provided context to answer questions accurately and comprehensively.

**Context from Documentation:**
---------------------
{context_str}
---------------------

**Query:** {query_str}

**Instructions:**
- Base your answer strictly on the provided context
- Include specific values, settings, and technical details exactly as mentioned
- If the context doesn't contain sufficient information, acknowledge the limitation
- Provide clear, structured answers

**Answer:**
"""

def get_qkd_answer_multi_index(
    question: str, 
    document_filter: str = None,
    top_k: int = 5
) -> Tuple[str, float]:
    """
    Retrieve context and generate answer using unified database architecture.
    
    This is significantly faster and more scalable than the original because:
    1. MySQL-based vector storage (no JSON file loading)
    2. Only loads needed chunks into memory
    3. Can handle millions of documents
    4. Direct similarity calculation on query results
    
    Args:
        question: User's question
        document_filter: Specific document to query (None for all)
        top_k: Number of chunks to retrieve
        
    Returns:
        Tuple of (answer, confidence_score)
    """
    # Initialize database manager
    db_manager = UnifiedDBManager()
    
    # Generate embedding for the question (using sentence-transformers)
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        question_embedding = embedding_model.encode(question).tolist()
    except ImportError:
        logger.error("SentenceTransformer not available - using placeholder embedding")
        question_embedding = [0.1] * 384  # Placeholder
    
    # Try parallel hybrid search first, fallback to semantic
    logger.info(f"Querying database with filter: {document_filter}")

    try:
        # Use parallel hybrid search (BM25 + FAISS + RRF)
        logger.info("Using parallel hybrid search with BM25 + FAISS GPU")
        chunks = db_manager.parallel_hybrid_search(
            query_text=question,
            query_embedding=question_embedding,
            top_k=top_k,
            use_reranker=True,  # Enable cross-encoder reranking
            document_filter=document_filter
        )
    except Exception as e:
        # Fallback to simple semantic search if hybrid not available
        logger.warning(f"Parallel hybrid search failed: {e}, falling back to semantic search")
        chunks = db_manager.semantic_search(
            query_embedding=question_embedding,
            top_k=top_k,
            document_filter=document_filter
        )
    
    # Check if we have any context
    if not chunks:
        logger.warning("No relevant chunks found")
        return "I am sorry, the provided documentation does not contain information on this topic.", 0.1
    
    # Format context
    context_str = format_context_unified(chunks)
    
    # Generate answer
    prompt = RAG_PROMPT_TEMPLATE.format(
        context_str=context_str,
        query_str=question
    )

    # Log prompt size for debugging
    logger.info(f"Prompt size: {len(prompt)} chars (~{len(prompt)//4} tokens estimated)")
    logger.info(f"Retrieved {len(chunks)} chunks for context")

    answer = generate_answer(prompt)

    # Debug: Log raw LLM output
    logger.info(f"Raw LLM output: {len(answer)} chars")
    logger.info(f"Output ends with: ...{answer[-100:]}" if len(answer) > 100 else f"Full output: {answer}")

    # Add citations
    answer = add_citations_unified(answer, chunks)
    
    # Calculate confidence based on scores (handle both similarity and final_score)
    scores = []
    for chunk in chunks:
        # Use final_score from reranker if available, else rrf_score, else similarity
        if "final_score" in chunk:
            scores.append(chunk["final_score"])
        elif "rrf_score" in chunk:
            scores.append(chunk["rrf_score"])
        elif "similarity" in chunk:
            scores.append(chunk["similarity"])
        else:
            scores.append(0.5)  # Default if no score available

    avg_score = sum(scores) / len(scores) if scores else 0.5
    confidence = min(0.95, max(0.1, avg_score))

    # Ensure confidence is a Python float, not numpy float32
    confidence = float(confidence)

    return answer, confidence


def format_context_unified(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into context string.
    
    Args:
        chunks: List of chunks from unified database
        
    Returns:
        Formatted context string
    """
    if not chunks:
        return "No relevant context found."
    
    formatted_sections = []
    
    for i, chunk in enumerate(chunks, 1):
        content = chunk["content"].strip()
        source = chunk["display_name"] or chunk["filename"]
        chunk_index = chunk["chunk_index"]

        # Handle different score field names from parallel_hybrid_search
        if "similarity" in chunk:
            score = chunk["similarity"]
            score_label = "Relevance"
        elif "final_score" in chunk:
            score = chunk["final_score"]
            score_label = "Final Score"
        elif "rrf_score" in chunk:
            score = chunk["rrf_score"]
            score_label = "RRF Score"
        else:
            score = 0.5
            score_label = "Score"

        section_text = f"""
**Context Section {i}:**
- Source: {source}
- Chunk: #{chunk_index}
- {score_label}: {score:.3f}
- Content: {content}
"""
        formatted_sections.append(section_text)
    
    return "\n".join(formatted_sections)


def add_citations_unified(answer: str, chunks: List[Dict]) -> str:
    """
    Add properly formatted citations to the answer.
    
    Args:
        answer: Generated answer
        chunks: Retrieved chunks from database
        
    Returns:
        Answer with citations
    """
    if not chunks or "I am sorry" in answer:
        return answer
    
    # Group citations by document
    citations_by_doc = {}
    
    for chunk in chunks:
        source = chunk["display_name"] or chunk["filename"]
        chunk_index = chunk["chunk_index"]

        # Handle different score field names from parallel_hybrid_search
        if "similarity" in chunk:
            score = chunk["similarity"]
        elif "final_score" in chunk:
            score = chunk["final_score"]
        elif "rrf_score" in chunk:
            score = chunk["rrf_score"]
        else:
            score = 0.5

        clean_source = source.replace('.pdf', '').replace('_', ' ')

        if clean_source not in citations_by_doc:
            citations_by_doc[clean_source] = []

        citations_by_doc[clean_source].append({
            'chunk': chunk_index,
            'similarity': score
        })
    
    # Format citations
    if citations_by_doc:
        citation_lines = []
        for doc, chunk_info in citations_by_doc.items():
            # Sort by similarity (highest first)
            chunk_info.sort(key=lambda x: x['similarity'], reverse=True)
            
            if len(chunk_info) == 1:
                chunk_str = f"Chunk #{chunk_info[0]['chunk']}"
            else:
                chunks_str = ', '.join([f"#{c['chunk']}" for c in chunk_info[:3]])  # Show top 3
                if len(chunk_info) > 3:
                    chunks_str += f" (+{len(chunk_info)-3} more)"
                chunk_str = f"Chunks {chunks_str}"
            
            citation_lines.append(f"{doc} | {chunk_str}")
        
        answer += f"\n\n**SOURCE:**\n" + "\n".join(citation_lines)
    
    return answer


