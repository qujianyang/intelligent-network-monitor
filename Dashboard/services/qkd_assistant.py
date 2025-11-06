# Dashboard/services/qkd_assistant.py

"""
Legacy wrapper for QKD Assistant - redirects to multi-index implementation
========================================================================
This module is a thin wrapper that maintains backward compatibility for:
  - streaming_response.py (uses get_qkd_answer_with_streaming)
  - visual_search.py (uses get_qkd_answer)

The actual implementation is in qkd_assistant_multi_index.py which uses
MySQL-based vector storage for better performance and scalability.

Entry points:
  - get_qkd_answer(...)
  - get_qkd_answer_with_streaming(...)
"""

import logging
from typing import Tuple
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

def get_qkd_answer_with_streaming(
    question: str,
    top_k: int = 3,
    progress_callback=None,
    document_filter: str = 'all'
) -> Tuple[str, float]:
    """
    Wrapper that redirects to multi-index implementation for streaming support.

    Args:
        question (str): User inquiry about Quantum Key Distribution.
        top_k (int): How many top-similarity chunks to retrieve.
        progress_callback (callable): Optional callback for progress updates.
        document_filter (str): Document to filter by ('all' for no filter).

    Returns:
        Tuple[str, float]: (LLM-generated response, confidence score in [0.0, 1.0])
    """
    try:
        # Import the multi-index implementation
        from Dashboard.services.qkd_assistant_multi_index import get_qkd_answer_multi_index

        # Log the redirection
        logger.info(f"Redirecting to multi-index implementation for query: {question[:50]}...")

        # Call multi-index version
        answer, confidence = get_qkd_answer_multi_index(
            question=question,
            document_filter=document_filter if document_filter != 'all' else None,
            top_k=top_k
        )

        # Add progress callback support if provided
        if progress_callback:
            progress_callback("complete", 100, "Response generated successfully")

        return answer, confidence

    except ImportError as e:
        # If multi-index module is not available, return error
        logger.error(f"Multi-index module not available: {e}")
        error_msg = ("The QKD Assistant system is not properly configured. "
                    "Please ensure qkd_assistant_multi_index.py is available.")
        return error_msg, 0.1

    except Exception as e:
        # Generic error handling
        logger.error(f"Multi-index processing failed: {e}")
        error_msg = ("I apologize, but I encountered an error while processing your request. "
                    "Please try again or contact support if the issue persists.")
        return error_msg, 0.1


def get_qkd_answer(question: str, top_k: int = 3) -> Tuple[str, float]:
    """
    Simple wrapper for backward compatibility.
    Used by visual_search.py for image-based queries.

    Args:
        question (str): User inquiry about Quantum Key Distribution.
        top_k (int): How many top-similarity chunks to retrieve.

    Returns:
        Tuple[str, float]: (LLM-generated response, confidence score in [0.0, 1.0])
    """
    # Just call the streaming version without progress callback
    return get_qkd_answer_with_streaming(
        question=question,
        top_k=top_k,
        progress_callback=None,
        document_filter='all'
    )