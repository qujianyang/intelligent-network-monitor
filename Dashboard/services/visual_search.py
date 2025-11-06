"""
Visual Search Enhancement for QKD Documentation System
=======================================================
KISS-compliant vision-based query enhancement using local LLaVA model.

This module converts images to natural language for RAG queries:
1. Get technical description from LLaVA (diagrams, photos, schematics)
2. Combine description with user question (no keyword extraction)
3. Feed to existing parallel hybrid search (BM25 + FAISS)

BM25 filters stop words naturally. Semantic search prefers natural language.
No fake parsing. No keyword soup. Just honest, simple description.
"""

import requests
import base64
import logging
import tempfile
import os
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Load configuration
try:
    from config import LOCAL_LLM_API
    api_url = LOCAL_LLM_API
except ImportError:
    api_url = "http://localhost:11434"

# Default vision configuration
VISION_CONFIG = {
    "model_name": "llava:7b",
    "api_url": api_url,
    "timeout_seconds": 120,
    "max_retries": 2,
    "confidence_threshold": 0.6
}


class VisualSearchProcessor:
    """
    Simplified vision processor for equipment identification and query enhancement.
    Uses local LLaVA model via Ollama to analyze equipment images.
    """

    def __init__(self, vision_config: Dict = None):
        """Initialize vision processor with LLaVA."""
        self.config = vision_config or VISION_CONFIG
        self.api_url = self.config.get("api_url", "http://localhost:11434")
        self.model_name = self.config.get("model_name", "llava:7b")
        self.timeout = self.config.get("timeout_seconds", 120)
        self.max_retries = self.config.get("max_retries", 2)

        logger.info(f"VisualSearchProcessor initialized with {self.model_name}")

    def analyze_equipment_image(self, image_path: str, user_question: str = "") -> Dict[str, Any]:
        """
        Get technical description from image using LLaVA.

        Args:
            image_path: Path to image file
            user_question: Optional user question for context

        Returns:
            Dict with description and status
        """
        try:
            prompt = self._create_analysis_prompt(user_question)
            response = self._call_llava_api(image_path, prompt)

            if response.get("success", False):
                return {
                    "success": True,
                    "description": response["response"],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                logger.warning(f"Vision analysis failed: {response.get('error', 'Unknown')}")
                return self._fallback_response(user_question)

        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return self._fallback_response(user_question)

    def _create_analysis_prompt(self, user_question: str = "") -> str:
        """
        Create simple prompt to extract technical keywords from image.
        Works for diagrams, photos, schematics - any technical image.
        """
        prompt = "Identify the equipment in this image. Include: device name, model number, manufacturer, type (e.g., QKD transmitter, receiver), and any visible text or labels. Be specific."

        if user_question:
            prompt += f"\n\nUser will ask: {user_question}\nInclude relevant details for this question."

        return prompt

    def _call_llava_api(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Call local LLaVA API via Ollama."""
        try:
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Prepare API request
            api_url = f"{self.api_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent analysis
                    "num_predict": 500
                }
            }

            logger.info(f"Calling LLaVA: {api_url} (model: {self.model_name})")

            # Make API call
            response = requests.post(
                api_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"LLaVA analysis completed successfully")
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "model": result.get("model", self.model_name)
                }
            else:
                logger.error(f"LLaVA API error: {response.status_code}")
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}"
                }

        except Exception as e:
            logger.error(f"LLaVA API call failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _fallback_response(self, user_question: str = "") -> Dict[str, Any]:
        """Fallback when vision analysis fails."""
        return {
            "success": False,
            "description": user_question if user_question else "technical equipment",
            "timestamp": datetime.now().isoformat(),
            "error": "Vision analysis unavailable"
        }


def enhance_query_with_vision(vision_result: Dict[str, Any], user_question: str) -> str:
    """
    Combine LLaVA description with user question. Simple and effective.

    No keyword extraction - BM25 already filters stop words, and semantic
    search works better with natural language.

    Args:
        vision_result: Output from analyze_equipment_image()
        user_question: User's original question

    Returns:
        Enhanced query string for RAG retrieval
    """
    description = vision_result.get("description", "").strip()

    # Take first 500 chars of description (preserve device names and model numbers)
    truncated = description[:500].strip() if description else ""

    # Combine with user question
    if truncated and user_question:
        enhanced = f"{truncated} {user_question}".strip()
    elif truncated:
        enhanced = truncated
    else:
        enhanced = user_question if user_question else "technical equipment documentation"

    logger.info(f"Enhanced query: '{enhanced[:100]}...'")
    return enhanced


def process_visual_query(image_file, user_question: str = "") -> Dict[str, Any]:
    """
    Main entry point for visual search processing.

    This function:
    1. Analyzes the image with LLaVA
    2. Enhances the query with visual context
    3. Retrieves relevant documentation via parallel hybrid search
    4. Generates answer with LLM

    Args:
        image_file: Uploaded image file (file-like object)
        user_question: User's question about the equipment

    Returns:
        Dict with answer, confidence, sources, and metadata
    """
    temp_path = None

    try:
        # Step 1: Save image temporarily
        image_file.seek(0)
        image_data = image_file.read()

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_file.write(image_data)
            temp_path = temp_file.name

        logger.info(f"Processing visual query with image: {temp_path}")

        # Step 2: Analyze image with LLaVA
        processor = VisualSearchProcessor()
        vision_result = processor.analyze_equipment_image(temp_path, user_question)

        if not vision_result.get("success", False):
            logger.warning("Vision analysis failed, using fallback")

        # Step 3: Enhance query with vision context
        enhanced_query = enhance_query_with_vision(vision_result, user_question)

        # Step 4: Retrieve documentation using existing parallel hybrid search
        try:
            from Dashboard.services.qkd_assistant_multi_index import get_qkd_answer_multi_index
            answer, rag_confidence = get_qkd_answer_multi_index(
                question=enhanced_query,
                document_filter=None,  # Search all documents
                top_k=5
            )
        except ImportError:
            # Fallback to older RAG system if multi_index not available
            logger.warning("Falling back to basic qkd_assistant")
            from Dashboard.services.qkd_assistant import get_qkd_answer
            answer, rag_confidence = get_qkd_answer(enhanced_query, top_k=5)

        # Step 5: Format response
        formatted_response = _format_visual_response(answer, user_question)

        return {
            "success": True,
            "response": formatted_response,
            "confidence": rag_confidence,
            "enhanced_query": enhanced_query,
            "vision_description": vision_result.get("description", "")[:500],  # First 500 chars for debugging
            "architecture": "visual_search_simplified",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Visual query processing error: {e}")
        return {
            "success": False,
            "error": f"Visual search processing failed: {str(e)}",
            "architecture": "visual_search_enhanced"
        }

    finally:
        # Cleanup temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.info(f"Cleaned up temp file: {temp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up temp file: {cleanup_error}")


def _format_visual_response(answer: str, user_question: str) -> str:
    """Format the final response with user question context."""
    if user_question:
        return f"**Question**: {user_question}\n\n{answer}"
    return answer
