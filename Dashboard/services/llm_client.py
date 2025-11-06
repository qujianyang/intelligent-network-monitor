"""
LLM Client Module
================
Handles all direct communication with the Ollama Large Language Model.
This module isolates LLM-specific logic for easy switching between providers.

Features:
- Ollama API client with retry/backoff
- Health checking
- Error handling with custom exceptions
- Configurable timeouts and parameters
"""

import time
import logging
import requests
from typing import Optional

from config import (
    LOCAL_LLM_API,
    LOCAL_LLM_MODEL,
    LLM_CONFIG
)

# Setup logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 1.0

# Build base URL for Ollama API
BASE_API_URL = LOCAL_LLM_API.rstrip('/')
if not BASE_API_URL.endswith('/api'):
    BASE_API_URL += '/api'

# Note: LOCAL_LLM_MODEL and LOCAL_LLM_API are imported from config.py

class OllamaError(Exception):
    """Custom exception for Ollama API errors."""
    pass


def _post(route: str, payload: dict, retries: int = DEFAULT_RETRIES, backoff: float = DEFAULT_BACKOFF) -> dict:
    """Internal POST helper with retry/backoff to Ollama REST API."""
    url = BASE_API_URL + route
    for attempt in range(1, retries + 1):
        try:
            start = time.perf_counter()
            # Use configurable timeout from config.py
            timeout = LLM_CONFIG.get("timeout_seconds", 120)
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.perf_counter() - start
            logger.debug(f"POST {url} succeeded in {elapsed:.2f}s on attempt {attempt}")
            return data
        except requests.RequestException as exc:
            logger.warning(f"Attempt {attempt}/{retries} failed for {url}: {exc}")
            if attempt == retries:
                logger.error(f"All {retries} attempts failed for {url}")
                raise OllamaError(f"Ollama request failed after {retries} attempts: {exc}") from exc
            time.sleep(backoff * attempt)


def generate_answer(
    prompt: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    top_k: int = 40,
    max_retries: int = None,
) -> str:
    """Generate a completion from the local Ollama LLM."""
    if not LOCAL_LLM_MODEL:
        raise OllamaError("LOCAL_LLM_MODEL must be set in config.py")

    payload = {
        "model": LOCAL_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,  # Low for technical accuracy
            "top_p": 0.8,  # More focused sampling (reduced from 0.9)
            "top_k": 20,  # Limit vocabulary for consistency (reduced from 40)
            "repeat_penalty": 1.05,  # Prevent repetition without being too harsh
            "repeat_last_n": 128,  # Look back window for repetition check
            "num_ctx": 4096,  # Good context window for RAG
            # Don't set num_predict - let model stop naturally
            "stop": ["\n\n\n", "###", "**Query:**", "**Instructions:**"],  # Stop sequences to prevent prompt repetition
        },
    }
    start = time.perf_counter()
    data = _post(
        route="/generate",
        payload=payload,
        retries=max_retries or DEFAULT_RETRIES,
        backoff=DEFAULT_BACKOFF,
    )
    elapsed = time.perf_counter() - start
    response = data.get("response", "")
    logger.info(f"generate_answer() succeeded in {elapsed:.2f}s; returned {len(response)} chars")
    return response


def health_check() -> bool:
    """Quick ping to confirm Ollama daemon and model are reachable."""
    if not LOCAL_LLM_MODEL:
        return False
    payload = {
        "model": LOCAL_LLM_MODEL,
        "prompt": "",
        "stream": False,
        "options": {"temperature": 0.0, "top_p": 1.0, "top_k": 1},
    }
    try:
        _post(route="/generate", payload=payload, retries=1, backoff=0)
        return True
    except OllamaError:
        return False

 