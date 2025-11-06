"""
Dashboard/services/streaming_response.py

Real-time streaming response system for QKD Assistant with actual backend progress tracking.
This module provides Server-Sent Events (SSE) streaming with genuine progress updates
from the RAG pipeline, document retrieval, and LLM generation processes.
"""

import json
import time
import logging
from typing import Generator, Dict, Any, Callable, Optional
from threading import Lock, Thread
from queue import SimpleQueue, Empty
import uuid

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Thread-safe progress tracking for streaming responses."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()
        self.current_step = ""
        self.progress = 0
        self.total_steps = 0
        self.step_details = {}
        self.lock = Lock()
        self.completed = False
        self.error = None
        
    def update(self, step: str, progress: int, message: str, details: Dict[str, Any] = None):
        """Update progress with thread safety."""
        with self.lock:
            self.current_step = step
            self.progress = min(progress, 100)
            self.step_details[step] = {
                "message": message,
                "timestamp": time.time(),
                "details": details or {}
            }
            logger.info(f"Progress [{self.session_id}]: {step} - {progress}% - {message}")
    
    def complete(self, final_data: Dict[str, Any]):
        """Mark as completed with final results."""
        with self.lock:
            self.completed = True
            self.progress = 100
            self.step_details["complete"] = {
                "message": "Response ready",
                "timestamp": time.time(),
                "details": final_data
            }
    
    def set_error(self, error_message: str):
        """Mark as failed with error."""
        with self.lock:
            self.error = error_message
            self.completed = True

class StreamingResponse:
    """Manages streaming responses with real backend progress tracking."""
    
    # Global registry for active progress trackers
    _active_sessions = {}
    _sessions_lock = Lock()
    
    def __init__(self, query: str, mode: str, document_filter: str = 'all'):
        self.query = query
        self.mode = mode
        self.document_filter = document_filter
        self.session_id = str(uuid.uuid4())
        self.tracker = ProgressTracker(self.session_id)
        self._message_queue = SimpleQueue()
        self._worker_thread = None
        
        # Register session
        with StreamingResponse._sessions_lock:
            StreamingResponse._active_sessions[self.session_id] = self.tracker
    
    def get_progress_callback(self) -> Callable:
        """Get a callback function for backend components to report progress."""
        def progress_callback(step: str, progress: int, message: str, details: Dict[str, Any] = None):
            # Update tracker
            self.tracker.update(step, progress, message, details)
            
            # Queue the progress update for streaming
            progress_data = {
                "type": "status",
                "step": step,
                "message": message,
                "progress": progress,
                "session_id": self.session_id,
                "details": details or {}
            }
            
            # Send immediately to queue
            self._message_queue.put(self._format_sse_data(progress_data))
            
        return progress_callback
    
    def _do_work(self):
        """Background worker that processes the request and queues results."""
        try:
            # Get progress callback that queues updates
            progress_callback = self.get_progress_callback()
            
            # 🎯 MODE-BASED ROUTING: Route to different processing functions based on mode
            logger.info(f" Streaming worker processing mode '{self.mode}' for query: {self.query[:50]}...")
            
            if self.mode == 'chat':
                # 💬 CHAT MODE: Pure LLM without RAG
                logger.info(" Using pure LLM processing for chat mode")
                try:
                    # Import from app.py where the functions now live
                    from GUI.app import process_pure_llm_with_streaming
                    answer, confidence = process_pure_llm_with_streaming(
                        self.query,
                        progress_callback=progress_callback
                    )
                    architecture = "pure_llm"
                    
                except ImportError as e:
                    logger.error(f"Failed to import process_pure_llm_with_streaming: {e}")
                    # Fallback to basic conversational response
                    answer = f"I understand you're asking: {self.query}\n\nI'm a conversational AI assistant. How can I help you today?"
                    confidence = 0.7
                    architecture = "fallback_chat"
                    
            elif self.mode == 'agent':
                # 🤖 AGENT MODE: Use agent processing with tools
                logger.info(" Using agent processing for agent mode")
                try:
                    # Import from app.py where the functions now live
                    from GUI.app import process_agent_with_streaming
                    answer, confidence, meta = process_agent_with_streaming(
                        self.query,
                        progress_callback=progress_callback
                    )
                    architecture = "agent_mode"
                    
                except ImportError as e:
                    logger.error(f"Failed to import process_agent_with_streaming: {e}")
                    # Fallback to RAG processing
                    from Dashboard.services.qkd_assistant import get_qkd_answer_with_streaming
                    answer, confidence = get_qkd_answer_with_streaming(
                        self.query, 
                        top_k=5, 
                        progress_callback=progress_callback
                    )
                    architecture = "fallback_rag"
                    
            else:
                # 📚 SOP MODE (default): Use RAG processing
                logger.info(f" Using RAG processing for SOP mode (mode='{self.mode}')")
                from Dashboard.services.qkd_assistant import get_qkd_answer_with_streaming
                
                # Use higher top_k when document filtering to ensure we find chunks from the target document
                base_top_k = 5
                if self.document_filter and self.document_filter != 'all':
                    base_top_k = 10  # Higher base for document filtering
                    logger.info(f" Document filter active, using base_top_k={base_top_k}")
                
                answer, confidence = get_qkd_answer_with_streaming(
                    self.query, 
                    top_k=base_top_k, 
                    progress_callback=progress_callback,
                    document_filter=self.document_filter
                )
                architecture = "streaming_rag"
            
            # Final response
            processing_time = time.time() - self.tracker.start_time

            # Convert numpy types to Python native types for JSON serialization
            if hasattr(confidence, 'item'):  # Check if it's a numpy type
                confidence = float(confidence.item())
            else:
                confidence = float(confidence)

            final_data = {
                "response": answer,
                "confidence": confidence,
                "processing_time": processing_time,
                "architecture": architecture,
                "mode": self.mode,
                "session_id": self.session_id
            }
            # Optionally attach tools_used if available
            try:
                if 'meta' in locals() and isinstance(meta, dict) and meta.get('tools_used'):
                    final_data["tools_used"] = meta.get('tools_used', [])
            except Exception:
                pass
            
            logger.info(f" Streaming processing complete: mode={self.mode}, architecture={architecture}, confidence={confidence}")
            
            self.tracker.complete(final_data)
            
            # Queue final response
            self._message_queue.put(self._format_sse_data({
                "type": "response",
                "step": "complete",
                "message": "Response ready",
                "progress": 100,
                "data": final_data
            }))
            
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            self.tracker.set_error(error_msg)
            logger.error(f"Streaming error for session {self.session_id}: {e}")
            
            # Queue error response
            self._message_queue.put(self._format_sse_data({
                "type": "error",
                "step": "error",
                "message": error_msg,
                "progress": 0,
                "session_id": self.session_id
            }))

    def stream_response(self) -> Generator[str, None, None]:
        """Generate streaming response with real-time backend progress."""
        try:
            logger.info(f"Starting streaming response for session {self.session_id}")
            
            # Step 1: Send initial message
            yield self._format_sse_data({
                "type": "status",
                "step": "initialize",
                "message": "Initializing AI processing...",
                "progress": 5,
                "session_id": self.session_id
            })
            
            # Step 2: Start background worker
            self._worker_thread = Thread(target=self._do_work, daemon=True)
            self._worker_thread.start()
            
            # Step 3: Stream messages from queue in real-time
            timeout_count = 0
            max_timeouts = 360  # 180 seconds (3 minutes) with 0.5s intervals
            
            while not self.tracker.completed or not self._message_queue.empty():
                try:
                    # Get message from queue with timeout
                    message = self._message_queue.get(timeout=0.5)
                    yield message
                    timeout_count = 0  # Reset timeout counter on successful message
                    
                except Empty:
                    timeout_count += 1
                    if timeout_count >= max_timeouts:
                        # Timeout - send error and break
                        logger.warning(f"Streaming timeout for session {self.session_id}")
                        yield self._format_sse_data({
                            "type": "error",
                            "step": "timeout",
                            "message": "Request timed out. Please try again.",
                            "progress": 0,
                            "session_id": self.session_id
                        })
                        break
                    
                    # Send keepalive every 10 seconds
                    if timeout_count % 20 == 0:
                        yield self._format_sse_data({
                            "type": "keepalive",
                            "session_id": self.session_id
                        })
            
        except Exception as e:
            logger.error(f"Critical streaming error for session {self.session_id}: {e}")
            yield self._format_sse_data({
                "type": "error",
                "step": "critical_error",
                "message": f"System error: {str(e)}",
                "progress": 0,
                "session_id": self.session_id
            })
        
        finally:
            # Cleanup session
            with StreamingResponse._sessions_lock:
                if self.session_id in StreamingResponse._active_sessions:
                    del StreamingResponse._active_sessions[self.session_id]
            logger.info(f"Cleaned up session {self.session_id}")
    
    def _format_sse_data(self, data: Dict[str, Any]) -> str:
        """Format data for Server-Sent Events."""
        return f"data: {json.dumps(data)}\n\n"
    
    @classmethod
    def get_session_progress(cls, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress for a session (for polling fallback)."""
        with cls._sessions_lock:
            tracker = cls._active_sessions.get(session_id)
            if not tracker:
                return None
            
            with tracker.lock:
                return {
                    "session_id": session_id,
                    "current_step": tracker.current_step,
                    "progress": tracker.progress,
                    "completed": tracker.completed,
                    "error": tracker.error,
                    "processing_time": time.time() - tracker.start_time,
                    "step_details": dict(tracker.step_details)
                }




def get_active_sessions() -> Dict[str, Dict[str, Any]]:
    """Get all active streaming sessions (for debugging/monitoring)."""
    with StreamingResponse._sessions_lock:
        return {
            session_id: {
                "query": tracker.query if hasattr(tracker, 'query') else "unknown",
                "mode": tracker.mode if hasattr(tracker, 'mode') else "unknown", 
                "progress": tracker.progress,
                "current_step": tracker.current_step,
                "processing_time": time.time() - tracker.start_time
            }
            for session_id, tracker in StreamingResponse._active_sessions.items()
        }