"""
QKD Agent Implementation
=======================
LLM-powered agent with ReAct reasoning for dynamic tool selection.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# Use the new, correct Ollama integration
from langchain_ollama import OllamaLLM

# Setup logger early
logger = logging.getLogger(__name__)

from Dashboard.services.tools import create_qkd_tools
try:
    from Dashboard.services.tools_enhanced import create_enhanced_qkd_tools
    ENHANCED_TOOLS_AVAILABLE = True
    # Log after initialization, not during import
except ImportError:
    ENHANCED_TOOLS_AVAILABLE = False
    # Log after initialization, not during import
from Dashboard.services.llm_client import LOCAL_LLM_MODEL, LOCAL_LLM_API
from config import LLM_CONFIG

# ReAct Prompt Template - Optimized for instruction following
REACT_PROMPT_TEMPLATE = """You are an expert QKD Network Assistant. You have access to various tools to help answer questions.

You have access to the following tools:

{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Example for multiple alarms:
If you find alarms "xcvr-rx-los" and "xcvr-rx-power-lo", you must:
Action: qkd_knowledge_base
Action Input: xcvr-rx-los
[Get observation]
Action: qkd_knowledge_base
Action Input: xcvr-rx-power-lo
[Get observation]
Then provide Final Answer with BOTH solutions

RULES:
1. Tool names: Never add parentheses (use "qkd_knowledge_base" not "qkd_knowledge_base()")
2. No input: Use "None" as Action Input when tool needs no input
3. Search terms: Use EXACT terms from user's question as Action Input
4. Multiple items: If you find multiple alarms/errors, you MUST search knowledge base for EACH one individually before Final Answer
5. Documentation first: For "how to fix", "what should I do", or "recommended action", search knowledge base
6. Current status: For "current state", "is it working", or real-time metrics, use diagnostic tools
7. Don't make things up: Only provide information from tool observations, never generate solutions

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


class QKDAgent:
    """
    Agentic AI system for QKD network management.
    Uses ReAct reasoning to dynamically select and chain tools.
    """
    
    def __init__(self,
                 model_name: str = LOCAL_LLM_MODEL,
                 temperature: float = 0.1,
                 max_iterations: int = 10,  # Increased from 8 to allow complete multi-alarm searches
                 memory_window_size: int = 3):  # Reduced from 5 to save tokens for tool outputs
        """
        Initialize the QKD Agent.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: LLM temperature for response generation
            max_iterations: Maximum reasoning steps allowed
            memory_window_size: Number of conversation turns to remember
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.memory_window_size = memory_window_size
        
        # Initialize components
        self._llm = None
        self._tools = None
        self._agent_executor = None
        self._memory = None
        self._initialized = False
        self.reasoning_log = []
        
    def initialize(self) -> bool:
        """Initialize the agent components."""
        try:
            logger.info("Initializing QKD Agent...")
            
            # 1. Initialize LLM
            self._llm = OllamaLLM(
                model=self.model_name,
                base_url=LOCAL_LLM_API,
                temperature=self.temperature,
                num_ctx=LLM_CONFIG.get("context_window", 4096),
                timeout=LLM_CONFIG.get("timeout_seconds", 120),
                top_p=LLM_CONFIG.get("top_p", 0.9),
                top_k=LLM_CONFIG.get("top_k", 40)
            )
            
            # 2. Load tools (use enhanced tools if available)
            if ENHANCED_TOOLS_AVAILABLE:
                self._tools = create_enhanced_qkd_tools()
                logger.info(f"Loaded {len(self._tools)} enhanced tools including fault detection")
            else:
                self._tools = create_qkd_tools()
                logger.info(f"Loaded {len(self._tools)} standard tools")
            
            # 3. Initialize memory - using native LangChain memory
            self._memory = ConversationBufferWindowMemory(
                k=self.memory_window_size,
                memory_key="chat_history",
                return_messages=False
            )
            
            # 4. Create ReAct agent
            prompt = PromptTemplate(
                template=REACT_PROMPT_TEMPLATE,
                input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
            )
            
            agent = create_react_agent(
                llm=self._llm,
                tools=self._tools,
                prompt=prompt
            )
            
            # 5. Create agent executor
            self._agent_executor = AgentExecutor(
                agent=agent,
                tools=self._tools,
                memory=self._memory,
                verbose=True,  # Enable detailed logging
                max_iterations=self.max_iterations,
                handle_parsing_errors=True,
                return_intermediate_steps=True,
                max_execution_time=120  # 120 second timeout for ML operations
            )
            
            self._initialized = True
            logger.info("QKD Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            return False
    
    def process_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a user query using ReAct reasoning with LangChain memory.

        Args:
            query: User's question or request

        Returns:
            Tuple of (response, metadata)
        """
        if not self._initialized:
            if not self.initialize():
                return "Agent initialization failed. Please check the system.", {"error": True}

        try:
            start_time = datetime.now()

            # Clean up the query
            query = query.strip()

            # Execute agent reasoning - memory context is automatically included by LangChain
            logger.info(f"Processing query: {query[:100]}...")
            result = self._agent_executor.invoke({"input": query})

            # Extract response and reasoning steps
            response = result.get("output", "No response generated")
            intermediate_steps = result.get("intermediate_steps", [])

            # Build metadata
            tools_used = []
            for step in intermediate_steps:
                if hasattr(step[0], 'tool'):
                    tools_used.append(step[0].tool)

            metadata = {
                "reasoning_steps": len(intermediate_steps),
                "tools_used": tools_used,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "model": self.model_name,
                "timestamp": start_time.isoformat(),
                "intermediate_steps": intermediate_steps
            }

            logger.info(f"Query processed successfully in {metadata['processing_time']:.2f}s")
            return response, metadata

        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            return f"Error processing request: {str(e)}", {"error": True, "exception": str(e)}

    # Simplified: Using process_query for all queries now
    # The LangChain memory automatically maintains context
    def process_query_with_investigation(self, query: str, context_id: str = None,
                                       network_state: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Simplified method that redirects to process_query.
        Kept for backward compatibility with existing API calls.

        The LangChain ConversationBufferWindowMemory automatically handles context,
        so we don't need the complex investigation tracking system.
        """
        # Simply use the standard process_query which includes memory context
        return self.process_query(query)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        if self._memory and hasattr(self._memory, 'chat_memory'):
            messages = []
            for msg in self._memory.chat_memory.messages:
                messages.append({
                    "role": getattr(msg, "type", "unknown"),
                    "content": getattr(msg, "content", "")
                })
            return messages
        return []
    
    def clear_memory(self):
        """Clear the conversation memory."""
        if self._memory:
            self._memory.clear()
            logger.info("Agent memory cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "initialized": self._initialized,
            "model": self.model_name,
            "tools_available": len(self._tools) if self._tools else 0,
            "memory_size": len(self.get_conversation_history()),
            "max_iterations": self.max_iterations
        }
    
    def extract_reasoning_steps(self, intermediate_steps: List) -> List[Dict[str, Any]]:
        """Extract clean reasoning steps for display."""
        clean_steps = []
        for i, (action, observation) in enumerate(intermediate_steps):
            step = {
                "number": i + 1,
                "thought": "",
                "action": "",
                "action_input": "",
                "observation": str(observation)[:200]
            }
            
            # Extract thought from log
            if hasattr(action, 'log'):
                log_text = action.log
                thought_match = re.search(r'Thought:\s*(.+?)(?:Action:|$)', log_text, re.DOTALL)
                if thought_match:
                    step["thought"] = thought_match.group(1).strip()
            
            # Extract action and input
            if hasattr(action, 'tool'):
                step["action"] = action.tool
            if hasattr(action, 'tool_input'):
                step["action_input"] = json.dumps(action.tool_input) if isinstance(action.tool_input, dict) else str(action.tool_input)
            
            clean_steps.append(step)
        
        return clean_steps
    



# Global agent instance
_global_agent = None

def get_agent() -> QKDAgent:
    """Get or create the global QKD agent instance."""
    global _global_agent
    if _global_agent is None:
        _global_agent = QKDAgent()
    return _global_agent

def reset_agent():
    """Reset the global agent instance - useful for testing or after configuration changes."""
    global _global_agent
    _global_agent = None
