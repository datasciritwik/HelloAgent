# config.py
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration for the agent system."""
    openai_api_key: str
    model_name: str = "gpt-4"
    temperature: float = 0.0
    max_retries: int = 3
    timeout: int = 30
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load configuration from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        return cls(
            openai_api_key=api_key,
            model_name=os.getenv("OPENAI_MODEL", "gpt-4"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


# exceptions.py
class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass


class ToolExecutionError(AgentError):
    """Error during tool execution."""
    pass


class StateValidationError(AgentError):
    """Error during state validation."""
    pass


class ConfigurationError(AgentError):
    """Error in agent configuration."""
    pass


# logger.py
import logging
import sys
from typing import Optional


class AgentLogger:
    """Centralized logging for the agent system."""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs)


# tools.py
import json
import requests
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Standardized tool result."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseTool(ABC):
    """Base class for all tools."""
    
    def __init__(self, logger: AgentLogger):
        self.logger = logger
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool."""
        pass
    
    def handle_error(self, error: Exception) -> ToolResult:
        """Handle tool execution errors."""
        self.logger.error(f"Tool execution failed: {str(error)}")
        return ToolResult(success=False, error=str(error))


class WebSearchTool(BaseTool):
    """Web search tool with proper error handling."""
    
    def __init__(self, logger: AgentLogger, api_key: Optional[str] = None):
        super().__init__(logger)
        self.api_key = api_key or os.getenv("SEARCH_API_KEY")
    
    def execute(self, query: str, max_results: int = 5) -> ToolResult:
        """Execute web search."""
        try:
            self.logger.info(f"Searching for: {query}")
            
            if not self.api_key:
                # Fallback to mock search for demo
                return ToolResult(
                    success=True,
                    data=f"Mock search results for: {query}",
                    metadata={"source": "mock", "query": query}
                )
            
            # Implement actual search API call here
            # This is a placeholder for real implementation
            response = {
                "results": [
                    {"title": f"Result for {query}", "snippet": "Sample snippet"}
                ]
            }
            
            return ToolResult(
                success=True,
                data=response,
                metadata={"query": query, "results_count": len(response["results"])}
            )
            
        except Exception as e:
            return self.handle_error(e)


class CalculatorTool(BaseTool):
    """Safe calculator tool."""
    
    ALLOWED_OPERATORS = {'+', '-', '*', '/', '(', ')', '.', ' '}
    
    def execute(self, expression: str) -> ToolResult:
        """Execute mathematical calculation."""
        try:
            self.logger.info(f"Calculating: {expression}")
            
            # Sanitize expression
            if not self._is_safe_expression(expression):
                return ToolResult(
                    success=False,
                    error="Invalid expression: contains unsafe characters"
                )
            
            result = eval(expression)
            
            return ToolResult(
                success=True,
                data=result,
                metadata={"expression": expression}
            )
            
        except Exception as e:
            return self.handle_error(e)
    
    def _is_safe_expression(self, expression: str) -> bool:
        """Check if expression is safe to evaluate."""
        allowed_chars = self.ALLOWED_OPERATORS | set('0123456789')
        return all(c in allowed_chars for c in expression)


class FileManagerTool(BaseTool):
    """File management tool with security checks."""
    
    def __init__(self, logger: AgentLogger, base_path: str = "./agent_files"):
        super().__init__(logger)
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def execute(self, operation: str, filename: str, content: str = "") -> ToolResult:
        """Execute file operations."""
        try:
            filepath = os.path.join(self.base_path, filename)
            
            # Security check
            if not filepath.startswith(self.base_path):
                return ToolResult(
                    success=False,
                    error="Invalid file path: outside allowed directory"
                )
            
            if operation == "write":
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return ToolResult(
                    success=True,
                    data=f"File {filename} written successfully",
                    metadata={"operation": "write", "filename": filename}
                )
            
            elif operation == "read":
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                return ToolResult(
                    success=True,
                    data=content,
                    metadata={"operation": "read", "filename": filename}
                )
            
            else:
                return ToolResult(
                    success=False,
                    error=f"Unsupported operation: {operation}"
                )
                
        except Exception as e:
            return self.handle_error(e)


# Create tool instances as LangChain tools
def create_langchain_tools(logger: AgentLogger) -> List:
    """Create LangChain compatible tools."""
    
    web_search = WebSearchTool(logger)
    calculator = CalculatorTool(logger)
    file_manager = FileManagerTool(logger)
    
    @tool
    def search_web(query: str) -> str:
        """Search the web for information."""
        result = web_search.execute(query)
        return json.dumps(result.dict())
    
    @tool
    def calculate(expression: str) -> str:
        """Calculate mathematical expressions safely."""
        result = calculator.execute(expression)
        return json.dumps(result.dict())
    
    @tool
    def manage_file(operation: str, filename: str, content: str = "") -> str:
        """Manage files (read/write operations)."""
        result = file_manager.execute(operation, filename, content)
        return json.dumps(result.dict())
    
    return [search_web, calculate, manage_file]


# state.py
from typing import Any, Dict, List, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Extended state for the agent with additional fields."""
    
    # Task management
    task_list: List[str] = []
    current_task: Optional[str] = None
    completed_tasks: List[str] = []
    
    # Execution context
    execution_count: int = 0
    max_iterations: int = 10
    
    # Metadata
    session_id: str = ""
    user_id: Optional[str] = None
    start_time: Optional[float] = None
    
    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None


class StateManager:
    """Manages agent state validation and transitions."""
    
    def __init__(self, logger: AgentLogger):
        self.logger = logger
    
    def validate_state(self, state: AgentState) -> bool:
        """Validate state consistency."""
        try:
            # Check iteration limits
            if state.execution_count >= state.max_iterations:
                self.logger.warning("Max iterations reached")
                return False
            
            # Check error limits
            if state.error_count >= 3:
                self.logger.error("Too many errors, stopping execution")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"State validation error: {e}")
            return False
    
    def update_state(self, state: AgentState, **updates) -> AgentState:
        """Update state with validation."""
        try:
            for key, value in updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)
                else:
                    self.logger.warning(f"Unknown state key: {key}")
            
            return state
            
        except Exception as e:
            self.logger.error(f"State update error: {e}")
            raise StateValidationError(f"Failed to update state: {e}")


# agents.py
import time
import uuid
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, config: AgentConfig, logger: AgentLogger):
        self.config = config
        self.logger = logger
        self.state_manager = StateManager(logger)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.openai_api_key,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        
        # Initialize tools
        self.tools = create_langchain_tools(logger)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)
    
    def create_system_prompt(self) -> str:
        """Create system prompt for the agent."""
        return """You are a helpful AI assistant with access to tools.
        
        Guidelines:
        - Use tools when necessary to help the user
        - Provide clear and concise responses
        - Handle errors gracefully
        - Be transparent about limitations
        - Always prioritize user safety and data privacy
        """
    
    def handle_error(self, state: AgentState, error: Exception) -> AgentState:
        """Handle agent errors."""
        self.logger.error(f"Agent error: {error}")
        
        error_message = AIMessage(
            content=f"I encountered an error: {str(error)}. Let me try a different approach."
        )
        
        return self.state_manager.update_state(
            state,
            messages=state["messages"] + [error_message],
            error_count=state.error_count + 1,
            last_error=str(error)
        )


class MainAgent(BaseAgent):
    """Main reasoning agent."""
    
    def __init__(self, config: AgentConfig, logger: AgentLogger):
        super().__init__(config, logger)
        self.system_prompt = self.create_system_prompt()
    
    def process(self, state: AgentState) -> AgentState:
        """Process user input and generate response."""
        try:
            # Validate state
            if not self.state_manager.validate_state(state):
                raise StateValidationError("Invalid state")
            
            # Prepare messages
            messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
            
            # Generate response
            self.logger.info("Generating agent response")
            response = self.llm_with_tools.invoke(messages)
            
            # Update state
            updated_state = self.state_manager.update_state(
                state,
                messages=state["messages"] + [response],
                execution_count=state.execution_count + 1
            )
            
            return updated_state
            
        except Exception as e:
            return self.handle_error(state, e)


class AgentWorkflow:
    """Main workflow orchestrator."""
    
    def __init__(self, config: AgentConfig, logger: AgentLogger):
        self.config = config
        self.logger = logger
        self.agent = MainAgent(config, logger)
        self.memory = MemorySaver()
        
        # Build workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def _build_workflow(self) -> StateGraph:
        """Build the agent workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self.agent.process)
        workflow.add_node("tools", self.agent.tool_node)
        
        # Add edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "end": "__end__"
            }
        )
        workflow.add_edge("tools", "agent")
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        return workflow
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if workflow should continue."""
        try:
            messages = state["messages"]
            if not messages:
                return "end"
            
            last_message = messages[-1]
            
            # Check for tool calls
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            
            # Check state validity
            if not self.agent.state_manager.validate_state(state):
                return "end"
            
            return "end"
            
        except Exception as e:
            self.logger.error(f"Decision error: {e}")
            return "end"
    
    def run(self, query: str, session_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
        """Run the agent workflow."""
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Create initial state
            initial_state = AgentState(
                messages=[HumanMessage(content=query)],
                session_id=session_id,
                user_id=user_id,
                start_time=time.time(),
                execution_count=0,
                error_count=0
            )
            
            # Configure execution
            config = {"configurable": {"thread_id": session_id}}
            
            # Run workflow
            self.logger.info(f"Running workflow for session: {session_id}")
            result = self.app.invoke(initial_state, config=config)
            
            # Extract response
            if result["messages"]:
                response = result["messages"][-1].content
                self.logger.info(f"Workflow completed successfully")
                return response
            else:
                return "I apologize, but I couldn't generate a response."
                
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return f"I encountered an error: {str(e)}"


# main.py
def main():
    """Main entry point for the agent system."""
    try:
        # Load configuration
        config = AgentConfig.from_env()
        
        # Initialize logger
        logger = AgentLogger("agent_system", config.log_level)
        
        # Create agent workflow
        workflow = AgentWorkflow(config, logger)
        
        # Interactive loop
        print("Agent system initialized. Type 'quit' to exit.")
        
        while True:
            user_input = input("\nUser: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Process user input
            response = workflow.run(user_input)
            print(f"Agent: {response}")
            
    except KeyboardInterrupt:
        print("\nSystem interrupted by user.")
    except Exception as e:
        print(f"System error: {e}")


if __name__ == "__main__":
    main()


# requirements.txt content (add this to a separate file)
"""
langchain-core>=0.1.0
langchain-openai>=0.1.0
langgraph>=0.0.40
pydantic>=2.0.0
requests>=2.31.0
python-dotenv>=1.0.0
"""

# .env.example content (add this to a separate file)
"""
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.0
MAX_RETRIES=3
REQUEST_TIMEOUT=30
LOG_LEVEL=INFO
SEARCH_API_KEY=your_search_api_key_here
"""