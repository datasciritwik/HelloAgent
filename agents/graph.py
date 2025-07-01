from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from agents.nodes import AgentState, GreetingNode, FactsNode
from database.storage import ConversationStorage
from database.models import Conversation
import uuid

class AIAgentGraph:
    def __init__(self, llm_provider: str = "openai", session_id: str = None):
        self.llm_provider = llm_provider
        self.session_id = session_id or str(uuid.uuid4())
        self.storage = ConversationStorage()
        
        # Initialize nodes
        self.greeting_node = GreetingNode(llm_provider)
        self.facts_node = FactsNode(llm_provider)
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Define the graph state
        graph = StateGraph(dict)
        
        # Add nodes
        graph.add_node("greeting", self._greeting_node_wrapper)
        graph.add_node("facts", self._facts_node_wrapper)
        
        # Add conditional routing
        graph.add_conditional_edges(
            START,
            self._route_input,
            {
                "greeting": "greeting",
                "facts": "facts"
            }
        )
        
        # Connect nodes to end
        graph.add_edge("greeting", END)
        graph.add_edge("facts", END)
        
        return graph.compile()
    
    def _route_input(self, state: Dict[str, Any]) -> Literal["greeting", "facts"]:
        """Route input to appropriate node based on content"""
        user_input = state.get("user_input", "").lower()
        
        # Keywords that suggest greeting
        greeting_keywords = [
            "hello", "hi", "hey", "good morning", "good afternoon", 
            "good evening", "greetings", "time", "date", "weather"
        ]
        
        # Keywords that suggest facts
        fact_keywords = [
            "fact", "facts", "tell me about", "what is", "explain", 
            "information", "learn", "know", "research", "find"
        ]
        
        # Check for greeting keywords
        if any(keyword in user_input for keyword in greeting_keywords):
            return "greeting"
        
        # Check for fact keywords
        if any(keyword in user_input for keyword in fact_keywords):
            return "facts"
        
        # Default to greeting for ambiguous inputs
        return "greeting"
    
    def _greeting_node_wrapper(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for greeting node"""
        agent_state = AgentState()
        agent_state.user_input = state["user_input"]
        agent_state.llm_provider = self.llm_provider
        agent_state.session_id = self.session_id
        
        result = self.greeting_node.process(agent_state)
        
        # Save to database
        conversation = Conversation(
            session_id=self.session_id,
            user_message=state["user_input"],
            agent_response=result["response"],
            node_type="greeting",
            llm_provider=self.llm_provider
        )
        self.storage.save_conversation(conversation)
        
        return {
            **state,
            "response": result["response"],
            "node_type": "greeting",
            "success": result["success"]
        }
    
    def _facts_node_wrapper(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for facts node"""
        agent_state = AgentState()
        agent_state.user_input = state["user_input"]
        agent_state.llm_provider = self.llm_provider
        agent_state.session_id = self.session_id
        
        result = self.facts_node.process(agent_state)
        
        # Save to database
        conversation = Conversation(
            session_id=self.session_id,
            user_message=state["user_input"],
            agent_response=result["response"],
            node_type="facts",
            llm_provider=self.llm_provider
        )
        self.storage.save_conversation(conversation)
        
        return {
            **state,
            "response": result["response"],
            "node_type": "facts",
            "success": result["success"]
        }
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input through the graph"""
        initial_state = {
            "user_input": user_input,
            "session_id": self.session_id
        }
        
        result = self.graph.invoke(initial_state)
        return result