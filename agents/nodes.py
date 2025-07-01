from typing import Dict, Any
from langchain.schema import HumanMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from tools.datetime_tool import DateTimeTool
from tools.weather_tool import WeatherTool
from tools.search_tool import WebSearchTool
from llm.factory import LLMFactory

class AgentState:
    def __init__(self):
        self.messages = []
        self.user_input = ""
        self.agent_response = ""
        self.node_type = ""
        self.llm_provider = "openai"
        self.session_id = ""

class GreetingNode:
    def __init__(self, llm_provider: str = "openai"):
        self.llm_provider = llm_provider
        self.llm = LLMFactory.create_llm(llm_provider)
        self.tools = [DateTimeTool(), WeatherTool()]
        
        # Create prompt template for greeting
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly AI assistant specialized in greetings. 
            Your job is to greet users warmly and provide helpful information using the available tools.
            You can check the current date/time and weather information.
            Keep your responses friendly, concise, and helpful.
            
            Available tools:
            - get_current_datetime: Get current date and time
            - get_weather: Get weather for a specific city
            """),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Create agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def process(self, state: AgentState) -> Dict[str, Any]:
        """Process greeting requests"""
        try:
            result = self.agent_executor.invoke({
                "input": state.user_input
            })
            
            state.agent_response = result["output"]
            state.node_type = "greeting"
            
            return {
                "response": state.agent_response,
                "node_type": state.node_type,
                "success": True
            }
        except Exception as e:
            state.agent_response = f"Sorry, I encountered an error: {str(e)}"
            state.node_type = "greeting"
            return {
                "response": state.agent_response,
                "node_type": state.node_type,
                "success": False,
                "error": str(e)
            }

class FactsNode:
    def __init__(self, llm_provider: str = "openai"):
        self.llm_provider = llm_provider
        self.llm = LLMFactory.create_llm(llm_provider)
        self.tools = [WebSearchTool()]
        
        # Create prompt template for facts
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable AI assistant specialized in providing interesting facts.
            Your job is to find and share fascinating, accurate facts based on user queries.
            Use the web search tool to find current and relevant information.
            Present facts in an engaging and educational manner.
            Always cite your sources when possible.
            
            Available tools:
            - web_search: Search the internet for current information
            """),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Create agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def process(self, state: AgentState) -> Dict[str, Any]:
        """Process fact requests"""
        try:
            result = self.agent_executor.invoke({
                "input": state.user_input
            })
            
            state.agent_response = result["output"]
            state.node_type = "facts"
            
            return {
                "response": state.agent_response,
                "node_type": state.node_type,
                "success": True
            }
        except Exception as e:
            state.agent_response = f"Sorry, I couldn't find facts about that topic: {str(e)}"
            state.node_type = "facts"
            return {
                "response": state.agent_response,
                "node_type": state.node_type,
                "success": False,
                "error": str(e)
            }