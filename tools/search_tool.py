from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel
import requests
from config.settings import settings

class SearchInput(BaseModel):
    query: str

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the internet for information"
    args_schema: Type[BaseModel] = SearchInput
    
    def _run(self, query: str) -> str:
        if not settings.SEARCH_API_KEY:
            return "Web search not configured. Please provide a search API key."
        
        try:
            # Using SerpAPI as example - you can replace with your preferred search API
            url = "https://serpapi.com/search.json"
            params = {
                "q": query,
                "api_key": settings.SEARCH_API_KEY,
                "engine": "google",
                "num": 3
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            if "organic_results" in data:
                results = []
                for result in data["organic_results"][:3]:
                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    results.append(f"Title: {title}\nSummary: {snippet}")
                return "\n\n".join(results)
            else:
                return f"No search results found for: {query}"
        except Exception as e:
            return f"Error performing search: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)