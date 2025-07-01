from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel
import requests
from config.settings import settings

class WeatherInput(BaseModel):
    city: str

class WeatherTool(BaseTool):
    name :str = "get_weather"
    description:str = "Get current weather information for a city"
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, city: str) -> str:
        if not settings.WEATHER_API_KEY:
            return f"Weather service not configured. Please provide a weather API key."
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": city,
                "appid": settings.WEATHER_API_KEY,
                "units": "metric"
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                temp = data["main"]["temp"]
                description = data["weather"][0]["description"]
                humidity = data["main"]["humidity"]
                return f"Weather in {city}: {temp}°C, {description}, Humidity: {humidity}%"
            else:
                return f"Could not get weather for {city}. Error: {data.get('message', 'Unknown error')}"
        except Exception as e:
            return f"Error getting weather: {str(e)}"
    
    async def _arun(self, city: str) -> str:
        return self._run(city)