from langchain.tools import BaseTool
from datetime import datetime
from typing import Type
from pydantic import BaseModel

class DateTimeInput(BaseModel):
    format: str = "%Y-%m-%d %H:%M:%S"

class DateTimeTool(BaseTool):
    name: str = "get_current_datetime"
    description: str = "Get the current date and time"
    args_schema: Type[BaseModel] = DateTimeInput
    
    def _run(self, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        return datetime.now().strftime(format)
    
    async def _arun(self, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        return self._run(format)