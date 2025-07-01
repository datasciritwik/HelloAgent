from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Conversation:
    id: Optional[str] = None
    session_id: str = ""
    user_message: str = ""
    agent_response: str = ""
    node_type: str = ""  # 'greeting' or 'facts'
    llm_provider: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()