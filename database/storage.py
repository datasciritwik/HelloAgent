import duckdb
from typing import List
from database.models import Conversation
from config.settings import settings

class ConversationStorage:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.DB_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize the database and create tables"""
        conn = duckdb.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id VARCHAR PRIMARY KEY,
                session_id VARCHAR,
                user_message TEXT,
                agent_response TEXT,
                node_type VARCHAR,
                llm_provider VARCHAR,
                timestamp TIMESTAMP
            )
        """)
        conn.close()
    
    def save_conversation(self, conversation: Conversation) -> str:
        """Save a conversation to the database"""
        import uuid
        conn = duckdb.connect(self.db_path)
        conversation_id = conversation.id or str(uuid.uuid4())
        result = conn.execute("""
            INSERT INTO conversations 
            (id, session_id, user_message, agent_response, node_type, llm_provider, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            RETURNING id
        """, [
            conversation_id,
            conversation.session_id,
            conversation.user_message,
            conversation.agent_response,
            conversation.node_type,
            conversation.llm_provider,
            conversation.timestamp
        ]).fetchone()
        conn.close()
        return result[0] if result else None
    
    def get_session_history(self, session_id: str) -> List[Conversation]:
        """Get conversation history for a session"""
        conn = duckdb.connect(self.db_path)
        rows = conn.execute("""
            SELECT * FROM conversations 
            WHERE session_id = ? 
            ORDER BY timestamp
        """, [session_id]).fetchall()
        conn.close()
        
        return [
            Conversation(
                id=row[0],
                session_id=row[1],
                user_message=row[2],
                agent_response=row[3],
                node_type=row[4],
                llm_provider=row[5],
                timestamp=row[6]
            )
            for row in rows
        ]
    
    def get_all_sessions(self) -> List[str]:
        """Get all unique session IDs, ordered by most recent activity"""
        conn = duckdb.connect(self.db_path)
        rows = conn.execute("""
            SELECT session_id
            FROM (
                SELECT session_id, MAX(timestamp) as last_activity
                FROM conversations
                GROUP BY session_id
            )
            ORDER BY last_activity DESC
        """).fetchall()
        conn.close()
        return [row[0] for row in rows]