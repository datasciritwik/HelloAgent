import re
from datetime import datetime
from typing import List

def clean_text(text: str) -> str:
    """Clean and normalize text input"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\-\.,\?!]', '', text)
    
    return text

def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text"""
    # Simple keyword extraction
    words = text.lower().split()
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }
    
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    return keywords

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display"""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

def validate_session_id(session_id: str) -> bool:
    """Validate session ID format"""
    if not session_id:
        return False
    
    # Check if it's a valid UUID format
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    return bool(re.match(uuid_pattern, session_id, re.IGNORECASE))

def get_node_type_emoji(node_type: str) -> str:
    """Get emoji for node type"""
    emoji_map = {
        'greeting': '👋',
        'facts': '📚',
        'unknown': '❓'
    }
    return emoji_map.get(node_type, '🤖')