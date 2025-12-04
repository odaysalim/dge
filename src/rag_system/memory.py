"""
Conversation Memory System for Agentic RAG.
Implements sliding window memory with token-based truncation.
"""

import json
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

from ..config.settings import CONFIG


class ConversationMemory:
    """
    Manages conversation history with sliding window and token-based truncation.
    """

    def __init__(
        self,
        max_tokens: int = None,
        window_size: int = None,
        persist_path: Optional[str] = None
    ):
        """
        Initialize conversation memory.

        Args:
            max_tokens: Maximum tokens to keep in memory (default from config)
            window_size: Number of message pairs to keep (default from config)
            persist_path: Optional path to persist conversation history
        """
        self.max_tokens = max_tokens or CONFIG['memory']['max_tokens']
        self.window_size = window_size or CONFIG['memory']['window_size']
        self.persist_path = persist_path
        self.messages: List[Dict[str, str]] = []

        # Load from disk if persist_path is provided
        if self.persist_path and Path(self.persist_path).exists():
            self.load()

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Add a message to conversation history.

        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional metadata (sources, timestamp, etc.)
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.messages.append(message)

        # Trim if exceeding window size
        if len(self.messages) > self.window_size * 2:  # 2 messages per turn
            self.messages = self.messages[-(self.window_size * 2):]

        # Persist if path is set
        if self.persist_path:
            self.save()

    def get_recent_context(self, n_turns: int = None) -> str:
        """
        Get recent conversation context as a formatted string.

        Args:
            n_turns: Number of conversation turns to include (default: window_size)

        Returns:
            Formatted conversation history
        """
        n_turns = n_turns or self.window_size
        recent_messages = self.messages[-(n_turns * 2):]  # 2 messages per turn

        if not recent_messages:
            return ""

        context_parts = ["**Previous Conversation:**\n"]
        for msg in recent_messages:
            role_label = "User" if msg['role'] == 'user' else "Assistant"
            context_parts.append(f"\n{role_label}: {msg['content']}\n")

        return "\n".join(context_parts)

    def get_messages_for_llm(self, n_turns: int = None) -> List[Dict[str, str]]:
        """
        Get recent messages formatted for LLM APIs.

        Args:
            n_turns: Number of conversation turns to include

        Returns:
            List of message dicts with 'role' and 'content'
        """
        n_turns = n_turns or self.window_size
        recent_messages = self.messages[-(n_turns * 2):]

        return [
            {'role': msg['role'], 'content': msg['content']}
            for msg in recent_messages
        ]

    def clear(self):
        """Clear all conversation history."""
        self.messages = []
        if self.persist_path:
            self.save()

    def save(self):
        """Save conversation history to disk."""
        if not self.persist_path:
            return

        path = Path(self.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump({
                'messages': self.messages,
                'max_tokens': self.max_tokens,
                'window_size': self.window_size
            }, f, indent=2)

    def load(self):
        """Load conversation history from disk."""
        if not self.persist_path or not Path(self.persist_path).exists():
            return

        with open(self.persist_path, 'r') as f:
            data = json.load(f)
            self.messages = data.get('messages', [])
            self.max_tokens = data.get('max_tokens', self.max_tokens)
            self.window_size = data.get('window_size', self.window_size)

    def get_summary(self) -> Dict:
        """Get a summary of the conversation memory state."""
        return {
            'total_messages': len(self.messages),
            'conversation_turns': len(self.messages) // 2,
            'max_tokens': self.max_tokens,
            'window_size': self.window_size,
            'persist_path': self.persist_path
        }

    def __repr__(self):
        return f"ConversationMemory(messages={len(self.messages)}, window={self.window_size})"


# Global memory instance (can be used across the application)
_global_memory: Optional[ConversationMemory] = None


def get_global_memory() -> ConversationMemory:
    """
    Get the global conversation memory instance.
    Creates one if it doesn't exist.
    """
    global _global_memory
    if _global_memory is None:
        _global_memory = ConversationMemory()
    return _global_memory


def create_session_memory(session_id: str, persist: bool = True) -> ConversationMemory:
    """
    Create a session-specific conversation memory.

    Args:
        session_id: Unique session identifier
        persist: Whether to persist to disk

    Returns:
        ConversationMemory instance for the session
    """
    if persist:
        from ..config.settings import PROJECT_ROOT
        persist_path = str(PROJECT_ROOT / "data" / "memory" / f"{session_id}.json")
    else:
        persist_path = None

    return ConversationMemory(persist_path=persist_path)
