"""
Zep Memory — agent memory via Zep Cloud
Preserved from original MiroFish, adapted for synthetic data context
"""

import os
from typing import Any, Optional
from dataclasses import dataclass

from ..config.settings import Settings


@dataclass
class MemoryEntry:
    """A single memory entry"""
    key: str
    content: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ZepMemory:
    """
    Agent memory backed by Zep Cloud.
    Used for:
    - Storing generation history
    - Remembering patterns across runs
    - Caching analysis results
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Settings.ZEP_API_KEY
        self._client = None
        self._memory: dict[str, MemoryEntry] = {}  # In-memory fallback

    @property
    def client(self):
        """Lazy-init Zep client"""
        if self._client is None and self.api_key:
            try:
                from zep_cloud.client import Zep
                self._client = Zep(api_key=self.api_key)
            except ImportError:
                pass
        return self._client

    def store(self, key: str, content: str, metadata: dict | None = None):
        """Store a memory entry"""
        self._memory[key] = MemoryEntry(key=key, content=content, metadata=metadata or {})

    def retrieve(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry"""
        return self._memory.get(key)

    def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Search memories (simple keyword match)"""
        results = []
        query_lower = query.lower()
        for entry in self._memory.values():
            if query_lower in entry.content.lower() or query_lower in entry.key.lower():
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def list_all(self) -> list[MemoryEntry]:
        """List all memories"""
        return list(self._memory.values())

    def clear(self):
        """Clear all memories"""
        self._memory.clear()

    def to_dict(self) -> dict:
        """Serialize memories"""
        return {
            k: {"key": v.key, "content": v.content, "metadata": v.metadata}
            for k, v in self._memory.items()
        }

    def load_from_dict(self, data: dict):
        """Load memories from dict"""
        for k, v in data.items():
            self._memory[k] = MemoryEntry(
                key=v.get("key", k),
                content=v.get("content", ""),
                metadata=v.get("metadata", {}),
            )
