"""
GraphRAG — data pattern graph for enhanced context retrieval
Adapted from original MiroFish for synthetic data patterns
"""

import json
from typing import Any
from dataclasses import dataclass, field


@dataclass
class PatternNode:
    """A node in the pattern graph"""
    id: str
    type: str  # pattern, constraint, entity, topic
    content: str
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class PatternEdge:
    """An edge between pattern nodes"""
    source: str
    target: str
    relationship: str  # implies, contradicts, requires, similar_to
    weight: float = 1.0


class GraphRAG:
    """
    Simple in-memory graph for storing and querying data patterns.
    Used for: pattern-based context retrieval during generation.
    """

    def __init__(self):
        self.nodes: dict[str, PatternNode] = {}
        self.edges: list[PatternEdge] = []

    def add_node(self, node: PatternNode):
        """Add a pattern node"""
        self.nodes[node.id] = node

    def add_edge(self, edge: PatternEdge):
        """Add a relationship edge"""
        self.edges.append(edge)

    def get_related(self, node_id: str, max_depth: int = 2) -> list[PatternNode]:
        """Get nodes related to a given node"""
        visited = set()
        queue = [(node_id, 0)]
        results = []

        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)

            if current != node_id and current in self.nodes:
                results.append(self.nodes[current])

            for edge in self.edges:
                if edge.source == current and edge.target not in visited:
                    queue.append((edge.target, depth + 1))
                elif edge.target == current and edge.source not in visited:
                    queue.append((edge.source, depth + 1))

        return results

    def search(self, query: str, limit: int = 5) -> list[PatternNode]:
        """Search nodes by content"""
        query_lower = query.lower()
        results = []
        for node in self.nodes.values():
            if query_lower in node.content.lower():
                results.append(node)
                if len(results) >= limit:
                    break
        return sorted(results, key=lambda n: n.weight, reverse=True)

    def build_from_patterns(self, patterns: dict):
        """
        Build graph from extracted patterns dict.
        """
        # Structural patterns
        structural = patterns.get("structural_patterns", {})
        if structural:
            self.add_node(PatternNode(
                id="structural_root",
                type="pattern",
                content=json.dumps(structural, ensure_ascii=False)[:500],
                weight=0.8,
            ))

        # Semantic patterns
        semantic = patterns.get("semantic_patterns", {})
        for topic in semantic.get("topics", [])[:10]:
            node_id = f"topic_{topic}"
            self.add_node(PatternNode(
                id=node_id,
                type="topic",
                content=topic,
                weight=0.7,
            ))

        for entity in semantic.get("entities", [])[:10]:
            node_id = f"entity_{entity}"
            self.add_node(PatternNode(
                id=node_id,
                type="entity",
                content=entity,
                weight=0.6,
            ))

        # Constraints
        for i, constraint in enumerate(patterns.get("constraints", [])[:10]):
            node_id = f"constraint_{i}"
            content = constraint if isinstance(constraint, str) else json.dumps(constraint)
            self.add_node(PatternNode(
                id=node_id,
                type="constraint",
                content=content,
                weight=0.9,
            ))

    def to_dict(self) -> dict:
        """Serialize graph"""
        return {
            "nodes": {k: {"id": v.id, "type": v.type, "content": v.content, "weight": v.weight}
                      for k, v in self.nodes.items()},
            "edges": [{"source": e.source, "target": e.target, "relationship": e.relationship}
                      for e in self.edges],
        }
