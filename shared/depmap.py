from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
import yaml


@dataclass
class DependencyLink:
    src: str
    dst: str
    relation: str


class DependencyMap:
    """
    Dependency map tracking upstream/downstream relationships between:
    - code:<path>
    - feature:<name>
    - dataset:<batch_id>
    - model:<name>@<version>
    - deployment:<target>
    """

    def __init__(self, data: Optional[dict] = None):
        self._raw_data = data or {}
        self._nodes: Set[str] = set()
        self._owners: Dict[str, str] = {}
        self._links: List[DependencyLink] = []
        if data:
            self._parse(data)

    def _parse(self, data: dict):
        # Parse owners
        for node, owner in data.get("owners", {}).items():
            self._nodes.add(node)
            self._owners[node] = owner

        # Parse links
        for link in data.get("links", []):
            src = link["src"]
            dst = link["dst"]
            relation = link.get("relation", "depends_on")
            self._nodes.add(src)
            self._nodes.add(dst)
            self._links.append(DependencyLink(src=src, dst=dst, relation=relation))

    @classmethod
    def from_yaml(cls, path: str | Path) -> DependencyMap:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    def to_dict(self) -> dict:
        return {
            "owners": dict(self._owners),
            "links": [
                {"src": l.src, "dst": l.dst, "relation": l.relation}
                for l in self._links
            ],
        }

    def save_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def upstream(self, node: str) -> List[str]:
        """
        Returns all nodes that directly feed into or are upstream of this node.
        If src -> dst, then src is upstream of dst.
        """
        return [l.src for l in self._links if l.dst == node]

    def downstream(self, node: str) -> List[str]:
        """
        Returns all nodes that depend directly on this node.
        If src -> dst, then dst is downstream of src.
        """
        return [l.dst for l in self._links if l.src == node]

    def owner(self, node: str) -> Optional[str]:
        """Returns the assigned owner agent for a given node."""
        if node in self._owners:
            return self._owners[node]
        # Infer owner by node prefix if not explicitly set
        prefix = node.split(":")[0] if ":" in node else node
        prefix_to_owner = {
            "code": "reviewer",
            "feature": "reviewer",
            "dataset": "checker",
            "model": "trainer",
            "deployment": "deployer",
            "pipeline": "setup",
        }
        return prefix_to_owner.get(prefix)

    def add_link(self, src: str, dst: str, relation: str = "depends_on") -> None:
        """Adds a directional link src -> dst."""
        self._nodes.add(src)
        self._nodes.add(dst)
        for link in self._links:
            if link.src == src and link.dst == dst:
                link.relation = relation
                return
        self._links.append(DependencyLink(src=src, dst=dst, relation=relation))

    def set_owner(self, node: str, owner: str) -> None:
        """Sets the owner for a node."""
        self._nodes.add(node)
        self._owners[node] = owner

    def walk_upstream(self, node: str) -> Set[str]:
        """Recursively traverses all upstream ancestor nodes."""
        visited: Set[str] = set()
        queue = [node]
        while queue:
            curr = queue.pop(0)
            for up in self.upstream(curr):
                if up not in visited:
                    visited.add(up)
                    queue.append(up)
        return visited
