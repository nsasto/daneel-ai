from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional

from .models import (
    DocRef,
    Entity,
    GraphDoc,
    MemobaseHit,
    MemobaseRecord,
    RagdollDocument,
    RagdollHit,
    Relationship,
)


class MemobaseClient:
    """In memory Memobase placeholder for development and tests."""

    def __init__(self) -> None:
        self._records: List[MemobaseRecord] = []

    async def search(self, user_id: str, query: str, topic: Optional[str], k: int) -> List[MemobaseHit]:
        hits: List[MemobaseHit] = []
        for record in self._records:
            if record["user_id"] != user_id:
                continue
            if topic and record["topic"] != topic:
                continue
            if query.lower() in record["content"].lower():
                hits.append(
                    {
                        "content": record["content"],
                        "score": record.get("strength", 1.0),
                        "topic": record["topic"],
                        "type": record["type"],
                        "id": record["id"],
                        "metadata": record.get("metadata", {}),
                    }
                )
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:k]

    async def write(self, records: List[MemobaseRecord]) -> None:
        now = time.time()
        for record in records:
            record.setdefault("created_at", now)
            record.setdefault("last_seen_at", now)
            record.setdefault("strength", 1.0)
            self._records.append(record)


class RagdollClient:
    """Lightweight in-memory RAGdoll style client for local development."""

    def __init__(self) -> None:
        self._collections: Dict[str, List[RagdollDocument]] = defaultdict(list)

    async def ingest(self, documents: List[RagdollDocument], collection: str, *, enable_graph: bool = True) -> None:
        self._collections[collection].extend(documents)

    async def search(
        self,
        query: str,
        collections: List[str],
        k: int,
        *,
        use_graph: bool = False,
        use_pagerank: bool = False,
        hybrid_mode: Optional[str] = None,
    ) -> List[RagdollHit]:
        hits: List[RagdollHit] = []
        for collection in collections:
            for doc in self._collections.get(collection, []):
                if query.lower() in doc["text"].lower():
                    hits.append({"text": doc["text"], "score": 1.0, "metadata": doc.get("metadata", {})})
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:k]


class GraphClient:
    """Graph RAG placeholder with PageRank-like retrieval."""

    def __init__(self) -> None:
        self._entities: List[Entity] = []
        self._relationships: List[Relationship] = []
        self._doc_refs: List[DocRef] = []

    async def query_with_pagerank(self, query: str, topic_graphs: List[str], max_docs: int) -> List[GraphDoc]:
        hits: List[GraphDoc] = []
        for doc_ref in self._doc_refs:
            if topic_graphs and doc_ref["topic"] not in topic_graphs:
                continue
            if query.lower() in doc_ref["doc_id"].lower():
                hits.append({"text": doc_ref["doc_id"], "score": 1.0, "metadata": {"topic": doc_ref["topic"]}})
        return hits[:max_docs]

    async def update_from_entities(
        self,
        entities: List[Entity],
        relationships: List[Relationship],
        doc_refs: List[DocRef],
    ) -> None:
        self._entities.extend(entities)
        self._relationships.extend(relationships)
        self._doc_refs.extend(doc_refs)
