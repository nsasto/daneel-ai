from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional

import httpx

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
    """Memobase client interface."""

    async def search(self, user_id: str, query: str, topic: Optional[str], k: int) -> List[MemobaseHit]:  # pragma: no cover - interface
        raise NotImplementedError

    async def write(self, records: List[MemobaseRecord]) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class RagdollClient:
    """RAGdoll client interface."""

    async def ingest(self, documents: List[RagdollDocument], collection: str, *, enable_graph: bool = True) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def search(
        self,
        query: str,
        collections: List[str],
        k: int,
        *,
        use_graph: bool = False,
        use_pagerank: bool = False,
        hybrid_mode: Optional[str] = None,
    ) -> List[RagdollHit]:  # pragma: no cover - interface
        raise NotImplementedError


class GraphClient:
    """Graph RAG client interface."""

    async def query_with_pagerank(self, query: str, topic_graphs: List[str], max_docs: int) -> List[GraphDoc]:  # pragma: no cover - interface
        raise NotImplementedError

    async def update_from_entities(
        self,
        entities: List[Entity],
        relationships: List[Relationship],
        doc_refs: List[DocRef],
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryMemobaseClient(MemobaseClient):
    """In-memory Memobase placeholder for development and tests."""

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


class InMemoryRagdollClient(RagdollClient):
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


class InMemoryGraphClient(GraphClient):
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


class HTTPMemobaseClient(MemobaseClient):
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 5.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout, headers=self._headers(api_key))

    @staticmethod
    def _headers(api_key: Optional[str]) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def search(self, user_id: str, query: str, topic: Optional[str], k: int) -> List[MemobaseHit]:
        payload = {"user_id": user_id, "query": query, "topic": topic, "k": k}
        try:
            resp = await self._client.post("/search", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("hits", [])
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Memobase search failed: {exc}") from exc

    async def write(self, records: List[MemobaseRecord]) -> None:
        try:
            resp = await self._client.post("/write", json={"records": records})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Memobase write failed: {exc}") from exc


class HTTPRagdollClient(RagdollClient):
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 5.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout, headers=self._headers(api_key))

    @staticmethod
    def _headers(api_key: Optional[str]) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def ingest(self, documents: List[RagdollDocument], collection: str, *, enable_graph: bool = True) -> None:
        payload = {"documents": documents, "collection": collection, "enable_graph": enable_graph}
        try:
            resp = await self._client.post("/ingest", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"RAGdoll ingest failed: {exc}") from exc

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
        payload = {
            "query": query,
            "collections": collections,
            "k": k,
            "use_graph": use_graph,
            "use_pagerank": use_pagerank,
            "hybrid_mode": hybrid_mode,
        }
        try:
            resp = await self._client.post("/search", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("hits", [])
        except httpx.HTTPError as exc:
            raise RuntimeError(f"RAGdoll search failed: {exc}") from exc


class HTTPGraphClient(GraphClient):
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 5.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout, headers=self._headers(api_key))

    @staticmethod
    def _headers(api_key: Optional[str]) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def query_with_pagerank(self, query: str, topic_graphs: List[str], max_docs: int) -> List[GraphDoc]:
        payload = {"query": query, "topic_graphs": topic_graphs, "max_docs": max_docs}
        try:
            resp = await self._client.post("/query-with-pagerank", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("hits", [])
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Graph query failed: {exc}") from exc

    async def update_from_entities(
        self,
        entities: List[Entity],
        relationships: List[Relationship],
        doc_refs: List[DocRef],
    ) -> None:
        payload = {"entities": entities, "relationships": relationships, "doc_refs": doc_refs}
        try:
            resp = await self._client.post("/update-from-entities", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Graph update failed: {exc}") from exc
