from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict


class InteractionKind(str, Enum):
    STORE_ONLY = "store_only"
    TRANSFORM_AND_STORE = "transform_and_store"
    ACT_NOW = "act_now"
    ACT_AND_STORE = "act_and_store"


class RetrievalIntent(str, Enum):
    MEMORY_PERSONAL = "memory_personal"
    FACT_LOOKUP = "fact_lookup"
    MULTI_HOP = "multi_hop_reasoning"
    MIXED = "mixed"


class RetrievalChunk(TypedDict):
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]


class PlannedToolCall(TypedDict):
    tool_name: str
    arguments: Dict[str, Any]


class TopicSelection(TypedDict):
    candidates: List[str]
    primary: Optional[str]
    confidence: float


class GraphState(TypedDict, total=False):
    user_id: str
    query: str
    history: List[Dict[str, Any]]

    interaction_kind: InteractionKind
    ingestion_type: Optional[str]

    topic_candidates: List[str]
    primary_topic: Optional[str]
    topic_confidence: float
    retrieval_intent: RetrievalIntent

    need_memobase: bool
    need_vector: bool
    need_graph: bool

    memobase_results: List[RetrievalChunk]
    vector_results: List[RetrievalChunk]
    graph_results: List[RetrievalChunk]
    final_results: List[RetrievalChunk]

    planned_tools: List[PlannedToolCall]
    tool_results: Dict[str, Any]

    answer: str
    metadata: Dict[str, Any]


class MemobaseRecord(TypedDict):
    id: str
    user_id: str
    topic: str
    type: str
    content: str
    metadata: Dict[str, Any]
    created_at: float
    last_seen_at: float
    strength: float
    expires_at: Optional[float]


class MemobaseHit(TypedDict):
    content: str
    score: float
    topic: str
    type: str
    id: str
    metadata: Dict[str, Any]


class RagdollDocument(TypedDict):
    doc_id: str
    text: str
    metadata: Dict[str, Any]


class RagdollHit(TypedDict):
    text: str
    score: float
    metadata: Dict[str, Any]


class GraphDoc(TypedDict):
    text: str
    score: float
    metadata: Dict[str, Any]


class Entity(TypedDict):
    name: str
    type: str
    metadata: Dict[str, Any]


class Relationship(TypedDict):
    source: str
    target: str
    type: str
    metadata: Dict[str, Any]


class DocRef(TypedDict):
    doc_id: str
    topic: str


DEFAULT_TOPICS = ["work", "projects", "family", "personal_admin"]
