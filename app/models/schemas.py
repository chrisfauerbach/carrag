from datetime import datetime

from pydantic import BaseModel, Field


# --- Ingest ---

class IngestURLRequest(BaseModel):
    url: str
    tags: list[str] = []


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    tags: list[str] = []
    status: str = "ingested"


# --- Models ---

class ModelsResponse(BaseModel):
    models: list[str]
    default: str


# --- Query ---

class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    model: str | None = None
    return_sources: bool = True
    history: list[ChatMessage] = []
    chat_id: str | None = None
    tags: list[str] = []


class SourceChunk(BaseModel):
    content: str
    score: float
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = []
    model: str
    duration_ms: float


# --- Documents ---

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    source_type: str
    chunk_count: int
    tags: list[str] = []
    created_at: datetime | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class DocumentDetailResponse(BaseModel):
    document_id: str
    filename: str
    source_type: str
    chunk_count: int
    tags: list[str] = []
    metadata: dict
    created_at: datetime | None = None


class UpdateTagsRequest(BaseModel):
    tags: list[str]


class UpdateTagsResponse(BaseModel):
    document_id: str
    tags: list[str]
    chunks_updated: int


class DocumentDeleteResponse(BaseModel):
    document_id: str
    chunks_deleted: int
    status: str = "deleted"


class ChunkInfo(BaseModel):
    content: str
    chunk_index: int
    char_start: int
    char_end: int


class DocumentChunksResponse(BaseModel):
    document_id: str
    filename: str
    source_type: str
    chunk_count: int
    chunks: list[ChunkInfo]


# --- Similarity ---

class SimilarityNode(BaseModel):
    document_id: str
    filename: str
    source_type: str
    chunk_count: int


class SimilarityEdge(BaseModel):
    source: str
    target: str
    similarity: float


class SimilarityResponse(BaseModel):
    nodes: list[SimilarityNode]
    edges: list[SimilarityEdge]
    threshold: float


# --- Chat Sessions ---

class ChatMessageStored(BaseModel):
    role: str
    content: str
    timestamp: str | None = None
    model: str | None = None
    duration_ms: float | None = None
    sources: list[SourceChunk] | None = None


class CreateChatRequest(BaseModel):
    title: str | None = None


class CreateChatResponse(BaseModel):
    chat_id: str
    title: str
    created_at: str
    updated_at: str


class ChatSessionInfo(BaseModel):
    chat_id: str
    title: str
    message_count: int
    created_at: str
    updated_at: str


class ChatListResponse(BaseModel):
    chats: list[ChatSessionInfo]
    total: int


class ChatDetailResponse(BaseModel):
    chat_id: str
    title: str
    messages: list[ChatMessageStored]
    message_count: int
    created_at: str
    updated_at: str


class RenameChatRequest(BaseModel):
    title: str


class RenameChatResponse(BaseModel):
    chat_id: str
    title: str
    updated_at: str


class ChatDeleteResponse(BaseModel):
    chat_id: str
    status: str = "deleted"


# --- Metrics ---

class MetricEvent(BaseModel):
    timestamp: str
    event_type: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    duration_ms: float | None = None
    ollama_total_ms: float | None = None
    metadata: dict | None = None


class MetricsResponse(BaseModel):
    events: list[MetricEvent]
    total: int
