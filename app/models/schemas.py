from datetime import datetime

from pydantic import BaseModel, Field


# --- Ingest ---

class IngestURLRequest(BaseModel):
    url: str


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
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
    created_at: datetime | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class DocumentDetailResponse(BaseModel):
    document_id: str
    filename: str
    source_type: str
    chunk_count: int
    metadata: dict
    created_at: datetime | None = None


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
