from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    DocumentListResponse,
    DocumentDetailResponse,
    DocumentDeleteResponse,
    DocumentChunksResponse,
    ChunkInfo,
    DocumentInfo,
)
from app.services.elasticsearch import es_service

router = APIRouter()


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """List all ingested documents."""
    docs = await es_service.list_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total=len(docs),
    )


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(document_id: str):
    """Get details for a specific document."""
    doc = await es_service.get_document(document_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    return DocumentDetailResponse(**doc)


@router.get("/{document_id}/chunks", response_model=DocumentChunksResponse)
async def get_document_chunks(document_id: str):
    """Get all chunks for a document."""
    doc = await es_service.get_document(document_id)
    if doc is None:
        raise HTTPException(404, "Document not found")

    chunks = await es_service.get_document_chunks(document_id)
    return DocumentChunksResponse(
        document_id=document_id,
        filename=doc["filename"],
        source_type=doc["source_type"],
        chunk_count=len(chunks),
        chunks=[ChunkInfo(**c) for c in chunks],
    )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(document_id: str):
    """Delete a document and all its chunks."""
    doc = await es_service.get_document(document_id)
    if doc is None:
        raise HTTPException(404, "Document not found")

    deleted = await es_service.delete_document(document_id)
    return DocumentDeleteResponse(document_id=document_id, chunks_deleted=deleted)
