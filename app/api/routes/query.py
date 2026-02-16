from fastapi import APIRouter

from app.models.schemas import QueryRequest, QueryResponse
from app.services.rag import query_rag

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Ask a question and get a RAG-generated answer."""
    result = await query_rag(
        question=request.question,
        top_k=request.top_k,
        model=request.model,
        history=request.history,
    )

    response = QueryResponse(
        answer=result["answer"],
        model=result["model"],
        duration_ms=result["duration_ms"],
    )

    if request.return_sources:
        response.sources = result["sources"]

    return response
