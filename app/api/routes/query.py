from httpx import AsyncClient as HttpxClient
from fastapi import APIRouter

from app.config import settings
from app.models.schemas import ModelsResponse, QueryRequest, QueryResponse
from app.services.rag import query_rag

router = APIRouter()


@router.get("/query/models", response_model=ModelsResponse)
async def list_models():
    """List locally available Ollama models and the configured default."""
    async with HttpxClient() as client:
        resp = await client.get(f"{settings.ollama_url}/api/tags")
        resp.raise_for_status()
        data = resp.json()

    embedding_families = {"nomic-bert", "bert"}
    names = sorted(
        m["name"]
        for m in data.get("models", [])
        if not embedding_families & set(m.get("details", {}).get("families", []))
    )
    return ModelsResponse(models=names, default=settings.llm_model)


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
