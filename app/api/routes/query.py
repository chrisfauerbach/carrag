import json
from datetime import datetime, timezone

from httpx import AsyncClient as HttpxClient
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models.schemas import ModelsResponse, QueryRequest, QueryResponse
from app.services.rag import query_rag, query_rag_stream
from app.services.chat import chat_service

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
        tags=request.tags or None,
    )

    response = QueryResponse(
        answer=result["answer"],
        model=result["model"],
        duration_ms=result["duration_ms"],
    )

    if request.return_sources:
        response.sources = result["sources"]

    if request.chat_id:
        now = datetime.now(timezone.utc).isoformat()
        user_msg = {"role": "user", "content": request.question, "timestamp": now}
        assistant_msg = {
            "role": "assistant",
            "content": result["answer"],
            "timestamp": now,
            "model": result["model"],
            "duration_ms": result["duration_ms"],
            "sources": result["sources"],
        }
        await chat_service.append_messages(request.chat_id, [user_msg, assistant_msg])

    return response


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream a RAG answer as Server-Sent Events."""

    async def event_generator():
        tokens = []
        sources = []
        done_data = None
        errored = False

        try:
            async for event in query_rag_stream(
                question=request.question,
                top_k=request.top_k,
                model=request.model,
                history=request.history,
                tags=request.tags or None,
            ):
                event_type = event["type"]
                data = json.dumps(event["data"])
                yield f"event: {event_type}\ndata: {data}\n\n"

                if event_type == "token":
                    tokens.append(event["data"]["token"])
                elif event_type == "sources":
                    sources = event["data"].get("sources", [])
                elif event_type == "done":
                    done_data = event["data"]
        except Exception as exc:
            errored = True
            error_data = json.dumps({"error": str(exc)})
            yield f"event: error\ndata: {error_data}\n\n"

        if request.chat_id and not errored and done_data:
            now = datetime.now(timezone.utc).isoformat()
            user_msg = {"role": "user", "content": request.question, "timestamp": now}
            assistant_msg = {
                "role": "assistant",
                "content": "".join(tokens),
                "timestamp": now,
                "model": done_data.get("model"),
                "duration_ms": done_data.get("duration_ms"),
                "sources": sources,
            }
            await chat_service.append_messages(request.chat_id, [user_msg, assistant_msg])

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )
