import asyncio
import json
import logging
import time
from typing import AsyncGenerator

import httpx

from app.config import settings
from app.services.embeddings import embedding_service
from app.services.elasticsearch import es_service
from app.services.metrics import metrics_service, extract_ollama_metrics
from app.services.prompts import prompts_service, DEFAULT_PROMPTS
from app.services.reranker import reranker_service

logger = logging.getLogger(__name__)


async def generate_tags(content: str, max_tags: int = 5, filename: str = "") -> list[str]:
    """Generate descriptive tags for a document using the LLM.

    Truncates content to ~8000 chars and asks the LLM for comma-separated tags.
    Returns [] on any failure — auto-tagging should never block ingestion.
    """
    truncated = content[:8000]
    filename_hint = f"Filename: {filename}\n\n" if filename else ""

    try:
        sys_prompt_doc = await prompts_service.get_prompt("autotag_system")
        user_prompt_doc = await prompts_service.get_prompt("autotag_user")
    except Exception:
        sys_prompt_doc = None
        user_prompt_doc = None

    system_prompt = (
        sys_prompt_doc["content"] if sys_prompt_doc
        else DEFAULT_PROMPTS["autotag_system"]["content"]
    )

    if user_prompt_doc:
        user_prompt = user_prompt_doc["content"].format(
            max_tags=max_tags, filename_hint=filename_hint, truncated=truncated
        )
    else:
        user_prompt = DEFAULT_PROMPTS["autotag_user"]["content"].format(
            max_tags=max_tags, filename_hint=filename_hint, truncated=truncated
        )

    try:
        async with httpx.AsyncClient(base_url=settings.ollama_url, timeout=120) as client:
            resp = await client.post(
                "/api/generate",
                json={
                    "model": settings.llm_model,
                    "prompt": user_prompt,
                    "system": system_prompt,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            result = resp.json()

        ollama_metrics = extract_ollama_metrics(result)
        metrics_service.record_background(
            "tag_generation",
            settings.llm_model,
            **ollama_metrics,
            metadata={"filename": filename, "content_length": len(content)},
        )

        raw = result.get("response", "")
        tags = [t.strip().lower() for t in raw.split(",") if t.strip()]
        return tags[:max_tags]
    except Exception:
        logger.warning("Auto-tag generation failed", exc_info=True)
        return []


async def _expand_context(chunks: list[dict]) -> list[dict]:
    """Fetch neighboring chunks for each reranked chunk and merge them.

    For each chunk, fetches chunk_index +/- 1 from same document.
    Deduplicates by (document_id, chunk_index) and merges adjacent texts.
    """
    if not chunks:
        return chunks

    # Fetch neighbors in parallel
    tasks = [
        es_service.get_neighboring_chunks(c["document_id"], c["chunk_index"], window=1)
        for c in chunks
    ]
    neighbor_results = await asyncio.gather(*tasks)

    expanded = []
    seen = set()

    for chunk, neighbors in zip(chunks, neighbor_results):
        # Merge neighbor texts in order, deduplicating
        merged_parts = []
        for n in neighbors:
            key = (n["document_id"], n["chunk_index"])
            if key not in seen:
                seen.add(key)
                merged_parts.append(n["content"])

        expanded.append({
            "content": "\n".join(merged_parts) if merged_parts else chunk["content"],
            "score": chunk["score"],
            "metadata": chunk["metadata"],
            "document_id": chunk["document_id"],
            "chunk_index": chunk["chunk_index"],
        })

    return expanded


async def _prepare_rag_context(
    question: str, top_k: int = 10, model: str | None = None, history: list | None = None,
    tags: list[str] | None = None, rerank: bool | None = None,
) -> tuple[str, str, list[dict], str]:
    """Shared retrieval logic: embed -> hybrid search -> rerank -> expand -> build prompt.

    Returns (prompt, system_prompt, sources, llm_model).
    """
    llm_model = model or settings.llm_model

    # Determine if reranking is active (per-query override or global setting)
    rerank_active = rerank if rerank is not None else reranker_service.enabled

    # 1. Embed the question (search_query prefix required by nomic-embed-text)
    query_vector = await embedding_service.embed_single(question, prefix="search_query: ")

    # 2. Retrieve similar chunks (over-retrieve if reranking)
    retrieval_k = top_k * settings.retrieval_k_multiplier if rerank_active else top_k
    chunks = await es_service.hybrid_search(query_vector, question, top_k=retrieval_k, tags=tags or None)

    # 3. Rerank if active
    if rerank_active:
        chunks = reranker_service.rerank(question, chunks, top_k)

    # 4. Context expansion — fetch neighbors for each reranked chunk
    if rerank_active and settings.context_expansion_enabled:
        chunks = await _expand_context(chunks)

    # 5. Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("filename", "unknown")
        context_parts.append(f"[Source {i}: {source}]\n{chunk['content']}")
    context = "\n\n---\n\n".join(context_parts)

    # 6. Build the prompt (with optional conversation history)
    history_block = ""
    if history:
        history_lines = []
        for msg in history:
            role = msg.role if hasattr(msg, "role") else msg["role"]
            content = msg.content if hasattr(msg, "content") else msg["content"]
            label = "User" if role == "user" else "Assistant"
            history_lines.append(f"{label}: {content}")
        history_block = "\n\nConversation history:\n" + "\n".join(history_lines) + "\n"

    try:
        sys_prompt_doc = await prompts_service.get_prompt("rag_system")
        user_prompt_doc = await prompts_service.get_prompt("rag_user")
    except Exception:
        sys_prompt_doc = None
        user_prompt_doc = None

    system_prompt = (
        sys_prompt_doc["content"] if sys_prompt_doc
        else DEFAULT_PROMPTS["rag_system"]["content"]
    )

    if user_prompt_doc:
        prompt = user_prompt_doc["content"].format(
            context=context, history_block=history_block, question=question
        )
    else:
        prompt = DEFAULT_PROMPTS["rag_user"]["content"].format(
            context=context, history_block=history_block, question=question
        )

    sources = [
        {
            "content": c["content"],
            "score": c["score"],
            "metadata": c["metadata"],
        }
        for c in chunks
    ]

    return prompt, system_prompt, sources, llm_model


async def query_rag(
    question: str, top_k: int = 10, model: str | None = None, history: list | None = None,
    tags: list[str] | None = None, rerank: bool | None = None,
) -> dict:
    """Full RAG pipeline: embed question -> retrieve chunks -> generate answer."""
    start = time.time()

    prompt, system_prompt, sources, llm_model = await _prepare_rag_context(
        question, top_k=top_k, model=model, history=history, tags=tags, rerank=rerank
    )

    # Generate answer via Ollama
    async with httpx.AsyncClient(base_url=settings.ollama_url, timeout=300) as client:
        resp = await client.post(
            "/api/generate",
            json={
                "model": llm_model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
            },
        )
        resp.raise_for_status()
        result = resp.json()

    duration_ms = (time.time() - start) * 1000

    ollama_metrics = extract_ollama_metrics(result)
    metrics_service.record_background(
        "query",
        llm_model,
        duration_ms=round(duration_ms, 1),
        **ollama_metrics,
        metadata={"question_length": len(question), "top_k": top_k},
    )

    return {
        "answer": result.get("response", ""),
        "sources": sources,
        "model": llm_model,
        "duration_ms": round(duration_ms, 1),
    }


async def query_rag_stream(
    question: str, top_k: int = 10, model: str | None = None, history: list | None = None,
    tags: list[str] | None = None, rerank: bool | None = None,
) -> AsyncGenerator[dict, None]:
    """Streaming RAG pipeline: yields SSE-style event dicts.

    Events: {type: "sources", data: ...}, {type: "token", data: ...}, {type: "done", data: ...}
    """
    start = time.time()

    prompt, system_prompt, sources, llm_model = await _prepare_rag_context(
        question, top_k=top_k, model=model, history=history, tags=tags, rerank=rerank
    )

    # Yield sources immediately (retrieval is done)
    yield {"type": "sources", "data": {"sources": sources}}

    # Stream generation from Ollama
    async with httpx.AsyncClient(base_url=settings.ollama_url, timeout=300) as client:
        async with client.stream(
            "POST",
            "/api/generate",
            json={
                "model": llm_model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("done"):
                    duration_ms = (time.time() - start) * 1000
                    ollama_metrics = extract_ollama_metrics(chunk)
                    metrics_service.record_background(
                        "query_stream",
                        llm_model,
                        duration_ms=round(duration_ms, 1),
                        **ollama_metrics,
                        metadata={"question_length": len(question), "top_k": top_k},
                    )
                    yield {
                        "type": "done",
                        "data": {
                            "model": llm_model,
                            "duration_ms": round(duration_ms, 1),
                        },
                    }
                    break
                token = chunk.get("response", "")
                if token:
                    yield {"type": "token", "data": {"token": token}}
