import logging
import time

import httpx

from app.config import settings
from app.services.embeddings import embedding_service
from app.services.elasticsearch import es_service

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the context below to answer the question. If the context doesn't contain enough information to answer, say so clearly.
Always cite which source(s) you used in your answer."""


async def query_rag(question: str, top_k: int = 5, model: str | None = None, history: list | None = None) -> dict:
    """Full RAG pipeline: embed question -> retrieve chunks -> generate answer."""
    start = time.time()
    llm_model = model or settings.llm_model

    # 1. Embed the question
    query_vector = await embedding_service.embed_single(question)

    # 2. Retrieve similar chunks
    chunks = await es_service.knn_search(query_vector, top_k=top_k)

    # 3. Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("filename", "unknown")
        context_parts.append(f"[Source {i}: {source}]\n{chunk['content']}")
    context = "\n\n---\n\n".join(context_parts)

    # 4. Build the prompt (with optional conversation history)
    history_block = ""
    if history:
        history_lines = []
        for msg in history:
            role = msg.role if hasattr(msg, "role") else msg["role"]
            content = msg.content if hasattr(msg, "content") else msg["content"]
            label = "User" if role == "user" else "Assistant"
            history_lines.append(f"{label}: {content}")
        history_block = "\n\nConversation history:\n" + "\n".join(history_lines) + "\n"

    prompt = f"""Context:
{context}
{history_block}
Question: {question}

Answer based on the context above:"""

    # 5. Generate answer via Ollama
    async with httpx.AsyncClient(base_url=settings.ollama_url, timeout=300) as client:
        resp = await client.post(
            "/api/generate",
            json={
                "model": llm_model,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
            },
        )
        resp.raise_for_status()
        result = resp.json()

    duration_ms = (time.time() - start) * 1000

    return {
        "answer": result.get("response", ""),
        "sources": [
            {
                "content": c["content"],
                "score": c["score"],
                "metadata": c["metadata"],
            }
            for c in chunks
        ],
        "model": llm_model,
        "duration_ms": round(duration_ms, 1),
    }
