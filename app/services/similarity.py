"""Document similarity computation — centroids + pairwise cosine similarity."""

import math

from app.services.elasticsearch import es_service


def compute_centroid(vectors: list[list[float]]) -> list[float]:
    """Average embedding vectors element-wise."""
    if not vectors:
        return []
    dim = len(vectors[0])
    centroid = [0.0] * dim
    for vec in vectors:
        for i in range(dim):
            centroid[i] += vec[i]
    n = len(vectors)
    return [c / n for c in centroid]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors. Returns 0.0 for zero vectors."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        norm_a += ai * ai
        norm_b += bi * bi
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0.0:
        return 0.0
    return dot / denom


async def compute_document_similarity(threshold: float = 0.3) -> dict:
    """Orchestrate: fetch embeddings -> centroids -> pairwise similarity -> filter edges."""
    embeddings_by_doc = await es_service.get_all_embeddings_by_document()

    if not embeddings_by_doc:
        return {"nodes": [], "edges": [], "threshold": threshold}

    # Get document metadata for nodes
    documents = await es_service.list_documents()
    doc_lookup = {d["document_id"]: d for d in documents}

    # Compute centroids
    centroids: dict[str, list[float]] = {}
    for doc_id, vectors in embeddings_by_doc.items():
        centroids[doc_id] = compute_centroid(vectors)

    # Build nodes
    doc_ids = list(centroids.keys())
    nodes = []
    for doc_id in doc_ids:
        meta = doc_lookup.get(doc_id, {})
        nodes.append({
            "document_id": doc_id,
            "filename": meta.get("filename", "unknown"),
            "source_type": meta.get("source_type", "unknown"),
            "chunk_count": len(embeddings_by_doc[doc_id]),
        })

    # Compute pairwise similarity, filter by threshold
    edges = []
    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            sim = cosine_similarity(centroids[doc_ids[i]], centroids[doc_ids[j]])
            if sim >= threshold:
                edges.append({
                    "source": doc_ids[i],
                    "target": doc_ids[j],
                    "similarity": round(sim, 4),
                })

    return {"nodes": nodes, "edges": edges, "threshold": threshold}
