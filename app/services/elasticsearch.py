import asyncio
import logging
from datetime import datetime, timezone

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from app.config import settings

logger = logging.getLogger(__name__)

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "content": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine",
            },
            "document_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "char_start": {"type": "integer"},
            "char_end": {"type": "integer"},
            "metadata": {
                "type": "object",
                "enabled": True,
            },
            "tags": {"type": "text"},
            "created_at": {"type": "date"},
        }
    }
}


class ElasticsearchService:
    def __init__(self):
        self._client: AsyncElasticsearch | None = None

    @property
    def client(self) -> AsyncElasticsearch:
        if self._client is None:
            self._client = AsyncElasticsearch(settings.es_url)
        return self._client

    async def init(self):
        """Create the index if it doesn't exist."""
        exists = await self.client.indices.exists(index=settings.es_index)
        if not exists:
            await self.client.indices.create(index=settings.es_index, body=INDEX_MAPPING)
            logger.info(f"Created index: {settings.es_index}")
        else:
            logger.info(f"Index {settings.es_index} already exists.")

    async def index_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        metadata: dict,
        tags: list[str] | None = None,
    ) -> int:
        """Bulk-index chunks with their embeddings.

        Each chunk dict must have: text, document_id, chunk_index, char_start, char_end.
        Returns the number of successfully indexed documents.
        """
        now = datetime.now(timezone.utc).isoformat()
        resolved_tags = tags or []

        def gen_actions():
            for chunk, embedding in zip(chunks, embeddings):
                yield {
                    "_index": settings.es_index,
                    "_source": {
                        "content": chunk["text"],
                        "embedding": embedding,
                        "document_id": chunk["document_id"],
                        "chunk_index": chunk["chunk_index"],
                        "char_start": chunk["char_start"],
                        "char_end": chunk["char_end"],
                        "metadata": metadata,
                        "tags": resolved_tags,
                        "created_at": now,
                    },
                }

        success, _ = await async_bulk(self.client, gen_actions())
        await self.client.indices.refresh(index=settings.es_index)
        return success

    async def hybrid_search(
        self, query_vector: list[float], query_text: str, top_k: int = 5, tags: list[str] | None = None
    ) -> list[dict]:
        """Find the top-k most relevant chunks using hybrid BM25 + kNN search with manual RRF."""
        source_fields = ["content", "document_id", "chunk_index", "metadata", "created_at"]

        if tags:
            tag_filter = {
                "bool": {
                    "should": [{"match": {"tags": tag}} for tag in tags],
                    "minimum_should_match": 1,
                }
            }
            bm25_query = {
                "bool": {
                    "must": {"match": {"content": {"query": query_text}}},
                    "filter": [tag_filter],
                }
            }
            knn_filter = tag_filter
        else:
            bm25_query = {"match": {"content": {"query": query_text}}}
            knn_filter = None

        knn_body = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": top_k * 10,
        }
        if knn_filter:
            knn_body["filter"] = knn_filter

        bm25_resp, knn_resp = await asyncio.gather(
            self.client.search(
                index=settings.es_index,
                body={
                    "query": bm25_query,
                    "_source": source_fields,
                },
                size=top_k,
            ),
            self.client.search(
                index=settings.es_index,
                body={
                    "knn": knn_body,
                    "_source": source_fields,
                },
                size=top_k,
            ),
        )

        return self._rrf_fuse(bm25_resp, knn_resp, top_k)

    @staticmethod
    def _rrf_fuse(bm25_resp: dict, knn_resp: dict, top_k: int, k: int = 60) -> list[dict]:
        """Fuse two ranked lists using Reciprocal Rank Fusion: score = sum(1/(k+rank))."""
        docs: dict[str, dict] = {}  # _id -> source data
        scores: dict[str, float] = {}  # _id -> cumulative RRF score

        for rank, hit in enumerate(bm25_resp["hits"]["hits"], start=1):
            _id = hit["_id"]
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank)
            docs[_id] = hit["_source"]

        for rank, hit in enumerate(knn_resp["hits"]["hits"], start=1):
            _id = hit["_id"]
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank)
            docs[_id] = hit["_source"]

        ranked_ids = sorted(scores, key=lambda _id: scores[_id], reverse=True)[:top_k]

        return [
            {
                "content": docs[_id]["content"],
                "score": scores[_id],
                "metadata": docs[_id].get("metadata", {}),
                "document_id": docs[_id]["document_id"],
                "chunk_index": docs[_id]["chunk_index"],
            }
            for _id in ranked_ids
        ]

    async def update_document_tags(self, document_id: str, tags: list[str]) -> int:
        """Update tags on all chunks belonging to a document. Returns count of updated docs."""
        resp = await self.client.update_by_query(
            index=settings.es_index,
            body={
                "query": {"term": {"document_id": document_id}},
                "script": {
                    "source": "ctx._source.tags = params.tags; ctx._source.metadata.tags = params.tags",
                    "lang": "painless",
                    "params": {"tags": tags},
                },
            },
        )
        await self.client.indices.refresh(index=settings.es_index)
        return resp.get("updated", 0)

    async def list_documents(self) -> list[dict]:
        """List all unique documents with their chunk counts."""
        resp = await self.client.search(
            index=settings.es_index,
            body={
                "size": 0,
                "aggs": {
                    "documents": {
                        "terms": {"field": "document_id", "size": 10000},
                        "aggs": {
                            "doc_info": {
                                "top_hits": {
                                    "_source": ["metadata", "created_at"],
                                    "size": 1,
                                }
                            }
                        },
                    }
                },
            },
        )
        documents = []
        for bucket in resp["aggregations"]["documents"]["buckets"]:
            hit = bucket["doc_info"]["hits"]["hits"][0]["_source"]
            meta = hit.get("metadata", {})
            documents.append({
                "document_id": bucket["key"],
                "filename": meta.get("filename", "unknown"),
                "source_type": meta.get("source_type", "unknown"),
                "chunk_count": bucket["doc_count"],
                "tags": meta.get("tags", []),
                "created_at": hit.get("created_at"),
            })
        return documents

    async def get_document(self, document_id: str) -> dict | None:
        """Get details for a specific document."""
        resp = await self.client.search(
            index=settings.es_index,
            body={
                "query": {"term": {"document_id": document_id}},
                "_source": ["metadata", "created_at"],
                "size": 1,
            },
        )
        hits = resp["hits"]["hits"]
        if not hits:
            return None

        count_resp = await self.client.count(
            index=settings.es_index,
            body={"query": {"term": {"document_id": document_id}}},
        )
        meta = hits[0]["_source"].get("metadata", {})
        return {
            "document_id": document_id,
            "filename": meta.get("filename", "unknown"),
            "source_type": meta.get("source_type", "unknown"),
            "chunk_count": count_resp["count"],
            "tags": meta.get("tags", []),
            "metadata": meta,
            "created_at": hits[0]["_source"].get("created_at"),
        }

    async def get_document_chunks(self, document_id: str) -> list[dict]:
        """Get all chunks for a document, sorted by chunk_index."""
        resp = await self.client.search(
            index=settings.es_index,
            body={
                "query": {"term": {"document_id": document_id}},
                "_source": ["content", "chunk_index", "char_start", "char_end"],
                "sort": [{"chunk_index": "asc"}],
                "size": 10000,
            },
        )
        return [hit["_source"] for hit in resp["hits"]["hits"]]

    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document. Returns count of deleted docs."""
        resp = await self.client.delete_by_query(
            index=settings.es_index,
            body={"query": {"term": {"document_id": document_id}}},
        )
        await self.client.indices.refresh(index=settings.es_index)
        return resp.get("deleted", 0)

    async def get_all_embeddings_by_document(self) -> dict[str, list[list[float]]]:
        """Fetch all chunk embeddings grouped by document_id using the scroll API."""
        result: dict[str, list[list[float]]] = {}
        resp = await self.client.search(
            index=settings.es_index,
            body={
                "_source": ["document_id", "embedding"],
                "size": 500,
            },
            scroll="2m",
        )
        scroll_id = resp["_scroll_id"]
        try:
            while True:
                hits = resp["hits"]["hits"]
                if not hits:
                    break
                for hit in hits:
                    doc_id = hit["_source"]["document_id"]
                    embedding = hit["_source"]["embedding"]
                    result.setdefault(doc_id, []).append(embedding)
                resp = await self.client.scroll(scroll_id=scroll_id, scroll="2m")
                scroll_id = resp["_scroll_id"]
        finally:
            await self.client.clear_scroll(scroll_id=scroll_id)
        return result

    async def find_document_by_source(self, filename: str, source_type: str) -> str | None:
        """Find an existing document_id by filename + source_type. Returns None if not found."""
        resp = await self.client.search(
            index=settings.es_index,
            body={
                "size": 0,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"metadata.filename.keyword": filename}},
                            {"term": {"metadata.source_type.keyword": source_type}},
                        ]
                    }
                },
                "aggs": {
                    "docs": {
                        "terms": {"field": "document_id", "size": 1}
                    }
                },
            },
        )
        buckets = resp["aggregations"]["docs"]["buckets"]
        if buckets:
            return buckets[0]["key"]
        return None

    async def close(self):
        if self._client:
            await self._client.close()


es_service = ElasticsearchService()
