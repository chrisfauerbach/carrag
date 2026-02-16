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
    ) -> int:
        """Bulk-index chunks with their embeddings.

        Each chunk dict must have: text, document_id, chunk_index, char_start, char_end.
        Returns the number of successfully indexed documents.
        """
        now = datetime.now(timezone.utc).isoformat()

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
                        "created_at": now,
                    },
                }

        success, _ = await async_bulk(self.client, gen_actions())
        await self.client.indices.refresh(index=settings.es_index)
        return success

    async def knn_search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        """Find the top-k most similar chunks using kNN search."""
        resp = await self.client.search(
            index=settings.es_index,
            body={
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": top_k * 10,
                },
                "_source": ["content", "document_id", "chunk_index", "metadata", "created_at"],
            },
            size=top_k,
        )
        results = []
        for hit in resp["hits"]["hits"]:
            results.append({
                "content": hit["_source"]["content"],
                "score": hit["_score"],
                "metadata": hit["_source"].get("metadata", {}),
                "document_id": hit["_source"]["document_id"],
                "chunk_index": hit["_source"]["chunk_index"],
            })
        return results

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
            "metadata": meta,
            "created_at": hits[0]["_source"].get("created_at"),
        }

    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document. Returns count of deleted docs."""
        resp = await self.client.delete_by_query(
            index=settings.es_index,
            body={"query": {"term": {"document_id": document_id}}},
        )
        await self.client.indices.refresh(index=settings.es_index)
        return resp.get("deleted", 0)

    async def close(self):
        if self._client:
            await self._client.close()


es_service = ElasticsearchService()
