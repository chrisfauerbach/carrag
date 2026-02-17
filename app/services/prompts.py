import logging
from datetime import datetime, timezone

from elasticsearch import NotFoundError

from app.config import settings
from app.services.elasticsearch import es_service

logger = logging.getLogger(__name__)

PROMPTS_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "key": {"type": "keyword"},
            "name": {"type": "text"},
            "description": {"type": "text"},
            "content": {"type": "text"},
            "variables": {"type": "keyword"},
            "default_content": {"type": "text"},
            "updated_at": {"type": "date"},
        }
    }
}

DEFAULT_PROMPTS = {
    "rag_system": {
        "key": "rag_system",
        "name": "RAG System Prompt",
        "description": "Instructions for how the LLM answers RAG queries",
        "content": (
            "You are a helpful assistant that answers questions based on the provided context.\n"
            "Use ONLY the context below to answer the question. If the context doesn't contain "
            "enough information to answer, say so clearly.\n"
            "Always cite which source(s) you used in your answer."
        ),
        "variables": [],
    },
    "rag_user": {
        "key": "rag_user",
        "name": "RAG User Template",
        "description": "Template for the user message sent to the LLM during RAG queries",
        "content": (
            "Context:\n{context}\n{history_block}\n"
            "Question: {question}\n\n"
            "Answer based on the context above:"
        ),
        "variables": ["context", "history_block", "question"],
    },
    "autotag_system": {
        "key": "autotag_system",
        "name": "Auto-Tag System Prompt",
        "description": "Instructions for LLM-based automatic document tagging",
        "content": (
            "You are a tagging assistant for car manuals and automotive documents. "
            "Return ONLY a comma-separated list of short, lowercase tags. "
            "No numbering, no explanation, no extra text. "
            "ALWAYS include the vehicle make (e.g. ford, toyota, bmw) and model "
            "(e.g. f-150, camry, 3 series) as separate tags if they can be identified. "
            "Also include the model year if present. "
            "Fill remaining tags with other useful descriptors like document type "
            "(e.g. owners manual, service manual, wiring diagram) or key topics."
        ),
        "variables": [],
    },
    "autotag_user": {
        "key": "autotag_user",
        "name": "Auto-Tag User Template",
        "description": "Template for the user message sent to the LLM during auto-tagging",
        "content": (
            "Generate up to {max_tags} short descriptive tags for this automotive "
            "document:\n\n{filename_hint}{truncated}"
        ),
        "variables": ["max_tags", "filename_hint", "truncated"],
    },
}


class PromptsService:
    @property
    def client(self):
        return es_service.client

    async def init_index(self):
        exists = await self.client.indices.exists(index=settings.es_prompts_index)
        if not exists:
            await self.client.indices.create(
                index=settings.es_prompts_index, body=PROMPTS_INDEX_MAPPING
            )
            logger.info(f"Created prompts index: {settings.es_prompts_index}")

        # Seed defaults if index is empty
        resp = await self.client.count(index=settings.es_prompts_index)
        if resp["count"] == 0:
            now = datetime.now(timezone.utc).isoformat()
            for key, prompt in DEFAULT_PROMPTS.items():
                doc = {
                    **prompt,
                    "default_content": prompt["content"],
                    "updated_at": now,
                }
                await self.client.index(
                    index=settings.es_prompts_index, id=key, body=doc, refresh="wait_for"
                )
            logger.info("Seeded default prompts into ES")
        else:
            logger.info(f"Prompts index {settings.es_prompts_index} already has data.")

    async def get_prompt(self, key: str) -> dict | None:
        try:
            resp = await self.client.get(index=settings.es_prompts_index, id=key)
            return resp["_source"]
        except NotFoundError:
            return None

    async def list_prompts(self) -> list[dict]:
        resp = await self.client.search(
            index=settings.es_prompts_index,
            body={"query": {"match_all": {}}, "size": 10},
        )
        return [hit["_source"] for hit in resp["hits"]["hits"]]

    async def update_prompt(self, key: str, content: str) -> dict | None:
        existing = await self.get_prompt(key)
        if existing is None:
            return None

        now = datetime.now(timezone.utc).isoformat()
        await self.client.update(
            index=settings.es_prompts_index,
            id=key,
            body={"doc": {"content": content, "updated_at": now}},
            refresh="wait_for",
        )
        return {"key": key, "content": content, "updated_at": now}

    async def reset_prompt(self, key: str) -> dict | None:
        if key not in DEFAULT_PROMPTS:
            return None

        now = datetime.now(timezone.utc).isoformat()
        default = DEFAULT_PROMPTS[key]
        await self.client.update(
            index=settings.es_prompts_index,
            id=key,
            body={"doc": {"content": default["content"], "updated_at": now}},
            refresh="wait_for",
        )
        return {"key": key, "content": default["content"], "updated_at": now}


prompts_service = PromptsService()
