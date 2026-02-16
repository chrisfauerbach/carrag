import logging
import uuid
from datetime import datetime, timezone

from elasticsearch import NotFoundError

from app.config import settings
from app.services.elasticsearch import es_service

logger = logging.getLogger(__name__)

CHAT_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "chat_id": {"type": "keyword"},
            "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "messages": {"type": "object", "enabled": False},
            "message_count": {"type": "integer"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
        }
    }
}


class ChatService:
    @property
    def client(self):
        return es_service.client

    async def init_index(self):
        exists = await self.client.indices.exists(index=settings.es_chat_index)
        if not exists:
            await self.client.indices.create(
                index=settings.es_chat_index, body=CHAT_INDEX_MAPPING
            )
            logger.info(f"Created chat index: {settings.es_chat_index}")
        else:
            logger.info(f"Chat index {settings.es_chat_index} already exists.")

    async def create_chat(self, title: str | None = None) -> dict:
        chat_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        doc = {
            "chat_id": chat_id,
            "title": title or "New Chat",
            "messages": [],
            "message_count": 0,
            "created_at": now,
            "updated_at": now,
        }
        await self.client.index(
            index=settings.es_chat_index, id=chat_id, body=doc, refresh="wait_for"
        )
        return doc

    async def get_chat(self, chat_id: str) -> dict | None:
        try:
            resp = await self.client.get(index=settings.es_chat_index, id=chat_id)
            return resp["_source"]
        except NotFoundError:
            return None

    async def list_chats(self) -> list[dict]:
        resp = await self.client.search(
            index=settings.es_chat_index,
            body={
                "_source": ["chat_id", "title", "message_count", "created_at", "updated_at"],
                "sort": [{"updated_at": "desc"}],
                "size": 100,
            },
        )
        return [hit["_source"] for hit in resp["hits"]["hits"]]

    async def append_messages(self, chat_id: str, new_messages: list[dict]) -> dict | None:
        chat = await self.get_chat(chat_id)
        if chat is None:
            return None

        now = datetime.now(timezone.utc).isoformat()
        messages = chat.get("messages", [])
        messages.extend(new_messages)

        # Auto-title from first user message when title is still default
        title = chat["title"]
        if title == "New Chat":
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg["content"].strip()
                    title = content[:60] + ("..." if len(content) > 60 else "")
                    break

        await self.client.index(
            index=settings.es_chat_index,
            id=chat_id,
            body={
                **chat,
                "messages": messages,
                "message_count": len(messages),
                "title": title,
                "updated_at": now,
            },
            refresh="wait_for",
        )

        return await self.get_chat(chat_id)

    async def rename_chat(self, chat_id: str, title: str) -> dict | None:
        chat = await self.get_chat(chat_id)
        if chat is None:
            return None

        now = datetime.now(timezone.utc).isoformat()
        await self.client.update(
            index=settings.es_chat_index,
            id=chat_id,
            body={"doc": {"title": title, "updated_at": now}},
            refresh="wait_for",
        )
        return {"chat_id": chat_id, "title": title, "updated_at": now}

    async def delete_chat(self, chat_id: str) -> bool:
        try:
            await self.client.delete(
                index=settings.es_chat_index, id=chat_id, refresh="wait_for"
            )
            return True
        except NotFoundError:
            return False


chat_service = ChatService()
