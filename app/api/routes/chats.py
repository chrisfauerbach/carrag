from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    ChatListResponse,
    ChatSessionInfo,
    ChatDetailResponse,
    ChatMessageStored,
    ChatDeleteResponse,
    CreateChatRequest,
    CreateChatResponse,
    RenameChatRequest,
    RenameChatResponse,
)
from app.services.chat import chat_service

router = APIRouter()


@router.post("", response_model=CreateChatResponse, status_code=201)
async def create_chat(request: CreateChatRequest | None = None):
    title = request.title if request else None
    doc = await chat_service.create_chat(title=title)
    return CreateChatResponse(
        chat_id=doc["chat_id"],
        title=doc["title"],
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )


@router.get("", response_model=ChatListResponse)
async def list_chats():
    chats = await chat_service.list_chats()
    return ChatListResponse(
        chats=[ChatSessionInfo(**c) for c in chats],
        total=len(chats),
    )


@router.get("/{chat_id}", response_model=ChatDetailResponse)
async def get_chat(chat_id: str):
    chat = await chat_service.get_chat(chat_id)
    if chat is None:
        raise HTTPException(404, "Chat not found")
    return ChatDetailResponse(
        chat_id=chat["chat_id"],
        title=chat["title"],
        messages=[ChatMessageStored(**m) for m in chat.get("messages", [])],
        message_count=chat["message_count"],
        created_at=chat["created_at"],
        updated_at=chat["updated_at"],
    )


@router.patch("/{chat_id}", response_model=RenameChatResponse)
async def rename_chat(chat_id: str, request: RenameChatRequest):
    result = await chat_service.rename_chat(chat_id, request.title)
    if result is None:
        raise HTTPException(404, "Chat not found")
    return RenameChatResponse(**result)


@router.delete("/{chat_id}", response_model=ChatDeleteResponse)
async def delete_chat(chat_id: str):
    deleted = await chat_service.delete_chat(chat_id)
    if not deleted:
        raise HTTPException(404, "Chat not found")
    return ChatDeleteResponse(chat_id=chat_id)
