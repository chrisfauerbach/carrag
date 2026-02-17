from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    PromptInfo,
    PromptListResponse,
    UpdatePromptRequest,
    PromptUpdateResponse,
    PromptResetResponse,
)
from app.services.prompts import prompts_service

router = APIRouter()


@router.get("", response_model=PromptListResponse)
async def list_prompts():
    prompts = await prompts_service.list_prompts()
    return PromptListResponse(
        prompts=[PromptInfo(**p) for p in prompts],
        total=len(prompts),
    )


@router.get("/{key}", response_model=PromptInfo)
async def get_prompt(key: str):
    prompt = await prompts_service.get_prompt(key)
    if prompt is None:
        raise HTTPException(404, "Prompt not found")
    return PromptInfo(**prompt)


@router.patch("/{key}", response_model=PromptUpdateResponse)
async def update_prompt(key: str, request: UpdatePromptRequest):
    result = await prompts_service.update_prompt(key, request.content)
    if result is None:
        raise HTTPException(404, "Prompt not found")
    return PromptUpdateResponse(**result)


@router.post("/{key}/reset", response_model=PromptResetResponse)
async def reset_prompt(key: str):
    result = await prompts_service.reset_prompt(key)
    if result is None:
        raise HTTPException(404, "Prompt not found")
    return PromptResetResponse(**result)
