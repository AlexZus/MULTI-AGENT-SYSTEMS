"""Prompt CRUD API routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from tracevault.models import PromptModel
from tracevault.store import PromptStore

router = APIRouter(prefix="/api/prompts", tags=["prompts"])


def get_prompt_store() -> PromptStore:
    from tracevault.server import prompt_store
    return prompt_store


class PromptUpdateRequest(BaseModel):
    template: str
    label: str = ""
    variables: list[str] = []


@router.get("", response_model=list[PromptModel])
async def list_prompts(store: PromptStore = Depends(get_prompt_store)):
    return await store.list_prompts()


@router.get("/{name}", response_model=PromptModel)
async def get_prompt(name: str, store: PromptStore = Depends(get_prompt_store)):
    prompt = await store.get_prompt(name)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt


@router.put("/{name}", response_model=PromptModel)
async def update_prompt(
    name: str,
    body: PromptUpdateRequest,
    store: PromptStore = Depends(get_prompt_store),
):
    existing = await store.get_prompt(name)
    if existing:
        updated = existing.model_copy(
            update={
                "template": body.template,
                "label": body.label or existing.label,
                "variables": body.variables or existing.variables,
            }
        )
    else:
        from tracevault.models import PromptModel as PM
        updated = PM(name=name, template=body.template, label=body.label, variables=body.variables)
    return await store.upsert_prompt(updated)


@router.delete("/{name}")
async def delete_prompt(name: str, store: PromptStore = Depends(get_prompt_store)):
    deleted = await store.delete_prompt(name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"deleted": name}


@router.post("/{name}/rollback/{version}", response_model=PromptModel)
async def rollback_prompt(
    name: str,
    version: int,
    store: PromptStore = Depends(get_prompt_store),
):
    result = await store.rollback_prompt(name, version)
    if result is None:
        raise HTTPException(status_code=404, detail="Prompt or version not found")
    return result
