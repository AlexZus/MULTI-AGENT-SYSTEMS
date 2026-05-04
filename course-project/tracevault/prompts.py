"""Prompt loading helpers — fetches from MongoDB via PromptStore at runtime."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tracevault.store import PromptStore

# Module-level store reference set by server.py on startup
_prompt_store: "PromptStore | None" = None


def set_prompt_store(store: "PromptStore") -> None:
    global _prompt_store
    _prompt_store = store


async def load_prompt(name: str, **variables: str) -> str:
    """Load a prompt template from MongoDB and interpolate variables.

    Falls back to an empty string if the prompt is not found.
    """
    if _prompt_store is None:
        raise RuntimeError("Prompt store not initialised — call set_prompt_store() first")
    prompt = await _prompt_store.get_prompt(name)
    if prompt is None:
        return ""
    template = prompt.template
    for key, value in variables.items():
        template = template.replace(f"{{{key}}}", value)
    return template


async def seed_from_files(prompts_dir: Path, store: "PromptStore") -> None:
    """Load all .md files from prompts_dir into the PromptStore.

    Only inserts prompts that do not already exist (idempotent).
    """
    from tracevault.models import PromptModel

    for md_file in sorted(prompts_dir.glob("*.md")):
        name = md_file.stem
        existing = await store.get_prompt(name)
        if existing is not None:
            continue  # already seeded
        template = md_file.read_text(encoding="utf-8")
        # Extract variable names: {variable_name}
        variables = list(dict.fromkeys(re.findall(r"\{(\w+)\}", template)))
        prompt = PromptModel(
            name=name,
            label=name.replace("_", " ").title(),
            template=template,
            variables=variables,
        )
        await store.upsert_prompt(prompt)
