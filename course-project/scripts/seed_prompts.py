#!/usr/bin/env python3
"""Seed prompt .md files into tracevault MongoDB.

Usage:
    python scripts/seed_prompts.py [--force]

--force: re-seed even if prompts already exist (will increment versions).
"""

import asyncio
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import motor.motor_asyncio

from tracevault.store import PromptStore
from tracevault.prompts import seed_from_files


async def main(force: bool = False) -> None:
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://admin:admin_password@172.20.0.1:27017")
    mongodb_db = os.getenv("MONGODB_DB", "course_project")

    client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_url)
    db = client[mongodb_db]
    store = PromptStore(db)

    prompts_dir = Path(__file__).parent.parent / "prompts"
    if not prompts_dir.exists():
        print(f"ERROR: prompts directory not found: {prompts_dir}")
        sys.exit(1)

    if force:
        # Drop existing prompts
        await db["prompts"].drop()
        print("Dropped existing prompts collection.")

    await seed_from_files(prompts_dir, store)
    prompts = await store.list_prompts()
    print(f"Seeded {len(prompts)} prompts:")
    for p in prompts:
        print(f"  {p.name}  v{p.version}  ({len(p.template)} chars)")

    client.close()


if __name__ == "__main__":
    force = "--force" in sys.argv
    asyncio.run(main(force=force))
