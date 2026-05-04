"""MongoDB data stores for tracevault."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import motor.motor_asyncio

from tracevault.models import EvaluationModel, PromptModel, SpanModel, TraceModel


class TraceStore:
    def __init__(self, db: motor.motor_asyncio.AsyncIOMotorDatabase) -> None:
        self._col = db["traces"]

    async def create_trace(self, trace: TraceModel) -> None:
        await self._col.insert_one(trace.model_dump())

    async def update_trace(self, trace_id: str, **fields: Any) -> None:
        fields["updated_at"] = datetime.utcnow()
        await self._col.update_one({"trace_id": trace_id}, {"$set": fields})

    async def add_span(self, trace_id: str, span: SpanModel) -> None:
        await self._col.update_one(
            {"trace_id": trace_id},
            {
                "$push": {"spans": span.model_dump()},
                "$inc": {"total_tokens": span.input_tokens + span.output_tokens},
                "$set": {"updated_at": datetime.utcnow()},
            },
        )

    async def get_trace(self, trace_id: str) -> TraceModel | None:
        doc = await self._col.find_one({"trace_id": trace_id}, {"_id": 0})
        return TraceModel(**doc) if doc else None

    async def list_traces(
        self,
        *,
        agent_name: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
        project_name: str | None = None,
        limit: int = 50,
    ) -> list[TraceModel]:
        query: dict = {}
        if status:
            query["status"] = status
        if session_id:
            query["session_id"] = session_id
        if project_name:
            query["project_name"] = project_name
        if agent_name:
            query["spans.agent_name"] = agent_name

        cursor = self._col.find(query, {"_id": 0}).sort("created_at", -1).limit(limit)
        return [TraceModel(**doc) async for doc in cursor]

    async def list_sessions(self) -> list[dict]:
        """Aggregate per-session stats."""
        pipeline = [
            {
                "$group": {
                    "_id": "$session_id",
                    "project_name": {"$first": "$project_name"},
                    "traces_count": {"$sum": 1},
                    "total_tokens": {"$sum": "$total_tokens"},
                    "completed": {
                        "$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}
                    },
                    "last_activity": {"$max": "$updated_at"},
                }
            },
            {"$sort": {"last_activity": -1}},
        ]
        cursor = self._col.aggregate(pipeline)
        results = []
        async for doc in cursor:
            total = doc["traces_count"]
            results.append(
                {
                    "session_id": doc["_id"],
                    "project_name": doc.get("project_name", ""),
                    "traces_count": total,
                    "total_tokens": doc["total_tokens"],
                    "success_rate": round(doc["completed"] / total, 2) if total else 0.0,
                    "last_activity": doc["last_activity"],
                }
            )
        return results


class PromptStore:
    def __init__(self, db: motor.motor_asyncio.AsyncIOMotorDatabase) -> None:
        self._col = db["prompts"]

    async def get_prompt(self, name: str) -> PromptModel | None:
        doc = await self._col.find_one({"name": name}, {"_id": 0})
        return PromptModel(**doc) if doc else None

    async def upsert_prompt(self, prompt: PromptModel) -> PromptModel:
        existing = await self.get_prompt(prompt.name)
        if existing:
            # Archive current version to history
            history_entry = {
                "version": existing.version,
                "template": existing.template,
                "label": existing.label,
                "updated_at": existing.updated_at.isoformat(),
            }
            new_version = existing.version + 1
            history = existing.history + [history_entry]
            prompt = prompt.model_copy(
                update={
                    "version": new_version,
                    "history": history,
                    "created_at": existing.created_at,
                    "updated_at": datetime.utcnow(),
                }
            )
            await self._col.replace_one({"name": prompt.name}, prompt.model_dump())
        else:
            prompt = prompt.model_copy(update={"updated_at": datetime.utcnow()})
            await self._col.insert_one(prompt.model_dump())
        return prompt

    async def list_prompts(self) -> list[PromptModel]:
        cursor = self._col.find({}, {"_id": 0}).sort("name", 1)
        return [PromptModel(**doc) async for doc in cursor]

    async def get_prompt_history(self, name: str) -> list[dict]:
        doc = await self._col.find_one({"name": name}, {"_id": 0, "history": 1})
        return doc.get("history", []) if doc else []

    async def rollback_prompt(self, name: str, version: int) -> PromptModel | None:
        """Copy old template as a new version (history stays linear)."""
        existing = await self.get_prompt(name)
        if not existing:
            return None
        # Find the target version in history
        target_template = None
        target_label = existing.label
        for entry in existing.history:
            if entry.get("version") == version:
                target_template = entry["template"]
                target_label = entry.get("label", existing.label)
                break
        if target_template is None:
            return None  # version not found in history
        rolled_back = existing.model_copy(
            update={"template": target_template, "label": target_label}
        )
        return await self.upsert_prompt(rolled_back)

    async def delete_prompt(self, name: str) -> bool:
        result = await self._col.delete_one({"name": name})
        return result.deleted_count > 0


class EvaluationStore:
    def __init__(self, db: motor.motor_asyncio.AsyncIOMotorDatabase) -> None:
        self._col = db["evaluations"]

    async def save_evaluation(self, evaluation: EvaluationModel) -> None:
        await self._col.insert_one(evaluation.model_dump())

    async def list_evaluations(
        self,
        *,
        agent_name: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[EvaluationModel]:
        query: dict = {}
        if agent_name:
            query["agent_name"] = agent_name
        if session_id:
            query["session_id"] = session_id
        cursor = self._col.find(query, {"_id": 0}).sort("created_at", -1).limit(limit)
        return [EvaluationModel(**doc) async for doc in cursor]

    async def get_agent_stats(self, agent_name: str | None = None) -> list[dict]:
        """Return per-agent aggregate stats: avg_score, pass_rate, count."""
        match: dict = {}
        if agent_name:
            match["agent_name"] = agent_name

        pipeline: list[dict] = []
        if match:
            pipeline.append({"$match": match})
        pipeline += [
            {
                "$group": {
                    "_id": "$agent_name",
                    "count": {"$sum": 1},
                    "avg_score": {"$avg": "$overall_score"},
                    "passes": {
                        "$sum": {"$cond": [{"$eq": ["$verdict", "pass"]}, 1, 0]}
                    },
                }
            },
            {"$sort": {"_id": 1}},
        ]

        cursor = self._col.aggregate(pipeline)
        results = []
        async for doc in cursor:
            count = doc["count"]
            results.append(
                {
                    "agent_name": doc["_id"],
                    "count": count,
                    "avg_score": round(doc["avg_score"] or 0.0, 3),
                    "pass_rate": round(doc["passes"] / count, 3) if count else 0.0,
                }
            )
        return results
