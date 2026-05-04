"""Pydantic models for tracevault data layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SpanModel(BaseModel):
    span_id: str
    trace_id: str
    agent_name: str
    iteration: int = 0
    input_messages: list[dict] = Field(default_factory=list)
    output_message: dict = Field(default_factory=dict)
    tool_calls: list[dict] = Field(default_factory=list)  # [{name, args, result, latency_ms}]
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tags: list[str] = Field(default_factory=list)


class TraceModel(BaseModel):
    trace_id: str
    session_id: str
    project_name: str
    user_id: str = "default"
    user_story: str = ""
    status: str = "running"  # "running" | "waiting_hitl" | "completed" | "failed"
    verdict: str | None = None
    total_tokens: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    spans: list[SpanModel] = Field(default_factory=list)


class PromptModel(BaseModel):
    name: str
    label: str = ""
    template: str
    variables: list[str] = Field(default_factory=list)
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    history: list[dict] = Field(default_factory=list)  # [{version, template, updated_at}]


class CriterionResult(BaseModel):
    name: str
    passed: bool
    score: float
    reasoning: str = ""


class EvaluationModel(BaseModel):
    eval_id: str
    trace_id: str
    session_id: str
    agent_name: str
    evaluator: str = "llm-judge"
    criteria: list[CriterionResult] = Field(default_factory=list)
    overall_score: float = 0.0  # 0.0–1.0
    verdict: str = "fail"  # "pass" | "fail"
    created_at: datetime = Field(default_factory=datetime.utcnow)
