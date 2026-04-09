"""Shared fixtures for DeepEval tests.

All tests use the same LLM  via local OpenAI-compatible API)
for both the agents under test and the DeepEval judge metrics.
"""

import json
import os
import sys

import pytest

# Make the project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.models import GPTModel

from config import Settings

_settings = Settings()


# ---------------------------------------------------------------------------
# Judge model — same local model used by agents
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def judge():
    """DeepEval-compatible GPTModel pointing to the local LLM server."""
    return GPTModel(
        model=_settings.model_name,
        api_key=_settings.api_key,
        base_url=_settings.openai_compatible_api_url,
    )


# ---------------------------------------------------------------------------
# Pre-built fixture data (avoids expensive LLM calls in component tests)
# ---------------------------------------------------------------------------

SAMPLE_PLAN_JSON = json.dumps({
    "goal": "Explain the main architecture patterns for AI-powered Telegram bots",
    "search_queries": [
        "Telegram bot architecture patterns LLM integration",
        "RAG Telegram bot implementation",
        "Telegram webhook vs polling comparison",
        "aiogram python-telegram-bot framework comparison",
    ],
    "sources_to_check": ["knowledge_base", "web"],
    "output_format": "Structured Markdown report with sections per architecture pattern and a comparison table",
})

SAMPLE_FINDINGS = """# AI-Powered Telegram Bot Architectures

## Overview
Telegram bots can be built with varying levels of AI integration, from simple LLM chatbots
to full multi-agent RAG systems.

## Architecture Patterns

### Pattern 1: Simple LLM Chatbot
Connects user messages directly to an LLM API (OpenAI, Anthropic). Suitable for FAQ bots.
No persistent state or retrieval.
**Source:** telegram_ai_bots.txt, page 1

### Pattern 2: Stateful Conversation Bot
Maintains conversation history in Redis or a database. Enables multi-turn interactions.
**Source:** telegram_ai_bots.txt, page 1

### Pattern 3: RAG Bot
Retrieves relevant documents from a vector store before calling the LLM. Grounded in
specific knowledge bases. Uses FAISS or Pinecone for storage.
**Source:** telegram_ai_bots.txt, page 2

### Pattern 4: Multi-Agent Bot
Orchestrates several agents (planner, researcher, critic) to handle complex research tasks.
**Source:** telegram_ai_bots.txt, page 3

## Webhook vs Long Polling
- **Webhook:** Server receives push updates; requires HTTPS public endpoint; efficient for production.
- **Long Polling:** Bot requests updates periodically; simpler setup; suitable for development.
**Source:** telegram_bots_guide.txt, page 1

## Rate Limits
Telegram enforces 30 messages/second globally and 1 message/second per chat.
Exponential backoff is recommended on 429 errors.
**Source:** telegram_bot_api_reference.txt, page 2
"""

SAMPLE_FINDINGS_POOR = """Telegram bots are good. You can use them for AI stuff.
There are different ways to make them. Some use Python, some use Node.
They work with webhooks or polling.
"""

SAMPLE_CRITIQUE_APPROVE = json.dumps({
    "verdict": "APPROVE",
    "is_fresh": True,
    "is_complete": True,
    "is_well_structured": True,
    "strengths": [
        "Covers all four architecture patterns with clear descriptions",
        "Includes comparison of webhook vs long polling",
        "All claims are backed by source citations",
        "Rate limit information is specific and actionable",
    ],
    "gaps": [],
    "revision_requests": [],
})

SAMPLE_CRITIQUE_REVISE = json.dumps({
    "verdict": "REVISE",
    "is_fresh": False,
    "is_complete": False,
    "is_well_structured": False,
    "strengths": ["Mentions webhooks and polling"],
    "gaps": [
        "No citations for any claims",
        "Missing architecture patterns (RAG, multi-agent)",
        "No specific technical details or code examples",
        "Insufficient depth on any topic",
    ],
    "revision_requests": [
        "Add citations for all factual claims (filename + page or URL)",
        "Expand architecture patterns section with at least 3 distinct patterns",
        "Include webhook vs long polling technical comparison",
        "Add rate limit specifics from the API reference",
    ],
})


@pytest.fixture
def sample_plan_json():
    return SAMPLE_PLAN_JSON


@pytest.fixture
def sample_findings():
    return SAMPLE_FINDINGS


@pytest.fixture
def sample_findings_poor():
    return SAMPLE_FINDINGS_POOR


@pytest.fixture
def sample_critique_approve():
    return SAMPLE_CRITIQUE_APPROVE


@pytest.fixture
def sample_critique_revise():
    return SAMPLE_CRITIQUE_REVISE


@pytest.fixture
def golden_dataset():
    dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)
