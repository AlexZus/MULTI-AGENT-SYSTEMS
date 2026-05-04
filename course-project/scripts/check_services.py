#!/usr/bin/env python3
"""
Check connectivity to all external services required by the application.

Usage:
    python scripts/check_services.py
    python scripts/check_services.py --json   # machine-readable output
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field

import httpx
import motor.motor_asyncio

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from config import Settings


@dataclass
class ServiceResult:
    name: str
    url: str
    ok: bool
    detail: str = ""
    error: str = ""


async def _check_llm(client: httpx.AsyncClient, settings: Settings) -> ServiceResult:
    url = f"{settings.openai_compatible_api_url}/models"
    try:
        r = await client.get(url, timeout=5.0)
        r.raise_for_status()
        data = r.json()
        models = [m.get("id", "") for m in data.get("data", [])]
        target = settings.model_name
        detail = f"{target}" if target in models else f"(available: {', '.join(models[:3])})"
        return ServiceResult("Local LLM", url, True, detail)
    except Exception as e:
        return ServiceResult("Local LLM", url, False, error=str(e))


async def _check_mcp(
    client: httpx.AsyncClient, name: str, url: str
) -> ServiceResult:
    """Ping MCP server — try GET /tools or just GET /."""
    try:
        # MCP streamable-http: OPTIONS or GET on base URL is enough to confirm it's up
        r = await client.get(url, timeout=5.0)
        # 4xx/5xx from MCP is still "up" (it's running but may reject GETs)
        detail = f"HTTP {r.status_code}"
        return ServiceResult(name, url, True, detail)
    except Exception as e:
        return ServiceResult(name, url, False, error=str(e))


async def _check_embedding(client: httpx.AsyncClient, settings: Settings) -> ServiceResult:
    url = settings.embedding_url
    try:
        # Send a minimal embedding request
        r = await client.post(
            url,
            json={"input": ["ping"], "model": "all-mpnet-base-v2"},
            timeout=10.0,
        )
        r.raise_for_status()
        data = r.json()
        dims = len(data["data"][0]["embedding"]) if data.get("data") else "?"
        return ServiceResult("Embedding Service", url, True, f"dims={dims}")
    except Exception as e:
        # Maybe the service exposes a /health endpoint
        try:
            health_url = url.rsplit("/", 2)[0] + "/health"
            r2 = await client.get(health_url, timeout=5.0)
            r2.raise_for_status()
            return ServiceResult("Embedding Service", url, True, "health OK")
        except Exception:
            return ServiceResult("Embedding Service", url, False, error=str(e))


async def _check_mongodb(settings: Settings) -> ServiceResult:
    url = settings.mongodb_url
    display_url = url.split("@")[-1] if "@" in url else url
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(
            url, serverSelectionTimeoutMS=3000
        )
        info = await client.admin.command("buildInfo")
        client.close()
        version = info.get("version", "?")
        return ServiceResult("MongoDB", f"mongodb://{display_url}", True, f"v{version}")
    except Exception as e:
        return ServiceResult("MongoDB", f"mongodb://{display_url}", False, error=str(e))


async def run_checks(settings: Settings) -> list[ServiceResult]:
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            _check_llm(client, settings),
            _check_mcp(client, "Filesystem MCP", settings.mcp_filesystem_url),
            _check_mcp(client, "Python REPL MCP", settings.mcp_repl_url),
            _check_embedding(client, settings),
            _check_mongodb(settings),
        )
    return list(results)


def _print_table(results: list[ServiceResult]) -> None:
    name_w = max(len(r.name) for r in results) + 2
    url_w = max(len(r.url) for r in results) + 2
    status_w = 20

    sep = "─" * (name_w + url_w + status_w)
    header = f"{'Service':<{name_w}}{'URL':<{url_w}}{'Status':<{status_w}}"
    print(header)
    print(sep)
    for r in results:
        if r.ok:
            status = f"✓ OK  ({r.detail})" if r.detail else "✓ OK"
        else:
            err = r.error[:40] if len(r.error) > 40 else r.error
            status = f"✗ DOWN  {err}"
        print(f"{r.name:<{name_w}}{r.url:<{url_w}}{status}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check all external service dependencies")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    settings = Settings()
    results = asyncio.run(run_checks(settings))

    if args.json:
        print(json.dumps([
            {"name": r.name, "url": r.url, "ok": r.ok, "detail": r.detail, "error": r.error}
            for r in results
        ], indent=2))
    else:
        _print_table(results)

    all_ok = all(r.ok for r in results)
    if not all_ok:
        down = [r.name for r in results if not r.ok]
        print(f"\nWARNING: {len(down)} service(s) unreachable: {', '.join(down)}")
        return 1
    print("\nAll services OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
