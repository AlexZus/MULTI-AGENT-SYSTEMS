# Python Standard Library Reference

## pathlib — File System Paths

```python
from pathlib import Path

# Create path objects
p = Path("src/main.py")
p = Path.home() / ".config" / "app.json"

# Operations
p.exists()          # True/False
p.is_file()         # True/False
p.is_dir()          # True/False
p.mkdir(parents=True, exist_ok=True)
p.read_text()       # read contents as string
p.write_text("...")  # write string to file
p.stat().st_mtime   # modification time (float)
list(p.glob("*.py"))  # glob matching
p.stem              # filename without extension
p.suffix            # ".py"
p.parent            # parent directory Path
```

## asyncio — Async I/O

```python
import asyncio

# Run async functions
asyncio.run(main())

# Gather concurrent tasks
results = await asyncio.gather(task1(), task2(), task3())

# Create tasks
task = asyncio.create_task(coro())
result = await task

# Run sync code in thread pool
result = await asyncio.to_thread(blocking_func, arg)

# Timeout
async with asyncio.timeout(5.0):
    await slow_operation()

# Queue
queue = asyncio.Queue(maxsize=100)
await queue.put(item)
item = await queue.get()
item = queue.get_nowait()  # raises QueueEmpty if empty
```

## dataclasses

```python
from dataclasses import dataclass, field

@dataclass
class Point:
    x: float
    y: float
    label: str = ""
    tags: list = field(default_factory=list)
```

## json

```python
import json

# Serialize
text = json.dumps(obj, indent=2, ensure_ascii=False)
# Deserialize
obj = json.loads(text)
# File I/O
with open("data.json", "w") as f:
    json.dump(obj, f)
with open("data.json") as f:
    obj = json.load(f)
```

## typing

```python
from typing import Any, Callable, Awaitable, Generator, AsyncGenerator

# Common aliases
list[str]           # Python 3.9+
dict[str, Any]
tuple[int, ...]
str | None          # Python 3.10+
Callable[[str], int]
Awaitable[str]
AsyncGenerator[str, None]
```

## contextlib

```python
from contextlib import asynccontextmanager, contextmanager

@asynccontextmanager
async def managed_resource():
    resource = await acquire()
    try:
        yield resource
    finally:
        await release(resource)

# AsyncExitStack for dynamic context managers
from contextlib import AsyncExitStack
async with AsyncExitStack() as stack:
    a = await stack.enter_async_context(ctx_a())
    b = await stack.enter_async_context(ctx_b())
```

## uuid

```python
import uuid

uid = uuid.uuid4()          # random UUID
uid_hex = uuid.uuid4().hex  # 32-char hex string
```

## datetime

```python
from datetime import datetime, timezone

now = datetime.now(timezone.utc)  # timezone-aware UTC
iso = now.isoformat()             # "2026-04-24T10:00:00+00:00"
dt = datetime.fromisoformat(iso)
```
