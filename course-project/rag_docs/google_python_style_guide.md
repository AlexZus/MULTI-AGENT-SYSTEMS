# Google Python Style Guide — Key Rules

## Imports

- Use `import x` for packages and modules.
- Use `from x import y` where `y` is the name to use.
- Use `from x import y as z` if `y` conflicts with a local name.
- Do not use relative names in imports.

```python
# Correct
import os
import sys
from typing import Any
from mypackage import mymodule

# Wrong
import mypackage.mymodule  # ok when needed for clarity
from mypackage import *    # never
```

## Naming

| Type | Convention | Example |
|------|-----------|---------|
| Package/module | lowercase_with_underscores | `my_module` |
| Class | CapWords | `MyClass` |
| Exception | CapWords + Error suffix | `ValueError` |
| Function/variable | lowercase_with_underscores | `get_user` |
| Constant | UPPER_CASE | `MAX_SIZE` |
| Private | Leading underscore | `_internal` |

## Docstrings

```python
def fetch_data(url: str, timeout: float = 5.0) -> dict:
    """Fetch JSON data from a URL.

    Args:
        url: The URL to fetch from.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response as a dictionary.

    Raises:
        httpx.HTTPError: If the request fails.
    """
```

## Type Annotations

All public APIs should be annotated:

```python
def process(items: list[str], *, max_count: int = 10) -> list[str]:
    ...

async def fetch(url: str) -> bytes | None:
    ...
```

## String Formatting

Prefer f-strings for interpolation:

```python
# Preferred
message = f"Hello, {name}! You have {count} messages."

# Acceptable for logging (lazy evaluation)
logger.info("User %s logged in", user_id)
```

## Exceptions

```python
# Catch specific exceptions
try:
    result = risky_operation()
except ValueError as e:
    logger.warning("Invalid value: %s", e)
    raise
except (TypeError, KeyError):
    raise RuntimeError("Unexpected data format") from None

# Don't swallow exceptions silently
try:
    ...
except Exception:
    pass  # WRONG — at minimum log it
```

## Comprehensions

Use list/dict/set comprehensions for simple transformations. Avoid nesting more than 2 levels:

```python
# Good
squares = [x**2 for x in range(10) if x % 2 == 0]

# Too complex — use a loop
result = [f(x) for outer in matrix for x in outer if predicate(x)]
```

## Default Arguments

Never use mutable objects as default arguments:

```python
# Wrong
def append_to(element, to=[]):
    to.append(element)

# Correct
def append_to(element, to=None):
    if to is None:
        to = []
    to.append(element)
```

## Context Managers

Always use `with` for resources that need cleanup:

```python
with open("file.txt") as f:
    data = f.read()

async with aiohttp.ClientSession() as session:
    resp = await session.get(url)
```
