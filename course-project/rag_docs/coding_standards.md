# Coding Standards

## Python Style

Follow the Google Python Style Guide. Key rules:

- Use 4 spaces for indentation, never tabs.
- Maximum line length: 100 characters.
- All public functions, classes, and methods must have docstrings.
- Use type annotations on all function signatures.
- Prefer `f-strings` over `.format()` or `%` formatting.

## File Organization

- One class per file for large classes; small related helpers may share a file.
- `__init__.py` should only contain imports, not logic.
- Tests go in a `tests/` directory mirroring the source layout.

## Naming Conventions

- `snake_case` for variables, functions, and module names.
- `PascalCase` for class names.
- `UPPER_SNAKE_CASE` for module-level constants.
- Prefix private attributes/methods with a single underscore `_`.

## Error Handling

- Use specific exception types, not bare `except:` clauses.
- Always log exceptions before re-raising or handling them.
- Define custom exceptions in a dedicated `exceptions.py` module.
- Validate inputs at system boundaries (user input, external API responses).

## Testing

- Use `pytest` for all tests.
- Unit tests must not touch the filesystem, network, or database (use mocks/fixtures).
- Integration tests may use real services but must clean up after themselves.
- Aim for at least one test per public function.
- Use `pytest.mark.asyncio` for async tests.

## Imports

- Standard library first, then third-party, then local — separated by blank lines.
- Use absolute imports, not relative imports (except in packages for clarity).
- Never use wildcard imports (`from module import *`).

## FastAPI Best Practices

- Use Pydantic models for all request and response bodies.
- Return appropriate HTTP status codes (404 for not found, 422 for validation errors, etc.).
- Use dependency injection (`Depends`) for shared resources (database, settings).
- All endpoints should be async.
- Use `APIRouter` to group related endpoints; include routers in the main `app`.

## Async Python

- Prefer `async/await` over callbacks.
- Use `asyncio.gather()` for parallel async tasks.
- Never block the event loop with synchronous I/O — use `asyncio.to_thread()` or `run_in_executor()`.
- Use `asynccontextmanager` for async context managers.
