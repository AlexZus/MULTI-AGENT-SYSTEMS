# FastAPI Tutorial

## Overview

FastAPI is a modern, fast web framework for building APIs with Python 3.7+. It is based on Pydantic and Starlette.

Key features:
- Automatic OpenAPI documentation
- Type hints → automatic validation and serialization
- Async support out of the box
- Dependency injection

## Basic App

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.get("/items/{item_id}")
async def read_item(item_id: int) -> Item:
    return Item(name="Foo", price=1.5)

@app.post("/items/")
async def create_item(item: Item) -> Item:
    return item
```

## Routers

Use `APIRouter` to organize endpoints:

```python
from fastapi import APIRouter

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}

# In main app:
app.include_router(router)
```

## Dependencies

```python
from fastapi import Depends

async def get_db():
    db = Database()
    try:
        yield db
    finally:
        await db.close()

@app.get("/items/")
async def list_items(db = Depends(get_db)):
    return await db.find_all()
```

## Lifespan (startup/shutdown)

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    await connect_db()
    yield
    # shutdown
    await disconnect_db()

app = FastAPI(lifespan=lifespan)
```

## SSE (Server-Sent Events)

```python
from fastapi.responses import StreamingResponse
import asyncio

@app.get("/events")
async def events():
    async def generator():
        while True:
            yield f"data: {get_data()}\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(generator(), media_type="text/event-stream")
```

## Static Files and Templates

```python
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
```

## Error Handling

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return items[item_id]
```
