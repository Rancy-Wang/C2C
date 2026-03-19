from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union
from rosetta.context.sglang_patch.contextual_system import ContextualSystem

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("api_server")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_SYSTEM: Optional["ContextualSystem"] = None


def get_system():
    if _SYSTEM is None:
        raise HTTPException(status_code=503, detail="ContextualSystem is not initialized")
    return _SYSTEM


# ---------------------------------------------------------------------------
# Pydantic data models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class GenerateRequest(BaseModel):
    """Simple /generate endpoint body."""
    prompt: Optional[str] = None
    messages: Optional[List[Message]] = None
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    enable_thinking: bool = False


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible /v1/chat/completions request."""
    model: str = ""
    messages: List[Message]

    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    n: int = 1
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    enable_thinking: bool = False
    tools: Optional[List[Dict[str, Any]]] = None

    # --- Custom extension for discardable KV cache ---
    # Accepts either:
    #   - a Dict[int, List[int]] mapping trigger_message_id -> list of msg ids to drop
    #   - a plain List[int] of message ids to drop unconditionally
    #   - a bool (True = auto-drop heuristic, False = no drop)
    #   - null / omitted = no drop
    drop_message: Optional[Union[Dict[int, List[int]], List[int], bool]] = None


class CompletionRequest(BaseModel):
    """OpenAI-compatible /v1/completions request (prompt-based)."""
    model: str = ""
    prompt: str

    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    n: int = 1
    stream: bool = False
    stop: Optional[List[str]] = None

    enable_thinking: bool = False

    drop_message: Optional[Union[Dict[int, List[int]], List[int], bool]] = None


# --- Response models (OpenAI-compatible) ---

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:12]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[CompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "contextual-system"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_drop_message(
    drop_message: Optional[Union[Dict[int, List[int]], List[int], bool]],
) -> Optional[Dict[int, List[int]]]:
    """Convert the flexible ``drop_message`` field to the Dict[int, List[int]]
    format expected by ContextualSystem, or None."""
    if drop_message is None or drop_message is False:
        return None
    if drop_message is True:
        # True means "let the system decide" – pass empty dict so the
        # backend can apply its own heuristic.
        return {}
    if isinstance(drop_message, list):
        # Treat as unconditional drop: trigger at message 0
        return {0: drop_message}
    # Already a dict
    return {int(k): v for k, v in drop_message.items()}


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    global _SYSTEM
    if _SYSTEM is not None:
        logger.info("Shutting down ContextualSystem ...")
        _SYSTEM.shutdown()
        _SYSTEM = None


app = FastAPI(title="Contextual System API", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/generate")
async def generate(req: GenerateRequest):
    """Simple generation endpoint.  Accepts either a raw ``prompt`` string or
    a ``messages`` list and returns the generated text."""
    system = get_system()

    if req.messages is not None:
        msgs = [m.model_dump(exclude_none=True) for m in req.messages]
    elif req.prompt is not None:
        msgs = req.prompt
    else:
        raise HTTPException(status_code=400, detail="Either 'prompt' or 'messages' must be provided")

    try:
        result = await system.generate_one_requests(
            messages=msgs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            enable_thinking=req.enable_thinking,
        )
    except Exception as e:
        logger.exception("generate_one_requests failed")
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({"text": result})


@app.post("/v1/chat/completions")
async def v1_chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint with ``drop_message``
    support for the discardable KV cache feature.

    Stateless: the caller sends the full conversation history every time.
    Session identification and KV-cache reuse are handled internally by
    ``ContextualSystem.generate_one_round``.
    """
    system = get_system()

    drop_config = _normalize_drop_message(req.drop_message)
    msgs = [m.model_dump(exclude_none=True) for m in req.messages]

    try:
        assistant_text = await system.generate_one_round(
            messages=msgs,
            drop_messages=drop_config,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            enable_thinking=req.enable_thinking,
            tools=[t for t in req.tools] if req.tools else None,
        )
    except Exception as e:
        logger.exception("chat completions failed")
        raise HTTPException(status_code=500, detail=str(e))

    response = ChatCompletionResponse(
        model=system.model_path,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=assistant_text),
                finish_reason="stop",
            )
        ],
    )
    return response


@app.post("/v1/completions")
async def v1_completions(req: CompletionRequest):
    """OpenAI-compatible text completions endpoint."""
    system = get_system()

    try:
        result = await system.generate_one_requests(
            messages=req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            enable_thinking=req.enable_thinking,
        )
    except Exception as e:
        logger.exception("completions failed")
        raise HTTPException(status_code=500, detail=str(e))

    return CompletionResponse(
        model=system.model_path,
        choices=[CompletionChoice(index=0, text=result, finish_reason="stop")],
    )


@app.get("/v1/models")
async def list_models():
    system = get_system()
    return ModelList(
        data=[ModelCard(id=system.model_path, root=system.model_path)]
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/shutdown")
async def shutdown():
    """Gracefully shut down the ContextualSystem and release GPU resources."""
    global _SYSTEM
    if _SYSTEM is None:
        return JSONResponse({"status": "already_stopped"})

    logger.info("Shutdown requested via /shutdown endpoint")
    try:
        _SYSTEM.shutdown()
    except Exception as e:
        logger.exception("Error during shutdown")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _SYSTEM = None

    return JSONResponse({"status": "shutdown_complete"})


# ---------------------------------------------------------------------------
# Initialization helper (called from launch.py)
# ---------------------------------------------------------------------------

def mount_system(system) -> None:
    """Attach an already-initialized ``ContextualSystem`` instance to the
    FastAPI app so that the route handlers can access it."""
    global _SYSTEM
    if _SYSTEM is not None:
        raise RuntimeError("ContextualSystem is already mounted")
    _SYSTEM = system
    logger.info("ContextualSystem mounted to API server")
