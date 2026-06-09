"""Memory routes — conversations (Tier 2) and user memories (Tier 3)."""

import uuid

from fastapi import APIRouter, HTTPException, Query, Request

from knowledge.api.middleware import get_request_id, get_tenant_id, get_user_id
from knowledge.api.schemas import (
    APIResponse,
    AddMemoryRequest,
    ConversationSummary,
    MemoryItem,
)

router = APIRouter(tags=["memory"])


# ── Conversations (Tier 2 episodic memory) ────────────────────────────────────

@router.get("/conversations", response_model=APIResponse[list[ConversationSummary]])
async def list_conversations(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
) -> APIResponse[list[ConversationSummary]]:
    """List conversations for the current user, newest first."""
    conv_store = getattr(request.app.state, "conversation_store", None)
    request_id = get_request_id() or str(uuid.uuid4())

    if conv_store is None:
        return APIResponse(request_id=request_id, data=[])

    conversations = await conv_store.list_conversations(
        user_id=get_user_id() or "",
        tenant_id=get_tenant_id() or "default",
        limit=limit,
    )
    return APIResponse(
        request_id=request_id,
        data=[
            ConversationSummary(
                id=str(c["id"]),
                session_id=c["session_id"],
                title=c.get("title"),
                summary=c.get("summary"),
                turn_count=c.get("turn_count", 0),
                last_turn_at=str(c.get("last_turn_at", "")),
            )
            for c in conversations
        ],
    )


@router.get("/conversations/{conversation_id}", response_model=APIResponse[dict])
async def get_conversation(conversation_id: str, request: Request) -> APIResponse[dict]:
    """Get conversation metadata + messages."""
    conv_store = getattr(request.app.state, "conversation_store", None)
    request_id = get_request_id() or str(uuid.uuid4())

    if conv_store is None:
        raise HTTPException(status_code=503, detail="Conversation store not initialised")

    conv = await conv_store.get_conversation(conversation_id, get_user_id() or "")
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return APIResponse(request_id=request_id, data=conv)


@router.delete("/conversations/{conversation_id}", response_model=APIResponse[dict])
async def delete_conversation(conversation_id: str, request: Request) -> APIResponse[dict]:
    """Soft-delete a conversation (GDPR erasure — hard delete after 7-day grace)."""
    conv_store = getattr(request.app.state, "conversation_store", None)
    request_id = get_request_id() or str(uuid.uuid4())

    if conv_store:
        await conv_store.delete_conversation(conversation_id, get_user_id() or "")

    return APIResponse(request_id=request_id, data={"deleted": True})


# ── User memories (Tier 3 semantic memory) ────────────────────────────────────

@router.get("/memories", response_model=APIResponse[list[MemoryItem]])
async def list_memories(request: Request) -> APIResponse[list[MemoryItem]]:
    """List all memories for the current user."""
    mem_store  = getattr(request.app.state, "mem0_store", None)
    request_id = get_request_id() or str(uuid.uuid4())

    if mem_store is None:
        return APIResponse(request_id=request_id, data=[])

    memories = await mem_store.list_memories(
        user_id=get_user_id() or "",
        tenant_id=get_tenant_id() or "default",
    )
    return APIResponse(
        request_id=request_id,
        data=[
            MemoryItem(
                id=str(m.get("id", "")),
                content=m.get("content", ""),
                created_at=str(m.get("created_at", "")),
            )
            for m in memories
        ],
    )


@router.post("/memories", response_model=APIResponse[MemoryItem])
async def add_memory(body: AddMemoryRequest, request: Request) -> APIResponse[MemoryItem]:
    """Manually add a user memory."""
    mem_store  = getattr(request.app.state, "mem0_store", None)
    request_id = get_request_id() or str(uuid.uuid4())

    if mem_store is None:
        raise HTTPException(status_code=503, detail="Memory store not initialised")

    mem_id = await mem_store.add_memory(
        content=body.content,
        user_id=get_user_id() or "",
        tenant_id=get_tenant_id() or "default",
    )
    return APIResponse(
        request_id=request_id,
        data=MemoryItem(id=str(mem_id), content=body.content),
    )


@router.delete("/memories/{memory_id}", response_model=APIResponse[dict])
async def delete_memory(memory_id: str, request: Request) -> APIResponse[dict]:
    """Delete one memory — immediate hard delete (GDPR requirement)."""
    mem_store  = getattr(request.app.state, "mem0_store", None)
    request_id = get_request_id() or str(uuid.uuid4())

    if mem_store:
        await mem_store.delete_memory(
            memory_id=memory_id,
            user_id=get_user_id() or "",
            tenant_id=get_tenant_id() or "default",
        )
    return APIResponse(request_id=request_id, data={"deleted": True})


@router.delete("/memories", response_model=APIResponse[dict])
async def delete_all_memories(request: Request) -> APIResponse[dict]:
    """Delete ALL memories for the current user (right to erasure)."""
    mem_store  = getattr(request.app.state, "mem0_store", None)
    request_id = get_request_id() or str(uuid.uuid4())

    count = 0
    if mem_store:
        count = await mem_store.delete_all_memories(
            user_id=get_user_id() or "",
            tenant_id=get_tenant_id() or "default",
        )
    return APIResponse(request_id=request_id, data={"deleted_count": count})
