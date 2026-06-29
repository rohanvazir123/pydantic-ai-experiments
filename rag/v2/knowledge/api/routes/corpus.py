"""Corpus management routes."""

import uuid

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from knowledge.api.middleware import get_request_id
from knowledge.api.schemas import APIResponse, CorpusInfo

router = APIRouter(prefix="/corpus", tags=["corpus"])


@router.get("", response_model=APIResponse[list[CorpusInfo]])
async def list_corpora(request: Request) -> APIResponse[list[CorpusInfo]]:
    """List all corpora accessible to the current JWT role."""
    settings   = getattr(request.app.state, "settings", None)
    request_id = get_request_id() or str(uuid.uuid4())

    corpora = []
    if settings:
        for c in settings.corpus_configs:
            corpora.append(CorpusInfo(
                id=c.id,
                display_name=c.display_name,
                source_folders=[str(p) for p in c.source_folders],
                allowed_roles=c.allowed_roles,
                enable_graph_extraction=c.enable_graph_extraction,
                graph_ontology_path=c.graph_ontology_path,
            ))

    return APIResponse(request_id=request_id, data=corpora)


@router.post("/cache/clear", response_model=APIResponse[dict])
async def clear_all_cache(request: Request) -> APIResponse[dict]:
    """Flush all L2 Redis search + embedding cache entries (fingerprints kept)."""
    cache      = getattr(request.app.state, "cache", None)
    request_id = get_request_id() or str(uuid.uuid4())

    deleted = 0
    if cache:
        deleted = await cache.flush()

    return APIResponse(
        request_id=request_id,
        data={"cache_keys_deleted": deleted, "message": "Cache cleared"},
    )


@router.post("/{corpus_id}/cache/invalidate", response_model=APIResponse[dict])
async def invalidate_cache(corpus_id: str, request: Request) -> APIResponse[dict]:
    """Flush L2 + L3 cache entries for a corpus."""
    cache      = getattr(request.app.state, "cache", None)
    request_id = get_request_id() or str(uuid.uuid4())

    deleted = 0
    if cache:
        deleted = await cache.invalidate_corpus(corpus_id, "")

    return APIResponse(
        request_id=request_id,
        data={"corpus_id": corpus_id, "cache_keys_deleted": deleted},
    )


@router.get("/{corpus_id}/ontology", response_model=APIResponse[dict])
async def get_ontology(corpus_id: str, request: Request) -> APIResponse[dict]:
    """Return the current ontology Python source for a corpus."""
    from knowledge.corpus.ontologies.loader import _ONTOLOGIES_DIR
    request_id = get_request_id() or str(uuid.uuid4())

    path = _ONTOLOGIES_DIR / f"{corpus_id}.py"
    if not path.exists():
        path = _ONTOLOGIES_DIR / "generic.py"

    content = path.read_text(encoding="utf-8")
    return APIResponse(
        request_id=request_id,
        data={"corpus_id": corpus_id, "source": content, "is_default": not ((_ONTOLOGIES_DIR / f"{corpus_id}.py").exists())},
    )


@router.post("/{corpus_id}/ontology", response_model=APIResponse[dict])
async def upload_ontology(
    corpus_id: str,
    request: Request,
    file: UploadFile = File(...),
) -> APIResponse[dict]:
    """Upload a Python ontology file for a corpus. Validates before saving."""
    from knowledge.corpus.ontologies.loader import _ONTOLOGIES_DIR, load_ontology
    request_id = get_request_id() or str(uuid.uuid4())

    if not file.filename or not file.filename.endswith(".py"):
        raise HTTPException(status_code=400, detail="File must be a .py Python file")

    content = await file.read()
    dest    = _ONTOLOGIES_DIR / f"{corpus_id}.py"
    dest.write_bytes(content)

    # Validate the file
    try:
        load_ontology.cache_clear()
        cls = load_ontology(f"{corpus_id}.py")
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Invalid ontology: {exc}")

    return APIResponse(
        request_id=request_id,
        data={"corpus_id": corpus_id, "root_class": cls.__name__, "saved": True},
    )


@router.delete("/{corpus_id}/ontology", response_model=APIResponse[dict])
async def delete_ontology(corpus_id: str, request: Request) -> APIResponse[dict]:
    """Remove custom ontology — reverts to generic default."""
    from knowledge.corpus.ontologies.loader import _ONTOLOGIES_DIR, load_ontology
    request_id = get_request_id() or str(uuid.uuid4())

    path = _ONTOLOGIES_DIR / f"{corpus_id}.py"
    existed = path.exists()
    path.unlink(missing_ok=True)
    load_ontology.cache_clear()

    return APIResponse(
        request_id=request_id,
        data={"corpus_id": corpus_id, "deleted": existed, "reverted_to": "generic"},
    )
