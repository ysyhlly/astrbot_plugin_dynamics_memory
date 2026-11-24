"""Shared query embeddings and bounded, scope-aware vector candidate expansion."""

import asyncio
import hashlib
import json
import time
from contextvars import ContextVar
from types import SimpleNamespace

from ...series_panels import record_series_llm_usage
from ..faiss_async_persist import get_async_persister
from ..memory_scope import adapter_sql, current_adapter_access
from ..passive_group_capture import get_active_plugin

query_embeddings = ContextVar("livingmemory_query_embeddings", default=None)

_TELEMETRY_PLUGIN_ID = "astrbot_plugin_dynamics_memory"


def _extract_embedding_provider_meta(provider) -> tuple[str, str]:
    """Extract (provider_id, model) from an AstrBot EmbeddingProvider."""
    provider_id = ""
    model = ""
    if provider is None:
        return "embedding", ""
    meta = getattr(provider, "meta", None)
    if callable(meta):
        try:
            meta = meta()
        except Exception:
            meta = None
    if meta is not None:
        for attr in ("id", "provider_id"):
            val = getattr(meta, attr, None)
            if isinstance(val, str) and val.strip():
                provider_id = val.strip()
                break
        meta_model = getattr(meta, "model", None)
        if isinstance(meta_model, str) and meta_model.strip():
            model = meta_model.strip()
    cfg = getattr(provider, "provider_config", None)
    if isinstance(cfg, dict):
        if not provider_id:
            val = cfg.get("id")
            if isinstance(val, str) and val.strip():
                provider_id = val.strip()
        if not model:
            for key in ("embedding_model", "model"):
                val = cfg.get(key)
                if isinstance(val, str) and val.strip():
                    model = val.strip()
                    break
            if not model:
                model_cfg = cfg.get("model_config")
                if isinstance(model_cfg, dict):
                    val = model_cfg.get("model")
                    if isinstance(val, str) and val.strip():
                        model = val.strip()
    if not provider_id:
        for attr in ("id", "provider_id"):
            val = getattr(provider, attr, None)
            if isinstance(val, str) and val.strip():
                provider_id = val.strip()
                break
    if not model:
        getter = getattr(provider, "get_model", None)
        if callable(getter):
            try:
                val = getter()
                if isinstance(val, str) and val.strip():
                    model = val.strip()
            except Exception:
                pass
        if not model:
            val = getattr(provider, "model_name", None) or getattr(
                provider, "model", None
            )
            if isinstance(val, str) and val.strip():
                model = val.strip()
    return provider_id or "embedding", model


def report_embedding_usage(
    provider,
    *,
    elapsed_seconds: float,
    ok: bool,
    error_type: str = "",
    context=None,
) -> bool:
    """Report one embedding call's duration and outcome to series telemetry."""
    resolved_ctx = context
    if resolved_ctx is None:
        plugin = get_active_plugin()
        resolved_ctx = getattr(plugin, "context", None) if plugin is not None else None
    if resolved_ctx is None:
        resolved_ctx = getattr(provider, "context", None)
    provider_id, model = _extract_embedding_provider_meta(provider)
    return record_series_llm_usage(
        resolved_ctx,
        plugin_id=_TELEMETRY_PLUGIN_ID,
        provider_id=provider_id,
        purpose="embedding",
        generation_seconds=max(0.0, float(elapsed_seconds)),
        ok=bool(ok),
        error_type=error_type,
        model=model,
    )


async def call_embedding_with_telemetry(provider, text: str, *, context=None):
    """Invoke ``provider.get_embedding(text)`` and report telemetry."""
    started = time.monotonic()
    try:
        value = await provider.get_embedding(text)
    except BaseException as exc:
        report_embedding_usage(
            provider,
            elapsed_seconds=time.monotonic() - started,
            ok=False,
            error_type=type(exc).__name__,
            context=context,
        )
        raise
    report_embedding_usage(
        provider,
        elapsed_seconds=time.monotonic() - started,
        ok=True,
        context=context,
    )
    return value


async def retrieve_documents(
    db, metadata_filters, ids=None, limit=100, offset=0, *, ids_only=False
):
    """Apply request ownership in SQL before limiting host document reads."""
    predicate, adapters = adapter_sql("metadata")
    if not adapters:
        return await db.document_storage.get_documents(
            metadata_filters=metadata_filters, ids=ids, limit=limit, offset=offset
        )
    from sqlalchemy import text

    values = {f"adapter_{i}": value for i, value in enumerate(adapters)}
    for i in range(len(adapters)):
        predicate = predicate.replace("?", f":adapter_{i}", 1)
    conditions = [predicate]
    for key, value in metadata_filters.items():
        conditions.append(f"json_extract(metadata, '$.{key}') = :filter_{key}")
        values[f"filter_{key}"] = value
    if ids is not None:
        if not ids:
            return []
        conditions.append(
            "id IN (" + ",".join(f":id_{i}" for i in range(len(ids))) + ")"
        )
        values.update({f"id_{i}": value for i, value in enumerate(ids)})
    columns = "id" if ids_only else "id, doc_id, text, metadata"
    sql = f"SELECT {columns} FROM documents WHERE " + " AND ".join(conditions)
    if limit is not None:
        sql += " LIMIT :limit OFFSET :offset"
        values.update(limit=limit, offset=offset or 0)
    async with db.document_storage.get_session() as session:
        rows = (await session.execute(text(sql), values)).mappings().all()
    if ids_only:
        return [dict(row) for row in rows]
    return [dict(row, metadata=json.loads(row["metadata"] or "{}")) for row in rows]


async def retrieve_vectors(db, query, k, filters, *, source_key="id"):
    """Retrieve unique active candidates without repeating the embedding request.

    Args:
        db: AstrBot vector database.
        query: Query text; the same budget applies to both vector routes.
        k: Maximum number of results.
        filters: Session and persona restrictions.
        source_key: Document field used to deduplicate graph source memories.

    Returns:
        Ranked vector hits. Candidate expansion is capped at 8192 vectors.
    """
    if k <= 0:
        return []
    query = query[:2000]
    storage = getattr(db, "embedding_storage", None)
    total = getattr(getattr(storage, "index", None), "ntotal", None)
    native = isinstance(total, int)
    cap = min(total, 8192) if native else 8192
    if native and not total:
        return []
    fetch_k = min(cap, max(k * (4 if filters else 2), k))
    storage_filters = {
        key: value for key, value in filters.items() if not key.startswith("__")
    }

    search_options = {}
    if native:
        # No vector can satisfy an empty exact-filter scope. Check locally
        # before paying for an embedding; do not cache this absence because
        # new memories can be inserted immediately after this search.
        if storage_filters and not await db.document_storage.get_documents(
            metadata_filters=storage_filters, limit=1
        ):
            return []

        import numpy as np

        access = current_adapter_access.get()
        if access is not None and access.adapter_ids is not None:
            import faiss

            candidates = await retrieve_documents(
                db, storage_filters, limit=None, ids_only=True
            )
            if not candidates:
                return []
            allowed_ids = np.asarray([row["id"] for row in candidates], dtype=np.int64)
            search_options["params"] = faiss.SearchParameters(
                sel=faiss.IDSelectorBatch(allowed_ids)
            )
            cap = min(len(candidates), cap)
            fetch_k = min(fetch_k, cap)

        provider = db.embedding_provider
        cache = query_embeddings.get()
        if cache is None:
            embedding = await call_embedding_with_telemetry(provider, query)
        else:
            key = (id(provider), query)
            if key not in cache:
                if isinstance(cache, QueryEmbeddingCache):
                    cache[key] = cache.start(provider, query)
                else:
                    cache[key] = asyncio.create_task(
                        call_embedding_with_telemetry(provider, query)
                    )
                    cache[key].add_done_callback(_observe_embedding)
            embedding = await asyncio.shield(cache[key])
        vector = np.asarray(embedding, dtype=np.float32).ravel()
        if vector.size != storage.dimension or not np.isfinite(vector).all():
            raise ValueError("Invalid query embedding")

    # 候选窗口翻倍重查时复用已取回的文档，避免每轮重复 SQL + JSON 解析；
    # 每轮只需查询新增的 kNN id。
    documents_by_id: dict[int, dict] = {}

    while True:
        if native:
            persister = get_async_persister(storage)
            if persister is not None:
                scores, indices = await persister.search(
                    vector, fetch_k, **search_options
                )
            elif search_options:
                scores, indices = storage.index.search(
                    vector.reshape(1, -1), fetch_k, **search_options
                )
            else:
                scores, indices = await storage.search(vector, fetch_k)
            ids = [int(value) for value in indices[0] if value >= 0]
            new_ids = [doc_id for doc_id in ids if doc_id not in documents_by_id]
            if new_ids:
                documents = await db.document_storage.get_documents(
                    metadata_filters=storage_filters,
                    ids=new_ids,
                    limit=len(new_ids),
                )
                documents_by_id.update(
                    {int(document["id"]): document for document in documents}
                )
            by_id = documents_by_id
            raw = [
                SimpleNamespace(
                    data=by_id[int(doc_id)], similarity=1.0 - float(score) / 2.0
                )
                for score, doc_id in zip(scores[0], indices[0])
                if int(doc_id) in by_id
            ]
            exhausted = len(ids) < fetch_k
        else:
            # Older host adapters expose only retrieve; preserve their public API.
            fallback_provider = getattr(db, "embedding_provider", None)
            retrieve_started = time.monotonic()
            try:
                raw = await db.retrieve(
                    query=query,
                    k=fetch_k,
                    fetch_k=fetch_k,
                    rerank=False,
                    metadata_filters=storage_filters or None,
                )
            except BaseException as exc:
                report_embedding_usage(
                    fallback_provider,
                    elapsed_seconds=time.monotonic() - retrieve_started,
                    ok=False,
                    error_type=type(exc).__name__,
                )
                raise
            report_embedding_usage(
                fallback_provider,
                elapsed_seconds=time.monotonic() - retrieve_started,
                ok=True,
            )
            exhausted = not filters and len(raw) < fetch_k

        results = []
        seen = set()
        for hit in raw:
            metadata = hit.data.get("metadata") or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (ValueError, TypeError):
                    continue
            if not isinstance(metadata, dict):
                continue
            if str(metadata.get("status") or "active") != "active":
                continue
            if not _metadata_matches_filters(metadata, filters):
                continue
            identity = (
                hit.data.get("id") if source_key == "id" else metadata.get(source_key)
            )
            if identity is None or identity in seen:
                continue
            seen.add(identity)
            results.append(hit)
            if len(results) >= k:
                return results
        if exhausted or fetch_k >= cap:
            return results
        fetch_k = min(cap, fetch_k * 2)


class QueryEmbeddingCache(dict):
    """One request's query vectors with an owner deadline, no source results."""

    def __init__(self, deadline, *, owner=None):
        super().__init__()
        self.deadline = float(deadline)
        self.owner = owner

    def start(self, provider, query):
        async def embed():
            loop = asyncio.get_running_loop()
            if loop.time() >= self.deadline:
                raise TimeoutError("query_embedding_deadline")
            async with asyncio.timeout_at(self.deadline):
                value = await call_embedding_with_telemetry(provider, query)
            if loop.time() >= self.deadline:
                raise TimeoutError("query_embedding_deadline")
            return value

        if self.owner is not None:
            # Independent of an individual vector waiter. A provider stuck in
            # cancellation still consumes a slot and is drained at shutdown.
            query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
            key = f"query-embedding:{id(self)}:{id(provider)}:{query_hash}"
            handle = self.owner.submit(key, embed, cache_result=False)
            task = handle.get("task")
            if task is None:
                raise RuntimeError("query_embedding_pool_" + handle["state"])
            return task
        task = asyncio.create_task(embed())
        task.add_done_callback(_observe_embedding)
        return task


def _observe_embedding(task):
    if not task.cancelled():
        task.exception()


def _metadata_matches_filters(metadata, filters):
    for key, value in filters.items():
        if key == "__source_platform__":
            prefix = str(value or "").casefold() + ":groupmessage:"
            declared = str(metadata.get("source_platform") or "").casefold()
            if declared == str(value or "").casefold():
                continue
            source_ids = (
                metadata.get("source_session_id"),
                metadata.get("session_id"),
            )
            if not any(
                isinstance(source_id, str) and source_id.casefold().startswith(prefix)
                for source_id in source_ids
            ):
                return False
        elif key == "__exclude_private_visibility__":
            if value and metadata.get("memory_visibility") == "private":
                return False
        elif metadata.get(key) != value:
            return False
    return True
