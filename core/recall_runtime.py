"""Bounded, request-keyed task ownership for optional narrative recall work."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import functools
import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from astrbot.api import logger

_ADAPTER_SNAPSHOT_UNSET = object()


async def timed_recall_phase(req, name, operation):
    started = time.monotonic()
    try:
        return await operation
    finally:
        phases = getattr(req, "_lm_private_phase_ms_v1", None)
        if not isinstance(phases, dict):
            phases = {}
            req._lm_private_phase_ms_v1 = phases
        phases[name] = phases.get(name, 0) + round(
            (time.monotonic() - started) * 1000, 1
        )


def trace_recall_hook(method):
    @functools.wraps(method)
    async def wrapped(self, event, req, *args, **kwargs):
        started = time.monotonic()
        req._lm_private_phase_ms_v1 = {}
        try:
            return await method(self, event, req, *args, **kwargs)
        finally:
            logger.info(
                "[LMRecall] hook outcome=%s local_only=%s elapsed_ms=%.1f phases=%s",
                str(getattr(req, "_lm_private_hook_outcome_v1", "unknown"))[:40],
                getattr(req, "_cd_local_recall_only_v1", False) is True,
                (time.monotonic() - started) * 1000,
                json.dumps(
                    getattr(req, "_lm_private_phase_ms_v1", {}), separators=(",", ":")
                ),
            )

    return wrapped


def recall_task_key(
    *,
    route: str,
    scope: dict,
    query: str,
    purpose: str,
    card_version: str,
    correction_revision: int | None,
    persona_ids: tuple[str, ...],
    origin: str,
    discriminator: Any = None,
) -> str:
    """Hash the complete request identity without storing or logging query text."""
    payload = {
        "route": str(route),
        "scope": scope,
        "query": str(query),
        "purpose": str(purpose),
        "card_version": str(card_version),
        "correction_revision": correction_revision,
        "persona_ids": list(persona_ids),
        "origin": str(origin),
        "discriminator": discriminator,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RecallDeadline:
    """One monotonic native-recall deadline with time reserved for final checks."""

    started_at: float
    deadline: float
    final_reserve_seconds: float = 1.0

    @classmethod
    def start(
        cls,
        total_seconds: float = 6.0,
        final_reserve_seconds: float = 1.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        started = clock()
        total = max(0.1, min(float(total_seconds), 7.0))
        reserve = max(0.0, min(float(final_reserve_seconds), total * 0.5))
        return cls(started, started + total, reserve)

    @property
    def search_deadline(self) -> float:
        return max(self.started_at, self.deadline - self.final_reserve_seconds)

    def remaining(self, *, clock: Callable[[], float] = time.monotonic) -> float:
        return max(0.0, self.deadline - clock())

    def search_remaining(self, *, clock: Callable[[], float] = time.monotonic) -> float:
        return max(0.0, self.search_deadline - clock())


class RecallOperationContext:
    """Body-free, monotonic diagnostics for the native recall handler."""

    def __init__(self, route: str = "handler"):
        self.route = str(route)
        self.started = 0.0

    async def __aenter__(self):
        self.started = time.monotonic()
        return self

    async def __aexit__(self, exc_type, _exc, _tb):
        elapsed = int(max(0.0, time.monotonic() - self.started) * 1000)
        outcome = (
            "ok"
            if exc_type is None
            else "cancelled"
            if getattr(exc_type, "__name__", "") == "CancelledError"
            else "error"
        )
        log = logger.debug if exc_type is None else logger.info
        log(
            "[LMRecall] route=%s elapsed_ms=%d outcome=%s exception_type=%s",
            self.route,
            elapsed,
            outcome,
            getattr(exc_type, "__name__", "none"),
        )
        return False


def remember_private_segment(req, part, *, source_level: str, item_count: int) -> None:
    """Mark one freshly appended LM part for the CD body-free receipt getter."""
    if source_level not in {"verified_quote", "legacy_summary", "other"}:
        source_level = "other"
    try:
        current = getattr(req, "_lm_private_segment_refs_v1", None)
        refs = dict(current) if isinstance(current, dict) else {}
        refs[id(part)] = {
            "source_level": source_level,
            "item_count": max(0, int(item_count)),
        }
        setattr(req, "_lm_private_segment_refs_v1", refs)
    except Exception:
        return


def discard_private_summary_segment(
    req, part, *, previous_sources=None, previous_validator=None
) -> None:
    """Roll back one unconfirmed injection before the Agent freezes its sources.

    Remove only this exact part/marker and newly installed matching summary
    receipt. Other validated sources are retained. A guard already frozen by
    an Agent still observes a receipt mismatch and therefore fails closed.
    """
    parts = getattr(req, "extra_user_content_parts", None)
    if isinstance(parts, list):
        req.extra_user_content_parts = [entry for entry in parts if entry is not part]
    refs = getattr(req, "_lm_private_segment_refs_v1", None)
    if isinstance(refs, dict):
        refs = dict(refs)
        refs.pop(id(part), None)
        req._lm_private_segment_refs_v1 = refs
    current = getattr(req, "_lm_private_summary_sources_v1", None)
    validator = getattr(req, "_lm_private_summary_validate_v1", None)
    text = getattr(part, "text", None)
    owned = (
        isinstance(current, dict)
        and current is not previous_sources
        and isinstance(text, str)
        and current.get("segment_sha256")
        == hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    if not owned:
        return
    if isinstance(previous_sources, dict) and callable(previous_validator):
        req._lm_private_summary_sources_v1 = previous_sources
        req._lm_private_summary_validate_v1 = previous_validator
    else:
        for key in (
            "_lm_private_summary_sources_v1",
            "_lm_private_summary_failure_reason_v1",
        ):
            try:
                delattr(req, key)
            except AttributeError:
                pass
        if validator is not previous_validator:
            try:
                delattr(req, "_lm_private_summary_validate_v1")
            except AttributeError:
                pass


class BoundedRecallSearchContext:
    """Own pure retrieval routes without retaining request/event objects.

    Callers pass only a retriever method and its query/scope arguments. The
    request waiter can time out while the global pool keeps any late route
    result owned; no callback can append it to the request.
    """

    def __init__(
        self,
        pool: BoundedRecallTaskPool,
        budget: RecallDeadline,
        *,
        scope: dict,
        query: str,
        purpose: str,
        card_version: str,
        correction_revision: int | None,
        persona_ids: tuple[str, ...],
        origin: str,
        max_parallel: int = 4,
        _parallel=None,
        _metrics=None,
        hard_deadline: bool = False,
        embedding_cache=None,
        local_only=False,
        adapter_access_snapshot=_ADAPTER_SNAPSHOT_UNSET,
        remote_deadline: float | None = None,
        _remote_parallel=None,
    ):
        self.pool = pool
        self.budget = budget
        self.scope = copy.deepcopy(scope)
        self.query = str(query)
        self.purpose = str(purpose)
        self.card_version = str(card_version)
        self.correction_revision = correction_revision
        self.persona_ids = tuple(str(value) for value in persona_ids)
        self.origin = str(origin)
        self.hard_deadline = bool(hard_deadline)
        self.embedding_cache = embedding_cache
        self.local_only = local_only is True
        self.remote_deadline = remote_deadline
        self._remote_parallel = _remote_parallel or asyncio.Semaphore(2)
        from .memory_scope import current_adapter_access

        self.adapter_access_snapshot = (
            current_adapter_access.get()
            if adapter_access_snapshot is _ADAPTER_SNAPSHOT_UNSET
            else adapter_access_snapshot
        )
        self._parallel = _parallel or asyncio.Semaphore(
            max(1, min(int(max_parallel), 4))
        )
        (self.route_elapsed_ms, self.route_failures, self.route_exception_types) = (
            _metrics or ({}, {}, {})
        )

    def for_scope(self, scope: dict, correction_revision: int | None):
        """Bind one authorized source scope while sharing this request's bounds."""
        return BoundedRecallSearchContext(
            self.pool,
            self.budget,
            scope=scope,
            query=self.query,
            purpose=self.purpose,
            card_version=self.card_version,
            correction_revision=correction_revision,
            persona_ids=self.persona_ids,
            origin=self.origin,
            _parallel=self._parallel,
            _metrics=(
                self.route_elapsed_ms,
                self.route_failures,
                self.route_exception_types,
            ),
            hard_deadline=self.hard_deadline,
            embedding_cache=self.embedding_cache,
            local_only=self.local_only,
            adapter_access_snapshot=self.adapter_access_snapshot,
            remote_deadline=self.remote_deadline,
            _remote_parallel=self._remote_parallel,
        )

    def with_deadline(self, deadline: float):
        """Shorten optional work while preserving the parent's validation budget."""
        child = self.for_scope(self.scope, self.correction_revision)
        child.budget = RecallDeadline(
            self.budget.started_at, min(self.budget.search_deadline, deadline), 0.0
        )
        return child

    async def run(
        self,
        route: str,
        method: Callable[..., Any],
        *args,
        discriminator: Any = None,
        remote: bool = False,
        **kwargs,
    ) -> dict:
        started = time.monotonic()
        cutoff = self.budget.search_deadline
        parallel = self._parallel
        if remote and self.remote_deadline is not None:
            cutoff = min(cutoff, self.remote_deadline)
            parallel = self._remote_parallel
        try:
            return await self._run(
                route, method, args, kwargs, discriminator, cutoff, parallel
            )
        except asyncio.CancelledError:
            self.route_failures[route] = "cancelled"
            raise
        finally:
            self.route_elapsed_ms[route] = self.route_elapsed_ms.get(route, 0) + int(
                (time.monotonic() - started) * 1000
            )

    async def _run(
        self, route, method, args, kwargs, discriminator, cutoff, parallel
    ) -> dict:
        started = time.monotonic()
        if not callable(method):
            self.route_failures[route] = "unavailable"
            return {"state": "unavailable", "value": None}
        key = recall_task_key(
            route=str(route),
            scope=self.scope,
            query=self.query,
            purpose=self.purpose,
            card_version=self.card_version,
            correction_revision=self.correction_revision,
            persona_ids=self.persona_ids,
            origin=self.origin,
            discriminator={
                "args": discriminator,
                "hard_deadline": self.hard_deadline,
                "local_only": self.local_only,
                "adapters": getattr(self.adapter_access_snapshot, "adapter_ids", None),
            },
        )

        slot_budget = max(0.0, cutoff - time.monotonic())
        if slot_budget <= 0:
            self.route_failures[route] = "timeout"
            return {"state": "timeout", "value": None}
        try:
            async with asyncio.timeout(slot_budget):
                await parallel.acquire()
        except asyncio.TimeoutError:
            self.route_failures[route] = "timeout"
            return {"state": "timeout", "value": None}
        finally:
            self.route_elapsed_ms[route + "_queue"] = self.route_elapsed_ms.get(
                route + "_queue", 0
            ) + int((time.monotonic() - started) * 1000)
        if time.monotonic() >= cutoff:
            parallel.release()
            self.route_failures[route] = "timeout"
            return {"state": "timeout", "value": None}
        dispatch_started = time.monotonic()

        async def execute_leaf():
            work_started = time.monotonic()
            self.route_elapsed_ms[route + "_dispatch"] = self.route_elapsed_ms.get(
                route + "_dispatch", 0
            ) + int((work_started - dispatch_started) * 1000)
            # The pool intentionally clears ambient ContextVars. Carry only
            # this request's pure query-vector cache, never req/event objects.
            from .memory_scope import current_adapter_access
            from .retrieval.vector_search import query_embeddings

            adapter_token = current_adapter_access.set(self.adapter_access_snapshot)
            token = (
                query_embeddings.set(self.embedding_cache)
                if self.embedding_cache is not None
                else None
            )
            try:
                if self.hard_deadline:
                    remaining = cutoff - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("recall_search_deadline")
                    async with asyncio.timeout(remaining):
                        return await method(*args, **kwargs)
                return await method(*args, **kwargs)
            finally:
                task = asyncio.current_task()
                if task is not None:
                    task._recall_finished_at = time.monotonic()
                self.route_elapsed_ms[route + "_work"] = self.route_elapsed_ms.get(
                    route + "_work", 0
                ) + int((time.monotonic() - work_started) * 1000)
                if token is not None:
                    query_embeddings.reset(token)
                current_adapter_access.reset(adapter_token)

        handle = self.pool.submit(key, execute_leaf, cache_result=False)
        task = handle.get("task") if isinstance(handle, dict) else None
        if handle.get("state") == "started" and isinstance(task, asyncio.Task):
            task.add_done_callback(
                lambda _completed: parallel.release(),
                context=contextvars.Context(),
            )
        else:
            parallel.release()
        outcome = await self.pool.wait(
            handle, deadline=cutoff, metrics=self.route_elapsed_ms, route=route
        )
        if self.hard_deadline and outcome.get("exception_type") == "TimeoutError":
            outcome = {**outcome, "state": "timeout"}
        state = str(outcome.get("state", "error"))
        if state != "completed":
            self.route_failures[route] = state
            exception_type = outcome.get("exception_type")
            if isinstance(exception_type, str):
                self.route_exception_types[route] = exception_type[:64]
        return outcome


class BoundedRecallTaskPool:
    """Own a bounded set of recall tasks and short-lived candidate snapshots.

    Request waiters may time out or be canceled without canceling shared work.
    Completed candidates are only hints: every caller must re-read current
    documents, source rows, and evidence revisions before using them.
    """

    _summary_cleanup_release = True
    _reply_scope_parallel_v1 = True

    def __init__(
        self,
        *,
        max_inflight: int = 16,
        max_cache: int = 32,
        cache_ttl_seconds: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_inflight = max(1, int(max_inflight))
        self.max_cache = max(1, int(max_cache))
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self.clock = clock
        self._tasks: dict[str, asyncio.Task] = {}
        self._scopes: set[asyncio.Task] = set()
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._closed = False
        self._drain_task: asyncio.Task | None = None
        self._copy_slots = asyncio.Semaphore(4)
        self._copy_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="recall-copy"
        )
        self._copy_tasks: set[asyncio.Task] = set()
        self._coalesced = 0
        self._busy = 0

    def submit_scope(
        self, operation, *, context: contextvars.Context
    ) -> asyncio.Task | None:
        """Own native scope orchestration separately from the leaf routes it awaits."""
        if self._closed or len(self._scopes) >= self.max_inflight:
            return None
        task = asyncio.create_task(
            operation(), name="livingmemory-recall-scope", context=context
        )
        self._scopes.add(task)
        task.add_done_callback(self._scope_completed, context=contextvars.Context())
        return task

    def _scope_completed(self, task: asyncio.Task) -> None:
        self._scopes.discard(task)
        if not task.cancelled():
            task.exception()

    def _prune_cache(self) -> None:
        now = self.clock()
        for key, (expires_at, _value) in list(self._cache.items()):
            if expires_at <= now:
                self._cache.pop(key, None)

    def submit(
        self, key: str, factory: Callable[[], Any], *, cache_result: bool = True
    ) -> dict:
        if self._closed:
            return {"state": "closed"}
        self._prune_cache()
        cached = self._cache.get(key) if cache_result else None
        if cached is not None:
            self._cache.move_to_end(key)
            try:
                return {"state": "cached", "value": copy.deepcopy(cached[1])}
            except Exception:
                self._cache.pop(key, None)
        task = self._tasks.get(key)
        if task is not None:
            self._coalesced += 1
            return {"state": "coalesced", "task": task}
        if len(self._tasks) >= self.max_inflight:
            self._busy += 1
            return {"state": "busy"}

        async def execute():
            return await factory()

        # Pure candidate leaves must not inherit req/event/request callback
        # ContextVars from the waiter. Route arguments are explicit and frozen
        # by the caller before submission.
        task = contextvars.Context().run(
            asyncio.create_task, execute(), name="livingmemory-recall-route"
        )
        self._tasks[key] = task
        task.add_done_callback(
            lambda completed, task_key=key, should_cache=cache_result: self._completed(
                task_key, completed, should_cache
            ),
            context=contextvars.Context(),
        )
        return {"state": "started", "task": task}

    def _completed(self, key: str, task: asyncio.Task, cache_result: bool) -> None:
        if self._tasks.get(key) is task:
            self._tasks.pop(key, None)
        if task.cancelled():
            return
        try:
            value = task.result()
        except BaseException:
            return
        if not cache_result or self._closed or self.cache_ttl_seconds <= 0:
            return
        try:
            stored = copy.deepcopy(value)
        except Exception:
            return
        self._cache[key] = (self.clock() + self.cache_ttl_seconds, stored)
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)

    async def _clone_result(self, value, deadline, metrics, route):
        """Bound and own worker copies; a cancelled waiter cannot free a busy slot."""
        if self._closed:
            raise RuntimeError("recall_pool_closed")
        queued = self.clock()
        remaining = deadline - queued
        if remaining <= 0:
            raise TimeoutError("recall_copy_deadline")
        async with asyncio.timeout(remaining):
            await self._copy_slots.acquire()
        if self._closed or self.clock() >= deadline:
            self._copy_slots.release()
            raise TimeoutError("recall_copy_deadline")

        def clone():
            started = self.clock()
            metrics[route + "_copyq"] = int((started - queued) * 1000)
            try:
                return copy.deepcopy(value)
            finally:
                metrics[route + "_copy"] = int((self.clock() - started) * 1000)

        # Worker copies need only explicit result data, never ambient request context.
        async def copy_owned():
            return await asyncio.get_running_loop().run_in_executor(
                self._copy_executor, contextvars.Context().run, clone
            )

        task = contextvars.Context().run(asyncio.create_task, copy_owned())
        self._copy_tasks.add(task)

        def completed(done):
            self._copy_tasks.discard(done)
            self._copy_slots.release()
            if not done.cancelled():
                done.exception()

        task.add_done_callback(completed, context=contextvars.Context())
        done, _ = await asyncio.wait({task}, timeout=max(0.0, deadline - self.clock()))
        if not done or self.clock() >= deadline:
            raise TimeoutError("recall_copy_deadline")
        return task.result()

    async def wait(
        self, handle: dict, *, deadline: float, metrics=None, route="result"
    ) -> dict:
        state = handle.get("state") if isinstance(handle, dict) else "invalid"
        if state == "cached":
            if self.clock() >= deadline:
                return {"state": "timeout", "value": None, "cached": False}
            return {"state": "completed", "value": handle.get("value"), "cached": True}
        if state in {"busy", "closed"}:
            return {"state": state, "value": None, "cached": False}
        task = handle.get("task") if isinstance(handle, dict) else None
        if not isinstance(task, asyncio.Task):
            return {"state": "invalid", "value": None, "cached": False}
        if not task.done():
            remaining = max(0.0, float(deadline) - self.clock())
            if remaining <= 0:
                return {"state": "timeout", "value": None, "cached": False}
            done, _pending = await asyncio.wait({task}, timeout=remaining)
            if not done:
                return {"state": "timeout", "value": None, "cached": False}
        if self.clock() >= deadline:
            return {"state": "timeout", "value": None, "cached": False}
        if task.cancelled():
            return {"state": "cancelled", "value": None, "cached": False}
        if metrics is not None:
            finished = getattr(task, "_recall_finished_at", None)
            if isinstance(finished, (int, float)):
                metrics[route + "_resume"] = int(
                    max(0.0, time.monotonic() - finished) * 1000
                )
        try:
            raw = task.result()
            value = (
                await self._clone_result(raw, deadline, metrics, route)
                if metrics is not None
                else copy.deepcopy(raw)
            )
            if self.clock() >= deadline:
                return {"state": "timeout", "value": None, "cached": False}
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return {"state": "timeout", "value": None, "cached": False}
        except Exception as exc:
            return {
                "state": "error",
                "value": None,
                "cached": False,
                "exception_type": type(exc).__name__,
            }
        return {"state": "completed", "value": value, "cached": False}

    def stats(self) -> dict[str, int | bool]:
        self._prune_cache()
        return {
            "inflight_count": len(self._tasks),
            "cache_entries": len(self._cache),
            "coalesced_count": self._coalesced,
            "busy_count": self._busy,
            "scope_inflight_count": len(self._scopes),
            "closed": self._closed,
        }

    async def close(self) -> None:
        """Stop accepting work and drain owned operations before backing stores close."""
        self._closed = True
        if self._drain_task is None:

            async def drain_owned():
                tasks = [*self._tasks.values(), *self._scopes]
                for task in tasks:
                    if not task.done():
                        task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                # Python threads cannot be cancelled safely; bounded copies
                # must finish before this owner is considered closed.
                copies = list(self._copy_tasks)
                if copies:
                    await asyncio.gather(*copies, return_exceptions=True)
                await asyncio.to_thread(self._copy_executor.shutdown, wait=True)
                self._tasks.clear()
                self._cache.clear()

            self._drain_task = asyncio.create_task(
                drain_owned(), name="livingmemory-recall-drain"
            )
        await asyncio.shield(self._drain_task)
