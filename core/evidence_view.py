"""Cached, provenance-limited evidence views for role replies and history projection."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import time
from collections import Counter, OrderedDict
from collections.abc import Awaitable, Callable
from datetime import datetime

from ..storage.evidence_sidecar import EvidenceSidecar
from .evidence_projection import (
    MAX_MATERIALS,
    PURPOSES,
    atom_id,
    event_evidence_id,
    normalize_conversation_row,
    normalize_event_material,
    normalize_host_material,
    normalize_legacy_summary,
    pending_continuity_spans,
)

PUBLIC_ATOM_STATUSES = frozenset(
    {
        "active",
        "verified",
        "verified_utterance",
        "reported_unverified",
        "pending_continuity",
        "observed_user_boundary",
    }
)


class EvidenceViewService:
    """Owns the sidecar and one bounded background derivation worker."""

    def __init__(
        self,
        sidecar: EvidenceSidecar,
        scope_key_fn: Callable[[dict], str],
        derive_callback: Callable[[list[dict]], Awaitable[dict[str, list[dict]]]]
        | None = None,
        *,
        queue_size: int = 128,
        per_scope_hourly_limit: int = 24,
        material_loader: Callable[[dict], Awaitable[dict | None]] | None = None,
        conversation_row_loader: Callable[[dict], Awaitable[dict | None]] | None = None,
        event_loader: Callable[[dict, str], Awaitable[dict | None]] | None = None,
        material_policy: Callable[[dict, dict], Awaitable[bool | None]] | None = None,
        owner_readiness: Callable[[str], Awaitable[bool | None]] | None = None,
        clock: Callable[[], float] = time.time,
        lease_seconds: float = 120,
        scheduler_poll_seconds: float = 30.0,
    ):
        self.sidecar = sidecar
        self.scope_key_fn = scope_key_fn
        self.derive_callback = derive_callback
        self.queue_size = max(1, int(queue_size))
        self.per_scope_hourly_limit = max(1, int(per_scope_hourly_limit))
        self.material_loader = material_loader
        self.conversation_row_loader = conversation_row_loader
        self.event_loader = event_loader
        self.material_policy = material_policy
        self.owner_readiness = owner_readiness
        self.clock = clock
        self.lease_seconds = max(15.0, float(lease_seconds))
        self.scheduler_poll_seconds = max(0.05, float(scheduler_poll_seconds))
        self._worker: asyncio.Task | None = None
        self._shutdown_task: asyncio.Task | None = None
        self._stopping = False
        self._wakeup = asyncio.Event()
        self._volatile_materials: OrderedDict[str, dict] = OrderedDict()
        self._volatile_materials_cap: int = max(64, self.queue_size * 2)
        self._admissions = OrderedDict()
        self._admission_worker = None
        self._admission_stats = Counter()
        self._policy_wakeups = OrderedDict()
        self._policy_wakeup_worker = None
        self._policy_wakeup_stats = Counter()

    async def initialize(self) -> None:
        if self._shutdown_task and not self._shutdown_task.done():
            await self.close()
        await self.sidecar.initialize()
        self._shutdown_task = None
        self._stopping = False
        if self.derive_callback and (self._worker is None or self._worker.done()):
            self._wakeup.set()
            self._worker = asyncio.create_task(
                self._worker_loop(), name="livingmemory-evidence-deriver"
            )

    async def close(self) -> None:
        self._stopping = True
        self._wakeup.set()
        task = self._shutdown_task
        if task is None:
            task = asyncio.create_task(
                self._shutdown(), name="livingmemory-evidence-shutdown"
            )
            self._shutdown_task = task
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        if task.cancelled():
            raise asyncio.CancelledError
        task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _shutdown(self) -> None:
        wakeup = self._policy_wakeup_worker
        if wakeup and not wakeup.done():
            wakeup.cancel()
        if wakeup:
            await asyncio.gather(wakeup, return_exceptions=True)
        self._policy_wakeup_worker = None
        self._policy_wakeups.clear()
        admission = self._admission_worker
        if admission and not admission.done():
            admission.cancel()
        if admission:
            await asyncio.gather(admission, return_exceptions=True)
        self._admission_worker = None
        self._admissions.clear()
        worker = self._worker
        if worker and not worker.done():
            worker.cancel()
        if worker:
            await asyncio.gather(worker, return_exceptions=True)
        # Keep the worker handle visible until cancellation cleanup, including
        # lease release, has fully settled.
        self._worker = None
        self._volatile_materials.clear()
        await self.sidecar.close()

    def _remember_volatile_material(self, material_id: str, material: dict) -> None:
        """Retain bounded volatile fallback payloads only when no material_loader is configured."""
        if self.material_loader is not None:
            return
        locator = material.get("source_locator") if isinstance(material, dict) else None
        if isinstance(locator, dict) and locator.get("kind") == "conversation_row":
            return
        cap = max(64, self.queue_size * 2)
        self._volatile_materials[material_id] = material
        self._volatile_materials.move_to_end(material_id)
        while len(self._volatile_materials) > cap:
            self._volatile_materials.popitem(last=False)

    def admission_diagnostics(self) -> dict:
        return {**self._admission_stats, "pending": len(self._admissions)}

    def _defer_admissions(self, prepared, scope, purpose, card_version):
        """Schedule durable admission without delaying a cached evidence read."""
        if (
            not self.derive_callback
            or self._stopping
            or not self._worker
            or self._worker.done()
        ):
            return {
                "queued": 0,
                "deferred": 0,
                "skipped": 0,
                "available": False,
                "admission_pending": 0,
            }
        accepted, deferred = 0, 0
        for descriptor, material in prepared:
            key = (
                descriptor["material_id"],
                descriptor.get("projection_key"),
                descriptor.get("correction_revision"),
                purpose,
                card_version,
            )
            if key in self._admissions:
                self._admission_stats["coalesced"] += 1
                accepted += 1
                continue
            if len(self._admissions) >= self.queue_size:
                deferred += 1
                continue
            self._admissions[key] = (
                copy.deepcopy(material),
                copy.deepcopy(scope),
                purpose,
                card_version,
            )
            accepted += 1
        if self._admissions and (
            self._admission_worker is None or self._admission_worker.done()
        ):
            self._admission_worker = asyncio.create_task(
                self._run_admissions(), name="livingmemory-evidence-admission"
            )
        self._admission_stats["scheduled"] += accepted
        self._admission_stats["deferred"] += deferred
        return {
            "queued": 0,
            "deferred": deferred,
            "skipped": 0,
            "available": True,
            "admission_pending": accepted,
        }

    async def _run_admissions(self):
        try:
            while self._admissions and not self._stopping:
                key, (material, scope, purpose, card) = next(
                    iter(self._admissions.items())
                )
                try:
                    # Admission rechecks revision and owner readiness. The
                    # derivation worker re-reads authoritative input and policy.
                    result = await asyncio.wait_for(
                        self.queue_material_derivation(
                            [material], scope=scope, purpose=purpose, card_version=card
                        ),
                        timeout=10,
                    )
                    self._admission_stats["persisted"] += result.get("queued", 0)
                    self._admission_stats["processed"] += 1
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self._admission_stats["failed"] += 1
                finally:
                    self._admissions.pop(key, None)
        finally:
            if self._admission_worker is asyncio.current_task():
                self._admission_worker = None

    def wake_scheduler(self) -> None:
        """Wake the durable scheduler after a fake-clock advance or new admission."""
        self._wakeup.set()

    async def wake_policy_jobs(self, scope: dict) -> int:
        """Wake all persona scopes for a UMO after its trusted owner arrives."""
        umo = scope.get("umo") if isinstance(scope, dict) else None
        if not isinstance(umo, str) or not umo:
            return 0
        count = await self.sidecar.wake_policy_deferred(umo, self.clock())
        self._wakeup.set()
        return count

    def defer_policy_wakeup(self, scope: dict) -> bool:
        """Queue scheduler bookkeeping after the trusted owner has been captured.

        Owner resolution and policy/source checks remain synchronous in their
        callers. This only wakes durable owner_wait jobs and never authorizes
        or derives a material. Repeated wakes for one UMO share one worker.
        """
        umo = scope.get("umo") if isinstance(scope, dict) else None
        if (
            not isinstance(umo, str)
            or not umo
            or self._stopping
            or self._worker is None
            or self._worker.done()
        ):
            return False
        if umo in self._policy_wakeups:
            self._policy_wakeup_stats["coalesced"] += 1
            return True
        if len(self._policy_wakeups) >= self.queue_size:
            self._policy_wakeup_stats["deferred"] += 1
            return False
        self._policy_wakeups[umo] = dict(scope)
        self._policy_wakeup_stats["scheduled"] += 1
        if self._policy_wakeup_worker is None or self._policy_wakeup_worker.done():
            self._policy_wakeup_worker = asyncio.create_task(
                self._run_policy_wakeups(), name="livingmemory-policy-wakeup"
            )
        return True

    def policy_wakeup_diagnostics(self) -> dict:
        return {**self._policy_wakeup_stats, "pending": len(self._policy_wakeups)}

    async def _run_policy_wakeups(self):
        try:
            while self._policy_wakeups and not self._stopping:
                umo, scope = next(iter(self._policy_wakeups.items()))
                try:
                    count = await self.wake_policy_jobs(scope)
                    self._policy_wakeup_stats["processed"] += 1
                    self._policy_wakeup_stats["jobs_woken"] += count
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self._policy_wakeup_stats["failed"] += 1
                finally:
                    self._policy_wakeups.pop(umo, None)
        finally:
            if self._policy_wakeup_worker is asyncio.current_task():
                self._policy_wakeup_worker = None

    async def ensure_conversation_row_ready(
        self,
        scope: dict,
        descriptor: dict,
        *,
        policy_check: Callable[[dict, dict], Awaitable[bool | None]] | None = None,
    ) -> dict:
        """Create a deterministic READY quotation from one freshly verified user row.

        This path is intentionally independent of the LLM deriver. The caller
        supplies a descriptor produced from an authorized memory-source
        snapshot; this method re-reads the exact row through the configured
        loader and verifies its canonical identity again before persistence.
        """
        if not isinstance(scope, dict) or not isinstance(descriptor, dict):
            return {"status": "invalid", "material_id": "", "atom_ids": []}
        locator = descriptor.get("source_locator")
        if (
            not isinstance(locator, dict)
            or locator.get("kind") != "conversation_row"
            or locator.get("source_store") != "livingmemory.conversations.messages"
            or not self.conversation_row_loader
        ):
            return {
                "status": "invalid",
                "material_id": str(descriptor.get("material_id", "")),
                "atom_ids": [],
            }
        try:
            scope_id = self.scope_key_fn(scope)
        except Exception:
            return {
                "status": "invalid",
                "material_id": str(descriptor.get("material_id", "")),
                "atom_ids": [],
            }
        if descriptor.get("scope_key") != scope_id or locator.get("scope") != {
            key: str(scope.get(key, ""))
            for key in ("platform", "umo", "persona_id", "instance_id")
        }:
            return {
                "status": "invalid",
                "material_id": str(descriptor.get("material_id", "")),
                "atom_ids": [],
            }
        revision = self.sidecar.peek_evidence_revision(scope_id)
        if not revision.get("available") or revision.get(
            "correction_revision"
        ) != descriptor.get("correction_revision"):
            return {
                "status": "invalid",
                "material_id": str(descriptor.get("material_id", "")),
                "atom_ids": [],
            }
        revoked = {str(row.get("target_id")) for row in revision.get("revocations", [])}
        for row in revision.get("revocations", []):
            revoked.update(str(value) for value in row.get("affected_ids", []))
        if descriptor.get("material_id") in revoked:
            return {
                "status": "invalid",
                "material_id": str(descriptor.get("material_id", "")),
                "atom_ids": [],
            }
        try:
            raw_row = await self.conversation_row_loader(descriptor)
        except asyncio.CancelledError:
            raise
        except Exception:
            raw_row = None
        if not isinstance(raw_row, dict):
            return {
                "status": "invalid",
                "material_id": str(descriptor.get("material_id", "")),
                "atom_ids": [],
            }
        raw_material = dict(raw_row)
        raw_material.update(
            {
                "source_kind": "conversation_row",
                "source_memory_id": locator.get("source_memory_id"),
            }
        )
        checked = self._normalize(
            raw_material,
            scope,
            str(descriptor.get("card_version", "")),
            str(locator.get("purpose", "reply")),
            int(descriptor.get("correction_revision", 0)),
        )
        if not checked:
            return {
                "status": "invalid",
                "material_id": str(descriptor.get("material_id", "")),
                "atom_ids": [],
            }
        current_descriptor, material = checked
        identity_fields = (
            "material_id",
            "scope_key",
            "material_kind",
            "text_hash",
            "projection_key",
            "correction_revision",
            "position",
            "conversation_id",
            "snapshot_fingerprint",
        )
        if any(
            current_descriptor.get(key) != descriptor.get(key)
            for key in identity_fields
        ):
            return {
                "status": "invalid",
                "material_id": str(descriptor.get("material_id", "")),
                "atom_ids": [],
            }
        current_locator = current_descriptor.get("source_locator", {})
        if any(
            current_locator.get(key) != locator.get(key)
            for key in ("kind", "session_id", "row_id", "row_hash", "source_memory_id")
        ):
            return {
                "status": "invalid",
                "material_id": str(descriptor.get("material_id", "")),
                "atom_ids": [],
            }
        policy_allowed = await self._check_material_policy(
            current_descriptor, material, policy_check
        )
        if policy_allowed is None:
            return {
                "status": "policy_unavailable",
                "material_id": descriptor["material_id"],
                "atom_ids": [],
            }
        if not policy_allowed:
            return {
                "status": "invalid",
                "material_id": descriptor["material_id"],
                "atom_ids": [],
            }
        await self.sidecar.upsert_material(current_descriptor)
        existing = await self.sidecar.get_materials(
            scope_id, [current_descriptor["material_id"]]
        )
        cached = existing.get(current_descriptor["material_id"], {})
        ready_ids = [
            entry.get("atom_id")
            for entry in cached.get("atoms", [])
            if entry.get("status") in PUBLIC_ATOM_STATUSES
            and isinstance(entry.get("atom_id"), str)
        ]
        if cached.get("status") == "ready" and ready_ids:
            return {
                "status": "ready",
                "material_id": current_descriptor["material_id"],
                "atom_ids": ready_ids,
            }
        candidate = {
            "kind": "utterance_fact",
            "claim": material["text"][:500],
            "span_start": 0,
            "span_end": min(len(material["text"]), 500),
        }
        atoms = self._validated_atoms(current_descriptor, material, [candidate])
        if not atoms:
            return {
                "status": "invalid",
                "material_id": current_descriptor["material_id"],
                "atom_ids": [],
            }
        # Re-read the source and policy immediately before committing, so a
        # changed/deleted row or a newly active shield cannot become READY.
        try:
            latest_row = await self.conversation_row_loader(current_descriptor)
        except asyncio.CancelledError:
            raise
        except Exception:
            latest_row = None
        latest_raw = dict(latest_row) if isinstance(latest_row, dict) else None
        if latest_raw is None:
            return {
                "status": "invalid",
                "material_id": current_descriptor["material_id"],
                "atom_ids": [],
            }
        latest_raw.update(
            {
                "source_kind": "conversation_row",
                "source_memory_id": locator.get("source_memory_id"),
            }
        )
        latest_checked = self._normalize(
            latest_raw,
            scope,
            str(descriptor.get("card_version", "")),
            str(locator.get("purpose", "reply")),
            int(descriptor.get("correction_revision", 0)),
        )
        current_revision = self.sidecar.peek_evidence_revision(scope_id)
        if (
            not latest_checked
            or latest_checked[0].get("material_id") != current_descriptor["material_id"]
            or not current_revision.get("available")
            or current_revision.get("correction_revision")
            != descriptor.get("correction_revision")
        ):
            return {
                "status": "invalid",
                "material_id": current_descriptor["material_id"],
                "atom_ids": [],
            }
        policy_allowed = await self._check_material_policy(
            current_descriptor, latest_checked[1], policy_check
        )
        if policy_allowed is None:
            return {
                "status": "policy_unavailable",
                "material_id": current_descriptor["material_id"],
                "atom_ids": [],
            }
        if not policy_allowed:
            return {
                "status": "invalid",
                "material_id": current_descriptor["material_id"],
                "atom_ids": [],
            }
        stored = await self.sidecar.store_atoms(
            current_descriptor, atoms, now=self.clock()
        )
        return {
            "status": "ready" if stored else "invalid",
            "material_id": current_descriptor["material_id"],
            "atom_ids": [atom["atom_id"] for atom in atoms] if stored else [],
        }

    async def list_ready_atoms(self, scope: dict, limit: int = 100) -> list[dict]:
        """Return canonical persisted atoms for a scope, with status and revision."""
        try:
            scope_id = self.scope_key_fn(scope)
        except Exception:
            return []
        revision = self.sidecar.peek_evidence_revision(scope_id)
        if not revision.get("available"):
            return []
        revoked = {str(row.get("target_id")) for row in revision.get("revocations", [])}
        invalidated = set()
        for row in revision.get("revocations", []):
            affected = {str(value) for value in row.get("affected_ids", [])}
            revoked.update(affected)
            invalidated.update(affected)
            if row.get("action") in {"withdraw", "dispute", "supersede"}:
                invalidated.add(str(row.get("target_id")))
        rows = await self.sidecar.list_atoms(scope_id, limit)
        result = []
        for row in rows:
            if row.get("status") not in PUBLIC_ATOM_STATUSES or not isinstance(
                row.get("atom"), dict
            ):
                continue
            atom = row["atom"]
            roots = {str(value) for value in atom.get("root_evidence_ids", [])}
            material_roots = {str(value) for value in atom.get("root_material_ids", [])}
            if (
                row.get("atom_id") in invalidated
                or not (roots or material_roots)
                or roots & revoked
                or material_roots & revoked
            ):
                continue
            result.append(
                {
                    "atom_id": row["atom_id"],
                    "status": row["status"],
                    "revision": row["revision"],
                    "correction_revision": revision["correction_revision"],
                    "scope_key": scope_id,
                    "atom": atom,
                }
            )
        return result

    async def validate_ready_material(
        self,
        scope: dict,
        descriptor: dict,
        *,
        policy_check: Callable[[dict, dict], Awaitable[bool | None]] | None = None,
    ) -> bool:
        """Revalidate a conversation-row locator and its READY source atom."""
        if not isinstance(scope, dict) or not isinstance(descriptor, dict):
            return False
        locator = descriptor.get("source_locator")
        if (
            not isinstance(locator, dict)
            or locator.get("kind") != "conversation_row"
            or locator.get("source_store") != "livingmemory.conversations.messages"
            or not self.conversation_row_loader
        ):
            return False
        try:
            scope_id = self.scope_key_fn(scope)
        except Exception:
            return False
        if descriptor.get("scope_key") != scope_id or locator.get("scope") != {
            key: str(scope.get(key, ""))
            for key in ("platform", "umo", "persona_id", "instance_id")
        }:
            return False
        revision = self.sidecar.peek_evidence_revision(scope_id)
        if not revision.get("available") or revision.get(
            "correction_revision"
        ) != descriptor.get("correction_revision"):
            return False
        revoked = {str(row.get("target_id")) for row in revision.get("revocations", [])}
        for row in revision.get("revocations", []):
            revoked.update(str(value) for value in row.get("affected_ids", []))
        if descriptor.get("material_id") in revoked:
            return False
        try:
            row = await self.conversation_row_loader(descriptor)
        except asyncio.CancelledError:
            raise
        except Exception:
            return False
        if not isinstance(row, dict):
            return False
        raw = {
            **row,
            "source_kind": "conversation_row",
            "source_memory_id": locator.get("source_memory_id"),
        }
        checked = self._normalize(
            raw,
            scope,
            str(descriptor.get("card_version", "")),
            str(locator.get("purpose", "reply")),
            int(descriptor.get("correction_revision", 0)),
        )
        if not checked:
            return False
        current, material = checked
        current_locator = current.get("source_locator", {})
        fields = (
            "material_id",
            "scope_key",
            "material_kind",
            "text_hash",
            "projection_key",
            "correction_revision",
            "position",
            "conversation_id",
            "snapshot_fingerprint",
        )
        if any(current.get(key) != descriptor.get(key) for key in fields):
            return False
        if any(
            current_locator.get(key) != locator.get(key)
            for key in ("kind", "session_id", "row_id", "row_hash", "source_memory_id")
        ):
            return False
        policy_allowed = await self._check_material_policy(
            current, material, policy_check
        )
        if policy_allowed is not True:
            return False
        canonical = await self.sidecar.get_materials(scope_id, [current["material_id"]])
        record = canonical.get(current["material_id"], {})
        if record.get("status") != "ready" or not record.get("atoms"):
            return False
        return any(
            entry.get("status") in PUBLIC_ATOM_STATUSES
            and current["material_id"] in entry.get("root_material_ids", [])
            and any(
                isinstance(ref, dict)
                and ref.get("source_kind") == "conversation_row"
                and ref.get("row_hash") == current_locator.get("row_hash")
                and ref.get("source_memory_id")
                == current_locator.get("source_memory_id")
                and ref.get("author_id") == material.get("author_id")
                for ref in entry.get("atom", {}).get("source_refs", [])
            )
            for entry in record.get("atoms", [])
        )

    async def get_material_descriptors(
        self, scope: dict, material_ids: list[str]
    ) -> dict[str, dict]:
        """Return exact-scope, body-free sidecar descriptors for READY recovery.

        The sidecar is selected by the canonical scope key. Its event locator
        is accepted only when its persisted event fingerprint recomputes to
        the same ev1 primary key; the current full scope is attached only after
        verifying that its key is the selected database scope.
        """
        if not isinstance(scope, dict) or not isinstance(material_ids, list):
            return {}
        try:
            scope_id = self.scope_key_fn(scope)
        except Exception:
            return {}
        rows = await self.sidecar.get_material_descriptors(scope_id, material_ids)
        current_scope = {
            key: str(scope.get(key, "")).strip()
            for key in ("platform", "umo", "persona_id", "instance_id")
        }
        if any(not value for value in current_scope.values()):
            return {}
        result = {}
        for material_id, raw in rows.items():
            if not isinstance(raw, dict) or raw.get("scope_key") != scope_id:
                continue
            descriptor = dict(raw)
            locator = descriptor.get("source_locator")
            if isinstance(locator, dict):
                if (
                    locator.get("scope_key") != scope_id
                    or locator.get("kind") != "event"
                ):
                    descriptor["source_locator"] = None
                else:
                    descriptor["source_locator"] = {
                        **locator,
                        "scope": dict(current_scope),
                    }
            result[material_id] = descriptor
        return result

    def peek_evidence_revision(self, scope: dict) -> dict:
        if self._stopping or not self.sidecar._ready:
            return {
                "schema_version": 1,
                "available": False,
                "scope_key": "",
                "correction_revision": None,
                "revocations": [],
            }
        try:
            key = self.scope_key_fn(scope)
        except Exception:
            return {
                "schema_version": 1,
                "available": False,
                "scope_key": "",
                "correction_revision": None,
                "revocations": [],
            }
        return self.sidecar.peek_evidence_revision(key)

    async def queue_material_derivation(
        self,
        materials: list[dict],
        *,
        scope: dict | None = None,
        purpose: str = "reply",
        card_version: str = "",
    ) -> dict:
        if purpose not in PURPOSES or not isinstance(materials, list):
            return {
                "schema_version": 1,
                "accepted": False,
                "queued": 0,
                "deferred": 0,
                "duplicates": 0,
                "skipped": 0,
                "material_ids": [],
            }
        if not materials:
            return {
                "schema_version": 1,
                "accepted": False,
                "queued": 0,
                "deferred": 0,
                "duplicates": 0,
                "skipped": 0,
                "available": True,
                "material_ids": [],
            }
        if (
            not self.derive_callback
            or self._stopping
            or not self._worker
            or self._worker.done()
        ):
            return {
                "schema_version": 1,
                "accepted": False,
                "queued": 0,
                "deferred": 0,
                "duplicates": 0,
                "skipped": 0,
                "available": False,
                "material_ids": [],
            }
        accepted = 0
        deferred = 0
        duplicates = 0
        terminal = 0
        skipped = 0
        ids: list[str] = []
        revision = (
            self.peek_evidence_revision(scope)
            if isinstance(scope, dict)
            else {"available": False}
        )
        if not revision.get("available") or not isinstance(
            revision.get("correction_revision"), int
        ):
            return {
                "schema_version": 1,
                "accepted": False,
                "queued": 0,
                "deferred": 0,
                "duplicates": 0,
                "skipped": min(len(materials), MAX_MATERIALS),
                "material_ids": [],
                "available": False,
            }
        correction_revision = revision["correction_revision"]
        for raw in materials[:MAX_MATERIALS]:
            prepared = self._normalize(
                raw, scope, card_version, purpose, correction_revision
            )
            if not prepared:
                skipped += 1
                continue
            descriptor, material = prepared
            material_id = descriptor["material_id"]
            ids.append(material_id)
            if await self.sidecar.has_ready_family(descriptor):
                duplicates += 1
                continue
            try:
                cache_status = await self.sidecar.upsert_material(descriptor)
            except Exception:
                skipped += 1
                continue
            if cache_status == "ready":
                duplicates += 1
                continue
            owner_ready = True
            if self.owner_readiness:
                source_scope = descriptor.get("source_locator", {}).get("scope", {})
                try:
                    owner_ready = (
                        await self.owner_readiness(str(source_scope.get("umo", "")))
                        is True
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    owner_ready = False
            outcome = await self.sidecar.admit_material_job(
                descriptor,
                now=self.clock(),
                queue_size=self.queue_size,
                owner_ready=owner_ready,
            )
            if outcome == "queued":
                accepted += 1
                self._remember_volatile_material(material_id, material)
            elif outcome == "deferred":
                deferred += 1
            elif outcome == "owner_wait":
                deferred += 1
                self._remember_volatile_material(material_id, material)
            elif outcome == "duplicate":
                duplicates += 1
            elif outcome == "terminal":
                terminal += 1
            else:
                skipped += 1
        if accepted or deferred:
            self._wakeup.set()
        return {
            "schema_version": 1,
            "accepted": accepted + deferred > 0,
            "queued": accepted,
            "deferred": deferred,
            "duplicates": duplicates,
            "terminal": terminal,
            "skipped": skipped,
            "material_ids": ids,
        }

    async def get_evidence_view(
        self,
        scope: dict,
        source_refs: list[dict] | None = None,
        purpose: str = "reply",
        card_version: str = "",
        conversation_id: str | None = None,
        snapshot_fingerprint: str | None = None,
        materials: list[dict] | None = None,
        typed_items: list[dict] | None = None,
        allow_derivation: bool = True,
    ) -> dict:
        """Read only ready cache, enqueue misses, and return a bounded immutable projection."""
        if purpose not in PURPOSES:
            return self._empty_view("invalid_purpose", purpose, card_version)
        try:
            scope_id = self.scope_key_fn(scope)
        except Exception:
            return self._empty_view("invalid_scope", purpose)
        revision = self.sidecar.peek_evidence_revision(scope_id)
        if not revision.get("available"):
            return self._empty_view("revision_unavailable", purpose)
        rows = []
        for row in (source_refs or [])[:MAX_MATERIALS]:
            if isinstance(row, dict):
                copy_row = dict(row)
                if conversation_id and not copy_row.get("conversation_id"):
                    copy_row["conversation_id"] = conversation_id
                if snapshot_fingerprint and not copy_row.get("snapshot_fingerprint"):
                    copy_row["snapshot_fingerprint"] = snapshot_fingerprint
                rows.append(copy_row)
        if materials:
            rows.extend(
                dict(row)
                for row in materials[: max(0, MAX_MATERIALS - len(rows))]
                if isinstance(row, dict)
            )
        prepared = []
        missing_identity = 0
        invalid_refs = 0
        for raw in rows[:MAX_MATERIALS]:
            item = self._normalize(
                raw, scope, card_version, purpose, revision["correction_revision"]
            )
            if item is None:
                if raw.get(
                    "source_kind", raw.get("material_kind", "host_history")
                ) == "host_history" and (
                    not raw.get("conversation_id", conversation_id)
                    or not raw.get("snapshot_fingerprint", snapshot_fingerprint)
                ):
                    missing_identity += 1
                else:
                    invalid_refs += 1
                continue
            prepared.append(item)
        material_ids = [desc["material_id"] for desc, _ in prepared]
        revoked_ids = {str(row.get("target_id")) for row in revision["revocations"]}
        invalidated_atom_ids = set()
        for row in revision["revocations"]:
            revoked_ids.update(str(value) for value in row.get("affected_ids", []))
            affected = {str(value) for value in row.get("affected_ids", [])}
            replacement_id = str(row.get("replacement_atom_id") or "")
            affected.discard(replacement_id)
            invalidated_atom_ids.update(affected)
            if row.get("action") in {"withdraw", "dispute", "supersede"}:
                invalidated_atom_ids.add(str(row.get("target_id")))
        typed_rows = [
            row for row in (typed_items or [])[:MAX_MATERIALS] if isinstance(row, dict)
        ]
        typed_ids = [
            row.get("source_id")
            for row in typed_rows
            if row.get("item_kind") == "evidence_atom"
            and isinstance(row.get("source_id"), str)
        ]
        stored_typed = await self.sidecar.get_atoms_by_ids(scope_id, typed_ids)
        typed_atoms = []
        typed_atom_ids: set[str] = set()
        atom_revisions: dict[str, int] = {}
        typed_rejected = 0
        typed_seen = set()
        for raw in typed_rows:
            atom_id_value = raw.get("source_id")
            valid = (
                raw.get("item_kind") == "evidence_atom"
                and raw.get("evidence_schema_version") == 1
                and isinstance(atom_id_value, str)
                and atom_id_value.startswith("atm1:")
                and atom_id_value not in typed_seen
                and isinstance(raw.get("scope"), dict)
                and isinstance(raw.get("correction_revision"), int)
                and not isinstance(raw.get("correction_revision"), bool)
                and raw.get("correction_revision") == revision["correction_revision"]
                and raw.get("scope_key") == scope_id
            )
            if valid:
                try:
                    valid = self.scope_key_fn(raw["scope"]) == scope_id
                except Exception:
                    valid = False
            stored = stored_typed.get(atom_id_value) if valid else None
            if not stored or stored.get("status") not in PUBLIC_ATOM_STATUSES:
                valid = False
            if valid:
                atom = stored.get("atom", {})
                canonical_roots = sorted(
                    {str(value) for value in stored.get("root_evidence_ids", [])}
                )
                canonical_material_roots = sorted(
                    {str(value) for value in stored.get("root_material_ids", [])}
                )
                raw_roots = raw.get("root_evidence_ids")
                raw_material_roots = raw.get("root_material_ids")
                valid = (
                    atom_id_value not in invalidated_atom_ids
                    and bool(canonical_roots or canonical_material_roots)
                    and isinstance(raw_roots, list)
                    and sorted({str(value) for value in raw_roots}) == canonical_roots
                    and isinstance(raw_material_roots, list)
                    and sorted({str(value) for value in raw_material_roots})
                    == canonical_material_roots
                    and not (set(canonical_roots) & revoked_ids)
                    and not (set(canonical_material_roots) & revoked_ids)
                    and (
                        not isinstance(atom, dict)
                        or atom.get("atom_id", atom_id_value) == atom_id_value
                    )
                )
                if valid:
                    typed_atoms.append(self._public_atom(atom, stored["status"]))
                    atom_revisions[atom_id_value] = int(stored.get("revision", 0))
                    if atom.get("source_status") == "exact_event" or any(
                        isinstance(ref, dict) and ref.get("source_event_id")
                        for ref in atom.get("source_refs", [])
                    ):
                        typed_atom_ids.add(atom_id_value)
                    typed_seen.add(atom_id_value)
            if not valid:
                typed_rejected += 1
        family_ids = [
            desc.get("source_family_id", "")
            for desc, _ in prepared
            if desc.get("source_family_id")
        ]
        cache_rows = await self.sidecar.get_materials(
            scope_id, material_ids, family_ids
        )
        cache = {}
        for descriptor, _ in prepared:
            material_id = descriptor["material_id"]
            direct = cache_rows.get(material_id)
            if (
                direct
                and direct.get("text_hash") == descriptor["text_hash"]
                and direct.get("projection_key") == descriptor["projection_key"]
            ):
                cache[material_id] = direct
                continue
            family_id = descriptor.get("source_family_id", "")
            if family_id:
                candidates = [
                    row
                    for row in cache_rows.values()
                    if row.get("source_family_id") == family_id
                    and row.get("text_hash") == descriptor["text_hash"]
                    and row.get("projection_key") == descriptor["projection_key"]
                    and row.get("status") == "ready"
                ]
                if candidates:
                    cache[material_id] = max(
                        candidates, key=lambda row: row.get("material_id", "")
                    )
        # 先查缓存、再只对「未 ready」的 material 排队派生。
        # 原实现先对全部 prepared 排队、之后才读缓存：每个 material 会依次走
        # has_ready_family + upsert_material + admit_material_job，而每次 sidecar 调用
        # 都要等锁并新建一个 sqlite 连接；已 ready 的 material 本会被判为 duplicates。
        # 生产实测该循环主导了 get_evidence_view 的 4~9s（最多 64 个 material × 3 次调用）。
        # 返回值原本未被任何调用方使用，故此处无需保留计数结果。
        queueable = [
            (desc, material)
            for desc, material in prepared
            if desc["material_id"] not in revoked_ids
            and desc.get("source_family_id", "") not in revoked_ids
            and desc["material_id"] not in cache
        ]
        queued = (
            self._defer_admissions(queueable, scope, purpose, card_version)
            if allow_derivation and queueable
            else {
                "queued": 0,
                "deferred": 0,
                "skipped": 0,
                "accepted": False,
                "available": True,
            }
        )
        atoms = list(typed_atoms)
        history_projection = {}
        pending_count = 0
        omitted_count = 0
        cached_count = 0
        for descriptor, material in prepared:
            material_id = descriptor["material_id"]
            cached = cache.get(material_id, {"status": "queued", "atoms": []})
            role = descriptor["role"]
            active_for_material = []
            for entry in cached.get("atoms", []):
                atom = entry["atom"]
                if entry["atom_id"] in invalidated_atom_ids:
                    continue
                if entry["status"] not in PUBLIC_ATOM_STATUSES:
                    continue
                roots = sorted({str(x) for x in atom.get("root_evidence_ids", [])})
                material_roots = sorted(
                    {str(x) for x in atom.get("root_material_ids", [])}
                )
                if not roots and not material_roots:
                    continue
                if any(
                    root in {r.get("target_id") for r in revision["revocations"]}
                    for root in roots
                ):
                    continue
                # This record is deliberately bounded and source-attributed.
                public_atom = self._public_atom(atom, entry["status"])
                atoms.append(public_atom)
                atom_revisions[public_atom["atom_id"]] = int(entry.get("revision", 0))
                active_for_material.append(public_atom["atom_id"])
            if descriptor["material_kind"] != "host_record_only":
                if cached.get("status") == "ready":
                    cached_count += 1
                elif descriptor["material_kind"] == "legacy_summary":
                    omitted_count += 1
                else:
                    pending_count += 1
                continue
            if role == "assistant" and (
                material_id in revoked_ids
                or descriptor.get("source_family_id", "") in revoked_ids
            ):
                omitted_count += 1
                history_projection[material_id] = {
                    "material_id": material_id,
                    "source_position": descriptor["position"],
                    "source_role": role,
                    "text_hash": descriptor["text_hash"],
                    "projection_status": "admin_corrected_omitted",
                    "preserve_as_historical_data": False,
                    "spans": [],
                    "atom_ids": [],
                    "source_status": descriptor["material_kind"],
                }
                continue
            if cached.get("status") == "ready":
                cached_count += 1
                projection = {
                    "material_id": material_id,
                    "source_position": descriptor["position"],
                    "source_role": role,
                    "text_hash": descriptor["text_hash"],
                    "projection_status": "historical_data_preserved"
                    if role == "user"
                    else "historical_claims_only"
                    if role == "summary" and active_for_material
                    else "derived_no_claim"
                    if role == "summary"
                    else "atoms_only"
                    if active_for_material
                    else "derived_no_continuity",
                    "preserve_as_historical_data": role == "user",
                    "spans": [],
                    "atom_ids": sorted(set(active_for_material)),
                    "source_status": descriptor["material_kind"],
                }
            elif role == "user":
                pending_count += 1
                projection = {
                    "material_id": material_id,
                    "source_position": descriptor["position"],
                    "source_role": role,
                    "text_hash": descriptor["text_hash"],
                    "projection_status": "historical_data_preserved_pending_derivation",
                    "preserve_as_historical_data": True,
                    "spans": [],
                    "atom_ids": [],
                    "source_status": descriptor["material_kind"],
                }
            elif role == "summary":
                omitted_count += 1
                projection = {
                    "material_id": material_id,
                    "source_position": descriptor["position"],
                    "source_role": role,
                    "text_hash": descriptor["text_hash"],
                    "projection_status": "legacy_summary_unparsed",
                    "preserve_as_historical_data": False,
                    "spans": [],
                    "atom_ids": [],
                    "source_status": "legacy_summary",
                }
            else:
                spans = pending_continuity_spans(material["text"])
                if spans:
                    pending_count += 1
                    projection_status = "degraded_pending_continuity"
                else:
                    omitted_count += 1
                    projection_status = "omitted_unclassified"
                projection = {
                    "material_id": material_id,
                    "source_position": descriptor["position"],
                    "source_role": role,
                    "text_hash": descriptor["text_hash"],
                    "projection_status": projection_status,
                    "preserve_as_historical_data": False,
                    "spans": spans,
                    "atom_ids": [],
                    "source_status": descriptor["material_kind"],
                }
            history_projection[material_id] = projection
        restored, invalid_event_atoms = await self._restore_event_provenance(
            atoms,
            atom_revisions,
            scope,
            scope_id,
            revision["correction_revision"],
            revoked_ids,
            invalidated_atom_ids,
            prepared,
        )
        if invalid_event_atoms:
            typed_rejected += len(invalid_event_atoms & typed_atom_ids)
            atoms = [
                atom for atom in atoms if atom.get("atom_id") not in invalid_event_atoms
            ]
            typed_atoms = [
                atom
                for atom in typed_atoms
                if atom.get("atom_id") not in invalid_event_atoms
            ]
        deduped_atoms = self._dedupe_atoms(atoms)
        slots = self._slots(deduped_atoms)
        fallback_span_count = sum(
            len(row.get("spans", [])) for row in history_projection.values()
        )
        policy_unavailable_count = sum(
            1
            for row in cache_rows.values()
            if row.get("diagnostics_code") == "policy_unavailable"
        )
        diagnostics = {
            "materials_total": len(prepared),
            "materials_cached": cached_count,
            "materials_queued": int(queued.get("queued", 0)),
            "materials_pending": pending_count,
            "materials_admission_pending": int(queued.get("admission_pending", 0)),
            "materials_terminal": int(queued.get("terminal", 0)),
            "materials_skipped": int(queued.get("skipped", 0)),
            "derivation_available": queued.get("available", True) is not False,
            "materials_omitted_unclassified": omitted_count,
            "materials_missing_snapshot_identity": missing_identity,
            "source_refs_rejected": invalid_refs,
            "typed_items_total": len(typed_rows),
            "typed_items_accepted": len(typed_atoms),
            "source_authors_restored": restored,
            "typed_items_rejected": typed_rejected,
            "materials_deferred": int(queued.get("deferred", 0)),
            "policy_unavailable": policy_unavailable_count,
            "fallback_spans_returned": fallback_span_count,
            "source_text_logged": False,
            "raw_source_returned": False,
            "projection_version": "evidence-view-v1",
            "purpose": purpose,
        }
        view = {
            "schema_version": 1,
            "available": True,
            "scope_key": scope_id,
            "correction_revision": revision["correction_revision"],
            "revocations": revision["revocations"],
            "purpose": purpose,
            "card_version": str(card_version)[:256],
            "slots": slots,
            "atoms": deduped_atoms,
            "history_projection": history_projection,
            "diagnostics": diagnostics,
        }
        view["evidence_status"] = self._evidence_status(view)
        return view

    async def correct_insight(
        self,
        target_id: str,
        expected_revision: int,
        action: str,
        replacement: str = "",
        reason: str = "",
        action_id: str = "",
    ) -> dict:
        return await self.sidecar.correct(
            target_id, expected_revision, action, replacement, reason, action_id
        )

    async def list_admin_insights(self, scope: dict, limit: int = 100) -> dict:
        try:
            scope_id = self.scope_key_fn(scope)
        except Exception:
            return {"schema_version": 1, "available": False, "items": []}
        rows = await self.sidecar.list_atoms(scope_id, limit)
        # Admin diagnostics expose identifiers and provenance labels, never source text.
        items = [
            {
                "atom_id": row["atom_id"],
                "kind": row["kind"],
                "status": row["status"],
                "revision": row["revision"],
                "source_count": len(row["atom"].get("root_evidence_ids", [])),
                "material_count": len(row["atom"].get("root_material_ids", [])),
                "source_kinds": sorted(
                    {
                        ref.get("source_status", "unknown")
                        for ref in row["atom"].get("source_refs", [])
                        if isinstance(ref, dict)
                    }
                ),
            }
            for row in rows
        ]
        rev = self.sidecar.peek_evidence_revision(scope_id)
        return {
            "schema_version": 1,
            "available": True,
            "scope_key": scope_id,
            "correction_revision": rev["correction_revision"],
            "items": items,
        }

    async def list_admin_insights_global(self, limit: int = 100) -> dict:
        rows = await self.sidecar.list_admin_atoms(limit)
        materials = await self.sidecar.list_admin_materials(limit)
        # Do not expose claim text, original source text, reasons, or provider data.
        return {
            "schema_version": 1,
            "available": True,
            "items": [
                *[
                    {
                        "atom_id": row["atom_id"],
                        "scope_key": row["scope_key"],
                        "kind": row["kind"],
                        "status": row["status"],
                        "source_count": row["source_count"],
                        "revision": row["revision"],
                        "correction_revision": row["correction_revision"],
                        "updated_at": row["updated_at"],
                    }
                    for row in rows
                ],
                *[
                    {
                        "atom_id": row["material_id"],
                        "scope_key": row["scope_key"],
                        "kind": "material:" + row["material_kind"],
                        "status": row["status"],
                        "attempt_count": row["attempt_count"],
                        "next_attempt_at": row["next_attempt_at"],
                        "outcome_code": row["outcome_code"],
                        "source_count": 0,
                        "revision": 0,
                        "correction_revision": row["correction_revision"],
                        "updated_at": row["updated_at"],
                    }
                    for row in materials
                ],
            ],
        }

    def _normalize(
        self,
        raw: dict,
        scope: dict | None,
        card_version: str,
        purpose: str,
        revision: int = 0,
    ):
        if not isinstance(scope, dict):
            return None
        try:
            scope_id = self.scope_key_fn(scope)
        except Exception:
            return None
        prepared = normalize_host_material(
            scope_id,
            raw,
            conversation_id=str(raw.get("conversation_id", "")),
            snapshot_fingerprint=str(raw.get("snapshot_fingerprint", "")),
            card_version=card_version,
            correction_revision=revision,
            purpose=purpose,
        )
        if not prepared:
            prepared = normalize_event_material(
                scope_id,
                scope,
                raw,
                card_version=card_version,
                correction_revision=revision,
                purpose=purpose,
            )
        if not prepared:
            prepared = normalize_legacy_summary(
                scope_id,
                raw,
                card_version=card_version,
                correction_revision=revision,
                purpose=purpose,
            )
        if not prepared and raw.get("source_kind") == "conversation_row":
            prepared = normalize_conversation_row(
                scope_id,
                scope,
                raw,
                source_memory_id=raw.get("source_memory_id"),
                card_version=card_version,
                correction_revision=revision,
                purpose=purpose,
            )
        if prepared:
            descriptor, material = prepared
            locator = dict(descriptor.get("source_locator", {}))
            locator["scope"] = {
                key: str(scope.get(key, ""))
                for key in ("platform", "umo", "persona_id", "instance_id")
            }
            descriptor["source_locator"] = locator
            material["source_locator"] = locator
        return prepared

    async def _worker_loop(self):
        while not self._stopping:
            if self.owner_readiness:
                try:
                    umos = await self.sidecar.due_policy_umos(self.clock())
                except Exception:
                    umos = []
                for umo in umos:
                    try:
                        owner_ready = await self.owner_readiness(umo)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        owner_ready = None
                    if owner_ready is not True:
                        await self.sidecar.pause_policy_jobs(umo, self.clock())
            jobs = await self.sidecar.claim_due_materials(
                now=self.clock(),
                queue_size=self.queue_size,
                per_scope_hourly_limit=self.per_scope_hourly_limit,
                lease_seconds=self.lease_seconds,
                batch_size=4,
            )
            if not jobs:
                self._wakeup.clear()
                try:
                    await asyncio.wait_for(
                        self._wakeup.wait(), timeout=self.scheduler_poll_seconds
                    )
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                batch = []
                for job in jobs:
                    volatile_material = self._volatile_materials.pop(
                        job["material_id"], None
                    )
                    try:
                        # Re-read authoritative source state even for jobs first
                        # admitted in this process; volatile payloads are only a
                        # fallback for isolated users without a loader.
                        locator = job.get("source_locator", {})
                        if locator.get("kind") == "conversation_row":
                            material = (
                                await self.conversation_row_loader(job)
                                if self.conversation_row_loader
                                else None
                            )
                        elif self.material_loader:
                            material = await self.material_loader(job)
                        else:
                            material = volatile_material
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        await self.sidecar.record_material_failure(
                            job["material_id"],
                            job["scope_key"],
                            self._failure_outcome(exc),
                            job["lease_id"],
                            now=self.clock(),
                        )
                        continue
                    if not isinstance(material, dict):
                        await self.sidecar.record_material_failure(
                            job["material_id"],
                            job["scope_key"],
                            "source_unavailable",
                            job["lease_id"],
                            now=self.clock(),
                        )
                        continue
                    locator = job.get("source_locator", {})
                    scope = locator.get("scope") if isinstance(locator, dict) else None
                    revision = (
                        self.peek_evidence_revision(scope)
                        if isinstance(scope, dict)
                        else {"available": False}
                    )
                    if not revision.get("available") or revision.get(
                        "correction_revision"
                    ) != job.get("correction_revision"):
                        await self.sidecar.record_material_failure(
                            job["material_id"],
                            job["scope_key"],
                            "source_unavailable",
                            job["lease_id"],
                            now=self.clock(),
                        )
                        continue
                    checked = self._normalize(
                        material,
                        scope,
                        job.get("card_version", ""),
                        str(locator.get("purpose", "reply")),
                        int(job.get("correction_revision", 0)),
                    )
                    if not checked or any(
                        checked[0].get(key) != job.get(key)
                        for key in (
                            "material_id",
                            "scope_key",
                            "material_kind",
                            "text_hash",
                            "projection_key",
                        )
                    ):
                        await self.sidecar.record_material_failure(
                            job["material_id"],
                            job["scope_key"],
                            "source_unavailable",
                            job["lease_id"],
                            now=self.clock(),
                        )
                        continue
                    policy_allowed = await self._material_allowed(job, checked[1])
                    if policy_allowed is None:
                        await self.sidecar.record_material_failure(
                            job["material_id"],
                            job["scope_key"],
                            "policy_unavailable",
                            job["lease_id"],
                            now=self.clock(),
                        )
                        continue
                    if not policy_allowed:
                        await self.sidecar.record_material_failure(
                            job["material_id"],
                            job["scope_key"],
                            "policy_refusal",
                            job["lease_id"],
                            now=self.clock(),
                        )
                        continue
                    renewed = await self.sidecar.renew_material_lease(
                        job["material_id"],
                        job["scope_key"],
                        job["lease_id"],
                        self.clock(),
                        self.lease_seconds,
                    )
                    if not renewed:
                        continue
                    batch.append((job, checked[1]))
                if not batch:
                    continue
                if not self.derive_callback:
                    for descriptor, _ in batch:
                        await self.sidecar.record_material_failure(
                            descriptor["material_id"],
                            descriptor["scope_key"],
                            "provider_unavailable",
                            descriptor["lease_id"],
                            now=self.clock(),
                        )
                    continue
                materials = [material for _, material in batch]
                results = await self._derive_under_leases(batch, materials)
                if results is None:
                    # A lease was lost; the new owner will load and process the
                    # durable source locator. Do not persist stale derivation.
                    continue
                if not isinstance(results, dict):
                    results = {}
                for descriptor, material in batch:
                    material_id = descriptor["material_id"]
                    if material_id not in results:
                        await self.sidecar.record_material_failure(
                            material_id,
                            descriptor["scope_key"],
                            "invalid_schema",
                            descriptor["lease_id"],
                            now=self.clock(),
                        )
                        continue
                    values = results[material_id]
                    if not isinstance(values, list):
                        await self.sidecar.record_material_failure(
                            material_id,
                            descriptor["scope_key"],
                            "invalid_schema",
                            descriptor["lease_id"],
                            now=self.clock(),
                        )
                        continue
                    commit_material = material
                    if self.material_loader:
                        try:
                            latest = await self.material_loader(descriptor)
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            latest = None
                        locator = descriptor.get("source_locator", {})
                        scope = (
                            locator.get("scope") if isinstance(locator, dict) else None
                        )
                        checked_latest = (
                            self._normalize(
                                latest,
                                scope,
                                descriptor.get("card_version", ""),
                                str(locator.get("purpose", "reply")),
                                int(descriptor.get("correction_revision", 0)),
                            )
                            if isinstance(latest, dict)
                            else None
                        )
                        if not checked_latest or any(
                            checked_latest[0].get(key) != descriptor.get(key)
                            for key in (
                                "material_id",
                                "scope_key",
                                "material_kind",
                                "text_hash",
                                "projection_key",
                            )
                        ):
                            await self.sidecar.record_material_failure(
                                material_id,
                                descriptor["scope_key"],
                                "source_unavailable",
                                descriptor["lease_id"],
                                now=self.clock(),
                            )
                            continue
                        commit_material = checked_latest[1]
                    valid = self._validated_atoms(descriptor, commit_material, values)
                    candidate_values = values[:12]
                    if candidate_values and len(valid) != len(candidate_values):
                        await self.sidecar.record_material_failure(
                            material_id,
                            descriptor["scope_key"],
                            "invalid_schema",
                            descriptor["lease_id"],
                            now=self.clock(),
                        )
                        continue
                    policy_allowed = await self._material_allowed(
                        descriptor, commit_material
                    )
                    if policy_allowed is None:
                        await self.sidecar.record_material_failure(
                            material_id,
                            descriptor["scope_key"],
                            "policy_unavailable",
                            descriptor["lease_id"],
                            now=self.clock(),
                            admission_spent=True,
                        )
                        continue
                    if not policy_allowed:
                        await self.sidecar.record_material_failure(
                            material_id,
                            descriptor["scope_key"],
                            "policy_refusal",
                            descriptor["lease_id"],
                            now=self.clock(),
                        )
                        continue
                    await self.sidecar.store_atoms(
                        descriptor, valid, descriptor["lease_id"], now=self.clock()
                    )
            except asyncio.CancelledError:
                releases = [
                    asyncio.create_task(
                        self.sidecar.release_material_lease(
                            descriptor["material_id"],
                            descriptor["scope_key"],
                            descriptor["lease_id"],
                            self.clock(),
                        )
                    )
                    for descriptor in jobs
                ]
                if releases:
                    release_group = asyncio.gather(*releases, return_exceptions=True)
                    while not release_group.done():
                        try:
                            await asyncio.shield(release_group)
                        except asyncio.CancelledError:
                            continue
                raise
            except Exception as exc:
                outcome = self._failure_outcome(exc)
                for descriptor in jobs:
                    await self.sidecar.record_material_failure(
                        descriptor["material_id"],
                        descriptor["scope_key"],
                        outcome,
                        descriptor["lease_id"],
                        now=self.clock(),
                    )
            finally:
                self._wakeup.set()

    async def _material_allowed(self, descriptor: dict, material: dict) -> bool | None:
        if self.material_policy is None:
            return True
        try:
            result = await self.material_policy(descriptor, material)
            return result if isinstance(result, bool) else None
        except asyncio.CancelledError:
            raise
        except Exception:
            return None

    async def _check_material_policy(
        self,
        descriptor: dict,
        material: dict,
        policy_check: Callable[[dict, dict], Awaitable[bool | None]] | None,
    ) -> bool | None:
        if policy_check is None:
            return await self._material_allowed(descriptor, material)
        try:
            result = await policy_check(descriptor, material)
            return result if isinstance(result, bool) else None
        except asyncio.CancelledError:
            raise
        except Exception:
            return None

    async def _derive_under_leases(
        self, batch: list[tuple[dict, dict]], materials: list[dict]
    ):
        lost = asyncio.Event()

        async def renew_loop():
            interval = max(1.0, self.lease_seconds / 3.0)
            while not self._stopping:
                await asyncio.sleep(interval)
                for descriptor, _ in batch:
                    renewed = await self.sidecar.renew_material_lease(
                        descriptor["material_id"],
                        descriptor["scope_key"],
                        descriptor["lease_id"],
                        self.clock(),
                        self.lease_seconds,
                    )
                    if not renewed:
                        lost.set()
                        return

        callback_task = asyncio.create_task(self.derive_callback(materials))
        renew_task = asyncio.create_task(renew_loop())
        lost_task = asyncio.create_task(lost.wait())
        try:
            done, _ = await asyncio.wait(
                (callback_task, lost_task), return_when=asyncio.FIRST_COMPLETED
            )
            if lost_task in done and lost.is_set() and not callback_task.done():
                callback_task.cancel()
                await asyncio.gather(callback_task, return_exceptions=True)
                return None
            return await callback_task
        finally:
            if not callback_task.done():
                callback_task.cancel()
            for task in (renew_task, lost_task):
                task.cancel()
            await asyncio.gather(
                callback_task, renew_task, lost_task, return_exceptions=True
            )

    @staticmethod
    def _failure_outcome(error: Exception) -> str:
        from .utils.api_failure import terminal_provider_failure

        if terminal_provider_failure(error):
            return "policy_refusal"
        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return "timeout"
        message = str(error)
        if message in {
            "summarizer_unavailable",
            "summarizer_changed",
            "LLM Provider 不可用",
        }:
            return "provider_unavailable"
        if message.startswith("invalid_derivation_"):
            return "invalid_schema"
        return "transport_failure"

    def _validated_atoms(
        self, descriptor: dict, material: dict, candidates: list[dict]
    ) -> list[dict]:
        if not isinstance(candidates, list):
            return []
        result = []
        role = descriptor["role"]
        exact_event = material.get("source_status") == "exact_event"
        exact_conversation_row = (
            material.get("source_status") == "conversation_row"
            and descriptor.get("source_kind") == "conversation_row"
            and role == "user"
            and isinstance(material.get("row_hash"), str)
            and bool(material.get("row_hash"))
            and isinstance(material.get("source_memory_id"), int)
            and not isinstance(material.get("source_memory_id"), bool)
        )
        event_type = material.get("event_type", "")
        roots = [
            str(x)
            for x in material.get("root_evidence_ids", [])
            if str(x).startswith("ev1:")
        ]
        material_roots = list(material.get("root_material_ids", []))
        for candidate in candidates[:12]:
            if not isinstance(candidate, dict):
                continue
            kind = str(candidate.get("kind", "reported_claim"))
            allowed = {
                "utterance_fact",
                "reported_claim",
                "commitment",
                "boundary",
                "refusal",
                "execution_claim",
                "execution_receipt",
            }
            if kind not in allowed:
                continue
            try:
                start, end = (
                    int(candidate.get("span_start", -1)),
                    int(candidate.get("span_end", -1)),
                )
            except (TypeError, ValueError):
                continue
            if start < 0 or end <= start or end > len(material["text"]):
                continue
            source_span = material["text"][start:end]
            if not source_span.strip() or len(source_span) > 1000:
                continue
            claim = str(candidate.get("claim", "")).strip()
            if not claim or len(claim) > 800:
                continue
            if role == "assistant":
                if kind not in {
                    "utterance_fact",
                    "commitment",
                    "boundary",
                    "refusal",
                    "execution_claim",
                    "execution_receipt",
                }:
                    continue
                if kind == "utterance_fact" and not exact_event:
                    continue
            if exact_conversation_row and kind != "utterance_fact":
                # The deterministic adapter only records the exact user
                # utterance; it does not infer claims, commitments, or actors.
                continue
            if role == "user" and kind == "refusal":
                kind = "boundary"
            elif role == "user" and kind in {
                "commitment",
                "execution_claim",
                "execution_receipt",
            }:
                kind = "reported_claim"
            if role == "summary":
                kind = "reported_claim"
            if kind == "execution_receipt" and not (
                exact_event
                and event_type == "tool.execution_receipt"
                and isinstance(material.get("receipt"), dict)
                and material["receipt"].get("schema_version") == 1
                and material["receipt"].get("source") == "astrbot_runtime"
                and material["receipt"].get("status") == "success"
                and bool(material["receipt"].get("operation_id"))
            ):
                # Text claiming something was done is never a program receipt.
                kind = "execution_claim"
            if kind == "execution_receipt":
                status = "verified"
            elif kind == "utterance_fact":
                status = (
                    "verified_utterance"
                    if exact_event or exact_conversation_row
                    else "reported_unverified"
                )
            elif role == "summary":
                status = "reported_unverified"
            elif role == "assistant" and not (
                exact_event and event_type == "delivery.success"
            ):
                status = "pending_continuity"
            elif role == "user" and kind == "boundary" and not exact_event:
                status = "observed_user_boundary"
            elif role == "user" and kind == "boundary" and exact_event:
                status = "active"
            elif kind == "execution_claim":
                status = "reported_unverified"
            elif kind in {"commitment", "boundary", "refusal"}:
                status = "active" if exact_event else "reported_unverified"
            else:
                status = "reported_unverified"
            evidence_ref = {
                "material_id": descriptor["material_id"],
                "source_event_id": material.get("source_event_id")
                if exact_event
                else None,
                "source_kind": "conversation_row"
                if exact_conversation_row
                else material.get("source_kind", ""),
                "source_memory_id": material.get("source_memory_id")
                if exact_conversation_row
                else None,
                "session_id": material.get("session_id")
                if exact_conversation_row
                else None,
                "row_id": material.get("id") if exact_conversation_row else None,
                "row_hash": material.get("row_hash")
                if exact_conversation_row
                else None,
                "author_id": material.get("author_id")
                if exact_event or exact_conversation_row
                else None,
                "author_kind": "bot"
                if role == "assistant"
                else "user"
                if exact_event or exact_conversation_row
                else "unknown",
                "occurred_at": material.get("occurred_at")
                if exact_event or exact_conversation_row
                else None,
                "delivery_status": material.get("delivery_status", "unknown"),
                "text_hash": descriptor["text_hash"],
                "exact_span": [start, end],
                "source_status": material["source_status"],
            }
            identity_refs = sorted(set(roots) | set(material_roots))
            atom_key = atom_id(
                descriptor["scope_key"],
                kind,
                claim,
                identity_refs,
                int(descriptor.get("correction_revision", 0)),
            )
            conditions = candidate.get("conditions", "")
            if not isinstance(conditions, str):
                conditions = ""
            atom = {
                "atom_id": atom_key,
                "kind": kind,
                "status": status,
                "scope": {"scope_key": descriptor["scope_key"]},
                "subject_ids": [material["author_id"]]
                if (exact_event or exact_conversation_row) and material.get("author_id")
                else [],
                "source_refs": [evidence_ref],
                "root_evidence_ids": roots,
                "root_material_ids": material_roots,
                "derived_from": [],
                "valid_from": material.get("occurred_at")
                if exact_event or exact_conversation_row
                else None,
                "valid_until": None,
                "conditions": conditions[:400],
                "polarity": str(candidate.get("polarity", "unknown"))[:24],
                "claim": claim,
                "source_status": material["source_status"],
                "delivery_status": material.get("delivery_status", "unknown"),
                "confidence": self._atom_confidence(
                    kind, material, exact_event or exact_conversation_row
                ),
                "extractor_version": "evidence-extract-v1",
                "revision": 0,
                "supersedes": None,
            }
            result.append(
                {
                    "atom_id": atom_key,
                    "kind": kind,
                    "status": status,
                    "root_evidence_ids": roots,
                    "root_material_ids": material_roots,
                    **atom,
                }
            )
        return result

    @staticmethod
    def _atom_confidence(kind: str, material: dict, exact_event: bool) -> float:
        if kind == "execution_receipt":
            return 1.0
        base = 0.85 if exact_event else 0.55
        # Only reported claims soften over time. Explicit boundaries/commitments
        # remain available until their own validity conditions or corrections change.
        if kind != "reported_claim" or not exact_event:
            return base
        value = material.get("occurred_at")
        try:
            if isinstance(value, (int, float)):
                timestamp = float(value)
                if timestamp > 100_000_000_000:
                    timestamp /= 1000.0
            elif isinstance(value, str):
                timestamp = datetime.fromisoformat(
                    value.replace("Z", "+00:00")
                ).timestamp()
            else:
                return base
        except (TypeError, ValueError, OverflowError):
            return base
        age_days = max(0.0, (time.time() - timestamp) / 86400.0)
        return max(0.15, base * math.exp(-age_days / 365.0))

    @staticmethod
    def _public_atom(atom: dict, status: str) -> dict:
        output = dict(atom)
        output["status"] = status
        output["root_evidence_ids"] = sorted(
            {str(x) for x in atom.get("root_evidence_ids", [])}
        )
        output["root_material_ids"] = sorted(
            {str(x) for x in atom.get("root_material_ids", [])}
        )
        output["source_refs"] = [
            dict(ref) for ref in atom.get("source_refs", []) if isinstance(ref, dict)
        ]
        return output

    @staticmethod
    def _dedupe_atoms(atoms: list[dict]) -> list[dict]:
        by_id = {}
        for atom in atoms:
            current = by_id.get(atom["atom_id"])
            if current is None:
                by_id[atom["atom_id"]] = atom
                continue
            roots = sorted(
                set(current.get("root_evidence_ids", []))
                | set(atom.get("root_evidence_ids", []))
            )
            material_roots = sorted(
                set(current.get("root_material_ids", []))
                | set(atom.get("root_material_ids", []))
            )
            sources = {
                json.dumps(ref, sort_keys=True, ensure_ascii=False): ref
                for ref in [
                    *current.get("source_refs", []),
                    *atom.get("source_refs", []),
                ]
            }
            current["root_evidence_ids"] = roots
            current["root_material_ids"] = material_roots
            current["source_refs"] = list(sources.values())[:20]
            # A repeated quotation changes provenance coverage, never confidence.
            current["confidence"] = min(
                float(current.get("confidence", 0.55)),
                float(atom.get("confidence", 0.55)),
            )
        return sorted(
            by_id.values(),
            key=lambda row: (row.get("kind", ""), row.get("atom_id", "")),
        )

    @staticmethod
    def _slots(atoms: list[dict]) -> dict[str, list[dict]]:
        slots = {
            name: []
            for name in (
                "current_evidence",
                "commitments_and_boundaries",
                "recent",
                "historical_claims",
            )
        }
        for atom in atoms:
            kind = atom.get("kind")
            status = atom.get("status")
            if status not in {
                "active",
                "verified",
                "verified_utterance",
                "observed_user_boundary",
                "reported_unverified",
                "pending_continuity",
            }:
                continue
            if atom.get("source_status") == "legacy_summary":
                slots["historical_claims"].append(atom)
            elif kind in {"commitment", "boundary", "refusal", "execution_receipt"}:
                slots["commitments_and_boundaries"].append(atom)
            elif status in {"reported_unverified", "pending_continuity"} or kind in {
                "reported_claim",
                "execution_claim",
            }:
                slots["historical_claims"].append(atom)
            elif (
                kind in {"utterance_fact", "observation"}
                and status == "verified_utterance"
            ):
                slots["recent"].append(atom)
            else:
                slots["current_evidence"].append(atom)
        from .context_reconciliation import reconcile_rows

        return {key: reconcile_rows(value)[:40] for key, value in slots.items()}

    async def _restore_event_provenance(
        self,
        atoms: list[dict],
        atom_revisions: dict[str, int],
        scope: dict,
        scope_id: str,
        current_revision: int,
        revoked_ids: set[str],
        invalidated_atom_ids: set[str],
        prepared: list[tuple[dict, dict]],
    ) -> tuple[int, set[str]]:
        """Repair response-only author fields from exact archived event sources."""
        if not atoms:
            return 0, set()
        prepared_by_id = {
            descriptor["material_id"]: (descriptor, material)
            for descriptor, material in prepared
            if descriptor.get("material_kind") in {"source_event", "program_receipt"}
        }
        material_ids = []
        for atom in atoms:
            refs = atom.get("source_refs", [])
            for ref in refs if isinstance(refs, list) else []:
                if isinstance(ref, dict) and isinstance(ref.get("material_id"), str):
                    material_ids.append(ref["material_id"])
        meta_rows = await self.get_material_descriptors(
            scope, list(dict.fromkeys(material_ids))
        )
        event_cache: dict[str, dict | None] = {}
        restored = 0
        invalid_event_atoms: set[str] = set()
        for atom in atoms[: MAX_MATERIALS * 12]:
            atom_id_value = str(atom.get("atom_id") or "")
            atom_refs = atom.get("source_refs", [])
            is_event_atom = (
                atom.get("source_status") == "exact_event"
                or isinstance(atom_refs, list)
                and any(
                    isinstance(ref, dict) and ref.get("source_status") == "exact_event"
                    for ref in atom_refs
                )
            )
            if (
                atom_id_value in invalidated_atom_ids
                or atom_id_value in revoked_ids
                or not is_event_atom
            ):
                continue
            refs = atom_refs
            if not isinstance(refs, list) or not refs:
                invalid_event_atoms.add(atom_id_value)
                continue
            material_id = (
                str(refs[0].get("material_id") or "")
                if isinstance(refs[0], dict)
                else ""
            )
            event_id_presence = [
                bool(isinstance(ref, dict) and ref.get("source_event_id"))
                for ref in refs
            ]
            if any(event_id_presence) and not all(event_id_presence):
                invalid_event_atoms.add(atom_id_value)
                continue
            event_ids = {
                str(ref.get("source_event_id"))
                for ref in refs
                if isinstance(ref, dict)
                and isinstance(ref.get("source_event_id"), str)
                and ref.get("source_event_id")
            }
            if len(event_ids) > 1 or not material_id or material_id in revoked_ids:
                invalid_event_atoms.add(atom_id_value)
                continue
            row = meta_rows.get(material_id)
            locator = row.get("source_locator") if isinstance(row, dict) else None
            canonical_event_id = (
                locator.get("event_id") if isinstance(locator, dict) else None
            )
            event_id = next(iter(event_ids), canonical_event_id)
            recovering_missing_event_id = not event_ids
            if (
                not isinstance(event_id, str)
                or not event_id
                or event_id != canonical_event_id
                or not isinstance(locator, dict)
                or locator.get("kind") != "event"
                or locator.get("scope_key") != scope_id
                or locator.get("scope")
                != {
                    key: str(scope.get(key, "")).strip()
                    for key in ("platform", "umo", "persona_id", "instance_id")
                }
                or event_evidence_id(scope_id, event_id) != material_id
            ):
                invalid_event_atoms.add(atom_id_value)
                continue
            if any(
                not isinstance(ref, dict)
                or str(ref.get("material_id") or "") != material_id
                or (
                    ref.get("source_event_id")
                    and str(ref.get("source_event_id")) != event_id
                )
                or ref.get("source_status") != "exact_event"
                for ref in refs
            ):
                invalid_event_atoms.add(atom_id_value)
                continue
            roots = {str(value) for value in atom.get("root_evidence_ids", [])}
            material_roots = {str(value) for value in atom.get("root_material_ids", [])}
            if roots != {material_id} or material_roots:
                invalid_event_atoms.add(atom_id_value)
                continue
            correction_revision = atom_revisions.get(atom_id_value)
            if (
                not isinstance(correction_revision, int)
                or isinstance(correction_revision, bool)
                or correction_revision < 0
                or correction_revision > current_revision
            ):
                invalid_event_atoms.add(atom_id_value)
                continue
            if (
                not row
                or row.get("status") != "ready"
                or row.get("material_kind") not in {"source_event", "program_receipt"}
            ):
                invalid_event_atoms.add(atom_id_value)
                continue
            if (
                row.get("scope_key") != scope_id
                or row.get("correction_revision") != current_revision
                or any(ref.get("text_hash") != row.get("text_hash") for ref in refs)
            ):
                invalid_event_atoms.add(atom_id_value)
                continue
            # Normalize the archive with this atom's persisted derivation
            # revision. The scope's current revision is checked above and all
            # current revocations/invalidations were checked before this path;
            # an unrelated later correction must not invalidate this atom.
            if material_id in prepared_by_id and not recovering_missing_event_id:
                descriptor, material = prepared_by_id[material_id]
                if material.get("source_event_id") != event_id:
                    invalid_event_atoms.add(atom_id_value)
                    continue
            else:
                if not self.event_loader:
                    invalid_event_atoms.add(atom_id_value)
                    continue
                if event_id not in event_cache:
                    try:
                        event_cache[event_id] = await self.event_loader(scope, event_id)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        event_cache[event_id] = None
                event = event_cache[event_id]
                if (
                    not isinstance(event, dict)
                    or event.get("event_id") != event_id
                    or not isinstance(event.get("scope"), dict)
                    or any(
                        event["scope"].get(key) != scope.get(key)
                        for key in ("platform", "umo", "persona_id", "instance_id")
                    )
                    or not isinstance(event.get("text"), str)
                ):
                    invalid_event_atoms.add(atom_id_value)
                    continue
                raw = {
                    "source_kind": "event",
                    "event_id": event_id,
                    "event_type": event.get("type", event.get("event_type")),
                    "scope": event.get("scope"),
                    "text": event.get("text"),
                    "actor_id": event.get("actor_id"),
                    "occurred_at": event.get("occurred_at"),
                    "source_position": row.get("position", 0),
                    "receipt": event.get("receipt"),
                }
                prepared_match = None
                for candidate_purpose in PURPOSES:
                    candidate = normalize_event_material(
                        scope_id,
                        scope,
                        raw,
                        card_version=row.get("card_version", ""),
                        correction_revision=correction_revision,
                        purpose=candidate_purpose,
                    )
                    if not candidate:
                        continue
                    candidate_descriptor, candidate_material = candidate
                    if all(
                        candidate_descriptor.get(key) == row.get(key)
                        for key in (
                            "material_id",
                            "scope_key",
                            "material_kind",
                            "role",
                            "conversation_id",
                            "snapshot_fingerprint",
                            "position",
                            "source_family_id",
                            "text_hash",
                            "projection_key",
                            "card_version",
                        )
                    ):
                        prepared_match = (candidate_descriptor, candidate_material)
                        break
                if not prepared_match:
                    invalid_event_atoms.add(atom_id_value)
                    continue
                descriptor, material = prepared_match
            if (
                descriptor.get("material_id") != material_id
                or descriptor.get("text_hash") != row.get("text_hash")
                or descriptor.get("projection_key") != row.get("projection_key")
                or material.get("source_event_id") != event_id
            ):
                invalid_event_atoms.add(atom_id_value)
                continue
            actor_id = material.get("author_id")
            expected_author_kind = (
                "bot"
                if descriptor.get("role") == "assistant"
                else "user"
                if descriptor.get("role") == "user"
                else "unknown"
            )
            existing_subjects = atom.get("subject_ids", [])
            if actor_id:
                if any(
                    (ref.get("author_id") and ref.get("author_id") != actor_id)
                    or (ref.get("actor_id") and ref.get("actor_id") != actor_id)
                    or (
                        ref.get("author_kind")
                        not in (None, "", "unknown", expected_author_kind)
                    )
                    for ref in refs
                ) or (existing_subjects and existing_subjects != [actor_id]):
                    invalid_event_atoms.add(atom_id_value)
                    continue
            elif existing_subjects or any(
                ref.get("author_id")
                or ref.get("actor_id")
                or ref.get("author_kind") not in (None, "", "unknown")
                for ref in refs
            ):
                invalid_event_atoms.add(atom_id_value)
                continue
            if recovering_missing_event_id:
                if (
                    not actor_id
                    or descriptor.get("role") != "user"
                    or atom.get("kind") != "utterance_fact"
                    or atom.get("status") != "verified_utterance"
                ):
                    invalid_event_atoms.add(atom_id_value)
                    continue
                text = material.get("text")
                spans_valid = isinstance(text, str) and all(
                    isinstance(ref.get("exact_span"), list)
                    and len(ref["exact_span"]) == 2
                    and all(
                        isinstance(value, int) and not isinstance(value, bool)
                        for value in ref["exact_span"]
                    )
                    and 0 <= ref["exact_span"][0] < ref["exact_span"][1] <= len(text)
                    for ref in refs
                )
                if not spans_valid:
                    invalid_event_atoms.add(atom_id_value)
                    continue
            # Archive loading above may await I/O. Do not return an author
            # projection from a revision that changed while it was in flight.
            latest_revision = self.sidecar.peek_evidence_revision(scope_id)
            if (
                not latest_revision.get("available")
                or latest_revision.get("correction_revision") != current_revision
            ):
                invalid_event_atoms.add(atom_id_value)
                continue
            updated_refs = []
            for ref in refs:
                updated = dict(ref)
                updated["source_event_id"] = event_id
                if actor_id:
                    updated["author_id"] = actor_id
                    updated["actor_id"] = actor_id
                    updated["author_kind"] = (
                        "bot" if descriptor.get("role") == "assistant" else "user"
                    )
                else:
                    updated.pop("author_id", None)
                    updated.pop("actor_id", None)
                    updated["author_kind"] = "unknown"
                updated["occurred_at"] = material.get("occurred_at")
                updated_refs.append(updated)
            atom["source_refs"] = updated_refs
            if actor_id and descriptor.get("role") == "user":
                atom["subject_ids"] = [actor_id]
            elif not actor_id:
                atom["subject_ids"] = []
            if actor_id:
                restored += 1
        return restored, invalid_event_atoms

    @staticmethod
    def _evidence_status(view: dict) -> str:
        if not view.get("available"):
            return (
                "invalid"
                if view.get("diagnostics", {}).get("reason") == "invalid_scope"
                else "unavailable"
            )
        diagnostics = view.get("diagnostics", {})
        atoms = view.get("atoms", [])
        if diagnostics.get("derivation_available") is False and not atoms:
            return "unavailable"
        if diagnostics.get("timeout_count"):
            return "partial" if atoms else "timeout"
        if diagnostics.get("route_error_count"):
            return "partial" if atoms else "unavailable"
        has_deferred = int(diagnostics.get("materials_deferred", 0) or 0) > 0
        has_pending = (
            int(diagnostics.get("materials_pending", 0) or 0) > 0
            or int(diagnostics.get("materials_queued", 0) or 0) > 0
        )
        has_invalid = (
            int(diagnostics.get("source_refs_rejected", 0) or 0) > 0
            or int(diagnostics.get("typed_items_rejected", 0) or 0) > 0
            or int(diagnostics.get("materials_terminal", 0) or 0) > 0
            or int(diagnostics.get("materials_skipped", 0) or 0) > 0
        )
        if atoms:
            return "partial" if has_deferred or has_pending or has_invalid else "ready"
        if has_invalid and has_pending:
            return "partial"
        if has_invalid:
            return "invalid"
        if has_deferred:
            return "deferred"
        if has_pending:
            return "pending"
        return "empty"

    @staticmethod
    def _empty_view(reason: str, purpose: str, card_version: str = "") -> dict:
        invalid = reason in {"invalid_scope", "invalid_purpose"}
        return {
            "schema_version": 1,
            "available": False,
            "scope_key": "",
            "correction_revision": None,
            "revocations": [],
            "purpose": purpose,
            "card_version": str(card_version)[:256],
            "slots": {
                name: []
                for name in (
                    "current_evidence",
                    "commitments_and_boundaries",
                    "recent",
                    "historical_claims",
                )
            },
            "atoms": [],
            "history_projection": {},
            "diagnostics": {
                "available": False,
                "reason": reason,
                "source_refs_rejected": 0,
                "typed_items_rejected": 0,
                "source_text_logged": False,
                "raw_source_returned": False,
            },
            "evidence_status": "invalid" if invalid else "unavailable",
        }
