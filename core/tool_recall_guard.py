"""Fresh, bounded validators for memory-tool results.

These callbacks run only at the request/effect boundary. Pure search leaves stay
in the shared recall pool and never capture an event or this validator.
"""

from __future__ import annotations

import asyncio
import copy
import weakref

from .event_handler_modules.message_utils import discard_unexpected_awaitable
from .evidence_projection import conversation_row_hash
from .memory_scope import (
    adapter_access,
    adapter_visible,
    is_event_memory_allowed,
    memory_metadata_matches_policy,
    resolve_memory_access_policy,
)
from .narrative_api import scope_key
from .passive_group_capture import get_active_plugin
from .recall_runtime import RecallDeadline
from .summary_source import project_live_atoms, summary_metadata_sha256, text_sha256
from .utils import get_persona_id


def _read_scope_matches(read_scopes, snapshot) -> bool:
    expected_scope_key = snapshot.get("scope_key")
    expected_read_only = snapshot.get("read_only") is True
    for source_scope, read_only, _policy in read_scopes:
        try:
            if (
                scope_key(source_scope) == expected_scope_key
                and bool(read_only) == expected_read_only
            ):
                return True
        except Exception:
            continue
    return False


async def _bounded(awaitable, deadline: RecallDeadline):
    remaining = deadline.remaining()
    if remaining <= 0:
        discard_unexpected_awaitable(awaitable)
        raise asyncio.TimeoutError
    return await asyncio.wait_for(awaitable, timeout=remaining)


def _verified_document_text(
    engine, current, snapshot, persona_filtering, persona_id
) -> str | None:
    if current is None:
        return None
    text, metadata, atom_rows = current
    if (
        not isinstance(text, str)
        or not isinstance(metadata, dict)
        or not adapter_visible(metadata)
        or text_sha256(text) != snapshot["document_text_sha256"]
        or summary_metadata_sha256(metadata) != snapshot["summary_metadata_sha256"]
        or str(metadata.get("status", "active")) != "active"
        or metadata.get("forgotten") is True
        or metadata.get("superseded_by") is not None
        or (metadata.get("source_session_id") or metadata.get("session_id"))
        != snapshot["source_session_id"]
        or metadata.get("source_platform") != snapshot.get("source_platform")
        or metadata.get("interaction_type") != snapshot.get("interaction_type")
        or metadata.get("memory_visibility") != snapshot.get("memory_visibility")
        or metadata.get("persona_id") != snapshot.get("persona_id")
        or (
            (persona_filtering or snapshot.get("read_only") is True)
            and metadata.get("persona_id") != persona_id
        )
        or not memory_metadata_matches_policy(
            metadata,
            interaction_type=snapshot.get("interaction_type"),
            source_platform=snapshot.get("source_platform"),
            source_session_id=snapshot["source_session_id"],
        )
    ):
        return None
    if snapshot["atom_proof"]:
        active = engine._agentic_live_rows(atom_rows)
        text, proof = project_live_atoms(active)
        if proof != snapshot["atom_proof"]:
            return None
    elif atom_rows or metadata.get("atom_types"):
        return None
    return text if text_sha256(text) == snapshot["projected_text_sha256"] else None


def make_memory_source_validator(
    *,
    plugin,
    event,
    engine,
    request_scope: dict,
    core_persona_id: str,
    persona_filtering: bool,
    document_snapshots: list[dict],
    source_row_snapshots: list[dict],
    owned_private: bool = False,
    cd_receipt=None,
    timeout_seconds: float = 1.0,
):
    """Return a zero-argument callback that rechecks tool documents and rows."""
    frozen_documents = tuple(copy.deepcopy(row) for row in document_snapshots)
    frozen_source_rows = tuple(copy.deepcopy(row) for row in source_row_snapshots)
    frozen_request_scope = copy.deepcopy(request_scope)
    document_ids = [snapshot["document_id"] for snapshot in frozen_documents]
    source_scopes = {
        snapshot["scope_key"]: snapshot["source_scope"] for snapshot in frozen_documents
    }
    request_ref = (
        getattr(event, "_lm_owned_memory_request_ref_v1", None)
        if owned_private
        else None
    )
    receipt_identity = (
        plugin._cd_owned_receipt_contract_identity(cd_receipt)
        if owned_private
        else None
    )

    def request_is_current():
        if (
            get_active_plugin() is not plugin
            or getattr(plugin.initializer, "memory_engine", None) is not engine
            or not plugin.series_runtime_allowed()
            or getattr(event, "unified_msg_origin", None) != frozen_request_scope["umo"]
            or not is_event_memory_allowed(plugin.config_manager, event)
        ):
            return False
        if not owned_private:
            return True
        current_request = (
            request_ref() if isinstance(request_ref, weakref.ReferenceType) else None
        )
        if (
            current_request is None
            or getattr(event, "_lm_owned_memory_request_ref_v1", None)
            is not request_ref
            or getattr(event, "_chat_dynamics_owned_request", None) is not True
            or getattr(event, "_cd_owned_original_input_v1", None) is not True
            or receipt_identity is None
        ):
            return False
        event_receipt = event.get_extra("_cd_memory_feedback_v1", None)
        if not plugin._fresh_worldbook_receipt_matches(
            event_receipt,
            frozen_request_scope,
            receipt_identity["card_version"],
        ):
            return False
        return (
            plugin._cd_owned_receipt_contract_identity(event_receipt)
            == receipt_identity
            and plugin._cd_owned_receipt_contract_identity(
                getattr(current_request, "_cd_memory_feedback_v1", None)
            )
            == receipt_identity
        )

    async def validate() -> bool:
        deadline = RecallDeadline.start(
            total_seconds=timeout_seconds, final_reserve_seconds=0
        )

        async def read_documents():
            with adapter_access(plugin.config_manager, event):
                return await _bounded(
                    engine.get_summary_source_snapshots(document_ids), deadline
                )

        async def worldbook_is_current():
            if not request_is_current():
                return False
            if not owned_private:
                return True
            fresh = await _bounded(
                plugin._fresh_cd_owned_worldbook_receipt(event, request_ref()), deadline
            )
            return request_is_current() and plugin._fresh_worldbook_receipt_matches(
                fresh,
                frozen_request_scope,
                receipt_identity["card_version"],
            )

        def persona_filtering_now():
            return persona_filtering or plugin.config_manager.get(
                "filtering_settings.use_persona_filtering", True
            )

        async def read_source_row(source, persona_id):
            with adapter_access(plugin.config_manager, event):
                return await _bounded(
                    plugin._native_verified_source_row(
                        source["scope"],
                        source["document_id"],
                        source["row_id"],
                        persona_id=persona_id,
                        persona_filtering=persona_filtering_now(),
                    ),
                    deadline,
                )

        try:
            if not await worldbook_is_current():
                return False
            persona_id = await _bounded(get_persona_id(plugin.context, event), deadline)
            if persona_id != core_persona_id:
                return False
            read_scopes = await _bounded(
                plugin._native_read_scopes(event, frozen_request_scope, persona_id),
                deadline,
            )
            for snapshot in frozen_documents:
                source_scope = snapshot.get("source_scope")
                if (
                    not isinstance(source_scope, dict)
                    or not _read_scope_matches(read_scopes, snapshot)
                    or not await _bounded(
                        plugin._narrative_scope_allowed(source_scope), deadline
                    )
                ):
                    return False
                revision = plugin.peek_evidence_revision(source_scope)
                if (
                    not isinstance(revision, dict)
                    or revision.get("available") is not True
                    or revision.get("correction_revision")
                    != snapshot.get("correction_revision")
                ):
                    return False
            current_documents = await read_documents()
            frames = []
            with adapter_access(plugin.config_manager, event):
                for snapshot in frozen_documents:
                    text = _verified_document_text(
                        engine,
                        current_documents.get(snapshot["document_id"]),
                        snapshot,
                        persona_filtering_now(),
                        persona_id,
                    )
                    if text is None:
                        return False
                    frames.append(
                        {
                            "atom_id": f"tool-summary:{snapshot['document_id']}",
                            "text": text,
                            "scope_umo": snapshot["source_session_id"],
                            "author_uid": "",
                            "source_uids": [],
                            "source_kind": "legacy_summary",
                            "provenance_complete": False,
                        }
                    )

            for source in frozen_source_rows:
                if not _read_scope_matches(read_scopes, source):
                    return False
                current = await read_source_row(source, persona_id)
                author_id = (
                    current.get("sender_id") if isinstance(current, dict) else None
                )
                if (
                    not isinstance(current, dict)
                    or current.get("role") != "user"
                    or not isinstance(author_id, str)
                    or not author_id.strip()
                    or author_id == source.get("scope", {}).get("umo")
                    or conversation_row_hash(current) != source.get("row_hash")
                ):
                    return False
                frames.append(
                    {
                        "atom_id": f"tool-source:{source['document_id']}:{source['row_id']}",
                        "text": current.get("content"),
                        "scope_umo": source["scope"].get("umo"),
                        "author_uid": author_id,
                        "source_uids": [author_id],
                        "source_kind": "verified_quote",
                        "provenance_complete": True,
                    }
                )

            if frames:
                allowed = await _bounded(
                    plugin._filter_native_frames(event, frozen_request_scope, frames),
                    deadline,
                )
                expected = {
                    (
                        row["atom_id"],
                        row["text"],
                        row["author_uid"],
                        tuple(row.get("source_uids", [])),
                    )
                    for row in frames
                }
                actual = (
                    {
                        (
                            row.get("atom_id"),
                            row.get("text"),
                            row.get("author_uid"),
                            tuple(row.get("source_uids", [])),
                        )
                        for row in allowed
                        if isinstance(row, dict)
                    }
                    if isinstance(allowed, list)
                    else set()
                )
                if actual != expected:
                    return False

            if not await worldbook_is_current():
                return False
            final_persona = await _bounded(
                get_persona_id(plugin.context, event), deadline
            )
            if final_persona != core_persona_id:
                return False
            final_scopes = await _bounded(
                plugin._native_read_scopes(event, frozen_request_scope, final_persona),
                deadline,
            )
            if any(
                not _read_scope_matches(final_scopes, snapshot)
                for snapshot in frozen_documents
            ):
                return False
            if any(
                not _read_scope_matches(final_scopes, source)
                for source in frozen_source_rows
            ):
                return False
            for source in source_scopes.values():
                if not await _bounded(
                    plugin._narrative_scope_allowed(source), deadline
                ):
                    return False

            async def recheck_row(source):
                row = await read_source_row(source, final_persona)
                if (
                    not isinstance(row, dict)
                    or row.get("role") != "user"
                    or row.get("sender_id") == source["scope"].get("umo")
                    or conversation_row_hash(row) != source.get("row_hash")
                ):
                    return False
                return True

            if frozen_source_rows and not all(
                await asyncio.gather(
                    *(recheck_row(source) for source in frozen_source_rows)
                )
            ):
                return False
            final_documents = await read_documents()
            # A configuration update can arrive while SQLite returns the final
            # snapshot. Reapply current scope rules to previously verified groups.
            verified_groups = tuple(
                scope["umo"] for scope, readonly, _ in final_scopes if readonly
            )
            latest_policy = resolve_memory_access_policy(
                plugin.config_manager,
                event,
                persona_id=final_persona,
                verified_group_sessions=verified_groups,
            )
            latest_origins = {
                (policy.source_session_id or origin, policy.read_only)
                for policy in latest_policy.read_scopes
                for origin in (policy.session_ids or (None,))
            }
            if any(
                (snapshot["source_session_id"], snapshot.get("read_only") is True)
                not in latest_origins
                for snapshot in frozen_documents
            ):
                return False
            with adapter_access(plugin.config_manager, event):
                if any(
                    _verified_document_text(
                        engine,
                        final_documents.get(snapshot["document_id"]),
                        snapshot,
                        persona_filtering_now(),
                        final_persona,
                    )
                    is None
                    for snapshot in frozen_documents
                ):
                    return False
            for snapshot in frozen_documents:
                revision = plugin.peek_evidence_revision(snapshot["source_scope"])
                if (
                    not isinstance(revision, dict)
                    or revision.get("available") is not True
                    or revision.get("correction_revision")
                    != snapshot.get("correction_revision")
                ):
                    return False
            return request_is_current()
        except asyncio.CancelledError:
            raise
        except Exception:
            return False

    return validate


def make_managed_evidence_validator(
    *,
    plugin,
    event,
    scope: dict,
    card_version: str,
    items: list[dict],
    timeout_seconds: float = 1.0,
):
    """Re-resolve the already selected canonical atoms before tool-derived effects."""
    frozen_scope = dict(scope)
    frozen_card = str(card_version or "")
    selected = tuple(dict(item) for item in items)
    source_refs = [
        dict(ref)
        for item in selected
        for ref in item.get("source_refs", [])
        if isinstance(ref, dict)
    ]
    typed_items = [
        {
            "item_kind": "evidence_atom",
            "evidence_schema_version": 1,
            "source_id": item.get("source_id"),
            "text": item.get("text", ""),
            "scope": dict(frozen_scope),
            "scope_key": item.get("scope_key"),
            "correction_revision": item.get("correction_revision"),
            "source_refs": [
                dict(ref)
                for ref in item.get("source_refs", [])
                if isinstance(ref, dict)
            ],
            "root_evidence_ids": list(item.get("root_evidence_ids", [])),
            "root_material_ids": list(item.get("root_material_ids", [])),
        }
        for item in selected
    ]

    async def validate() -> bool:
        deadline = RecallDeadline.start(
            total_seconds=timeout_seconds, final_reserve_seconds=0
        )
        try:
            current_view = await _bounded(
                plugin.get_evidence_view(
                    frozen_scope,
                    source_refs=source_refs,
                    typed_items=typed_items,
                    purpose="reply",
                    card_version=frozen_card,
                    allow_derivation=False,
                ),
                deadline,
            )
            if (
                not isinstance(current_view, dict)
                or current_view.get("available") is not True
                or current_view.get("evidence_status") not in {"ready", "partial"}
            ):
                return False
            by_id = {
                atom.get("atom_id"): atom
                for atom in current_view.get("atoms", [])
                if isinstance(atom, dict)
            }
            frames = []
            for item in selected:
                atom = by_id.get(item.get("source_id"))
                if (
                    not isinstance(atom, dict)
                    or atom.get("claim") != item.get("text")
                    or atom.get("kind") != item.get("fact_kind")
                ):
                    return False
                refs = atom.get("source_refs", [])
                authors = list(
                    dict.fromkeys(
                        str(ref.get("author_id") or ref.get("author_uid") or "")
                        for ref in refs
                        if isinstance(ref, dict)
                        and (ref.get("author_id") or ref.get("author_uid"))
                    )
                )
                if not authors:
                    return False
                frames.append(
                    {
                        "atom_id": atom.get("atom_id"),
                        "text": atom.get("claim"),
                        "scope_umo": frozen_scope.get("umo"),
                        "author_uid": authors[0],
                        "source_uids": authors,
                        "source_kind": atom.get("source_status"),
                        "provenance_complete": True,
                    }
                )
            allowed = await _bounded(
                plugin._filter_native_frames(event, frozen_scope, frames), deadline
            )
            expected = {
                (row["atom_id"], row["text"], row["author_uid"]) for row in frames
            }
            actual = (
                {
                    (row.get("atom_id"), row.get("text"), row.get("author_uid"))
                    for row in allowed
                    if isinstance(row, dict)
                }
                if isinstance(allowed, list)
                else set()
            )
            return actual == expected
        except asyncio.CancelledError:
            raise
        except Exception:
            return False

    return validate
