# Evidence view and corrections

LivingMemory 2.7.0-rc.7 adds an evidence projection sidecar for managed character
turns. It leaves conversation rows, historical memory documents, relationship
scores, and persona records unchanged. The sidecar stores only material
identifiers and hashes, derived atoms, and append-only administrator
corrections; raw source messages are held only in the bounded in-memory
derivation queue and are never written to diagnostics.

## Public interfaces

```python
async get_evidence_view(
    scope,
    source_refs=None,
    purpose="reply",
    card_version="",
    conversation_id=None,
    snapshot_fingerprint=None,
    materials=None,
)

async queue_material_derivation(materials, scope=None, purpose="reply", card_version="")
peek_evidence_revision(scope)  # synchronous RAM read
async correct_insight(target_id, expected_revision, action, replacement="", reason="", action_id="")
```

`scope` uses the existing four fields: `platform`, `umo`, `persona_id`, and
`instance_id`. Supported purposes are `reply`, `decision`, `history_quote`,
`reflection`, and `replay`; the caller chooses the purpose. A model supplied
string cannot change it.

A host-history row has only host-confirmed identity fields:

```json
{
  "source_kind": "host_history",
  "conversation_id": "host conversation id",
  "snapshot_fingerprint": "hash of the captured host history",
  "position": 17,
  "history_epoch": "opaque trusted conversation generation",
  "source_prefix_anchor": "hash of the complete raw history prefix through this row",
  "role": "assistant",
  "text": "recorded historical text"
}
```

Its `material_id` hashes the four-key scope, trusted conversation ID, snapshot
fingerprint, original position, and text hash. It is not an AstrBot event ID.
The sidecar uses a `fam1:` correction-closure key across snapshots only when
both the trusted history epoch and complete-prefix anchor are present. The
prefix anchor stays stable when later rows are appended and changes when the
history is truncated or reset. Missing either value leaves the row material-only
and disables cross-snapshot cache reuse and correction closure. Old sidecar
families without anchors are migrated as ambiguous and removed. The helper key
is never emitted as event evidence or counted as independent support.
The adapter must not add an author, timestamp, event ID, or delivery result to a
host-only row. Only an independently recorded event reference can produce an
`ev1:` evidence ID. Host and legacy materials use separate
`root_material_ids`; their `mat1:`/`leg1:` identifiers do not count as event
evidence or independent support.

An event-only reference may contain `source_event_id`, canonical `scope_key`,
and optional expected `actor_id`/`event_type`. References passed through either
`source_refs` or `materials` are resolved against LivingMemory's exact-scope
narrative event archive; missing or mismatched rows are rejected. A
caller-supplied event ID alone is never accepted as a source.

The view contains four bounded slots: `current_evidence`,
`commitments_and_boundaries`, `recent`, and `historical_claims`. It also returns
`atoms`, `history_projection` keyed by material ID, `correction_revision`,
`revocations`, and metadata-only diagnostics. A host user row remains marked as
historical data. A resolved bot row projects to atoms and omits its old full
assistant text. While derivation is pending, only complete candidate sentences
that may contain a commitment, refusal, or boundary are returned with
`pending_continuity` and `unverified` labels. Unclassified bot text is omitted.
If a legacy mixed summary has no source rows, reply projection omits its full
text until background derivation has produced historical claims marked
`reported_unverified`; `history_quote` remains the explicit route for a full
historical quote.

Material cache keys include the projection version, material and content
hashes, purpose, and correction revision. The stable effective-core
`card_version` is returned for compatibility checks; ordinary evidence
derivation does not depend on it and is not repeated when the character card
changes.

## Background derivation and source limits

Cache misses enqueue in a bounded queue (128 globally, 24 materials per scope
per hour, at most four per extraction batch). The worker uses only the
already-selected LivingMemory summarizer provider. It does not select or enable
a disabled provider and does not add a model call to the reply path. Missing
providers, invalid output, and queue limits leave a visible pending/degraded
status. The queue is in memory; after restart, a later request can enqueue the
material again from its current host snapshot.

User speech can produce reported claims or scoped user boundaries; it is not
upgraded to an external fact. Bot history without an exact delivery receipt can
produce only pending continuity candidates. A bot statement that it completed
an action is an `execution_claim`, never an `execution_receipt`. Tool execution
receipts require a separate program-verified source event.

Repeated references to the same root evidence do not increase confidence.
Changing a host snapshot or re-running summary derivation does not create new
evidence support or raise confidence.
Missing speaker identity, time, or delivery remains unknown. Legacy source
messages are never rewritten or deleted by evidence derivation.

## Administrator corrections

The chat command is restricted to AstrBot administrators:

```text
/lmem evidence list
/lmem evidence <atom-or-material-id>|<correction-revision>|withdraw|<reason>
/lmem evidence <atom-id>|<correction-revision>|supersede|<replacement>|<reason>
```

The authenticated LivingMemory Pages API also exposes
`/evidence/insights` and `/evidence/correct`. Corrections are append-only and
use a compare-and-swap revision. `withdraw`, `dispute`, and `supersede` mark
known dependent atoms and material caches for invalidation. A stale revision
fails without modifying the source. Corrections do not edit core persona text,
affection, or source conversations. Public revision responses include IDs and
actions, never correction reasons or original chat text.

The first release does not prove complete coverage of arbitrary paraphrases,
screenshots, or references that lack attributable source events. LivingMemory
uses Chat Dynamics' public preparation receipt to type-filter its own recalled
summaries. Raw native SDK history is covered only when Chat Dynamics confirms
`native_history_covered=true` after its reversible provider wrapper is installed.
A `prepared` or `bypass` receipt alone does not prove that the runner's actual
history was projected. On an unavailable or unsafe boundary neither plugin
guesses speaker or event identity, and the current conversation history remains
an explicit coverage gap.
