"""Per-search candidate budgets and an early, validated document signal."""
from __future__ import annotations

import asyncio


class NarrativeSearchPlan:
    def __init__(self, target_k, validator, *, initial=20, document_ceiling=120,
                 graph_ceiling=40):
        self.target_k = max(1, int(target_k))
        self.initial = max(self.target_k, int(initial))
        self.document_ceiling = max(self.initial, int(document_ceiling))
        self.graph_ceiling = max(self.initial, int(graph_ceiling))
        self.validator = validator
        self.documents_ready = asyncio.Event()
        self.documents = []
        self.document_fallback = []
        self.graph_fallback = []

    async def validate(self, rows, context, route):
        if context is None:
            return await self.validator(rows)
        outcome = await context.run(route, self.validator, rows,
                                    discriminator={"ids": [r.doc_id for r in rows]})
        return outcome.get("value", []) if outcome.get("state") == "completed" else []

    def publish_documents(self, rows):
        self.documents = list(rows)
        self.documents_ready.set()

    def documents_sufficient(self):
        # Scope and lifecycle have already passed validator; lexical evidence
        # remains mandatory. Vector-only matches must keep graph rescue.
        return len({r.doc_id for r in self.documents if r.content and r.final_score > 0
                    and float(r.bm25_score or 0) > 0}) >= self.target_k

    @staticmethod
    def can_expand(context):
        return context is None or context.budget.remaining() > .25
