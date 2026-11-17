"""Validated, provenance-preserving contracts for managed character turns."""

import hashlib
import json
from datetime import datetime


def scope_key(scope: dict) -> str:
    """Partition managed conversations by full origin and character instance."""
    if not isinstance(scope, dict):
        raise ValueError("scope must be an object")
    values = {
        key: scope.get(key, "")
        for key in ("platform", "umo", "persona_id", "instance_id")
    }
    if not all(isinstance(value, str) and value.strip() for value in values.values()):
        raise ValueError("scope requires platform, umo, persona_id and instance_id")
    values = {key: value.strip() for key, value in values.items()}
    return (
        "narrative:v1:"
        + hashlib.sha256(
            json.dumps(values, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
    )


def timestamp(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def is_managed_event(event) -> bool:
    """Recognize public markers without treating dynamic mock attributes as flags."""
    getter = getattr(event, "get_extra", None)
    if not callable(getter):
        return False
    value = getter("_narrative_managed_v1")
    return isinstance(value, (dict, bool, str, int)) and bool(value)


class ScopeEvent:
    """Expose only supplied scope information to the existing access checks."""

    def __init__(self, scope, actor_id=""):
        self.scope = scope
        self.unified_msg_origin = scope["umo"]
        self.actor_id = actor_id

    def get_sender_id(self):
        return self.actor_id

    def get_sender_name(self):
        return self.actor_id

    def get_platform_name(self):
        return self.scope["platform"]

    def get_group_id(self):
        return (
            self.unified_msg_origin.rsplit(":", 1)[-1]
            if ":GroupMessage:" in self.unified_msg_origin
            else ""
        )

    def get_message_type(self):
        from astrbot.api.platform import MessageType

        return (
            MessageType.GROUP_MESSAGE
            if self.get_group_id()
            else MessageType.FRIEND_MESSAGE
        )
