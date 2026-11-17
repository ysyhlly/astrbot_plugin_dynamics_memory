"""Regression: runtime-only config keys must survive ConfigManager validation.

These keys exist in _conf_schema.json and are read by memory_scope / recall
code, but were previously dropped by the pydantic validators because the
models did not declare them. Silent drops disable group-to-private recall.
"""

from astrbot_plugin_dynamics_memory.core.base.config_manager import ConfigManager
from astrbot_plugin_dynamics_memory.core.memory_scope import resolve_memory_access_policy


GROUP_A = "A:GroupMessage:1"
GROUP_B = "A:GroupMessage:2"


class _FakeEvent:
    unified_msg_origin = "A:FriendMessage:u1"

    def get_platform_id(self):
        return "A"

    def get_platform_name(self):
        return "aiocqhttp"

    def get_sender_id(self):
        return "u1"


def _manager():
    raw = {
        "filtering_settings": {
            "use_persona_filtering": True,
            "use_session_filtering": False,
            "enable_group_to_private_recall": True,
            "private_shared_group_sources": f"{GROUP_A}\n{GROUP_B}",
            "private_shared_group_instance_id": "persona-x",
            "persona_aliases": "default=persona-x",
        },
        "recall_engine": {
            "narrative_local_timeout_seconds": 1.25,
            "narrative_hybrid_timeout_seconds": 0.5,
        },
    }
    return ConfigManager(raw)


def test_runtime_keys_survive_validation():
    manager = _manager()
    assert manager.get("filtering_settings.enable_group_to_private_recall") is True
    assert manager.get("filtering_settings.private_shared_group_instance_id") == "persona-x"
    assert manager.get("filtering_settings.persona_aliases") == "default=persona-x"
    assert manager.get("recall_engine.narrative_local_timeout_seconds") == 1.25
    assert manager.get("recall_engine.narrative_hybrid_timeout_seconds") == 0.5


def test_private_policy_reads_shared_group_scopes():
    manager = _manager()
    policy = resolve_memory_access_policy(
        manager,
        _FakeEvent(),
        persona_id="persona-x",
        verified_group_sessions=(GROUP_A, GROUP_B),
    )
    own = [scope for scope in policy.read_scopes if scope.session_ids]
    groups = [
        scope for scope in policy.read_scopes
        if scope.read_only and scope.source_session_id in (GROUP_A, GROUP_B)
    ]
    assert len(own) == 1
    assert len(groups) == 2
    assert policy.read_scopes[0].session_ids == ("A:FriendMessage:u1",)
