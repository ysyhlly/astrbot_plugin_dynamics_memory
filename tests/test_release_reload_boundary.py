import sys
from types import ModuleType

from astrbot_plugin_dynamics_memory.reload_bootstrap import ensure_current


def test_release_marker_invalidates_cached_children_even_with_new_parent(
    tmp_path, monkeypatch
):
    package = "fixture_memory_reload"
    parent = ModuleType(package)
    entry = ModuleType(package + ".main")
    old = ModuleType(package + ".core.plugin_initializer")
    old.PluginInitializer = type("OldInitializer", (), {})
    unrelated = ModuleType("unrelated_memory_consumer")
    for module in (parent, entry, old, unrelated):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    path = tmp_path / "main.py"
    path.write_text("release = 2\n")
    ensure_current(
        package,
        entry.__name__,
        str(path),
        checks=[
            ("core.plugin_initializer", "PluginInitializer", "_release_readiness_v2")
        ],
    )
    assert old.__name__ not in sys.modules
    assert sys.modules[entry.__name__] is entry
    assert sys.modules[unrelated.__name__] is unrelated
    assert getattr(parent, "_narrative_loaded_source_fingerprint", None)
