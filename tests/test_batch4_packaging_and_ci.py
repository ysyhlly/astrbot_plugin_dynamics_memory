"""Batch 4 regression tests: dynamic version, CI gates, packaging hygiene, metadata/deps."""

from __future__ import annotations

import importlib.util
import json
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrbot_plugin_dynamics_memory.core.managers.backup_manager import PLUGIN_VERSION
from astrbot_plugin_dynamics_memory.core.series import (
    SeriesMixin,
    _load_plugin_version,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_package_release_module():
    spec = importlib.util.spec_from_file_location(
        "package_release_under_test", REPO_ROOT / "tools" / "package_release.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_metadata_version(path: Path) -> str:
    raw = path.read_text(encoding="utf-8-sig")
    match = re.search(r"^version:\s*['\"]?([^\s'\"#]+)", raw, re.MULTILINE)
    assert match is not None
    return match.group(1)


@pytest.mark.asyncio
async def test_series_status_reads_dynamic_version_from_metadata_yaml(tmp_path: Path):
    expected = _parse_metadata_version(REPO_ROOT / "metadata.yaml")
    assert expected != "0.1.0"
    assert _load_plugin_version() == expected

    # Custom metadata.yaml parsing (quoted & unquoted) + safe fallback.
    quoted = tmp_path / "m1.yaml"
    quoted.write_text('name: test\nversion: "9.8.7-rc.1"\n', encoding="utf-8")
    assert _load_plugin_version(quoted) == "9.8.7-rc.1"

    missing = tmp_path / "does_not_exist.yaml"
    assert _load_plugin_version(missing, fallback="0.0.0-fallback") == "0.0.0-fallback"

    empty = tmp_path / "empty.yaml"
    empty.write_text("name: only_name\n", encoding="utf-8")
    assert _load_plugin_version(empty, fallback="0.0.0-fallback") == "0.0.0-fallback"

    # SeriesMixin.series_status() returns dynamic metadata.yaml version.
    class _DummySeries(SeriesMixin):
        def __init__(self):
            self.context = SimpleNamespace(get_all_stars=lambda: [])

        async def _series_admin(self, *, check_csrf: bool = False):
            return "admin"

        def get_dynamics_capabilities(self):
            return {"role": "memory"}

    res = await _DummySeries().series_status()
    assert res["status"] == "ok"
    assert res["data"]["version"] == expected


def test_package_release_install_zip_single_root_and_blacklist(tmp_path: Path):
    pkg = _load_package_release_module()
    records = pkg.build(REPO_ROOT, tmp_path)
    install_rec = next(r for r in records if r["kind"] == "install")
    install_zip = tmp_path / install_rec["file"]
    assert install_zip.is_file()

    with zipfile.ZipFile(install_zip) as zf:
        names = zf.namelist()
        roots = {n.split("/", 1)[0] for n in names}
        assert roots == {"astrbot_plugin_dynamics_memory"}

        # Required runtime files present
        for required in (
            "astrbot_plugin_dynamics_memory/main.py",
            "astrbot_plugin_dynamics_memory/metadata.yaml",
            "astrbot_plugin_dynamics_memory/_conf_schema.json",
            "astrbot_plugin_dynamics_memory/requirements.txt",
            "astrbot_plugin_dynamics_memory/pages/dashboard/assets/app.min.js",
        ):
            assert required in names, f"missing required file: {required}"

        # Dev/test/CI files strictly excluded
        forbidden_top_dirs = {
            ".github",
            "tests",
            "docs",
            "tools",
            "scripts",
            "frontend",
        }
        forbidden_any_dirs = {"node_modules", "__pycache__", ".git", ".pytest_cache"}
        forbidden_filenames = {
            "package.json",
            "package-lock.json",
            "build_release.py",
            ".gitignore",
            ".gitattributes",
            "pyproject.toml",
            "pytest.ini",
            ".coveragerc",
            "requirements-dev.txt",
            "IMPLEMENTATION_PLAN.md",
        }
        for name in names:
            rel_parts = Path(name.split("/", 1)[1]).parts
            assert rel_parts[0] not in forbidden_top_dirs, (
                f"blacklisted dir leaked: {name}"
            )
            assert not (set(rel_parts) & forbidden_any_dirs), (
                f"blacklisted dir leaked: {name}"
            )
            assert rel_parts[-1] not in forbidden_filenames, (
                f"blacklisted file leaked: {name}"
            )

    # Verify the validator itself catches leaks and bad roots.
    base_valid = [
        "astrbot_plugin_dynamics_memory/main.py",
        "astrbot_plugin_dynamics_memory/metadata.yaml",
        "astrbot_plugin_dynamics_memory/_conf_schema.json",
        "astrbot_plugin_dynamics_memory/requirements.txt",
    ]
    pkg.verify_install_archive_members(base_valid, "astrbot_plugin_dynamics_memory")

    for bad_entry in (
        "other_root/main.py",
        "astrbot_plugin_dynamics_memory/.github/workflows/ci.yml",
        "astrbot_plugin_dynamics_memory/package-lock.json",
        "astrbot_plugin_dynamics_memory/build_release.py",
        "astrbot_plugin_dynamics_memory/.gitignore",
        "astrbot_plugin_dynamics_memory/IMPLEMENTATION_PLAN.md",
    ):
        with pytest.raises(ValueError):
            pkg.verify_install_archive_members(
                [*base_valid, bad_entry], "astrbot_plugin_dynamics_memory"
            )


def test_ci_workflow_covers_offline_pytest_frontend_package_and_deps():
    ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert ci_path.is_file()
    text = ci_path.read_text(encoding="utf-8")

    for required_token in (
        "offline-python:",
        "frontend:",
        "package-hygiene:",
        "pytest",
        "npm run check:dashboard",
        "npm run test:frontend",
        "tools/package_release.py",
        "verify_install_archive_members",
        "metadata.yaml",
        "requirements.txt",
    ):
        assert required_token in text, (
            f"ci.yml missing required check: {required_token}"
        )


def test_metadata_yaml_requirements_and_version_parity():
    meta_raw = (REPO_ROOT / "metadata.yaml").read_text(encoding="utf-8-sig")
    for field in (
        "name:",
        "display_name:",
        "author:",
        "version:",
        "desc:",
        "astrbot_version:",
        "repo:",
    ):
        assert re.search(rf"^{re.escape(field)}\s*\S+", meta_raw, re.MULTILINE), (
            f"metadata.yaml missing {field}"
        )

    version = _parse_metadata_version(REPO_ROOT / "metadata.yaml")
    pkg_json = json.loads((REPO_ROOT / "package.json").read_text(encoding="utf-8"))
    assert pkg_json["version"] == version
    assert PLUGIN_VERSION == version
    assert _load_plugin_version() == version

    req_lines = [
        line.strip()
        for line in (REPO_ROOT / "requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(req_lines) >= 6
    req_re = re.compile(r"^[A-Za-z0-9_.-]+(\[[A-Za-z0-9_,.-]+\])?([<>=!~].+)?$")
    for line in req_lines:
        assert req_re.match(line), f"invalid requirement specifier: {line}"
