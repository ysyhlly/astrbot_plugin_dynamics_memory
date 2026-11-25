"""
Backend i18n module for bot command responses.
Loads JSON translation files from core/i18n/ directory.
"""

import json
from collections.abc import Callable
from pathlib import Path

from astrbot.api import logger

_SUPPORTED_LANGS = ("zh", "en", "ru", "ja")
_fallback: dict = {}
_translations: dict = {}
_loaded_langs: dict[str, dict] = {}
_current_lang: str = "zh"
_language_resolver: Callable[[], str] | None = None


def _load_lang_file(language: str) -> dict:
    cached = _loaded_langs.get(language)
    if cached is not None:
        return cached
    base = Path(__file__).parent / "i18n"
    target_path = base / f"{language}.json"
    if target_path.exists():
        try:
            with open(target_path, encoding="utf-8") as f:
                data = json.load(f)
            _loaded_langs[language] = data
            return data
        except Exception as exc:
            logger.error(f"Failed to load i18n {language}.json: {exc}")
            return {}
    logger.warning(f"i18n file not found for {language}, falling back to zh")
    return {}


def _normalize_language(language: str | None) -> str:
    raw = str(language or "").strip().replace("_", "-").split("-", 1)[0].lower()
    return raw if raw in _SUPPORTED_LANGS else "zh"


def init(language: str = "zh", *, resolver: Callable[[], str] | None = None):
    """Initialize translations. Fallback always loaded as zh.

    When ``resolver`` is provided, language inheritance (e.g. from the series
    master control plugin) is resolved lazily on each lookup so plugin startup
    order never freezes an outdated default language.
    """
    global _fallback, _translations, _current_lang, _language_resolver
    _language_resolver = resolver
    _loaded_langs.clear()
    _fallback = _load_lang_file("zh")
    lang = _normalize_language(language)
    _current_lang = lang
    _translations = _load_lang_file(lang) or _fallback


def _sync_language() -> None:
    global _current_lang, _translations
    if _language_resolver is None:
        return
    try:
        resolved = _normalize_language(_language_resolver())
    except Exception:
        return
    if resolved != _current_lang:
        _current_lang = resolved
        _translations = _load_lang_file(resolved) or _fallback


def get_language() -> str:
    """Return the active language code after lazy resolution."""
    _sync_language()
    return _current_lang


def _get(data: dict, key: str):
    parts = key.split(".")
    for part in parts:
        if isinstance(data, dict) and part in data:
            data = data[part]
        else:
            return None
    return data


def t(key: str, **kwargs) -> str:
    """Get translated string by dot-notation key."""
    _sync_language()
    value = _get(_translations, key)
    if value is None:
        value = _get(_fallback, key)
    if value is None:
        logger.warning(f"i18n key missing: {key}")
        return key
    if not isinstance(value, str):
        return str(value)
    try:
        return value.format(**kwargs)
    except Exception as exc:
        logger.warning(f"i18n format error for key '{key}': {exc}")
        return value


def t_list(key: str) -> list[str]:
    """Get translated list of strings by dot-notation key."""
    _sync_language()
    value = _get(_translations, key)
    if value is None:
        value = _get(_fallback, key)
    if value is None:
        logger.warning(f"i18n list key missing: {key}")
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return []
