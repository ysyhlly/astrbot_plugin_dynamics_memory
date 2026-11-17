import asyncio
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_dynamics_memory.core.series import SeriesMixin, find_role
from astrbot_plugin_dynamics_memory.migration import migrate, snapshot
from astrbot_plugin_dynamics_memory.storage.conversation_store import ConversationStore
from astrbot_plugin_dynamics_memory.core.narrative_api import scope_key

SCOPE = dict(platform='qq', umo='qq:GroupMessage:7', persona_id='p', instance_id='i')


def event(identifier='1', text='我喜欢茶', **changes):
    return dict(schema_version=1, event_id=identifier, type='message.received', scope=dict(SCOPE),
                actor_id='u', text=text, occurred_at=1234, **changes)


@pytest.mark.asyncio
async def test_duplicate_collision_restart_and_review(tmp_path):
    path = str(tmp_path / 'history.db')
    store = ConversationStore(path)
    await store.initialize()
    try:
        accepted = await asyncio.gather(*(store.ingest_narrative_event(event()) for _ in range(8)))
        assert sum(accepted) == 1
        assert not (await store.list_series_conflicts())['items']
        assert not await store.ingest_narrative_event(event(text='我不喜欢茶'))
        assert not await store.ingest_narrative_event(event(text='我不喜欢茶'))
        rows = (await store.list_series_conflicts())['items']
        assert len(rows) == 1 and 'incoming_envelope' not in rows[0]
        assert await store.get_message_count(scope_key(SCOPE)) == 1
        row = rows[0]
        detail = await store.get_series_conflict(row['id'])
        assert detail['original']['text'] == '我喜欢茶'
        assert detail['incoming']['text'] == '我不喜欢茶'
        result = await store.resolve_series_conflict(row['id'], 0, '上游重放内容异常，保留原文', 'admin')
        assert result['source_unchanged'] and result['revision'] == 1
        with pytest.raises(ValueError):
            await store.resolve_series_conflict(row['id'], 0, 'stale', 'admin')
    finally:
        await store.close()
    store = ConversationStore(path)
    await store.initialize()
    try:
        assert not await store.ingest_narrative_event(event())
        assert (await store.list_series_conflicts())['items'][0]['status'] == 'reviewed'
        assert (await store.load_narrative_event(SCOPE, '1'))['text'] == '我喜欢茶'
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_real_change_other_member_other_scope_are_not_duplicates(tmp_path):
    store = ConversationStore(str(tmp_path / 'history.db'))
    await store.initialize()
    try:
        assert await store.ingest_narrative_event(event())
        assert await store.ingest_narrative_event(event('2', '现在不喜欢茶'))
        for field, other in [('umo', 'qq:GroupMessage:8'), ('persona_id', 'q'), ('instance_id', 'j')]:
            data = event()
            data['scope'][field] = other
            assert await store.ingest_narrative_event(data)
        assert not (await store.list_series_conflicts())['items']
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('field,bad', [('schema_version', True), ('text', ''), ('actor_id', ''), ('occurred_at', float('nan')), ('type', 'plan.created')])
async def test_invalid_events_do_not_write(tmp_path, field, bad):
    store = ConversationStore(str(tmp_path / 'history.db'))
    await store.initialize()
    try:
        data = event()
        data[field] = bad
        assert not await store.ingest_narrative_event(data)
        assert not (await store.list_series_conflicts())['items']
    finally:
        await store.close()


def entry(name, owner, active=True):
    return SimpleNamespace(name=name, star_cls=owner, activated=active)


def plugin(rows):
    owner = SeriesMixin()
    owner.context = SimpleNamespace(get_all_stars=lambda: rows)
    owner.initializer = SimpleNamespace(is_initialized=True)
    owner._series_enabled = True
    owner._terminating = False
    return owner


def test_ownership_detects_disable_reload_and_default_manifest():
    rows = []
    owner = plugin(rows)
    assert owner.get_dynamics_capabilities()['ready']
    rows.append(entry('astrbot_plugin_livingmemory', object()))
    assert not owner.series_runtime_allowed()
    rows[0].activated = False
    assert owner.series_runtime_allowed()
    owner._series_enabled = False
    assert not owner.get_dynamics_capabilities()['enabled']


def test_capability_conflict_and_not_ready():
    world = SimpleNamespace(get_dynamics_capabilities=lambda: dict(schema_version=1, enabled=True, ready=True, capabilities={'character': 1}))
    rows = [entry('world', world)]
    owner = plugin(rows)
    assert find_role(owner.context, 'character') is world
    rows.append(entry('other', world))
    assert find_role(owner.context, 'character') is None


@pytest.mark.asyncio
async def test_world_learning_readthrough_has_no_writes():
    world = SimpleNamespace(get_dynamics_capabilities=lambda: dict(schema_version=1, enabled=True, ready=True, capabilities={'character': 1}),
                            get_character_context=AsyncMock(return_value={'card_version': 'v', 'text': '角色设定'}))
    requests = []
    learning = SimpleNamespace(get_learning_context=lambda **kw: requests.append(kw) or {'context': '已审查的表达'})
    rows = [entry('world', world), entry('astrbot_plugin_dynamics_learning', learning)]
    owner = plugin(rows)
    owner._narrative_scope_allowed = AsyncMock(return_value=True)
    owner.get_evidence_view = AsyncMock(return_value={'atoms': []})
    request = dict(scope=SCOPE['umo'], persona='fingerprint', member='u', baseline='b', host_version='v')
    result = await owner.get_series_context(SCOPE, [{'source_event_id': '1'}], learning_request=request)
    assert result['world']['card_version'] == 'v'
    assert result['learning']['context'] == '已审查的表达'
    assert owner.get_evidence_view.await_args.kwargs['allow_derivation'] is False
    assert requests == [request]
    owner._narrative_scope_allowed = AsyncMock(return_value=False)
    assert not (await owner.get_series_context(SCOPE, []))['available']


@pytest.mark.asyncio
async def test_world_late_result_is_discarded():
    rows = []
    async def read(scope):
        rows.clear()
        return {'text': 'stale'}
    world = SimpleNamespace(get_dynamics_capabilities=lambda: dict(schema_version=1, enabled=True, ready=True, capabilities={'character': 1}),
                            get_character_context=read)
    rows.append(entry('world', world))
    owner = plugin(rows)
    owner._narrative_scope_allowed = AsyncMock(return_value=True)
    owner.get_evidence_view = AsyncMock(return_value={})
    assert (await owner.get_series_context(SCOPE, []))['world'] is None


def source_fixture(tmp_path):
    source = tmp_path / 'old'
    source.mkdir()
    db = sqlite3.connect(source / 'conversations.db')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE history(id INTEGER PRIMARY KEY, text TEXT)')
    db.execute("INSERT INTO history(text) VALUES ('原文')")
    db.commit()
    (source / 'livingmemory.index').write_bytes(b'unchanged index')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'llm_provider_id': 'existing', 'daily_limit': 0}), encoding='utf-8')
    return source, config, db


def test_migration_preserves_wal_index_config_and_source(tmp_path):
    source, config, db = source_fixture(tmp_path)
    try:
        preview = snapshot(source, config)
        target = tmp_path / 'new'
        target.mkdir()  # New disabled plugin may have created its empty data directory.
        result = migrate(source, config, target, preview['fingerprint'], offline=True)
        assert result['source_unchanged']
        assert snapshot(source, config)['fingerprint'] == preview['fingerprint']
        with sqlite3.connect(target / 'conversations.db') as restored:
            assert restored.execute('SELECT text FROM history').fetchone()[0] == '原文'
        assert (target / 'livingmemory.index').read_bytes() == b'unchanged index'
        settings = json.loads((target / 'migration-settings.json').read_text())
        assert settings == {'llm_provider_id': 'existing', 'daily_limit': 0, 'series_runtime_enabled': False}
        with pytest.raises(ValueError):
            migrate(source, config, target, preview['fingerprint'], offline=True)
    finally:
        db.close()


def test_migration_rejects_changed_source_and_no_offline(tmp_path):
    source, config, db = source_fixture(tmp_path)
    try:
        preview = snapshot(source, config)
        with pytest.raises(ValueError):
            migrate(source, config, tmp_path / 'new', preview['fingerprint'])
        db.execute("INSERT INTO history(text) VALUES ('新增')")
        db.commit()
        with pytest.raises(ValueError):
            migrate(source, config, tmp_path / 'new', preview['fingerprint'], offline=True)
        assert not (tmp_path / 'new').exists()
    finally:
        db.close()


def test_context_dedup_does_not_merge_people_time_polarity_or_receipts():
    from astrbot_plugin_dynamics_memory.core.context_reconciliation import reconcile_rows
    row = dict(kind='reported_claim', status='reported_unverified', subject_ids=['u'], scope={'scope_key': 's'},
               claim='喜欢 茶', conditions='', polarity='positive', valid_from=1, valid_until=None)
    original = json.loads(json.dumps(row))
    assert len(reconcile_rows([row, {**row, 'atom_id': 'other'}])) == 1
    for change in ({'subject_ids': ['v']}, {'scope': {'scope_key': 't'}}, {'valid_from': 2},
                   {'polarity': 'negative'}, {'conditions': '今天'}, {'claim': '不喜欢茶'}):
        assert len(reconcile_rows([row, {**row, **change}])) == 2
    receipt = {**row, 'kind': 'execution_receipt'}
    assert len(reconcile_rows([receipt, receipt])) == 2
    assert row == original


@pytest.mark.asyncio
async def test_public_api_does_not_initialize_or_expose_disabled_owner():
    from astrbot_plugin_dynamics_memory.api.public import ensure_ready, get_memory_engine
    owner = plugin([])
    owner._series_enabled = False
    owner.initializer.memory_engine = object()
    owner.initializer.ensure_initialized = AsyncMock(return_value=True)
    assert not await ensure_ready(owner)
    assert get_memory_engine(owner) is None
    owner.initializer.ensure_initialized.assert_not_awaited()


@pytest.mark.asyncio
async def test_registered_tool_stops_after_owner_revocation():
    from astrbot_plugin_dynamics_memory.core.tools.memory_search_tool import MemorySearchTool
    tool = MemorySearchTool()
    tool._series_owner_guard = lambda: False
    result = await tool.call(None, 'private query')
    assert '不可用' in result


@pytest.mark.asyncio
async def test_forbidden_admin_cannot_read_conflict_sources():
    owner = plugin([])
    owner._series_admin = AsyncMock(side_effect=PermissionError())
    owner.initializer.conversation_manager = SimpleNamespace(store=SimpleNamespace(get_series_conflict=AsyncMock()))
    with pytest.raises(PermissionError):
        await owner.series_conflict_detail()
    owner.initializer.conversation_manager.store.get_series_conflict.assert_not_awaited()


@pytest.mark.asyncio
async def test_private_original_collision_is_also_audited(tmp_path):
    store = ConversationStore(str(tmp_path / 'history.db'))
    await store.initialize()
    try:
        data = event('cd-original-input-v1:message-7')
        data['scope']['umo'] = 'qq:FriendMessage:u'
        assert await store.ingest_owned_private_original_event(data, source_platform='qq')
        assert not await store.ingest_owned_private_original_event({**data, 'text': '内容不同'}, source_platform='qq')
        assert len((await store.list_series_conflicts())['items']) == 1
        assert await store.get_message_count(data['scope']['umo']) == 1
    finally:
        await store.close()


def test_migration_empty_wal_appearance_does_not_change_fingerprint(tmp_path):
    source, config, db = source_fixture(tmp_path)
    db.close()
    before = snapshot(source, config)
    # Read-only SQLite connections can create empty WAL sidecars on a quiescent
    # WAL-mode database. Such a file contains no committed source changes.
    journal = source / 'unrelated.db-wal'
    journal.write_bytes(b'')
    after = snapshot(source, config)
    assert before['fingerprint'] == after['fingerprint']
    journal.write_bytes(b'nonempty journal must affect snapshot')
    assert snapshot(source, config)['fingerprint'] != after['fingerprint']
