"""Browser acceptance for series status and conflict review; isolated SQLite only."""
import asyncio
import json
import mimetypes
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace as NS
from urllib.parse import parse_qs, urlsplit
from unittest.mock import AsyncMock

from playwright.async_api import async_playwright
from astrbot_plugin_dynamics_memory.core.series import SeriesMixin
from astrbot_plugin_dynamics_memory.storage.conversation_store import ConversationStore


async def main():
    output = Path('/tmp/dynamics-memory-validation/browser')
    output.mkdir(exist_ok=True)
    store = ConversationStore(str(Path(tempfile.mkdtemp(prefix='dynamics-memory-browser-')) / 'ui-test.db'))
    await store.initialize()
    event = dict(schema_version=1, event_id='test-source', type='message.received', actor_id='demo',
                 text='原始陈述', occurred_at=123, scope=dict(platform='qq', umo='qq:GroupMessage:demo', persona_id='p', instance_id='i'))
    await store.ingest_narrative_event(event)
    await store.ingest_narrative_event({**event, 'text': '不同内容'})
    owner = SeriesMixin()
    owner.context = NS(get_all_stars=lambda: [])
    owner.initializer = NS(is_initialized=True, conversation_manager=NS(store=store))
    owner._series_enabled = True
    owner._terminating = False
    owner._series_admin = AsyncMock(return_value='test-admin')
    owner._ensure_plugin_ready = AsyncMock(return_value=(True, ''))
    loop = asyncio.get_running_loop()
    pages = Path(__file__).resolve().parents[1] / 'pages/dashboard'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.startswith('/api/'):
                endpoint = self.path.split('?', 1)[0][5:]
                if endpoint == 'page/series':
                    data = asyncio.run_coroutine_threadsafe(owner.series_status(), loop).result(10)
                elif endpoint == 'page/series/conflicts':
                    data = asyncio.run_coroutine_threadsafe(owner.series_conflict_list(), loop).result(10)
                elif endpoint == 'page/series/detail':
                    identifier = parse_qs(urlsplit(self.path).query)['id'][0]
                    detail = asyncio.run_coroutine_threadsafe(store.get_series_conflict(identifier), loop).result(10)
                    data = {'status': 'ok', 'data': detail}
                elif endpoint == 'page/readiness':
                    data = {'status': 'ok', 'data': {'ready': False, 'message': '离线验收环境'}}
                else:
                    data = {'status': 'ok', 'data': {'items': [], 'nodes': [], 'edges': [], 'total': 0}}
                body, mime = json.dumps(data).encode(), 'application/json'
            else:
                path = (pages / (self.path.split('?', 1)[0].lstrip('/') or 'index.html')).resolve()
                if not path.is_relative_to(pages) or not path.is_file():
                    self.send_error(404)
                    return
                body, mime = path.read_bytes(), mimetypes.guess_type(path.name)[0] or 'text/plain'
            self.send_response(200)
            self.send_header('Content-Type', mime + '; charset=utf-8')
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            result = asyncio.run_coroutine_threadsafe(store.resolve_series_conflict(data['id'], data['revision'], data['reason'], 'test-admin'), loop).result(10)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok', 'data': result}).encode())

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    errors = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=['--no-sandbox'])
            page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
            page.on('pageerror', lambda error: errors.append(str(error)))
            await page.add_init_script('''window.AstrBotPluginView = {
              ready: async()=>({theme:'dark',pluginName:'astrbot_plugin_dynamics_memory'}),
              getContext: ()=>({theme:'dark'}),
              apiGet: async(path,params)=>fetch('/api/'+path+'?'+new URLSearchParams(params||{})).then(r=>r.json()),
              apiPost: async(path,data)=>fetch('/api/'+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}).then(r=>r.json())
            };''')
            await page.goto(f'http://127.0.0.1:{server.server_port}/')
            await page.locator('#series-panel summary').click()
            await page.wait_for_function("document.getElementById('series-status').textContent.includes('已就绪')")
            await page.get_by_role('button', name='查看原文与冲突内容').click()
            await page.wait_for_function("document.getElementById('series-conflicts').textContent.includes('不同内容')")
            await page.get_by_placeholder('审查说明（不会覆盖原文）').fill('核实上游重复上报，保留原文')
            await page.screenshot(path=str(output / 'desktop.png'), full_page=True)
            await page.get_by_role('button', name='保留原文并标记已审查').click()
            await page.wait_for_function("document.getElementById('series-conflicts').textContent.includes('已审查')")
            owner._series_enabled = False
            await page.locator('#series-refresh').click()
            await page.wait_for_function("document.getElementById('series-status').textContent.includes('请先完成迁移')")
            await page.set_viewport_size({'width': 390, 'height': 844})
            await page.screenshot(path=str(output / 'mobile.png'), full_page=True)
            assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
            await browser.close()
        assert not errors, errors
        assert (await store.load_narrative_event(event['scope'], event['event_id']))['text'] == '原始陈述'
        print(json.dumps({'browser_errors': errors, 'reviewed': True, 'original_preserved': True, 'mobile_no_overflow': True}))
    finally:
        server.shutdown()
        await store.close()


if __name__ == '__main__':
    asyncio.run(main())
