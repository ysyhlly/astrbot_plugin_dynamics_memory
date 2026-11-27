import {msg,tr,setText,apply,feedback} from './series-ui.js';
import { ApiClient } from './modules/api-client.js';

const api = new ApiClient();
const status = document.getElementById('series-status');
const rows = document.getElementById('series-conflicts');
const button = document.getElementById('series-refresh');
let loading = false;

async function refresh() {
  if (loading || !status || !rows || !button) return;
  loading = true;
  button.disabled = true;
  setText(rows,'');
  try {
    await api.ready();
    const data = await api.get('series');
    setText(status,msg('{0} {1} · {2} · {3}: {4} · {5}: {6}{7}{8}',msg('群间记忆'),data.version,msg(data.ready?'已就绪':'未接入或正在初始化'),msg('群间世界'),msg(data.world_ready?'已就绪':'未就绪'),msg('群间学习'),msg(data.learning_installed?'已发现':'未安装'),data.conflicts.length?' '+tr('冲突提供者：{0}。请停用旧记忆插件后重载。',data.conflicts.join(', ')):'',!data.enabled?' '+tr('请先完成迁移，再开启原生配置中的“接入聊天与记忆记录”并重载。'):''));
    if (!data.ready) return;
    const conflicts = await api.get('series/conflicts');
    if (!conflicts.items.length) {
      setText(rows,msg("当前没有来源冲突。同一来源重试不会重复入库。"));
    }
    for (const row of conflicts.items) {
      const card = document.createElement('div');
      card.className = 'card';
      card.style.padding = '12px';
      const label = document.createElement('p');
      setText(label,msg('来源 {0} · {1} · 原文保持不变',row.event_id,msg(row.status==='pending'?'待审查':'已审查')));
      card.append(label);
      const detail = document.createElement('button');
      detail.className = 'btn btn-secondary';
      detail.type = 'button';
      setText(detail,msg("查看原文与冲突内容"));
      const preview = document.createElement('pre');
      preview.style.cssText = 'white-space:pre-wrap;overflow-wrap:anywhere;';
      detail.onclick = async () => {
        detail.disabled = true;
        try {
          const data = await api.get('series/detail', {id: row.id});
          setText(preview,msg('原文（{0}）：\n{1}\n\n冲突内容（{2}）：\n{3}',data.original.actor_id||msg('未知'),data.original.text||msg('来源已不可用'),data.incoming.actor_id||msg('未知'),data.incoming.text||''));
        } catch (error) { setText(preview,String(error.message || error)); }
        finally { detail.disabled = false; }
      };
      card.append(detail, preview);
      if (row.status === 'pending') {
        const reason = document.createElement('input');
        reason.className = 'input';
        setText(reason,msg("审查说明（不会覆盖原文）"),"placeholder");
        setText(reason,msg('来源冲突审查说明'),'aria-label');
        reason.maxLength = 1000;
        const resolve = document.createElement('button');
        resolve.type = 'button';
        resolve.className = 'btn btn-secondary';
        setText(resolve,msg("保留原文并标记已审查"));
        resolve.onclick = async () => {
          if (!reason.value.trim()) { reason.focus(); return; }
          resolve.disabled = true;
          try {
            await api.post('series/resolve', {id: row.id, revision: row.revision, reason: reason.value.trim()});
            await refresh();
          } catch (error) { setText(status,String(error.message || error)); }
          finally { resolve.disabled = false; }
        };
        card.append(reason, resolve);
      }
      rows.append(card);
    }
  } catch (error) { setText(status,msg('读取失败：{0}',error.message||error)); }
  finally { loading = false; button.disabled = false; }
}
if (button) button.addEventListener('click', refresh);
window.lmRefreshSeriesStatus = refresh;

const initWave2 = () => {
  const modPath = './assets/wave2-governance.min.js';
  import(modPath).then((mod) => mod?.initWave2Governance?.(api)).catch(() => {});
};
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initWave2, { once: true });
} else {
  initWave2();
}
