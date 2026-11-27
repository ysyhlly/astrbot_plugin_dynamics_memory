/**
 * Wave 2 Lazy Governance & UX Enhancement Module
 *
 * Loaded asynchronously outside the first-load bundle budget. Progressively
 * enhances:
 *  - System page: auto pending conflict/correction badges, batch conflict
 *    resolution + memory jump, member correction review queue, and memory
 *    health checkup (duplicates, contradictions, dead memories, orphan nodes).
 *  - Memories page: quick filter preset chips (low importance, zero-access /
 *    stale, source evidence, origin) & canonical type group selector.
 *  - Graph page: node type filter chips (person/topic/fact/summary) and
 *    synonym entity merge / orphan node cleanup governance drawer.
 *  - Settings page: explicit hot-reloaded vs reload-required feedback banner.
 */

import { ApiClient } from "./api-client.js";
import { esc, showToast } from "./utils.js";

const STYLE_ID = "wave2-governance-style";

const WAVE2_CSS = `
.w2-badge{display:inline-flex;align-items:center;justify-content:center;min-width:18px;height:18px;padding:0 5px;margin-left:6px;border-radius:999px;font-size:11px;font-weight:600;background:#dc2626;color:#fff;line-height:1}
.w2-badge[hidden]{display:none}
.w2-panel{margin-top:14px}
.w2-row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.w2-chip-bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:8px 0 10px}
.w2-chip{padding:4px 10px;border-radius:999px;border:1px solid var(--border-color,#cbd5e1);background:transparent;color:inherit;font-size:12px;cursor:pointer}
.w2-chip.active{background:var(--accent-color,#2563eb);border-color:var(--accent-color,#2563eb);color:#fff}
.w2-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:10px 0}
.w2-kpi{padding:10px 12px;border:1px solid var(--border-color,#e2e8f0);border-radius:10px}
.w2-kpi-val{font-size:20px;font-weight:700}
.w2-kpi-lbl{font-size:12px;opacity:.75}
.w2-item{padding:10px 12px;border:1px solid var(--border-color,#e2e8f0);border-radius:10px;margin-top:8px}
.w2-item-meta{font-size:12px;opacity:.75;margin-bottom:4px}
.w2-item-text{font-size:13px;line-height:1.45;word-break:break-word}
.w2-tabs{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}
.w2-drawer{margin-top:8px;padding:10px 12px;border:1px solid var(--border-color,#cbd5e1);border-radius:10px;background:var(--surface-bg,rgba(15,23,42,.04))}
.w2-pill-hot{display:inline-block;padding:1px 6px;border-radius:999px;font-size:11px;background:rgba(22,163,74,.14);color:#15803d;margin-left:6px}
.w2-pill-reload{display:inline-block;padding:1px 6px;border-radius:999px;font-size:11px;background:rgba(217,119,6,.16);color:#b45309;margin-left:6px}
`;

export function injectWave2Styles(doc = globalThis.document) {
  if (!doc || !doc.head || doc.getElementById(STYLE_ID)) return;
  const style = doc.createElement("style");
  style.id = STYLE_ID;
  style.textContent = WAVE2_CSS;
  doc.head.appendChild(style);
}

export function formatSettingsSaveMessage(result) {
  const hot = Array.isArray(result?.hot_applied) ? result.hot_applied : [];
  const reload = Array.isArray(result?.reload_required) ? result.reload_required : [];
  if (!hot.length && !reload.length) {
    return typeof window !== "undefined" && typeof window.t === "function"
      ? window.t("settings.saved")
      : "设置已保存";
  }
  if (!reload.length) {
    return `已保存：${hot.length} 项配置已立即热生效（无需重载插件）。`;
  }
  const preview = reload.slice(0, 3).join(", ") + (reload.length > 3 ? " 等" : "");
  if (!hot.length) {
    return `已保存：${reload.length} 项配置需重载插件后生效（${preview}）。`;
  }
  return `已保存：${hot.length} 项已立即热生效；${reload.length} 项需重载插件后生效（${preview}）。`;
}

export class Wave2GovernanceController {
  constructor(apiClient = new ApiClient(), doc = globalThis.document) {
    this.api = apiClient;
    this.doc = doc;
    this.memPreset = {
      lowImportance: false,
      staleZeroAccess: false,
      hasSource: "",
      origin: "",
      canonicalGroup: "",
    };
    this.graphNodeTypes = new Set();
    this.healthData = null;
    this.healthTab = "duplicates";
    this.selectedDeadIds = new Set();
    this.pendingConflicts = [];
    this.pendingCorrectionsCount = 0;
  }

  init() {
    if (!this.doc) return;
    injectWave2Styles(this.doc);
    this._installGlobalHooks();
    this._mountSystemBadges();
    this._mountConflictBatchBar();
    this._mountCorrectionQueuePanel();
    this._mountHealthCheckupPanel();
    this._mountMilestonesPanel();
    this._mountMemoryQuickFilters();
    this._mountGraphGovernanceControls();
    this.refreshBadges().catch(() => {});
    this.loadMilestones().catch(() => {});
  }

  _installGlobalHooks() {
    if (typeof window === "undefined") return;
    window.lmWave2MemoryParams = () => this.buildMemoryFilterParams();
    window.lmOnSettingsSaved = (result, settingsPage) => {
      const msg = formatSettingsSaveMessage(result);
      settingsPage?.renderFeedback?.(msg, false);
      showToast(msg);
    };
    const bridge =
      window.DynamicsPanelBridge ||
      window.AstrBotPluginView ||
      window.AstrBotPluginPage;
    if (bridge && !bridge._w2Wrapped) {
      bridge._w2Wrapped = true;
      if (typeof bridge.apiGet === "function") {
        const origGet = bridge.apiGet.bind(bridge);
        bridge.apiGet = (endpoint, params) => {
          const clean = String(endpoint || "").replace(/^\/+/, "");
          if (clean === "page/memories" || clean === "memories") {
            const extra = this.buildMemoryFilterParams();
            if (Object.keys(extra).length) {
              return origGet(endpoint, { ...(params || {}), ...extra });
            }
          }
          return origGet(endpoint, params);
        };
      }
      if (typeof bridge.apiPost === "function") {
        const origPost = bridge.apiPost.bind(bridge);
        bridge.apiPost = async (endpoint, body) => {
          const res = await origPost(endpoint, body);
          const clean = String(endpoint || "").replace(/^\/+/, "");
          if (clean === "page/config/update" || clean === "config/update") {
            const payload = res && typeof res === "object" && res.data ? res.data : res;
            if (payload && (Array.isArray(payload.hot_applied) || Array.isArray(payload.reload_required))) {
              const msg = formatSettingsSaveMessage(payload);
              setTimeout(() => {
                const bar = this.doc?.getElementById("settings-feedback");
                const lbl = this.doc?.getElementById("settings-feedback-message");
                if (bar && lbl) {
                  bar.hidden = false;
                  bar.classList.remove("is-error");
                  lbl.textContent = msg;
                }
                showToast(msg);
              }, 80);
            }
          }
          return res;
        };
      }
    }
  }

  buildMemoryFilterParams() {
    const out = {};
    if (this.memPreset.lowImportance) out.importance_max = "0.35";
    if (this.memPreset.staleZeroAccess) {
      out.zero_access = "true";
      out.stale_days = "30";
    }
    if (this.memPreset.hasSource === "true" || this.memPreset.hasSource === "false") {
      out.has_source = this.memPreset.hasSource;
    }
    if (this.memPreset.origin) out.memory_origin = this.memPreset.origin;
    if (this.memPreset.canonicalGroup) out.type = this.memPreset.canonicalGroup;
    return out;
  }

  /* ================================================================
     1. System Nav Badge & Auto Pending Counts
     ================================================================ */
  _mountSystemBadges() {
    const navBtn = this.doc.querySelector('.nav-item[data-page="system"]');
    if (!navBtn || this.doc.getElementById("w2-system-badge")) return;
    const badge = this.doc.createElement("span");
    badge.id = "w2-system-badge";
    badge.className = "w2-badge";
    badge.hidden = true;
    navBtn.appendChild(badge);
  }

  _updateSystemBadge() {
    const badge = this.doc?.getElementById("w2-system-badge");
    if (!badge) return;
    const total = (this.pendingConflicts?.length || 0) + (this.pendingCorrectionsCount || 0);
    badge.textContent = String(total);
    badge.hidden = total <= 0;
  }

  async refreshBadges() {
    let conflictPending = 0;
    try {
      const data = await this.api.get("series/conflicts");
      const items = Array.isArray(data?.items) ? data.items : [];
      this.pendingConflicts = items.filter((item) => item && item.status === "pending");
      conflictPending = Number.isFinite(Number(data?.pending_count))
        ? Number(data.pending_count)
        : this.pendingConflicts.length;
      this._renderConflictBatchBar();
    } catch {
      this.pendingConflicts = [];
    }

    try {
      const corr = await this.api.get("corrections", { status: "pending", limit: 20 });
      this.pendingCorrectionsCount = Number(corr?.pending_count || 0);
      this._renderCorrections(corr);
    } catch {
      this.pendingCorrectionsCount = 0;
    }

    this._updateSystemBadge();
    return {
      pendingConflicts: conflictPending,
      pendingCorrections: this.pendingCorrectionsCount,
    };
  }

  /* ================================================================
     2. Batch Source Conflict Resolution & Jump to Related Memories
     ================================================================ */
  _mountConflictBatchBar() {
    const panel = this.doc.getElementById("system-series-panel");
    if (!panel || this.doc.getElementById("w2-conflict-batch")) return;
    const bar = this.doc.createElement("div");
    bar.id = "w2-conflict-batch";
    bar.className = "w2-drawer";
    bar.hidden = true;
    panel.insertBefore(bar, this.doc.getElementById("series-conflicts"));
  }

  _renderConflictBatchBar() {
    const bar = this.doc?.getElementById("w2-conflict-batch");
    if (!bar) return;
    const pending = this.pendingConflicts || [];
    if (!pending.length) {
      bar.hidden = true;
      bar.innerHTML = "";
      return;
    }
    bar.hidden = false;
    const listHtml = pending
      .map((item) => {
        const sid = String(item.session_id || "");
        const jumpBtn = sid
          ? `<button type="button" class="btn btn-ghost btn-sm" data-w2-jump-session="${esc(sid)}">查看相关记忆</button>`
          : "";
        return `<label class="w2-row" style="margin-top:4px;font-size:12px">
          <input type="checkbox" class="w2-conflict-check" value="${esc(item.id)}" data-rev="${esc(item.revision)}" checked />
          <span>#${esc(item.id)} · ${esc(item.event_id || "")}</span>
          ${jumpBtn}
        </label>`;
      })
      .join("");
    bar.innerHTML = `
      <div class="w2-row" style="justify-content:space-between">
        <strong>待审查来源冲突 (${pending.length})</strong>
        <div class="w2-row">
          <input id="w2-conflict-batch-reason" class="input input-sm" type="text" placeholder="批量审查说明（保留原文）" style="min-width:200px" />
          <button type="button" id="w2-conflict-batch-btn" class="btn btn-primary btn-sm">批量标记已审查</button>
        </div>
      </div>
      <div style="margin-top:6px">${listHtml}</div>
      <p id="w2-conflict-batch-msg" class="text-tertiary" style="font-size:12px;margin-top:6px"></p>
    `;

    bar.querySelectorAll("[data-w2-jump-session]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const session = btn.dataset.w2JumpSession || "";
        if (typeof window !== "undefined") {
          window.lmCommitScope?.(session, "wave2");
          window.lmSwitchPage?.("memory");
        }
      });
    });

    const submitBtn = bar.querySelector("#w2-conflict-batch-btn");
    submitBtn?.addEventListener("click", async () => {
      const reasonInput = bar.querySelector("#w2-conflict-batch-reason");
      const msgEl = bar.querySelector("#w2-conflict-batch-msg");
      const reason = String(reasonInput?.value || "").trim();
      if (!reason) {
        if (msgEl) msgEl.textContent = "请先填写批量审查说明。";
        reasonInput?.focus?.();
        return;
      }
      const items = Array.from(bar.querySelectorAll(".w2-conflict-check:checked")).map((cb) => ({
        id: cb.value,
        revision: Number(cb.dataset.rev),
      }));
      if (!items.length) {
        if (msgEl) msgEl.textContent = "请至少勾选一条待审查冲突。";
        return;
      }
      submitBtn.disabled = true;
      try {
        const res = await this.api.post("series/resolve_batch", { items, reason });
        const count = Number(res?.resolved_count || 0);
        if (msgEl) msgEl.textContent = `已批量处理 ${count} 条来源冲突。`;
        showToast(`已批量处理 ${count} 条来源冲突`);
        await this.refreshBadges();
        window.lmRefreshSeriesStatus?.();
      } catch (err) {
        if (msgEl) msgEl.textContent = `批量处理失败：${err.message || err}`;
      } finally {
        submitBtn.disabled = false;
      }
    });
  }

  /* ================================================================
     3. Member Correction Proposal Review Queue (System page)
     ================================================================ */
  _mountCorrectionQueuePanel() {
    const sysBody = this.doc.querySelector("#page-system .page-body");
    if (!sysBody || this.doc.getElementById("w2-corrections-panel")) return;
    const panel = this.doc.createElement("div");
    panel.id = "w2-corrections-panel";
    panel.className = "system-panel mb-6 w2-panel";
    panel.innerHTML = `
      <div class="system-panel-header">
        <div class="system-panel-title">群友记忆纠错待审队列 <span id="w2-corr-badge" class="w2-badge" hidden>0</span></div>
        <div class="system-panel-tools">
          <button type="button" class="btn btn-secondary btn-sm" id="w2-corr-refresh">刷新队列</button>
        </div>
      </div>
      <p class="workflow-hint">群友通过 /纠错 提交的个人记忆修正申请；批准后自动修订目标记忆并记录修订历史。</p>
      <div id="w2-corr-list"></div>
    `;
    const repairPanel = this.doc.getElementById("system-repair-panel");
    sysBody.insertBefore(panel, repairPanel || null);
    panel.querySelector("#w2-corr-refresh")?.addEventListener("click", () => {
      this.loadCorrections();
    });
  }

  async loadCorrections() {
    try {
      const data = await this.api.get("corrections", { status: "all", limit: 30 });
      this.pendingCorrectionsCount = Number(data?.pending_count || 0);
      this._renderCorrections(data);
      this._updateSystemBadge();
    } catch (err) {
      const listEl = this.doc?.getElementById("w2-corr-list");
      if (listEl) listEl.innerHTML = `<p class="text-tertiary">加载失败：${esc(err.message || err)}</p>`;
    }
  }

  _renderCorrections(data) {
    const listEl = this.doc?.getElementById("w2-corr-list");
    const badge = this.doc?.getElementById("w2-corr-badge");
    const pendingCount = Number(data?.pending_count || 0);
    if (badge) {
      badge.textContent = String(pendingCount);
      badge.hidden = pendingCount <= 0;
    }
    if (!listEl) return;
    const items = Array.isArray(data?.items) ? data.items : [];
    if (!items.length) {
      listEl.innerHTML = `<p class="text-tertiary" style="font-size:12px">当前没有群友提交的记忆纠错申请。</p>`;
      return;
    }
    listEl.innerHTML = items
      .map((item) => {
        const isPending = item.status === "pending";
        const controls = isPending
          ? `<div class="w2-row" style="margin-top:8px">
               <input type="text" class="input input-sm" data-w2-corr-text="${item.id}" value="${esc(item.proposed_text || item.reason)}" style="flex:1;min-width:220px" />
               <button type="button" class="btn btn-primary btn-sm" data-w2-corr-act="approve" data-id="${item.id}">批准并修订</button>
               <button type="button" class="btn btn-secondary btn-sm" data-w2-corr-act="reject" data-id="${item.id}">驳回</button>
             </div>`
          : `<div class="w2-item-meta">状态：${esc(item.status)} ${item.review_note ? "· " + esc(item.review_note) : ""}</div>`;
        return `<div class="w2-item">
          <div class="w2-item-meta">申请 #${esc(item.id)} · 记忆 #${esc(item.memory_id)} · 提交人：${esc(item.requester_name || item.requester_id)} · 状态：${esc(item.status)}</div>
          <div class="w2-item-text"><strong>原记忆预览：</strong>${esc(item.original_preview)}</div>
          <div class="w2-item-text"><strong>纠错说明：</strong>${esc(item.reason)}</div>
          ${controls}
        </div>`;
      })
      .join("");

    listEl.querySelectorAll("[data-w2-corr-act]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = Number(btn.dataset.id);
        const decision = btn.dataset.w2CorrAct;
        const input = listEl.querySelector(`[data-w2-corr-text="${id}"]`);
        const replacement = String(input?.value || "").trim();
        btn.disabled = true;
        try {
          await this.api.post("corrections/review", {
            id,
            decision,
            replacement: decision === "approve" ? replacement : "",
            review_note: decision === "approve" ? "管理员审核通过" : "管理员驳回",
          });
          showToast(decision === "approve" ? "已批准并修订目标记忆" : "已驳回该纠错申请");
          await this.loadCorrections();
        } catch (err) {
          showToast(`操作失败：${err.message || err}`, true);
          btn.disabled = false;
        }
      });
    });
  }

  /* ================================================================
     4. Memory Health Checkup & Smart Dedup / Contradiction Governance
     ================================================================ */
  _mountHealthCheckupPanel() {
    const sysBody = this.doc.querySelector("#page-system .page-body");
    if (!sysBody || this.doc.getElementById("w2-health-panel")) return;
    const panel = this.doc.createElement("div");
    panel.id = "w2-health-panel";
    panel.className = "system-panel mb-6 w2-panel";
    panel.innerHTML = `
      <div class="system-panel-header">
        <div class="system-panel-title">记忆库健康体检与智能治理</div>
        <div class="system-panel-tools">
          <button type="button" class="btn btn-primary btn-sm" id="w2-health-scan-btn">运行健康体检</button>
        </div>
      </div>
      <p class="workflow-hint">扫描高相似重复记忆、疑似矛盾偏好、长期零召回冷记忆及图谱孤立节点，支持一键预览合并与批量治理。</p>
      <div id="w2-health-body"><p class="text-tertiary" style="font-size:12px">点击“运行健康体检”开始扫描。</p></div>
    `;
    const corrPanel = this.doc.getElementById("w2-corrections-panel");
    sysBody.insertBefore(panel, corrPanel || null);
    panel.querySelector("#w2-health-scan-btn")?.addEventListener("click", () => {
      this.runHealthCheckup();
    });
  }

  async runHealthCheckup() {
    const body = this.doc?.getElementById("w2-health-body");
    const btn = this.doc?.getElementById("w2-health-scan-btn");
    if (btn) btn.disabled = true;
    if (body) body.innerHTML = `<p class="text-tertiary" style="font-size:12px">正在扫描记忆库与图谱…</p>`;
    try {
      const data = await this.api.get("governance/health", { limit: 20 });
      this.healthData = data;
      this.selectedDeadIds.clear();
      this._renderHealthReport();
      return data;
    } catch (err) {
      if (body) body.innerHTML = `<p class="text-tertiary">扫描失败：${esc(err.message || err)}</p>`;
      return null;
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  _renderHealthReport() {
    const body = this.doc?.getElementById("w2-health-body");
    if (!body || !this.healthData) return;
    const s = this.healthData.summary || {};
    const tab = this.healthTab;
    body.innerHTML = `
      <div class="w2-grid">
        <div class="w2-kpi"><div class="w2-kpi-val">${esc(s.health_score ?? "--")}</div><div class="w2-kpi-lbl">健康分 (已扫 ${esc(s.scanned_memories ?? 0)})</div></div>
        <div class="w2-kpi"><div class="w2-kpi-val">${esc(s.duplicate_count ?? 0)}</div><div class="w2-kpi-lbl">疑似重复对</div></div>
        <div class="w2-kpi"><div class="w2-kpi-val">${esc(s.contradiction_count ?? 0)}</div><div class="w2-kpi-lbl">疑似矛盾</div></div>
        <div class="w2-kpi"><div class="w2-kpi-val">${esc(s.dead_count ?? 0)}</div><div class="w2-kpi-lbl">长期零召回</div></div>
        <div class="w2-kpi"><div class="w2-kpi-val">${esc(s.orphan_node_count ?? 0)}</div><div class="w2-kpi-lbl">孤立图谱节点</div></div>
      </div>
      <div class="w2-tabs">
        <button type="button" class="w2-chip ${tab === "duplicates" ? "active" : ""}" data-w2-tab="duplicates">重复记忆 (${esc(s.duplicate_count ?? 0)})</button>
        <button type="button" class="w2-chip ${tab === "contradictions" ? "active" : ""}" data-w2-tab="contradictions">矛盾记忆 (${esc(s.contradiction_count ?? 0)})</button>
        <button type="button" class="w2-chip ${tab === "dead" ? "active" : ""}" data-w2-tab="dead">冷记忆 (${esc(s.dead_count ?? 0)})</button>
        <button type="button" class="w2-chip ${tab === "orphans" ? "active" : ""}" data-w2-tab="orphans">孤立节点 (${esc(s.orphan_node_count ?? 0)})</button>
      </div>
      <div id="w2-health-tab-content"></div>
    `;

    body.querySelectorAll("[data-w2-tab]").forEach((tBtn) => {
      tBtn.addEventListener("click", () => {
        this.healthTab = tBtn.dataset.w2Tab;
        this._renderHealthReport();
      });
    });

    const content = body.querySelector("#w2-health-tab-content");
    if (!content) return;

    if (tab === "duplicates") {
      const items = this.healthData.duplicates || [];
      if (!items.length) {
        content.innerHTML = `<p class="text-tertiary" style="font-size:12px">未发现高相似重复记忆。</p>`;
        return;
      }
      content.innerHTML = items
        .map(
          (d) => `<div class="w2-item">
            <div class="w2-item-meta">相似度 ${esc(Math.round(Number(d.similarity || 0) * 100))}% · 保留 #${esc(d.keep_id)} ← 合并 #${esc(d.duplicate_id)}</div>
            <div class="w2-item-text"><strong>主记忆：</strong>${esc(d.keep_preview)}</div>
            <div class="w2-item-text"><strong>重复项：</strong>${esc(d.duplicate_preview)}</div>
            <div class="w2-row" style="margin-top:8px">
              <button type="button" class="btn btn-secondary btn-sm" data-w2-merge-preview data-keep="${esc(d.keep_id)}" data-dup="${esc(d.duplicate_id)}">预览合并</button>
              <button type="button" class="btn btn-primary btn-sm" data-w2-merge-exec data-keep="${esc(d.keep_id)}" data-dup="${esc(d.duplicate_id)}">一键合并并归档重复项</button>
            </div>
            <div class="w2-item-meta" data-w2-merge-out="${esc(d.keep_id)}-${esc(d.duplicate_id)}" style="margin-top:6px"></div>
          </div>`
        )
        .join("");

      content.querySelectorAll("[data-w2-merge-preview]").forEach((b) => {
        b.addEventListener("click", async () => {
          const keepId = Number(b.dataset.keep);
          const dupId = Number(b.dataset.dup);
          const out = content.querySelector(`[data-w2-merge-out="${keepId}-${dupId}"]`);
          b.disabled = true;
          try {
            const res = await this.api.post("governance/merge", {
              keep_id: keepId,
              merge_ids: [dupId],
              dry_run: true,
            });
            if (out) out.textContent = `预览合并结果：${res.merged_content || ""}`;
          } catch (err) {
            if (out) out.textContent = `预览失败：${err.message || err}`;
          } finally {
            b.disabled = false;
          }
        });
      });

      content.querySelectorAll("[data-w2-merge-exec]").forEach((b) => {
        b.addEventListener("click", async () => {
          const keepId = Number(b.dataset.keep);
          const dupId = Number(b.dataset.dup);
          b.disabled = true;
          try {
            await this.api.post("governance/merge", {
              keep_id: keepId,
              merge_ids: [dupId],
              dry_run: false,
            });
            showToast(`已合并 #${dupId} 至 #${keepId} 并软归档重复项`);
            await this.runHealthCheckup();
          } catch (err) {
            showToast(`合并失败：${err.message || err}`, true);
            b.disabled = false;
          }
        });
      });
    } else if (tab === "contradictions") {
      const items = this.healthData.contradictions || [];
      if (!items.length) {
        content.innerHTML = `<p class="text-tertiary" style="font-size:12px">未发现疑似矛盾记忆。</p>`;
        return;
      }
      content.innerHTML = items
        .map(
          (c) => `<div class="w2-item">
            <div class="w2-item-meta">${esc(c.reason)} · 记忆 #${esc(c.memory_a_id)} vs #${esc(c.memory_b_id)}</div>
            <div class="w2-item-text"><strong>#${esc(c.memory_a_id)}：</strong>${esc(c.preview_a)}</div>
            <div class="w2-item-text"><strong>#${esc(c.memory_b_id)}：</strong>${esc(c.preview_b)}</div>
            <div class="w2-row" style="margin-top:8px">
              <button type="button" class="btn btn-secondary btn-sm" data-w2-peek="${esc(c.memory_a_id)}">查看 #${esc(c.memory_a_id)}</button>
              <button type="button" class="btn btn-secondary btn-sm" data-w2-peek="${esc(c.memory_b_id)}">查看 #${esc(c.memory_b_id)}</button>
            </div>
          </div>`
        )
        .join("");
      content.querySelectorAll("[data-w2-peek]").forEach((b) => {
        b.addEventListener("click", () => {
          window.lmOpenPeekMemory?.({ memory_id: Number(b.dataset.w2Peek) });
        });
      });
    } else if (tab === "dead") {
      const items = this.healthData.dead_memories || [];
      if (!items.length) {
        content.innerHTML = `<p class="text-tertiary" style="font-size:12px">未发现长期零召回低价值冷记忆。</p>`;
        return;
      }
      const rowsHtml = items
        .map(
          (m) => `<label class="w2-item" style="display:block;cursor:pointer">
            <div class="w2-row">
              <input type="checkbox" class="w2-dead-cb" value="${esc(m.id)}" checked />
              <span class="w2-item-meta">#${esc(m.id)} · ${esc(m.age_days)} 天未召回 · 重要度 ${esc(m.importance)}</span>
            </div>
            <div class="w2-item-text">${esc(m.preview)}</div>
          </label>`
        )
        .join("");
      content.innerHTML = `
        <div class="w2-row" style="margin-bottom:8px">
          <button type="button" class="btn btn-secondary btn-sm" id="w2-dead-downweight">批量降权 (×0.5)</button>
          <button type="button" class="btn btn-danger btn-sm" id="w2-dead-archive">批量软归档</button>
        </div>
        ${rowsHtml}
      `;
      const runBulk = async (action) => {
        const ids = Array.from(content.querySelectorAll(".w2-dead-cb:checked")).map((cb) =>
          Number(cb.value)
        );
        if (!ids.length) {
          showToast("请先勾选要处理的记忆", true);
          return;
        }
        try {
          const res = await this.api.post("governance/bulk_action", { ids, action });
          showToast(
            action === "archive"
              ? `已软归档 ${res.count || ids.length} 条冷记忆`
              : `已降权 ${res.count || ids.length} 条冷记忆`
          );
          await this.runHealthCheckup();
        } catch (err) {
          showToast(`批量操作失败：${err.message || err}`, true);
        }
      };
      content.querySelector("#w2-dead-downweight")?.addEventListener("click", () => runBulk("downweight"));
      content.querySelector("#w2-dead-archive")?.addEventListener("click", () => runBulk("archive"));
    } else if (tab === "orphans") {
      const items = this.healthData.orphan_nodes || [];
      if (!items.length) {
        content.innerHTML = `<p class="text-tertiary" style="font-size:12px">图谱中没有孤立节点。</p>`;
        return;
      }
      content.innerHTML = `
        <div class="w2-row" style="margin-bottom:8px">
          <button type="button" class="btn btn-danger btn-sm" id="w2-orphan-clean-all">一键清理全部孤立节点 (${items.length})</button>
        </div>
        ${items
          .map(
            (n) => `<div class="w2-item">
              <div class="w2-item-meta">节点 #${esc(n.id)} · 类型：${esc(n.type)}</div>
              <div class="w2-item-text">${esc(n.label)}</div>
            </div>`
          )
          .join("")}
      `;
      content.querySelector("#w2-orphan-clean-all")?.addEventListener("click", async (e) => {
        e.currentTarget.disabled = true;
        try {
          const res = await this.api.post("graph/cleanup-orphans", {});
          showToast(`已清理 ${res.deleted_count || 0} 个孤立图谱节点`);
          await this.runHealthCheckup();
        } catch (err) {
          showToast(`清理失败：${err.message || err}`, true);
          e.currentTarget.disabled = false;
        }
      });
    }
  }

  /* ================================================================
     5. Memories Page: Quick Filter Presets & Canonical Type Groups
     ================================================================ */
  _mountMemoryQuickFilters() {
    const form = this.doc.getElementById("memory-filters");
    if (!form || this.doc.getElementById("w2-mem-presets")) return;

    const bar = this.doc.createElement("div");
    bar.id = "w2-mem-presets";
    bar.className = "w2-chip-bar";
    bar.innerHTML = `
      <span class="text-tertiary" style="font-size:12px">快捷筛选：</span>
      <button type="button" class="w2-chip" data-w2-preset="lowImportance">低重要度 (≤0.35)</button>
      <button type="button" class="w2-chip" data-w2-preset="staleZeroAccess">长期零召回 (≥30天)</button>
      <button type="button" class="w2-chip" data-w2-source="true">有原始来源</button>
      <button type="button" class="w2-chip" data-w2-source="false">无原始来源</button>
      <select id="w2-mem-origin" class="select input input-sm" style="width:auto" aria-label="记忆来源">
        <option value="">全部来源</option>
        <option value="auto">自动总结 (auto)</option>
        <option value="agent">Agent 主动记忆</option>
        <option value="import">外部导入 (import)</option>
        <option value="resummarize">重新总结</option>
      </select>
      <select id="w2-mem-type-group" class="select input input-sm" style="width:auto" aria-label="归类类型">
        <option value="">类型归类（合并旧枚举）</option>
        <option value="GROUP:FACT">事实类 (FACT + FACTUAL)</option>
        <option value="GROUP:EVENT">事件类 (EVENT + EPISODIC)</option>
        <option value="GROUP:PREFERENCE">偏好观点 (PREFERENCE + OPINION)</option>
        <option value="GROUP:RELATIONAL">人际关系 (RELATIONAL)</option>
        <option value="GROUP:PLANNED">计划待办 (PLANNED)</option>
        <option value="GROUP:GENERAL">通用摘要 (GENERAL)</option>
      </select>
    `;
    form.insertAdjacentElement("afterend", bar);

    const triggerRefresh = () => {
      if (typeof window !== "undefined" && window.lmState?.memory) {
        window.lmState.memory.page = 1;
        window.lmRefreshMemories?.();
      }
    };

    bar.querySelectorAll("[data-w2-preset]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const key = btn.dataset.w2Preset;
        this.memPreset[key] = !this.memPreset[key];
        btn.classList.toggle("active", this.memPreset[key]);
        triggerRefresh();
      });
    });

    bar.querySelectorAll("[data-w2-source]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const val = btn.dataset.w2Source;
        this.memPreset.hasSource = this.memPreset.hasSource === val ? "" : val;
        bar.querySelectorAll("[data-w2-source]").forEach((other) => {
          other.classList.toggle("active", other.dataset.w2Source === this.memPreset.hasSource);
        });
        triggerRefresh();
      });
    });

    const originSel = bar.querySelector("#w2-mem-origin");
    originSel?.addEventListener("change", () => {
      this.memPreset.origin = originSel.value;
      triggerRefresh();
    });

    const groupSel = bar.querySelector("#w2-mem-type-group");
    groupSel?.addEventListener("change", () => {
      this.memPreset.canonicalGroup = groupSel.value;
      triggerRefresh();
    });

    // Clear quick filters when the native reset button is clicked
    this.doc.getElementById("mem-reset-filters")?.addEventListener("click", () => {
      this.memPreset = {
        lowImportance: false,
        staleZeroAccess: false,
        hasSource: "",
        origin: "",
        canonicalGroup: "",
      };
      bar.querySelectorAll(".w2-chip").forEach((c) => c.classList.remove("active"));
      if (originSel) originSel.value = "";
      if (groupSel) groupSel.value = "";
    });
  }

  /* ================================================================
     6. Graph Page: Node Type Filter Chips & Entity Governance Drawer
     ================================================================ */
  _mountGraphGovernanceControls() {
    const hud = this.doc.querySelector("#page-graph .graph-floating-hud");
    if (!hud || this.doc.getElementById("w2-graph-controls")) return;

    const wrap = this.doc.createElement("div");
    wrap.id = "w2-graph-controls";
    wrap.innerHTML = `
      <div class="w2-chip-bar" style="margin:6px 0 2px">
        <span class="text-tertiary" style="font-size:12px">节点筛选：</span>
        <button type="button" class="w2-chip active" data-w2-ntype="all">全部</button>
        <button type="button" class="w2-chip" data-w2-ntype="person">人物</button>
        <button type="button" class="w2-chip" data-w2-ntype="topic">话题</button>
        <button type="button" class="w2-chip" data-w2-ntype="fact">事实</button>
        <button type="button" class="w2-chip" data-w2-ntype="summary">摘要</button>
        <button type="button" class="btn btn-secondary btn-sm" id="w2-graph-gov-toggle" style="margin-left:auto">实体治理</button>
      </div>
      <div id="w2-graph-gov-drawer" class="w2-drawer" hidden>
        <div class="w2-row" style="justify-content:space-between;margin-bottom:6px">
          <strong>图谱实体治理（合并同义实体 / 清理孤立节点）</strong>
          <button type="button" class="btn btn-danger btn-sm" id="w2-graph-clean-orphans">清理孤立节点</button>
        </div>
        <div class="w2-row">
          <input id="w2-merge-target" class="input input-sm" type="number" min="1" placeholder="保留节点 ID" style="width:120px" />
          <input id="w2-merge-sources" class="input input-sm" type="text" placeholder="待合并同义节点 ID（逗号分隔）" style="flex:1;min-width:180px" />
          <button type="button" class="btn btn-primary btn-sm" id="w2-merge-nodes-btn">合并同义节点</button>
        </div>
        <p id="w2-graph-gov-msg" class="text-tertiary" style="font-size:12px;margin-top:6px"></p>
      </div>
    `;
    hud.appendChild(wrap);

    const applyGraphFilter = async () => {
      const types = Array.from(this.graphNodeTypes);
      const session = this.doc.getElementById("graph-session-filter")?.value?.trim() || "";
      const params = {};
      if (session) params.session_id = session;
      if (types.length) params.node_types = types.join(",");
      try {
        const data = await this.api.get("graph/overview", params);
        const msgEl = wrap.querySelector("#w2-graph-gov-msg");
        if (msgEl) {
          const nCount = (data?.nodes || []).length;
          const eCount = (data?.edges || []).length;
          msgEl.textContent = types.length
            ? `已按节点类型 [${types.join(", ")}] 过滤：${nCount} 节点 / ${eCount} 关系`
            : "";
        }
        if (typeof window !== "undefined" && typeof window.Graph2D?.setData === "function" && data) {
          window.Graph2D.setData(data);
        }
      } catch (err) {
        showToast(`节点筛选失败：${err.message || err}`, true);
      }
    };

    wrap.querySelectorAll("[data-w2-ntype]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const t = btn.dataset.w2Ntype;
        if (t === "all") {
          this.graphNodeTypes.clear();
        } else if (this.graphNodeTypes.has(t)) {
          this.graphNodeTypes.delete(t);
        } else {
          this.graphNodeTypes.add(t);
        }
        wrap.querySelectorAll("[data-w2-ntype]").forEach((b) => {
          const bt = b.dataset.w2Ntype;
          b.classList.toggle(
            "active",
            bt === "all" ? this.graphNodeTypes.size === 0 : this.graphNodeTypes.has(bt)
          );
        });
        applyGraphFilter();
      });
    });

    const drawer = wrap.querySelector("#w2-graph-gov-drawer");
    wrap.querySelector("#w2-graph-gov-toggle")?.addEventListener("click", () => {
      if (drawer) drawer.hidden = !drawer.hidden;
    });

    const msgEl = wrap.querySelector("#w2-graph-gov-msg");
    wrap.querySelector("#w2-graph-clean-orphans")?.addEventListener("click", async (e) => {
      e.currentTarget.disabled = true;
      try {
        const res = await this.api.post("graph/cleanup-orphans", {});
        const text = `已清理 ${res.deleted_count || 0} 个孤立节点`;
        if (msgEl) msgEl.textContent = text;
        showToast(text);
        window.lmFetchGraphStats?.();
      } catch (err) {
        if (msgEl) msgEl.textContent = `清理失败：${err.message || err}`;
      } finally {
        e.currentTarget.disabled = false;
      }
    });

    wrap.querySelector("#w2-merge-nodes-btn")?.addEventListener("click", async (e) => {
      const targetId = Number(wrap.querySelector("#w2-merge-target")?.value);
      const rawSrc = String(wrap.querySelector("#w2-merge-sources")?.value || "");
      const sourceIds = rawSrc
        .split(/[,，\s]+/)
        .map((s) => Number(s.trim()))
        .filter((n) => Number.isInteger(n) && n > 0);
      if (!Number.isInteger(targetId) || targetId <= 0 || !sourceIds.length) {
        if (msgEl) msgEl.textContent = "请输入有效的保留节点 ID 与待合并节点 ID。";
        return;
      }
      e.currentTarget.disabled = true;
      try {
        const res = await this.api.post("graph/merge-nodes", {
          target_node_id: targetId,
          source_node_ids: sourceIds,
        });
        const text = `已合并 ${(res.merged_node_ids || []).length} 个同义节点至 #${targetId}（重连 ${res.relinked_edges || 0} 条边）`;
        if (msgEl) msgEl.textContent = text;
        showToast(text);
        window.lmFetchGraphStats?.();
      } catch (err) {
        if (msgEl) msgEl.textContent = `合并失败：${err.message || err}`;
      } finally {
        e.currentTarget.disabled = false;
      }
    });
  }

  /* ================================================================
     7. Wave 3: Event Bus & Cross-Plugin Milestone Keepsake Panel
     ================================================================ */
  _mountMilestonesPanel() {
    const sysBody = this.doc.querySelector("#page-system .page-body");
    if (!sysBody || this.doc.getElementById("w3-milestones-panel")) return;
    const panel = this.doc.createElement("div");
    panel.id = "w3-milestones-panel";
    panel.className = "system-panel mb-6 w2-panel";
    panel.innerHTML = `
      <div class="system-panel-header">
        <div class="system-panel-title">事件总线沉淀 &amp; 3D 小屋里程碑纪念品</div>
        <div class="system-panel-tools">
          <button type="button" class="btn btn-secondary btn-sm" id="w3-ms-refresh">刷新状态</button>
        </div>
      </div>
      <p class="workflow-hint">订阅核心 topic_archived 事件高信噪比沉淀；高重要度共同回忆/纪念日自动生成拍立得并上架 3D 小屋（每群每日上限 1 次）。</p>
      <div id="w3-ms-bus-status" class="w2-item-meta" style="margin-bottom:8px"></div>
      <div class="w2-row" style="margin-bottom:8px">
        <input id="w3-ms-mem-id" class="input input-sm" type="number" min="1" placeholder="输入记忆 ID 生成拍立得纪念品" style="width:220px" />
        <label class="w2-row" style="font-size:12px"><input id="w3-ms-force" type="checkbox" /> <span>忽略每日频控</span></label>
        <button type="button" class="btn btn-primary btn-sm" id="w3-ms-trigger-btn">生成纪念品</button>
      </div>
      <p id="w3-ms-feedback" class="text-tertiary" style="font-size:12px;margin-bottom:6px"></p>
      <div id="w3-ms-list"></div>
    `;
    const healthPanel = this.doc.getElementById("w2-health-panel");
    sysBody.insertBefore(panel, healthPanel || null);
    panel.querySelector("#w3-ms-refresh")?.addEventListener("click", () => this.loadMilestones());
    panel.querySelector("#w3-ms-trigger-btn")?.addEventListener("click", async (e) => {
      const memId = Number(panel.querySelector("#w3-ms-mem-id")?.value);
      const force = Boolean(panel.querySelector("#w3-ms-force")?.checked);
      const fb = panel.querySelector("#w3-ms-feedback");
      if (!Number.isInteger(memId) || memId <= 0) {
        if (fb) fb.textContent = "请输入有效的记忆编号。";
        return;
      }
      e.currentTarget.disabled = true;
      try {
        await this.api.post("milestones/trigger", { memory_id: memId, force });
        if (fb) fb.textContent = `已为记忆 #${memId} 生成里程碑纪念品。`;
        showToast(`已生成记忆 #${memId} 的里程碑纪念品`);
        await this.loadMilestones();
      } catch (err) {
        if (fb) fb.textContent = `生成失败：${err.message || err}`;
      } finally {
        e.currentTarget.disabled = false;
      }
    });
  }

  async loadMilestones() {
    const statusEl = this.doc?.getElementById("w3-ms-bus-status");
    const listEl = this.doc?.getElementById("w3-ms-list");
    if (!statusEl && !listEl) return null;
    try {
      const data = await this.api.get("milestones", { limit: 20 });
      this._renderMilestones(data);
      return data;
    } catch (err) {
      if (listEl) listEl.innerHTML = `<p class="text-tertiary" style="font-size:12px">加载失败：${esc(err.message || err)}</p>`;
      return null;
    }
  }

  _renderMilestones(data) {
    const statusEl = this.doc?.getElementById("w3-ms-bus-status");
    const listEl = this.doc?.getElementById("w3-ms-list");
    const bus = data?.event_bus || {};
    if (statusEl) {
      const subText = bus.subscribed ? "已订阅 topic_archived" : "未连接核心总线（独立降级运行）";
      statusEl.textContent = `Dynamics Event Bus：${subText} · 累计归档话题 ${Number(bus.topic_archived_total || 0)} 条 · 里程碑记录 ${Number(data?.total || 0)} 条`;
    }
    if (!listEl) return;
    const items = Array.isArray(data?.items) ? data.items : [];
    if (!items.length) {
      listEl.innerHTML = `<p class="text-tertiary" style="font-size:12px">暂无已生成的里程碑纪念品记录。</p>`;
      return;
    }
    listEl.innerHTML = items
      .map((item) => {
        const badges = [
          item.image_path ? "已配拍立得" : "纯文本摘要",
          item.keepsake_id ? `小屋纪念品 #${esc(item.keepsake_id)}` : "待同步小屋",
          item.event_emitted ? "已广播总线" : "",
        ]
          .filter(Boolean)
          .join(" · ");
        return `<div class="w2-item">
          <div class="w2-item-meta">记忆 #${esc(item.memory_id)} · ${esc(item.day_key || "")} · ${badges}</div>
          <div class="w2-item-text"><strong>${esc(item.title)}</strong> — ${esc(item.summary)}</div>
          <div class="w2-row" style="margin-top:6px">
            <button type="button" class="btn btn-secondary btn-sm" data-w3-ms-peek="${esc(item.memory_id)}">查看原记忆</button>
            ${item.world_deep_link ? `<span class="w2-item-meta">深链：${esc(item.world_deep_link)}</span>` : ""}
          </div>
        </div>`;
      })
      .join("");
    listEl.querySelectorAll("[data-w3-ms-peek]").forEach((btn) => {
      btn.addEventListener("click", () => {
        window.lmOpenPeekMemory?.({ memory_id: Number(btn.dataset.w3MsPeek) });
      });
    });
  }
}

export function initWave2Governance(apiClient) {
  const controller = new Wave2GovernanceController(apiClient || new ApiClient());
  controller.init();
  if (typeof window !== "undefined") {
    window.lmWave2Governance = controller;
  }
  return controller;
}
