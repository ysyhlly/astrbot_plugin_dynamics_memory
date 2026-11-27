/**
 * System Page - 系统统计页面
 * 负责展示系统概览和统计信息
 */

import { esc, atomLabel, confirmAction } from "./utils.js";

export class SystemPage {
  constructor(state, apiClient) {
    this.state = state;
    this.api = apiClient;
  }

  /**
   * 获取系统概览数据
   */
  async fetch() {
    // 如果有缓存且未过期，直接使用
    if (this.state._systemCache && Date.now() - this.state._systemCache.timestamp < 10000) {
      this.render(this.state._systemCache.data);
      return;
    }

    try {
      const data = await this.api.get("stats");
      this.state._systemCache = { data, timestamp: Date.now() };
      this.state._systemError = "";
      this.updateFeedback();
      this.render(data);
      /* 子面板与统计互不依赖：失败不影响其余内容，也不必在每次重绘时重复请求 */
      await Promise.all([
        this.fetchAndRenderBackups(),
        this.fetchConsolidation(),
        this.fetchEvidenceInsights(),
      ]);
      window.lmRefreshSeriesStatus?.();
    } catch (e) {
      this.state._systemError = e.message || window.t("system.fetchFailed");
      this.updateFeedback();
    }
  }

  /**
   * 渲染页面级请求反馈
   */
  updateFeedback() {
    const feedback = document.getElementById("system-feedback");
    const message = document.getElementById("system-feedback-message");
    const retry = document.getElementById("system-retry");
    const error = this.state._systemError || "";
    if (feedback) feedback.hidden = !error;
    if (message) message.textContent = error;
    if (retry) retry.hidden = !error;
  }

  /**
   * 渲染系统概览
   * @param {Object} data - 统计数据
   */
  render(data) {
    // 更新统计卡片
    this.renderStatCards(data);

    // 更新重要性分布图表
    this.renderImportanceChart(data.importance_distribution || {});

    // 更新 Atom 类型图表 - 后端返回 atom_breakdown
    this.renderAtomChart(data.atom_breakdown || {});

    // 更新会话列表 - 后端返回 recent_sessions
    this.renderSessionList(data.recent_sessions || []);
  }

  /**
   * 渲染统计卡片
   * @param {Object} data - 统计数据
   */
  renderStatCards(data) {
    // 后端返回字段：total_memories, status_breakdown, graph_nodes, atom_count
    const statusBreakdown = data.status_breakdown || {};

    document.getElementById("ss-total").textContent = data.total_memories || 0;
    document.getElementById("ss-active").textContent = statusBreakdown.active || 0;
    document.getElementById("ss-archived").textContent = statusBreakdown.archived || 0;
    document.getElementById("ss-deleted").textContent = statusBreakdown.deleted || 0;
    document.getElementById("ss-nodes").textContent = data.graph_nodes || 0;
    document.getElementById("ss-atoms").textContent = data.atom_count || 0;
  }

  /**
   * 渲染重要性分布图表
   * @param {Object} distribution - 分布数据 {"0-1": N, "1-2": N, ...}
   */
  renderImportanceChart(distribution) {
    const chartEl = document.getElementById("importance-chart");
    if (!chartEl) return;

    // 后端返回的是 {"0-1": 10, "1-2": 20, ...} 格式
    const bins = ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10"];
    const values = bins.map(bin => Number(distribution[bin]) || 0);
    if (!values.some(value => value > 0)) {
      chartEl.innerHTML = '<div class="bar-chart-empty">' + esc(window.t("common.noData")) + '</div>';
      return;
    }
    const maxValue = Math.max(...values, 1);

    let html = '';
    bins.forEach((bin, idx) => {
      const value = values[idx];
      const percentage = ((value / maxValue) * 100).toFixed(0);
      html += '<div class="bar-row">';
      html += '<span class="bar-row-label">' + esc(bin) + '</span>';
      html += '<div class="bar-row-track">';
      html += '<div class="bar-row-fill" style="width:' + esc(percentage) + '%"></div>';
      html += '</div>';
      html += '<span class="bar-row-value">' + esc(value) + '</span>';
      html += '</div>';
    });

    chartEl.innerHTML = html;
  }

  /**
   * 渲染 Atom 类型图表
   * @param {Object} types - 类型数据 {episodic: N, factual: N, ...}
   */
  renderAtomChart(types) {
    const chartEl = document.getElementById("atom-chart");
    if (!chartEl) return;

    // 后端返回 atom_breakdown
    const atomBreakdown = types || {};
    /* SQL 不保证顺序，按数量降序读起来才有主次 */
    const entries = Object.entries(atomBreakdown)
      .map(([type, count]) => [type, Number(count) || 0])
      .sort((a, b) => b[1] - a[1]);

    if (entries.length === 0) {
      chartEl.innerHTML = '<div class="bar-chart-empty">' + esc(window.t("system.noAtoms")) + '</div>';
      return;
    }

    const maxValue = Math.max(...entries.map(([_, count]) => count), 1);

    let html = '';
    entries.forEach(([type, count]) => {
      const percentage = ((count / maxValue) * 100).toFixed(0);
      html += '<div class="bar-row">';
      html += '<span class="bar-row-label">' + esc(atomLabel(type)) + '</span>';
      html += '<div class="bar-row-track">';
      html += '<div class="bar-row-fill" style="width:' + esc(percentage) + '%"></div>';
      html += '</div>';
      html += '<span class="bar-row-value">' + esc(count) + '</span>';
      html += '</div>';
    });

    chartEl.innerHTML = html;
  }

  /**
   * 渲染活跃会话列表
   * @param {Array} sessions - 会话列表
   */
  renderSessionList(sessions) {
    const listEl = document.getElementById("session-list");
    if (!listEl) return;

    if (!sessions || sessions.length === 0) {
      listEl.innerHTML = '<div class="table-empty">' + esc(window.t("system.noSessions")) + '</div>';
      return;
    }

    let html = '';
    sessions.forEach(session => {
      const sessionId = session.session_id || "--";
      const memoryCount = Number(session.message_count) || 0;

      html += '<div class="session-item">';
      html += '<div class="session-item-header">';
      html += '<span class="session-item-id">' + esc(String(sessionId)) + '</span>';
      html += '<span class="session-item-count">' + esc(memoryCount) + ' ' + esc(window.t("system.memories")) + '</span>';
      html += '</div>';
      html += '</div>';
    });

    listEl.innerHTML = html;
  }

  /**
   * 渲染备份列表
   * @param {Array} backups - 备份列表
   */
  renderBackupList(backups) {
    const listEl = document.getElementById("backup-list");
    if (!listEl) return;

    if (!backups || backups.length === 0) {
      listEl.innerHTML = '<div class="table-empty">' + esc(window.t("system.noBackups")) + '</div>';
      return;
    }

    let html = '';
    /* 后端按目录名字典序返回（v2.9 会排在 v2.10 前面），按 unix 时间重排；
       file_count 含目录内全部文件（多一个 backup_info.json），取 files_copied 更贴近可恢复内容。 */
    [...backups]
      .sort((a, b) => (b.backup_unix_time || 0) - (a.backup_unix_time || 0))
      .forEach(backup => {
        const name = backup.name || backup.directory || "Unknown";
        const timestamp = backup.backup_unix_time
          ? new Date(backup.backup_unix_time * 1000).toLocaleString()
          : (backup.backup_timestamp || "--");
        const fileCount = Number(backup.files_copied) || 0;

        html += '<div class="backup-item">';
        html += '<div class="backup-item-name">' + esc(name) + '</div>';
        html += '<div class="backup-item-meta">';
        html += '<span>' + esc(timestamp) + '</span>';
        html += '<span>' + esc(fileCount) + ' ' + esc(window.t("system.files")) + '</span>';
        html += '</div>';
        html += '</div>';
      });

    listEl.innerHTML = html;
  }

  /**
   * 获取并渲染备份列表
   */
  async fetchAndRenderBackups() {
    try {
      const data = await this.api.get("backups");
      this.renderBackupList(data.backups || []);
    } catch (e) {
      // 备份功能可能不可用，静默失败
      const listEl = document.getElementById("backup-list");
      if (listEl) {
        listEl.innerHTML = '<div class="backup-empty">' + window.t("common.unavailable") + '</div>';
      }
    }
  }

  /**
   * 获取并渲染记忆整合状态
   */
  async fetchConsolidation() {
    try {
      const data = await this.api.get("consolidation/status");
      this.state._consolidationEnabled = Boolean(data.config && data.config.enabled);
      this.renderConsolidation(data);
    } catch (e) {
      const metaEl = document.getElementById("consolidation-meta");
      if (metaEl) metaEl.innerHTML = '<div class="session-empty">' + window.t("common.unavailable") + '</div>';
    }
  }

  /**
   * 渲染记忆整合面板
   * @param {Object} data - {config, consolidated_count, archived_count}
   */
  renderConsolidation(data) {
    const config = data.config || {};
    const metaEl = document.getElementById("consolidation-meta");
    if (metaEl) {
      const enabled = config.enabled ? window.t("common.enabled") : window.t("common.disabled");
      const trigger = config.trigger === "reflection"
        ? window.t("system.consTriggerReflection")
        : window.t("system.consTriggerDaily");
      const granularity = config.granularity === "semantic"
        ? window.t("system.consGranularitySemantic")
        : window.t("system.consGranularitySession");
      const keep = config.keep_original === "delete"
        ? window.t("system.consKeepDelete")
        : window.t("system.consKeepArchive");
      metaEl.innerHTML =
        '<div class="session-item-meta">' +
        window.t("system.consStatus") + ': ' + esc(enabled) +
        ' · ' + window.t("system.consTrigger") + ': ' + esc(trigger) +
        ' · ' + window.t("system.consGranularity") + ': ' + esc(granularity) +
        ' · ' + window.t("system.consKeep") + ': ' + esc(keep) +
        '</div>';
    }

    const consolidatedEl = document.getElementById("cons-consolidated");
    if (consolidatedEl) consolidatedEl.textContent = data.consolidated_count || 0;
    const archivedEl = document.getElementById("cons-archived");
    if (archivedEl) archivedEl.textContent = data.archived_count || 0;

    /* 未启用时点击只会返回 skipped，控件状态应直接说明这一点 */
    const runBtn = document.getElementById("cons-run-btn");
    if (runBtn) runBtn.disabled = !config.enabled;
  }

  /**
   * 手动触发一轮记忆整合
   */
  async runConsolidation() {
    const button = document.getElementById("cons-run-btn");
    const resultEl = document.getElementById("cons-run-result");
    /* 整合会按配置归档或删除原记忆，启用时先二次确认；宿主 iframe 不能用 window.confirm。 */
    if (this.state._consolidationEnabled) {
      const confirmed = await confirmAction({
        title: window.t("system.consConfirmTitle"),
        message: window.t("system.consConfirmMessage"),
        confirmLabel: window.t("system.consolidationRun"),
        destructive: false,
      });
      if (!confirmed) return;
    }
    if (button) button.disabled = true;
    if (resultEl) resultEl.textContent = window.t("system.consRunning");
    try {
      const result = await this.api.post("consolidation/run", {}, { retries: 0 });
      if (result.error) {
        if (resultEl) resultEl.textContent = result.error;
      } else if (result.skipped) {
        if (resultEl) resultEl.textContent = window.t("system.consSkipped");
      } else if (!result.candidates) {
        if (resultEl) resultEl.textContent = window.t("system.consNoCandidates");
      } else {
        if (resultEl) {
          resultEl.textContent = window.t(
            "system.consResult",
            result.groups || 0,
            result.merged || 0,
            result.archived || 0,
            result.deleted || 0,
            result.failed || 0
          );
        }
      }
      await this.fetchConsolidation();
    } catch (e) {
      if (resultEl) resultEl.textContent = e.message || window.t("misc.requestFailed");
    } finally {
      if (button) button.disabled = !this.state._consolidationEnabled;
    }
  }

  /**
   * 执行存储自愈修复（不调用模型或嵌入）
   */
  async runStorageRepair() {
    const btn = document.getElementById("repair-run-btn");
    const resultEl = document.getElementById("repair-result");
    const limitEl = document.getElementById("repair-limit");
    const afterEl = document.getElementById("repair-after");
    const requeueEl = document.getElementById("repair-requeue-sqlite");

    const rawLimit = Number.parseInt(limitEl?.value ?? "50", 10);
    const limit = Number.isFinite(rawLimit) ? Math.max(1, Math.min(rawLimit, 200)) : 50;
    const rawAfter = Number.parseInt(afterEl?.value ?? "0", 10);
    const after = Number.isFinite(rawAfter) ? Math.max(0, rawAfter) : 0;
    const requeueSqlite = Boolean(requeueEl?.checked);

    if (btn) btn.disabled = true;
    if (resultEl) resultEl.textContent = window.t("flow.working");
    try {
      const res = await this.api.post(
        "storage/repair",
        { limit, after, requeue_sqlite: requeueSqlite },
        { retries: 0 }
      );
      const scanned = Number(res.scanned ?? res.checked ?? 0) || 0;
      const repaired = Number(res.repaired ?? res.updated ?? 0) || 0;
      const requeued = Number(res.requeued_sqlite_operations ?? 0) || 0;
      const nextCursor = Number(res.next_cursor ?? res.last_id ?? after) || 0;
      if (afterEl && res.has_more && nextCursor > after) {
        afterEl.value = String(nextCursor);
      }
      if (resultEl) {
        resultEl.textContent = window.t(
          "system.repairResult",
          scanned,
          repaired,
          requeued,
          nextCursor
        );
      }
    } catch (e) {
      if (resultEl) resultEl.textContent = e.message || window.t("misc.requestFailed");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  /**
   * 拉取派生见解列表（仅 ID 与来源元数据，不含群友原文）
   */
  async fetchEvidenceInsights() {
    const statusEl = document.getElementById("evidence-status");
    const listEl = document.getElementById("evidence-list");
    if (!listEl && !statusEl) return;
    try {
      const data = await this.api.get("evidence/insights");
      this.renderEvidenceInsights(data);
    } catch (e) {
      if (statusEl) statusEl.textContent = e.message || window.t("common.unavailable");
      if (listEl) {
        listEl.innerHTML =
          '<div class="table-empty">' + esc(window.t("common.unavailable")) + "</div>";
      }
    }
  }

  /**
   * 渲染派生见解列表
   * @param {Object} data - {available, items}
   */
  renderEvidenceInsights(data) {
    const statusEl = document.getElementById("evidence-status");
    const listEl = document.getElementById("evidence-list");
    if (!listEl) return;

    if (!data || data.available === false) {
      if (statusEl) statusEl.textContent = window.t("common.unavailable");
      listEl.innerHTML =
        '<div class="table-empty">' + esc(window.t("common.unavailable")) + "</div>";
      return;
    }

    const items = Array.isArray(data.items) ? data.items : [];
    if (statusEl) statusEl.textContent = window.t("system.evidenceCount", items.length);
    if (items.length === 0) {
      listEl.innerHTML =
        '<div class="table-empty">' + esc(window.t("common.noData")) + "</div>";
      return;
    }

    listEl.innerHTML = items.map((item) => {
      const id = esc(String(item.atom_id || ""));
      const rev = esc(Number(item.correction_revision ?? 0) || 0);
      return `<div class="evidence-item"><div class="evidence-item-info"><div class="evidence-item-id">${id}</div><div class="evidence-item-meta"><span>${esc(item.kind || "insight")}</span><span>${esc(item.status || "")}</span><span>rev ${rev}</span><span>sources ${esc(Number(item.source_count ?? 0) || 0)}</span></div></div><button type="button" class="btn btn-secondary btn-sm evidence-select-btn" data-atom-id="${id}" data-revision="${rev}">${esc(window.t("detail.editBtn"))}</button></div>`;
    }).join("");
  }

  /**
   * 选中见解填入下方纠错表单
   */
  selectEvidenceItem(atomId, revision) {
    const targetEl = document.getElementById("evidence-target-id");
    const revEl = document.getElementById("evidence-revision");
    if (targetEl) targetEl.value = String(atomId || "");
    if (revEl) revEl.value = String(Number.isFinite(Number(revision)) ? Number(revision) : 0);
  }

  /**
   * 切换撤回/替代操作时显隐替代内容输入框
   */
  syncEvidenceActionFields() {
    const actionEl = document.getElementById("evidence-action");
    const wrapEl = document.getElementById("evidence-replacement-wrap");
    if (actionEl && wrapEl) wrapEl.hidden = actionEl.value !== "supersede";
  }

  /**
   * 提交见解纠错（withdraw / supersede）
   */
  async submitEvidenceCorrection() {
    const targetEl = document.getElementById("evidence-target-id");
    const revEl = document.getElementById("evidence-revision");
    const actionEl = document.getElementById("evidence-action");
    const replacementEl = document.getElementById("evidence-replacement");
    const reasonEl = document.getElementById("evidence-reason");
    const btn = document.getElementById("evidence-submit-btn");
    const resultEl = document.getElementById("evidence-correct-result");

    const targetId = String(targetEl?.value || "").trim();
    const action = String(actionEl?.value || "withdraw").trim().toLowerCase();
    const expectedRevision = Number.parseInt(revEl?.value ?? "0", 10);
    const replacement = action === "supersede" ? String(replacementEl?.value || "").trim() : "";
    const reason = String(reasonEl?.value || "").trim();

    if (!targetId || (action === "supersede" && !replacement)) {
      if (resultEl) resultEl.textContent = window.t("batchEdit.valueRequired");
      return;
    }

    if (btn) btn.disabled = true;
    if (resultEl) resultEl.textContent = window.t("flow.working");
    try {
      const res = await this.api.post(
        "evidence/correct",
        {
          target_id: targetId,
          expected_revision: Number.isFinite(expectedRevision) ? expectedRevision : 0,
          action,
          replacement,
          reason,
        },
        { retries: 0 }
      );
      if (resultEl) {
        resultEl.textContent = window.t(
          "system.evidenceSuccess",
          res.action || action,
          res.correction_revision ?? expectedRevision
        );
      }
      if (replacementEl) replacementEl.value = "";
      if (reasonEl) reasonEl.value = "";
      await this.fetchEvidenceInsights();
    } catch (e) {
      if (resultEl) resultEl.textContent = e.message || window.t("misc.requestFailed");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  /**
   * 初始化事件监听
   */
  initEventListeners() {
    const runBtn = document.getElementById("cons-run-btn");
    if (runBtn) runBtn.addEventListener("click", () => this.runConsolidation());
    const retry = document.getElementById("system-retry");
    if (retry) retry.addEventListener("click", () => this.fetch());
    const repairBtn = document.getElementById("repair-run-btn");
    if (repairBtn) repairBtn.addEventListener("click", () => this.runStorageRepair());
    const evidenceRefreshBtn = document.getElementById("evidence-refresh-btn");
    if (evidenceRefreshBtn) {
      evidenceRefreshBtn.addEventListener("click", () => this.fetchEvidenceInsights());
    }
    const evidenceAction = document.getElementById("evidence-action");
    if (evidenceAction) {
      evidenceAction.addEventListener("change", () => this.syncEvidenceActionFields());
      this.syncEvidenceActionFields();
    }
    const evidenceSubmitBtn = document.getElementById("evidence-submit-btn");
    if (evidenceSubmitBtn) {
      evidenceSubmitBtn.addEventListener("click", () => this.submitEvidenceCorrection());
    }
    const evidenceList = document.getElementById("evidence-list");
    if (evidenceList) {
      evidenceList.addEventListener("click", (evt) => {
        const btn = evt.target?.closest?.(".evidence-select-btn");
        if (!btn) return;
        this.selectEvidenceItem(btn.dataset.atomId || "", btn.dataset.revision || "0");
      });
    }
  }

}
