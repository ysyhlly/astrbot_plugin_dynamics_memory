import test from "node:test";
import assert from "node:assert/strict";

import {
  formatSettingsSaveMessage,
  Wave2GovernanceController,
} from "../../pages/dashboard/modules/wave2-governance.js";

function makeElement(tag = "div") {
  const attrs = new Map();
  const listeners = new Map();
  const classes = new Set();
  const el = {
    tagName: tag.toUpperCase(),
    id: "",
    className: "",
    innerHTML: "",
    textContent: "",
    value: "",
    hidden: false,
    disabled: false,
    dataset: {},
    children: [],
    classList: {
      add: (...cs) => cs.forEach((c) => classes.add(c)),
      remove: (...cs) => cs.forEach((c) => classes.delete(c)),
      toggle: (c, force) => (force === undefined ? (classes.has(c) ? classes.delete(c) : classes.add(c)) : force ? classes.add(c) : classes.delete(c)),
      contains: (c) => classes.has(c),
    },
    appendChild(child) {
      this.children.push(child);
      return child;
    },
    insertBefore(child) {
      this.children.unshift(child);
      return child;
    },
    insertAdjacentElement(_pos, child) {
      this.children.push(child);
      return child;
    },
    setAttribute: (k, v) => attrs.set(k, v),
    getAttribute: (k) => attrs.get(k),
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(fn);
    },
    querySelector: () => null,
    querySelectorAll: () => [],
  };
  return el;
}

test("formatSettingsSaveMessage distinguishes hot-reloaded vs reload-required fields", () => {
  const hotOnly = formatSettingsSaveMessage({
    hot_applied: ["recall_engine.top_k", "graph_memory.graph_route_timeout_seconds"],
    reload_required: [],
  });
  assert.match(hotOnly, /2 项配置已立即热生效/);
  assert.match(hotOnly, /无需重载插件/);

  const reloadOnly = formatSettingsSaveMessage({
    hot_applied: [],
    reload_required: ["graph_memory.enabled"],
  });
  assert.match(reloadOnly, /1 项配置需重载插件后生效/);
  assert.match(reloadOnly, /graph_memory\.enabled/);

  const mixed = formatSettingsSaveMessage({
    hot_applied: ["recall_engine.top_k"],
    reload_required: ["graph_memory.enabled"],
  });
  assert.match(mixed, /1 项已立即热生效/);
  assert.match(mixed, /1 项需重载插件后生效/);
});

test("Wave2GovernanceController builds quick memory filter presets cleanly", () => {
  const ctrl = new Wave2GovernanceController(null, null);
  assert.deepEqual(ctrl.buildMemoryFilterParams(), {});

  ctrl.memPreset.lowImportance = true;
  ctrl.memPreset.staleZeroAccess = true;
  ctrl.memPreset.hasSource = "true";
  ctrl.memPreset.origin = "auto";
  ctrl.memPreset.canonicalGroup = "GROUP:FACT";

  assert.deepEqual(ctrl.buildMemoryFilterParams(), {
    importance_max: "0.35",
    zero_access: "true",
    stale_days: "30",
    has_source: "true",
    memory_origin: "auto",
    type: "GROUP:FACT",
  });
});

test("Wave2GovernanceController escapes XSS in correction queue and updates system badge", async () => {
  const elements = new Map();
  const badge = makeElement("span");
  badge.id = "w2-system-badge";
  const corrBadge = makeElement("span");
  corrBadge.id = "w2-corr-badge";
  const corrList = makeElement("div");
  corrList.id = "w2-corr-list";
  const batchBar = makeElement("div");
  batchBar.id = "w2-conflict-batch";

  elements.set("w2-system-badge", badge);
  elements.set("w2-corr-badge", corrBadge);
  elements.set("w2-corr-list", corrList);
  elements.set("w2-conflict-batch", batchBar);

  const doc = {
    head: makeElement("head"),
    getElementById: (id) => elements.get(id) || null,
    querySelector: () => null,
    createElement: (tag) => makeElement(tag),
  };

  const api = {
    async get(path) {
      if (path === "series/conflicts") {
        return {
          pending_count: 2,
          items: [
            { id: "c1", revision: 1, status: "pending", event_id: "ev<1>", session_id: "g1" },
            { id: "c2", revision: 1, status: "pending", event_id: "ev2", session_id: "g1" },
          ],
        };
      }
      if (path === "corrections") {
        return {
          pending_count: 1,
          items: [
            {
              id: 7,
              memory_id: 42,
              requester_name: "<img src=x onerror=alert(1)>",
              original_preview: "<b>old</b>",
              reason: "<script>evil()</script>",
              proposed_text: "safe text",
              status: "pending",
            },
          ],
        };
      }
      return {};
    },
  };

  const ctrl = new Wave2GovernanceController(api, doc);
  const counts = await ctrl.refreshBadges();
  assert.equal(counts.pendingConflicts, 2);
  assert.equal(counts.pendingCorrections, 1);
  assert.equal(badge.textContent, "3");
  assert.equal(badge.hidden, false);

  // Verify strict HTML escaping in rendered correction card & conflict batch bar
  assert.ok(!corrList.innerHTML.includes("<script>"));
  assert.ok(corrList.innerHTML.includes("&lt;script&gt;evil()&lt;/script&gt;"));
  assert.ok(!corrList.innerHTML.includes("<img src=x"));
  assert.ok(batchBar.innerHTML.includes("ev&lt;1&gt;"));
});

test("Wave2GovernanceController renders Wave 3 event bus status & milestone keepsakes safely", async () => {
  const elements = new Map();
  const statusEl = makeElement("div");
  statusEl.id = "w3-ms-bus-status";
  const listEl = makeElement("div");
  listEl.id = "w3-ms-list";
  elements.set("w3-ms-bus-status", statusEl);
  elements.set("w3-ms-list", listEl);

  const doc = {
    head: makeElement("head"),
    getElementById: (id) => elements.get(id) || null,
    querySelector: () => null,
    createElement: (tag) => makeElement(tag),
  };

  const api = {
    async get(path) {
      if (path === "milestones") {
        return {
          event_bus: { subscribed: true, topic_archived_total: 5 },
          total: 1,
          items: [
            {
              id: 1,
              memory_id: 42,
              day_key: "2026-11-27",
              title: "一周年<script>x</script>",
              summary: "共同回忆摘要",
              image_path: "/data/p.png",
              keepsake_id: "ks_99",
              world_deep_link: "astrbot_plugin_dynamics_world?page=cabin&keepsake=ks_99",
              event_emitted: true,
            },
          ],
        };
      }
      return {};
    },
  };

  const ctrl = new Wave2GovernanceController(api, doc);
  await ctrl.loadMilestones();
  assert.match(statusEl.textContent, /已订阅 topic_archived/);
  assert.match(statusEl.textContent, /累计归档话题 5 条/);
  assert.ok(!listEl.innerHTML.includes("<script>"));
  assert.ok(listEl.innerHTML.includes("一周年&lt;script&gt;x&lt;/script&gt;"));
  assert.ok(listEl.innerHTML.includes("小屋纪念品 #ks_99"));
});
