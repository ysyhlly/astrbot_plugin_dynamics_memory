import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { MemoryPage } from "../../pages/dashboard/modules/memory-page.js";

const here = dirname(fileURLToPath(import.meta.url));
const dashboard = join(here, "../../pages/dashboard");
const read = (name) => readFileSync(join(dashboard, name), "utf-8");

test("redundant top series-panel drawer is removed and moved into system overview", () => {
  const html = read("index.html");
  assert.doesNotMatch(html, /id="series-panel"/);
  assert.doesNotMatch(html, /dm-top-utility/);
  assert.match(html, /id="dynamics-series-panel"/);

  const systemStart = html.indexOf('id="page-system"');
  const promptsStart = html.indexOf('id="page-prompts"');
  assert.ok(systemStart > 0 && promptsStart > systemStart);
  const systemSection = html.slice(systemStart, promptsStart);
  assert.match(systemSection, /id="system-series-panel"/);
  assert.match(systemSection, /id="series-refresh"/);
  assert.match(systemSection, /id="series-status"/);
  assert.match(systemSection, /id="series-conflicts"/);
});

test("core 4 tabs remain top-level while prompts and settings are grouped under nav-more", () => {
  const html = read("index.html");
  const navMatch = html.match(/<nav class="sidebar-nav series-tabs"[\s\S]*?<\/nav>/);
  assert.ok(navMatch, "series-tabs nav should exist");
  const nav = navMatch[0];
  const moreIdx = nav.indexOf('id="nav-more"');
  assert.ok(moreIdx > 0, "nav-more dropdown should exist");

  const corePart = nav.slice(0, moreIdx);
  const morePart = nav.slice(moreIdx);
  for (const page of ["graph", "memory", "recall", "system"]) {
    assert.match(corePart, new RegExp(`data-page="${page}"`));
  }
  for (const page of ["prompts", "settings"]) {
    assert.match(morePart, new RegExp(`data-page="${page}"`));
    assert.doesNotMatch(corePart, new RegExp(`data-page="${page}"`));
  }
  assert.match(morePart, /role="menu"/);
  assert.match(morePart, /role="menuitem"/);
});

test("memory rows render provenance tone and confidence badge cleanly", (t) => {
  globalThis.window = { t: (key, ...args) => [key, ...args].join(" "), requestAnimationFrame() {} };
  let html = "";
  const tbody = { style: {}, set innerHTML(v) { html = v; }, get innerHTML() { return html; } };
  const scroll = { scrollTop: 0, clientHeight: 400, addEventListener() {} };
  globalThis.document = {
    getElementById: (id) => (id === "memories-body" ? tbody : id === "memories-scroll" ? scroll : null),
  };
  t.after(() => { delete globalThis.window; delete globalThis.document; });

  const page = new MemoryPage(
    {
      memory: {
        items: [
          { memory_id: 1, summary: "verified fact", importance: 8.5, confidence: 0.92, status: "active", updated_at: "now", origin: "agent_memorize_tool", summary_quality: "normal" },
          { memory_id: 2, summary: "imported note", importance: 4.0, status: "active", updated_at: "now", origin: "memory_import", summary_quality: "normal" },
        ],
        total: 2, page: 1, pageSize: 20, selectedIds: new Set(),
      },
    },
    {},
    {},
  );
  page.updateSelectionControls = () => {};
  page.renderVirtual();

  assert.match(html, /origin-tag is-verified/);
  assert.match(html, /origin-tag is-import/);
  assert.match(html, /conf-tag is-conf">92%</);
  assert.match(html, /conf-tag is-medium"/);
});

test("app wires dynamics-cmdk-deeplink handler for same-plugin Ctrl+K results", () => {
  const app = read("app.js");
  assert.match(app, /addEventListener\("dynamics-cmdk-deeplink"/);
  assert.match(app, /peekPanel\.renderMemory/);
  assert.match(app, /graph-query-input/);
});

test("graph-shared hexToRgba parses rgba/rgb/hex tokens without producing NaN", async () => {
  const vm = await import("node:vm");
  const ctx = {
    document: {
      documentElement: {
        getAttribute: (name) => (name === "data-ui-theme" ? "night" : "light"),
      },
    },
    getComputedStyle: () => ({ getPropertyValue: () => "rgba(36, 66, 51, 0.24)" }),
  };
  ctx.window = ctx;
  ctx.self = ctx;
  vm.runInNewContext(read("graph-shared.js"), ctx);
  const { hexToRgba, isDark } = ctx.GraphShared;

  assert.equal(isDark(), true);
  assert.equal(hexToRgba("#246246", 0.5), "rgba(36,98,70,0.5)");
  assert.equal(hexToRgba("#abc", 0.5), "rgba(170,187,204,0.5)");
  assert.equal(hexToRgba("rgba(36, 66, 51, 0.24)", 0.5), "rgba(36,66,51,0.12)");
  assert.equal(hexToRgba("rgb(36 66 51 / 50%)", 0.5), "rgba(36,66,51,0.25)");
  assert.equal(hexToRgba("not-a-color", 0.2), "rgba(128,128,128,0.2)");
  assert.doesNotMatch(hexToRgba("rgba(36, 66, 51, 0.24)", 0.05), /NaN/);
});

test("graph renderer community ellipses draw translucent colors under v2 rgba border tokens", async () => {
  const vm = await import("node:vm");
  let currentTheme = "day";
  const styles = [];
  const fakeCtx = {
    fillStyle: "#000000",
    strokeStyle: "#000000",
    lineWidth: 1,
    save() {},
    restore() {},
    setLineDash() {},
    beginPath() {},
    ellipse() {},
    fill() { styles.push({ op: "fill", style: this.fillStyle }); },
    stroke() { styles.push({ op: "stroke", style: this.strokeStyle }); },
  };
  const fakeCanvas = { getContext: () => fakeCtx };
  const ctx = {
    document: {
      documentElement: {
        getAttribute: (name) => (name === "data-ui-theme" ? currentTheme : "light"),
      },
    },
    getComputedStyle: () => ({
      getPropertyValue: (name) => {
        if (name === "--border-normal") {
          return currentTheme === "night"
            ? "rgba(176, 214, 188, 0.28)"
            : "rgba(36, 66, 51, 0.24)";
        }
        return "";
      },
    }),
  };
  ctx.window = ctx;
  ctx.self = ctx;
  vm.runInNewContext(read("graph-shared.js"), ctx);
  vm.runInNewContext(read("graph-renderer.js"), ctx);

  const renderer = new ctx.GraphRenderer(fakeCanvas);
  renderer.width = 800;
  renderer.height = 600;
  const nodes = [
    { id: 1, x: -40, y: 0, community: 0 },
    { id: 2, x: -20, y: 10, community: 0 },
    { id: 3, x: 40, y: 0, community: 1 },
    { id: 4, x: 60, y: 10, community: 1 },
  ];
  const layout = { _done: true };

  renderer._drawCommunities(nodes, layout, 1, false);
  assert.ok(styles.length >= 4);
  for (const s of styles) {
    assert.doesNotMatch(s.style, /NaN/);
    assert.notEqual(s.style, "#000000");
    assert.match(s.style, /^rgba\(36,66,51,/);
  }

  styles.length = 0;
  currentTheme = "night";
  renderer._drawCommunities(nodes, layout, 1, true);
  assert.ok(styles.length >= 4);
  for (const s of styles) {
    assert.match(s.style, /^rgba\(176,214,188,/);
  }
});
