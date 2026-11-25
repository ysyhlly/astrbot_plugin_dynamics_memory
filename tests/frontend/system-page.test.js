import assert from "node:assert/strict";
import test from "node:test";

import { SystemPage } from "../../pages/dashboard/modules/system-page.js";

test("consolidation asks for confirmation and skips the request when cancelled", async () => {
  const elements = {
    "cons-run-btn": { disabled: false },
    "cons-run-result": { textContent: "" },
  };
  const posts = [];
  globalThis.window = { t: (key) => key };
  globalThis.document = { getElementById: (id) => elements[id] || null };
  try {
    const state = { _consolidationEnabled: true };
    const page = new SystemPage(state, {
      post: async (...args) => {
        posts.push(args);
        return {};
      },
    });

    await page.runConsolidation();

    assert.equal(posts.length, 0, "取消确认后不得触发整合");
    assert.equal(elements["cons-run-btn"].disabled, false);
    assert.equal(elements["cons-run-result"].textContent, "");
  } finally {
    delete globalThis.document;
    delete globalThis.window;
  }
});

test("renderAtomChart escapes unknown atom types and resists prototype keys", () => {
  const chart = { innerHTML: "" };
  globalThis.window = { t: (key) => key };
  globalThis.document = { getElementById: (id) => (id === "atom-chart" ? chart : null) };
  try {
    const page = new SystemPage({}, {});
    page.renderAtomChart({
      '<img src=x onerror=alert(1)>': 5,
      constructor: 2,
    });

    assert.doesNotMatch(chart.innerHTML, /<img/i, "未知 atom 类型不得作为原始 HTML 注入");
    assert.match(chart.innerHTML, /&lt;img src=x onerror=alert\(1\)&gt;/);
    assert.doesNotMatch(chart.innerHTML, /function Object/, "原型键不得命中 Object.prototype");
    assert.match(chart.innerHTML, />constructor</);
  } finally {
    delete globalThis.document;
    delete globalThis.window;
  }
});
