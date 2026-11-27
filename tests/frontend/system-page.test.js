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

test("runStorageRepair posts bounded repair payload and advances cursor when has_more", async () => {
  const elements = {
    "repair-run-btn": { disabled: false },
    "repair-result": { textContent: "" },
    "repair-limit": { value: "80" },
    "repair-after": { value: "10" },
    "repair-requeue-sqlite": { checked: true },
  };
  const posts = [];
  globalThis.window = { t: (key, ...args) => `${key}:${args.join(",")}` };
  globalThis.document = { getElementById: (id) => elements[id] || null };
  try {
    const page = new SystemPage({}, {
      post: async (path, body, opts) => {
        posts.push({ path, body, opts });
        return { scanned: 80, repaired: 5, requeued_sqlite_operations: 2, next_cursor: 90, has_more: true };
      },
    });

    await page.runStorageRepair();

    assert.equal(posts.length, 1);
    assert.equal(posts[0].path, "storage/repair");
    assert.deepEqual(posts[0].body, { limit: 80, after: 10, requeue_sqlite: true });
    assert.equal(elements["repair-after"].value, "90");
    assert.equal(elements["repair-run-btn"].disabled, false);
    assert.match(elements["repair-result"].textContent, /80,5,2,90/);
  } finally {
    delete globalThis.document;
    delete globalThis.window;
  }
});

test("renderEvidenceInsights escapes fields and submitEvidenceCorrection posts and refreshes", async () => {
  const elements = {
    "evidence-status": { textContent: "" },
    "evidence-list": { innerHTML: "" },
    "evidence-target-id": { value: "" },
    "evidence-revision": { value: "0" },
    "evidence-action": { value: "supersede" },
    "evidence-replacement": { value: "Updated fact" },
    "evidence-reason": { value: "Admin verified" },
    "evidence-submit-btn": { disabled: false },
    "evidence-correct-result": { textContent: "" },
  };
  const posts = [];
  let getCalls = 0;
  globalThis.window = { t: (key, ...args) => `${key}:${args.join(",")}` };
  globalThis.document = { getElementById: (id) => elements[id] || null };
  try {
    const page = new SystemPage({}, {
      get: async (path) => {
        getCalls += 1;
        assert.equal(path, "evidence/insights");
        return {
          available: true,
          items: [
            {
              atom_id: 'atom-<script>1</script>',
              kind: "preference",
              status: "ready",
              correction_revision: 3,
              source_count: 2,
            },
          ],
        };
      },
      post: async (path, body) => {
        posts.push({ path, body });
        return { ok: true, action: body.action, correction_revision: 4 };
      },
    });

    await page.fetchEvidenceInsights();
    assert.equal(getCalls, 1);
    assert.doesNotMatch(elements["evidence-list"].innerHTML, /<script>/i);
    assert.match(elements["evidence-list"].innerHTML, /&lt;script&gt;/);

    page.selectEvidenceItem("atom-1", 3);
    assert.equal(elements["evidence-target-id"].value, "atom-1");
    assert.equal(elements["evidence-revision"].value, "3");

    await page.submitEvidenceCorrection();
    assert.equal(posts.length, 1);
    assert.equal(posts[0].path, "evidence/correct");
    assert.deepEqual(posts[0].body, {
      target_id: "atom-1",
      expected_revision: 3,
      action: "supersede",
      replacement: "Updated fact",
      reason: "Admin verified",
    });
    assert.equal(getCalls, 2, "提交纠错成功后应自动刷新列表");
    assert.equal(elements["evidence-replacement"].value, "");
    assert.equal(elements["evidence-reason"].value, "");
  } finally {
    delete globalThis.document;
    delete globalThis.window;
  }
});
