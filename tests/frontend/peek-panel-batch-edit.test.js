import assert from "node:assert/strict";
import test from "node:test";

import { PeekPanel } from "../../pages/dashboard/modules/peek-panel.js";

test("showBatchEditDialog renders without ReferenceError and escapes title", async () => {
  const peekBody = { innerHTML: "<p>prev</p>" };
  const listeners = {};
  const elements = {
    "peek-body": peekBody,
    "batch-edit-field": {
      value: "importance",
      addEventListener(type, fn) {
        listeners[`field:${type}`] = fn;
      },
    },
    "batch-edit-value-wrap": { innerHTML: "" },
    "batch-edit-apply-btn": {
      addEventListener(type, fn) {
        listeners[`apply:${type}`] = fn;
      },
    },
    "batch-edit-cancel-btn": {
      addEventListener(type, fn) {
        listeners[`cancel:${type}`] = fn;
      },
    },
  };

  globalThis.window = {
    t: (key, count) => (key === "batchEdit.title" ? `Batch Edit <${count}>` : key),
  };
  globalThis.document = {
    activeElement: null,
    getElementById: (id) => elements[id] || null,
  };

  try {
    const panel = new PeekPanel({}, {});
    const pending = panel.showBatchEditDialog(3);
    assert.ok(peekBody.innerHTML.includes('aria-label="Batch Edit &lt;3&gt;"'));
    assert.ok(peekBody.innerHTML.includes("Batch Edit &lt;3&gt;"));
    assert.equal(typeof listeners["cancel:click"], "function");
    listeners["cancel:click"]();
    assert.equal(await pending, null);
  } finally {
    delete globalThis.document;
    delete globalThis.window;
  }
});
