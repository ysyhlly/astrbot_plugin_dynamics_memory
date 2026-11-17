import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runInThisContext } from "node:vm";

const dashboard = new URL("../../pages/dashboard/", import.meta.url);

class Element extends EventTarget {
  attributes = new Map();
  children = [];
  dataset = {};
  style = {};
  isConnected = true;

  setAttribute(name, value) { this.attributes.set(name, value); }
  getAttribute(name) { return this.attributes.get(name); }
  get lang() { return this.getAttribute("lang"); }
  set lang(value) { this.setAttribute("lang", value); }
  append(...children) { this.children.push(...children); }
}

test("the series picker renders Japanese and shares its selection with the host", async t => {
  const contextListeners = [];
  const navigation = [];
  const preferences = new Map();
  const host = {
    getContext: () => ({ locale: "ja-JP" }),
    onContext: callback => contextListeners.push(callback),
    setNavigationContext: async context => { navigation.push(context); },
  };
  const window = Object.assign(new EventTarget(), {
    location: new URL("https://example.invalid/"),
    AstrBotPluginPage: host,
  });
  const document = Object.assign(new EventTarget(), {
    readyState: "loading",
    documentElement: new Element(),
    body: new Element(),
    createElement: () => new Element(),
  });
  const memoryTab = new Element();
  memoryTab.setAttribute("data-i18n", "nav.memory");
  const queryInput = new Element();
  queryInput.setAttribute("data-i18n-placeholder", "graph.queryPh");
  queryInput.setAttribute("data-i18n-aria", "graph.queryPh");
  let picker;
  document.querySelectorAll = selector => ({
    "[data-i18n]": [memoryTab],
    "[data-i18n-placeholder]": [queryInput],
    "[data-i18n-aria]": [queryInput],
    "[data-series-language]": picker ? [picker] : [],
  })[selector] ?? [];

  t.mock.method(globalThis, "setInterval", () => {});
  for (const [key, value] of Object.entries({
    window,
    document,
    location: window.location,
    localStorage: {
      getItem: key => preferences.get(key),
      setItem: (key, value) => preferences.set(key, value),
    },
    MutationObserver: class { observe() {} },
  })) {
    const previous = Object.getOwnPropertyDescriptor(globalThis, key);
    Object.defineProperty(globalThis, key, { configurable: true, value });
    t.after(() => {
      if (previous) Object.defineProperty(globalThis, key, previous);
      else delete globalThis[key];
    });
  }

  runInThisContext(readFileSync(new URL("assets/base.min.js", dashboard), "utf8"));
  const { getLanguage, languagePicker } = await import(
    new URL("series-ui.js", dashboard).href
  );
  document.dispatchEvent(new Event("DOMContentLoaded"));
  picker = languagePicker().children[1];
  assert.equal(picker.children.find(option => option.value === "ja").textContent, "日本語");
  assert.equal(picker.value, "ja");
  assert.equal(getLanguage(), "ja");
  assert.equal(document.documentElement.lang, "ja-JP");
  assert.equal(memoryTab.textContent, "記憶管理");
  assert.match(window.t("edit.conflict"), /最新の内容を確認してから保存/);
  assert.match(queryInput.getAttribute("placeholder"), /^人物、テーマ、事実/);
  assert.equal(queryInput.getAttribute("aria-label"), queryInput.getAttribute("placeholder"));
  assert.deepEqual(navigation, []);

  for (const receive of contextListeners) receive({ locale: "en-US" });
  assert.equal(getLanguage(), "en");
  assert.equal(memoryTab.textContent, "Memory");

  picker.value = "ja";
  picker.dispatchEvent(new Event("change"));
  assert.deepEqual(navigation, [{ locale: "ja" }]);
  assert.equal(memoryTab.textContent, "記憶管理");
  assert.equal(document.documentElement.lang, "ja-JP");
});
