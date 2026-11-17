import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("../../pages/dashboard/appearance.js", import.meta.url), "utf8");
const sharedKey = "chat_dynamics_ui";

function boot({ saved = {}, search = "", blocked = false, deviceDark = false } = {}) {
  const storage = new Map(Object.entries(saved));
  const attributes = new Map();
  const initialUrl = new URL(`https://example.test/plugin/index.html${search}#workspace`);
  const location = new URL(initialUrl);
  const root = { dataset: {}, style: {} };
  const label = { textContent: "日间模式" };
  const window = new EventTarget();
  const toggle = new EventTarget();
  const denyStorage = () => { throw new DOMException("Storage denied", "SecurityError"); };
  let initialized = false;

  toggle.setAttribute = (name, value) => attributes.set(name, value);
  Object.assign(window, {
    URL,
    URLSearchParams,
    location,
    localStorage: {
      getItem(key) { return storage.get(key) ?? null; },
      setItem(key, value) { storage.set(key, String(value)); },
    },
    history: {
      state: { view: "memory" },
      replaceState(state, _, url) {
        if (blocked) throw new DOMException("History denied", "SecurityError");
        assert.deepEqual(state, this.state);
        location.href = String(url);
      },
    },
    matchMedia: () => ({ matches: deviceDark }),
    document: {
      documentElement: root,
      getElementById(id) {
        if (!initialized) return null;
        return { "theme-toggle": toggle, "theme-label": label }[id] ?? null;
      },
    },
  });
  if (blocked) Object.defineProperty(window, "localStorage", { get: denyStorage });
  window.window = window;
  vm.runInNewContext(source, window, { filename: "appearance.js" });

  return {
    root, storage, location, label, attributes, window,
    init() {
      initialized = true;
      window.LMAppearance.init();
    },
    click() {
      toggle.dispatchEvent(new Event("click"));
    },
    hostContext(context) {
      if (typeof context.isDark === "boolean") root.dataset.theme = context.isDark ? "dark" : "light";
      window.LMAppearance.setContext(context);
    },
    changeStorage(key, newValue) {
      if (key === null) storage.clear();
      else if (newValue === null) storage.delete(key);
      else storage.set(key, newValue);
      window.dispatchEvent(Object.assign(new Event("storage"), { key, newValue }));
    },
  };
}

function assertTheme({ root }, mode) {
  const color = mode === "night" ? "dark" : "light";
  assert.equal(root.dataset.uiTheme, mode);
  assert.equal(root.dataset.theme, color);
  assert.equal(root.style.colorScheme, color);
}

test("first paint applies the series URL or shared preference before host and legacy fallbacks", () => {
  assertTheme(boot({ search: "?ui=day&theme=dark", saved: { [sharedKey]: "night", lmem_theme: "dark" } }), "day");
  assertTheme(boot({ search: "?ui=night&theme=light", saved: { [sharedKey]: "day" } }), "night");
  assertTheme(boot({ search: "?theme=light", saved: { [sharedKey]: "night" } }), "night");
  assertTheme(boot({ search: "?theme=dark", saved: { lmem_theme: "light" } }), "night");
  assertTheme(boot({ saved: { lmem_style: "paper", lmem_theme: "dark" } }), "night");
});

test("device and bridge color changes preserve the independent series mode", () => {
  const standalone = boot({ deviceDark: true });
  assertTheme(standalone, "day");
  standalone.hostContext({ isDark: true });
  assertTheme(standalone, "day");

  const embedded = boot({ search: "?ui=night&theme=light" });
  embedded.hostContext({ isDark: false, ui: "day", uiOverride: false });
  assertTheme(embedded, "night");
  embedded.hostContext({ locale: "en-US" });
  assertTheme(embedded, "night");
});

test("clicking the switch persists the shared selection and keeps scope, locale and history", () => {
  const page = boot({ search: "?ui=day&umo=qq%3AGroupMessage%3Ademo&lang=ja" });
  page.init();
  page.init();
  assert.equal(page.attributes.get("aria-pressed"), "false");
  assert.equal(page.attributes.get("aria-label"), "切换到夜间模式");
  page.click();
  assertTheme(page, "night");
  assert.equal(page.storage.get(sharedKey), "night");
  assert.equal(page.storage.has("lmem_theme"), false);
  assert.equal(page.label.textContent, "夜间模式");
  assert.equal(page.attributes.get("aria-pressed"), "true");
  assert.equal(page.attributes.get("aria-label"), "切换到日间模式");
  assert.equal(page.location.searchParams.get("ui"), "night");
  assert.equal(page.location.searchParams.get("umo"), "qq:GroupMessage:demo");
  assert.equal(page.location.searchParams.get("lang"), "ja");
  assert.equal(page.location.hash, "#workspace");

  page.window.DynamicsUI = { tr: value => ({ 夜间模式: "Night mode", 切换到日间模式: "Switch to day mode" })[value] };
  page.window.dispatchEvent(new Event("series-languagechange"));
  assert.equal(page.label.textContent, "Night mode");
  assert.equal(page.attributes.get("aria-label"), "Switch to day mode");
  assertTheme(page, "night");
});

test("invalid URL and saved preferences fall through to valid inputs or the series default", () => {
  const saved = { [sharedKey]: "unknown", lmem_theme: "unknown", lmem_style: "terminal" };
  const page = boot({ search: "?ui=unknown&theme=unknown", saved });
  assertTheme(page, "day");
  assert.equal(page.root.dataset.style, undefined);
  assertTheme(boot({ search: "?ui=unknown&theme=unknown", saved: { ...saved, lmem_theme: "dark" } }), "night");
  assertTheme(boot({ search: "?ui=unknown&theme=unknown", saved: { [sharedKey]: "night" } }), "night");
});

test("opaque storage and history leave URL fallback and the current page switch usable", () => {
  const page = boot({ search: "?theme=dark", blocked: true });
  assertTheme(page, "night");
  page.init();
  page.click();
  assertTheme(page, "day");
  assert.equal(page.attributes.get("aria-pressed"), "false");
  assert.equal(page.storage.has(sharedKey), false);
  assert.equal(page.location.searchParams.get("ui"), null);
  page.hostContext({ isDark: true, locale: "ja-JP" });
  assertTheme(page, "day");
});

test("other tabs update the switch with a valid shared mode and ignore removal or unrelated changes", () => {
  const page = boot({ search: "?ui=day&lang=ja" });
  page.init();
  page.changeStorage("lmem_style", "paper");
  page.changeStorage("lmem_theme", "dark");
  page.changeStorage(sharedKey, "unknown");
  assertTheme(page, "day");

  page.changeStorage(sharedKey, "night");
  assertTheme(page, "night");
  assert.equal(page.attributes.get("aria-pressed"), "true");
  assert.equal(page.location.searchParams.get("ui"), "night");
  assert.equal(page.location.searchParams.get("lang"), "ja");
  page.changeStorage(sharedKey, null);
  assertTheme(page, "night");
  page.changeStorage(null, null);
  assertTheme(page, "night");
  assert.equal(page.attributes.get("aria-pressed"), "true");
  assertTheme(boot({ saved: Object.fromEntries(page.storage) }), "day");
});
