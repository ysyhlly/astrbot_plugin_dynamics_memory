import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const dashboard = join(here, "../../pages/dashboard");
const read = (name) => readFileSync(join(dashboard, name), "utf-8");

function rgb(hex) {
  assert.match(hex, /^#(?:[\da-f]{3}|[\da-f]{6})$/i, "expected an opaque hex palette color");
  const value = hex.length === 4 ? [...hex.slice(1)].map((digit) => digit.repeat(2)).join("") : hex.slice(1);
  return [0, 2, 4].map((offset) => parseInt(value.slice(offset, offset + 2), 16));
}

function luminance(hex) {
  const channel = (value) => {
    const normalized = value / 255;
    return normalized <= 0.04045
      ? normalized / 12.92
      : ((normalized + 0.055) / 1.055) ** 2.4;
  };
  const [r, g, b] = rgb(hex);
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

function contrast(foreground, background) {
  const [high, low] = [luminance(foreground), luminance(background)].sort((a, b) => b - a);
  return (high + 0.05) / (low + 0.05);
}

function paletteDeclarations(css, selector) {
  const rules = [...css.replace(/\/\*[\s\S]*?\*\//g, "").matchAll(/([^{}]+)\{([^{}]*)\}/g)];
  const declarations = rules.filter(([, selectors]) => selectors.trim() === selector)
    .flatMap(([, , block]) => [...block.matchAll(/(--[\w-]+)\s*:\s*([^;{}]+)/g)]);
  return declarations.map(([, name, value]) => [name, value.trim()]);
}

function paletteColor(tokens, name, chain = []) {
  assert.ok(!chain.includes(name), `circular palette variable ${name}`);
  const value = tokens.get(name);
  assert.ok(value, `missing palette variable ${name}`);
  const reference = value.match(/^var\(\s*(--[\w-]+)\s*\)$/);
  return reference ? paletteColor(tokens, reference[1], [...chain, name]) : value;
}

for (const mode of ["day", "night"]) {
  test(`${mode} theme focus ring and tertiary text meet WCAG contrast`, () => {
    // The dashboard loads the Memory styles first and the shared palette last.
    const css = ["styles.css", "art-direction.css", "themes.css", "workflows.css", "series-ui.css"]
      .map(read).join("\n");
    const tokens = new Map(paletteDeclarations(css, ":root"));
    if (mode === "night") {
      const night = paletteDeclarations(css, 'html[data-ui-theme="night"]');
      assert.ok(night.length, "night theme palette is defined");
      for (const [name, value] of night) tokens.set(name, value);
    }

    const checks = [["--text-tertiary", 4.5], ["--border-focus", 3], ["--ds-accent", 3]];
    const surfaces = ["--bg-app", "--bg-card", "--bg-input", "--bg-tint", "--bg-hover", "--bg-active"];
    for (const [foreground, minimum] of checks) {
      for (const background of surfaces) {
        const ratio = contrast(paletteColor(tokens, foreground), paletteColor(tokens, background));
        assert.ok(ratio >= minimum, `${foreground} on ${background}: ${ratio.toFixed(2)} < ${minimum}`);
      }
    }
  });
}

test("graph canvas, peek panel and action dialog expose accessible semantics", () => {
  const index = read("index.html");
  const peek = read("modules/peek-panel.js");
  const graph = read("graph-ui.js");
  const settings = read("modules/settings-page.js");

  assert.match(index, /id="graph-canvas-state"[^>]*role="status"[^>]*aria-live="polite"/);
  assert.match(index, /id="peek-panel"[^>]*role="dialog"[^>]*aria-modal="true"[^>]*aria-labelledby="peek-title"/);
  assert.match(index, /id="graph-node-list-btn"/);
  assert.match(index, /id="graph-node-list-dialog"[^>]*aria-labelledby="graph-node-list-title"/);
  assert.match(index, /id="action-dialog"[^>]*aria-labelledby="action-dialog-title"/);

  assert.match(peek, /_trapFocus\(event\)/);
  assert.match(peek, /_previousFocus/);
  assert.match(peek, /role="dialog" aria-modal="true"/);

  assert.match(graph, /setAttribute\("role", "img"\)/);
  assert.match(graph, /graph\.canvasAria/);
  assert.match(graph, /graph-node-list/);

  assert.match(settings, /type === "text"/);
  assert.match(settings, /<textarea/);
  assert.match(settings, /confirmAction/);
  assert.doesNotMatch(settings, /window\.confirm/);
});

test("dashboard language handling covers ja everywhere", () => {
  const i18n = read("i18n.js");
  const app = read("app.js");
  assert.match(i18n, /lang === "ja" \? "ja"/);
  assert.match(i18n, /function languageTag/);
  assert.match(app, /\["zh", "en", "ru", "ja"\]/);
});
