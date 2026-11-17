/* Apply the World day/night preference before the dashboard paints. */
(() => {
  "use strict";

  const key = "chat_dynamics_ui";
  const root = document.documentElement;
  const valid = (value) => value === "day" || value === "night";
  const params = new URLSearchParams(window.location.search);
  let toggle;
  let mode = params.get("ui");
  if (!valid(mode)) mode = stored(key);
  if (!valid(mode)) {
    // The host injects its own light/dark theme, which is only a fallback.
    mode = fromLegacy(params.get("theme")) ?? fromLegacy(stored("lmem_theme")) ?? "day";
  }

  function stored(preference) {
    try { return localStorage.getItem(preference); } catch { return null; }
  }

  function fromLegacy(value) {
    if (value === "dark") return "night";
    if (value === "light") return "day";
    return valid(value) ? value : null;
  }

  function apply() {
    const night = mode === "night";
    root.dataset.uiTheme = mode;
    // Legacy graph renderers read data-theme. The host can rewrite that attribute.
    root.dataset.theme = night ? "dark" : "light";
    root.style.colorScheme = night ? "dark" : "light";
    toggle?.setAttribute("aria-pressed", String(night));
    if (!toggle) return;
    const label = document.getElementById("theme-label");
    const translate = window.DynamicsUI?.tr || ((text) => text);
    if (label) label.textContent = translate(night ? "夜间模式" : "日间模式");
    toggle.setAttribute("aria-label", translate(night ? "切换到日间模式" : "切换到夜间模式"));
  }

  function rememberUrl() {
    try {
      const url = new URL(window.location.href);
      url.searchParams.set("ui", mode);
      window.history.replaceState(window.history.state, "", url);
    } catch {
      // The AstrBot bridge can keep the preference in the host tab.
    }
  }

  apply();
  window.addEventListener("storage", ({key: changedKey, newValue}) => {
    if (changedKey !== key || !valid(newValue)) return;
    mode = newValue;
    apply();
    rememberUrl();
  });

  window.LMAppearance = {
    setContext() { apply(); },
    init() {
      if (toggle) return;
      toggle = document.getElementById("theme-toggle");
      if (!toggle) return;
      toggle.addEventListener("click", () => {
        mode = mode === "night" ? "day" : "night";
        apply();
        try { localStorage.setItem(key, mode); } catch { /* Current page still updates. */ }
        rememberUrl();
      });
      window.addEventListener("series-languagechange", apply);
      apply();
    },
  };
})();
