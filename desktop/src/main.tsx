import { createRoot } from "react-dom/client";
import { App } from "./App";
import { bridge } from "./lib/bridge";
import { engine } from "./lib/engine";
import { app, loadSettings, newChat, refreshChats } from "./state";
import { startDesktop } from "./lib/desktop";
import { QuickAsk } from "./components/QuickAsk";
import "./styles.css";
import "./desktop.css";
import "./features.css";
import "./panel.css";
import "highlight.js/styles/github-dark.css";

// theme and text size follow the settings
app.sub(() => {
  const s = app.get().settings;
  const el = document.documentElement;
  if (el.dataset.theme !== s.theme) el.dataset.theme = s.theme;
  if (el.dataset.material !== s.material) el.dataset.material = bridge ? s.material : "none";
  el.style.setProperty("--font-size", `${s.fontSize}px`);
  el.classList.toggle("no-anim", !s.animations);
});

// the engine's status: often while loading, now and then otherwise
let lastPath = "";
async function poll() {
  try {
    const st = await engine.status();
    app.set({ status: st });
    if (st.state === "ready" && st.path !== lastPath) {
      lastPath = st.path || "";
      app.set({ context: null });
    }
    setTimeout(poll, st.state === "loading" ? 700 : 2500);
  } catch {
    app.set({ status: { state: "error", error: "the engine isn't answering" } });
    setTimeout(poll, 3000);
  }
}

// the quick-ask bar (its own small window) is this page with #quick
const quick = location.hash === "#quick";

(async () => {
  await loadSettings();
  if (quick) {
    document.documentElement.classList.add("quick-mode");
    createRoot(document.getElementById("root")!).render(<QuickAsk />);
    return;
  }
  app.set({ chat: newChat(null) });
  refreshChats();
  poll();
  if (bridge) bridge.mcp.list().then((mcp) => app.set({ mcp })).catch(() => {});
  startDesktop();
  createRoot(document.getElementById("root")!).render(<App />);
})();
