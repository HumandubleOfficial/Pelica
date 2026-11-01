import { createRoot } from "react-dom/client";
import { App } from "./App";
import { bridge } from "./lib/bridge";
import { engine } from "./lib/engine";
import { app, loadSettings, newChat, openChat, persist, refreshChats } from "./state";
import { startDesktop } from "./lib/desktop";
import { QuickAsk } from "./components/QuickAsk";
import "./styles.css";
import "./desktop.css";
import "./features.css";
import "./panel.css";
import "./rail.css";
import "./polish.css";
import "./settings.css";
import { checkWhatsNew } from "./components/WhatsNew";
import { checkForUpdates } from "./components/Updates";
import { startScheduler } from "./agent/tasks";
import "highlight.js/styles/github-dark.css";

// the store (and chat saving), for looking at the app from DevTools and tests
(window as any).__pelica = { app, persist, openChat, refreshChats };

// theme and text size follow the settings
app.sub(() => {
  const s = app.get().settings;
  const el = document.documentElement;
  if (el.dataset.theme !== s.theme) el.dataset.theme = s.theme;
  if (el.dataset.material !== s.material) el.dataset.material = bridge ? s.material : "none";
  el.style.setProperty("--font-size", `${s.fontSize}px`);
  el.style.setProperty("--left-w", `${s.sideWidth}px`);
  el.style.setProperty("--mono", `${s.codeFont ? `"${s.codeFont.replace(/"/g, "")}", ` : ""}"Cascadia Code", "Cascadia Mono", Consolas, monospace`);
  el.classList.toggle("serif-replies", s.replyFont !== "sans");
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
  checkWhatsNew();
  checkForUpdates();
  poll();
  if (bridge) bridge.mcp.list().then((mcp) => app.set({ mcp })).catch(() => {});
  startDesktop();
  startScheduler();
  createRoot(document.getElementById("root")!).render(<App />);
})();
