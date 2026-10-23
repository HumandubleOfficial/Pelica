// The window: sidebar, title bar, the current page (chat, models, brain),
// and the overlays (settings, command approval, toasts).
import { useEffect, useState } from "react";
import { app, startNewChat, useApp } from "./state";
import { ChatView } from "./components/ChatView";
import { Composer } from "./components/Composer";
import { Sidebar, TitleBar } from "./components/Sidebar";
import { SettingsModal } from "./components/Settings";
import { Library } from "./components/Library";
import { Brain } from "./components/Brain";
import { ExpertsBg } from "./components/ExpertsBg";
import { Icon, Logo } from "./components/Icons";
import { Palette } from "./components/Palette";
import { ArtifactPanel } from "./components/ArtifactPanel";
import { PhotoViewer } from "./components/Attachments";
import { PromptMaker } from "./components/PromptMaker";
import { Compare } from "./components/Compare";
import { SidePanel } from "./components/SidePanel";

/** Keyboard shortcuts that work anywhere in the window. */
function useShortcuts() {
  useEffect(() => {
    const f = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
      const k = e.key.toLowerCase();
      const run = (fn: () => void) => (e.preventDefault(), fn());
      if (k === "k") run(() => app.set((s) => ({ palette: !s.palette })));
      else if (k === "n" && !e.shiftKey) run(() => (startNewChat(), setTimeout(() => window.dispatchEvent(new Event("pelica:focus")), 50)));
      else if (k === ",") run(() => app.set({ settingsOpen: true }));
      else if (k === "b" && !e.shiftKey) run(() => app.set((s) => ({ sidebar: !s.sidebar })));
      else if (k === "l") run(() => window.dispatchEvent(new Event("pelica:focus")));
    };
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, []);
}

/** Files dropped anywhere on the window go to the message box. */
function DropZone() {
  const [over, setOver] = useState(false);
  useEffect(() => {
    let depth = 0;
    const hasFiles = (e: DragEvent) => !!e.dataTransfer && Array.from(e.dataTransfer.types).includes("Files");
    const enter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth++;
      setOver(true);
    };
    const leave = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth = Math.max(0, depth - 1);
      if (!depth) setOver(false);
    };
    // (without this the window would navigate to the dropped file)
    const overFn = (e: DragEvent) => e.preventDefault();
    const drop = (e: DragEvent) => {
      e.preventDefault();
      depth = 0;
      setOver(false);
      const files = Array.from(e.dataTransfer?.files || []);
      if (!files.length) return;
      if (app.get().view !== "chat") app.set({ view: "chat" });
      setTimeout(() => window.dispatchEvent(new CustomEvent("pelica:drop", { detail: files })), 30);
    };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragleave", leave);
    window.addEventListener("dragover", overFn);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("dragover", overFn);
      window.removeEventListener("drop", drop);
    };
  }, []);
  return over ? (
    <div className="drop-zone">
      <div className="drop-card">
        <Icon name="clip" size={26} />
        <div>Drop to attach</div>
        <div className="muted small">Text and code files go into your message</div>
      </div>
    </div>
  ) : null;
}

function greeting(name: string) {
  const h = new Date().getHours();
  const part = h < 5 ? "Up late" : h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening";
  return name ? `${part}, ${name}` : part;
}

const IDEAS: [string, string, string][] = [
  ["terminal", "Build an app", "Build a small to-do web app with HTML, CSS and JavaScript in this folder, then tell me how to run it."],
  ["globe", "What's new", "What are today's top technology news stories?"],
  ["edit", "Write", "Help me write a friendly email declining a meeting."],
  ["scan", "Review code", "Look through this project and point out bugs or risky code."],
];

function Home() {
  const name = useApp((s) => s.settings.userName);
  const st = useApp((s) => s.status);
  return (
    <div className="home">
      <div className="home-center">
        <h1 className="greet">
          <span className="greet-mark">
            <Logo size={34} />
          </span>
          {greeting(name)}
        </h1>
        <Composer big />
        <div className="ideas">
          {IDEAS.map(([icon, label, text]) => (
            <button key={label} className="idea" onClick={() => window.dispatchEvent(new CustomEvent("pelica:compose", { detail: text }))}>
              <Icon name={icon} size={14} /> {label}
            </button>
          ))}
        </div>
        {st?.state === "loading" && (
          <div className="load-note">
            <div className="load-bar">
              <span style={{ width: `${st.progress || 2}%` }} />
            </div>
            Loading {st.model}… {st.progress ? `${Math.round(st.progress)}%` : ""}
          </div>
        )}
        {st?.state === "error" && <div className="err-text center">Loading failed: {st.error}</div>}
        {st?.state === "ready" && st.experts_ram ? (
          <div className="muted small center">
            {st.model}: {st.experts! - st.experts_ram} experts in VRAM, {st.experts_ram} in RAM · {Math.round((st.ctx || 0) / 1024)}K context
          </div>
        ) : null}
      </div>
    </div>
  );
}

function Approval() {
  const a = useApp((s) => s.approval);
  useEffect(() => {
    if (!a) return;
    const f = (e: KeyboardEvent) => {
      if (e.key === "Escape") a.resolve("deny");
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) a.resolve("allow");
    };
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, [a]);
  if (!a) return null;
  return (
    <div className="modal-back">
      <div className="modal approval">
        <div className="approval-head">
          <Icon name="terminal" size={18} /> {a.title}
        </div>
        {a.chatTitle && a.chatTitle !== app.get().chat?.title && <div className="approval-chat">From the chat “{a.chatTitle}”</div>}
        {a.why && <p className="muted small">{a.why}</p>}
        <pre className="approval-cmd">{a.command}</pre>
        {a.folder && <div className="muted small">in {a.folder}</div>}
        <div className="approval-btns">
          <button className="btn ghost" onClick={() => a.resolve("deny")}>
            Deny <kbd>Esc</kbd>
          </button>
          <button className="btn ghost" onClick={() => a.resolve("always")}>
            Allow this command for the chat
          </button>
          <button className="btn primary" autoFocus onClick={() => a.resolve("allow")}>
            Allow <kbd>Ctrl+Enter</kbd>
          </button>
        </div>
      </div>
    </div>
  );
}

function Toast() {
  const t = useApp((s) => s.toast);
  return t ? (
    <div className="toast" onClick={() => app.set({ toast: null })}>
      {t}
    </div>
  ) : null;
}

export function App() {
  useShortcuts();
  const view = useApp((s) => s.view);
  const sidebar = useApp((s) => s.sidebar);
  const empty = useApp((s) => !s.chat || (s.chat.messages.length === 0 && !s.chat.archive?.length));
  const panel = useApp((s) => s.settings.sidePanel && !s.artifact);
  return (
    <div className={`shell ${sidebar ? "" : "no-side"}`}>
      <ExpertsBg />
      {sidebar && <Sidebar />}
      <main className="main">
        <TitleBar />
        <div className="content">
          <div className="center-pane">
            {view === "library" ? (
              <Library />
            ) : view === "brain" ? (
              <Brain />
            ) : empty ? (
              <Home />
            ) : (
              <>
                <ChatView />
                <div className="composer-dock">
                  <Composer />
                </div>
              </>
            )}
          </div>
          <ArtifactPanel />
          {panel && view === "chat" && !empty && <SidePanel />}
        </div>
      </main>
      <Palette />
      <DropZone />
      <SettingsModal />
      <Approval />
      <PhotoViewer />
      <PromptMaker />
      <Compare />
      <Toast />
    </div>
  );
}
