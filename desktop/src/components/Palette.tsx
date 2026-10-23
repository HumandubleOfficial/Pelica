// The command palette (Ctrl+K): every action and chat, a few keystrokes
// away. Type to filter, arrows to move, Enter to run.
import { useEffect, useMemo, useRef, useState } from "react";
import { compactNow } from "../agent/agent";
import { pickFolder } from "../agent/commands";
import { bridge, type Material } from "../lib/bridge";
import { copyChat, exportChat } from "../lib/exportChat";
import { app, openChat, PERM_MODES, setSettings, startNewChat, useApp, type Settings } from "../state";
import { Icon } from "./Icons";

interface Item {
  id: string;
  group: string;
  label: string;
  icon: string;
  keys?: string;
  hint?: string;
  run: () => void;
}

/** How well `q` matches `s`: a substring beats a scattered match; 0 is none. */
function score(q: string, s: string): number {
  if (!q) return 1;
  const t = s.toLowerCase();
  const i = t.indexOf(q);
  if (i >= 0) return 200 - i + (i === 0 || t[i - 1] === " " ? 100 : 0);
  let from = 0;
  let sc = 0;
  let last = -2;
  for (const ch of q) {
    if (ch === " ") continue;
    const k = t.indexOf(ch, from);
    if (k < 0) return 0;
    sc += k === last + 1 ? 4 : 1;
    last = k;
    from = k + 1;
  }
  return sc;
}

const ago = (t: number) => {
  const m = Math.round((Date.now() - t) / 60000);
  return m < 1 ? "now" : m < 60 ? `${m} min ago` : m < 1440 ? `${Math.round(m / 60)} h ago` : `${Math.round(m / 1440)} d ago`;
};

export function setMaterial(m: Material) {
  setSettings({ material: m as Settings["material"] });
  bridge?.win.material(m).catch(() => {});
}

function actions(): Item[] {
  const s = app.get().settings;
  const close = () => app.set({ palette: false });
  const go = (fn: () => void) => () => (close(), fn());
  const items: Item[] = [
    { id: "new", group: "Actions", label: "New chat", icon: "plus", keys: "Ctrl+N", run: go(() => (startNewChat(), setTimeout(() => window.dispatchEvent(new Event("pelica:focus")), 50))) },
    { id: "folder", group: "Actions", label: "Choose this chat's folder…", icon: "folder", run: go(pickFolder) },
    { id: "prompt", group: "Actions", label: "Prompt maker", icon: "wand", hint: "answer a few questions, get a tuned prompt", run: go(() => app.set({ promptMaker: true })) },
    { id: "compare", group: "Go to", label: "Compare models with flagships", icon: "compare", hint: "Astra, Fable, Opus, Kimi K3, GLM 5.3", run: go(() => app.set({ compare: "" })) },
    ...PERM_MODES.map((m) => ({ id: `perm-${m.id}`, group: "Permissions", label: `Permissions: ${m.label}`, icon: m.id === "plan" ? "list" : m.id === "auto" ? "bolt" : m.id === "ask" ? "lock" : "shield", hint: s.permMode === m.id ? "current" : m.desc, run: go(() => setSettings({ permMode: m.id })) })),
    { id: "web", group: "Actions", label: s.web ? "Turn web search off" : "Turn web search on", icon: "globe", run: go(() => setSettings({ web: !s.web })) },
    { id: "attach", group: "Actions", label: "Attach files…", icon: "clip", hint: "or drop them on the window", run: go(() => document.querySelector<HTMLInputElement>(".composer input[type=file]")?.click()) },
    { id: "compact", group: "Actions", label: "Compact this conversation", icon: "layers", run: go(() => compactNow()) },
    { id: "export", group: "Actions", label: "Export this chat as Markdown…", icon: "download", run: go(() => exportChat()) },
    { id: "copy-chat", group: "Actions", label: "Copy this whole chat", icon: "copy", hint: "as Markdown", run: go(() => copyChat()) },
    { id: "models", group: "Go to", label: "Models", icon: "library", run: go(() => app.set({ view: "library" })) },
    { id: "brain", group: "Go to", label: "Brain: the model's experts in 3D", icon: "brain", run: go(() => app.set({ view: "brain" })) },
    { id: "chat", group: "Go to", label: "Chat", icon: "chat", run: go(() => app.set({ view: "chat" })) },
    { id: "settings", group: "Go to", label: "Settings", icon: "settings", keys: "Ctrl+,", run: go(() => app.set({ settingsOpen: true })) },
    { id: "desktop", group: "Go to", label: "Desktop settings: shortcut, tray, notifications", icon: "monitor", run: go(() => app.set({ settingsOpen: true, settingsTab: "Desktop" })) },
    { id: "side", group: "View", label: app.get().sidebar ? "Hide the sidebar" : "Show the sidebar", icon: "sidebar", keys: "Ctrl+B", run: go(() => app.set((x) => ({ sidebar: !x.sidebar }))) },
    ...(["dark", "warm", "light", "midnight", "neural"] as const).map((t) => ({ id: `theme-${t}`, group: "View", label: `Theme: ${t[0].toUpperCase() + t.slice(1)}`, icon: "palette", hint: s.theme === t ? "current" : undefined, run: go(() => setSettings({ theme: t })) })),
    ...(bridge
      ? ([
          ["none", "Solid"],
          ["mica", "Mica"],
          ["acrylic", "Acrylic"],
        ] as const).map(([m, l]) => ({ id: `mat-${m}`, group: "View", label: `Window backdrop: ${l}`, icon: "window", hint: s.material === m ? "current" : undefined, run: go(() => setMaterial(m)) }))
      : []),
  ];
  return items;
}

export function Palette() {
  const open = useApp((s) => s.palette);
  const chats = useApp((s) => s.chats);
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open) {
      setQ("");
      setSel(0);
      setTimeout(() => input.current?.focus(), 0);
    }
  }, [open]);

  const items = useMemo(() => {
    if (!open) return [];
    const query = q.trim().toLowerCase();
    const chatItems: Item[] = chats.map((c) => ({ id: `chat-${c.id}`, group: "Chats", label: c.title || "Untitled", icon: c.pinned ? "pin" : "chat", hint: ago(c.updated), run: () => (app.set({ palette: false }), openChat(c.id)) }));
    const all = [...actions(), ...chatItems];
    if (!query) return [...all.filter((i) => i.group !== "Chats"), ...chatItems.slice(0, 6)];
    return all
      .map((i) => ({ i, s: score(query, i.label) }))
      .filter((x) => x.s > 0)
      // (actions a little ahead of chats with the same match)
      .map((x) => ({ i: x.i, s: x.s + (x.i.group === "Chats" ? 0 : 5) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, 40)
      .map((x) => x.i);
  }, [open, q, chats]);

  useEffect(() => setSel(0), [q]);
  useEffect(() => {
    list.current?.querySelector(".pal-item.sel")?.scrollIntoView({ block: "nearest" });
  }, [sel]);

  if (!open) return null;
  const groups: [string, Item[]][] = [];
  for (const it of items) {
    const g = groups.find((x) => x[0] === it.group);
    if (g) g[1].push(it);
    else groups.push([it.group, [it]]);
  }
  const flat = groups.flatMap((g) => g[1]);

  return (
    <div className="modal-back pal-back" onMouseDown={(e) => e.target === e.currentTarget && app.set({ palette: false })}>
      <div className="palette">
        <div className="pal-input">
          <Icon name="search" size={16} />
          <input
            ref={input}
            value={q}
            placeholder="Search actions and chats…"
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape") app.set({ palette: false });
              else if (e.key === "ArrowDown") (e.preventDefault(), setSel((sel + 1) % Math.max(1, flat.length)));
              else if (e.key === "ArrowUp") (e.preventDefault(), setSel((sel - 1 + flat.length) % Math.max(1, flat.length)));
              else if (e.key === "Enter") flat[sel]?.run();
            }}
          />
          <kbd>Esc</kbd>
        </div>
        <div className="pal-list" ref={list}>
          {flat.length === 0 && <div className="muted small pad">Nothing matches “{q}”.</div>}
          {groups.map(([g, its]) => (
            <div key={g}>
              <div className="pal-group">{g}</div>
              {its.map((it) => {
                const i = flat.indexOf(it);
                return (
                  <button key={it.id} className={`pal-item ${i === sel ? "sel" : ""}`} onMouseMove={() => i !== sel && setSel(i)} onClick={it.run}>
                    <Icon name={it.icon} size={15} />
                    <span className="pal-label">{it.label}</span>
                    {it.hint && <span className="pal-hint">{it.hint}</span>}
                    {it.keys && <kbd>{it.keys}</kbd>}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
        <div className="pal-foot">
          <span>
            <kbd>↑</kbd>
            <kbd>↓</kbd> move
          </span>
          <span>
            <kbd>Enter</kbd> run
          </span>
          <span className="grow" />
          <span>
            <Icon name="command" size={11} /> Ctrl+K anywhere
          </span>
        </div>
      </div>
    </div>
  );
}
