// The panel on the right of a chat, as in the Claude app: where the chat
// works (this computer, its folder, permissions, web), how full the context
// is and what fills it, the connectors, and the files the chat wrote.
import { useMemo, useState } from "react";
import { compactNow } from "../agent/agent";
import { bridge } from "../lib/bridge";
import { app, PERM_MODES, setSettings, useApp, type Msg } from "../state";
import { openArtifact } from "./ArtifactPanel";
import { FolderList } from "./Composer";
import { presetOf } from "./Connectors";
import { Icon } from "./Icons";

const fmt = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 100000 ? 0 : 1)}k` : String(Math.round(n)));
const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

/** A section with a title that folds it (remembered in the settings). */
function Section({ id, title, children, action }: { id: string; title: string; children: React.ReactNode; action?: React.ReactNode }) {
  const folded = useApp((s) => s.settings.panelFolded.includes(id));
  const toggle = () => {
    const f = app.get().settings.panelFolded;
    setSettings({ panelFolded: folded ? f.filter((x) => x !== id) : [...f, id] });
  };
  return (
    <section className={`sp-card ${folded ? "folded" : ""}`}>
      <div className="sp-head">
        <button className="sp-title" onClick={toggle}>
          {title}
          <Icon name="down" size={12} className="sp-caret" />
        </button>
        <span className="grow" />
        {action}
      </div>
      {!folded && <div className="sp-body">{children}</div>}
    </section>
  );
}

function Environment() {
  const st = useApp((s) => s.status);
  const folder = useApp((s) => s.chat?.folder ?? null);
  const mode = useApp((s) => s.settings.permMode);
  const web = useApp((s) => s.settings.web);
  const [pick, setPick] = useState<"folder" | "perm" | null>(null);
  const parts = folder ? folder.replace(/\\/g, "/").split("/").filter(Boolean) : [];
  const name = parts.pop();
  const parent = parts.length ? (parts.length > 2 ? "…/" : "") + parts.slice(-2).join("/") : "";
  const cur = PERM_MODES.find((m) => m.id === mode) || PERM_MODES[1];
  return (
    <Section id="env" title="Environment">
      <div className="sp-row" title={st?.gpu ? `Runs on ${st.gpu}` : "Runs on this PC"}>
        <Icon name="monitor" size={15} />
        <span className="sp-label">This computer</span>
        <span className="sp-meta ellipsis">{st?.gpu?.replace(/^(NVIDIA|AMD|Intel\(R\))\s+(GeForce\s+)?/i, "") || ""}</span>
      </div>
      <button className="sp-row click" onClick={() => setPick(pick === "folder" ? null : "folder")} title={folder || "Choose a folder for this chat"}>
        <Icon name="folder" size={15} />
        <span className="sp-label ellipsis">
          {folder ? (
            <>
              <b>{name}</b>
              {parent && <span className="muted"> · {parent}</span>}
            </>
          ) : (
            <span className="muted">No folder</span>
          )}
        </span>
        <Icon name={pick === "folder" ? "down" : "chevron"} size={12} className="muted" />
      </button>
      {pick === "folder" && (
        <div className="sp-sub">
          <FolderList folder={folder} done={() => setPick(null)} />
        </div>
      )}
      <button className="sp-row click" onClick={() => setPick(pick === "perm" ? null : "perm")} title={cur.desc}>
        <Icon name={mode === "plan" ? "list" : mode === "auto" ? "bolt" : mode === "ask" ? "lock" : "shield"} size={15} />
        <span className="sp-label">Permissions</span>
        <span className={`sp-meta perm-${mode}`}>{cur.label}</span>
      </button>
      {pick === "perm" && (
        <div className="sp-sub">
          {PERM_MODES.map((m) => (
            <button key={m.id} className={`menu-row ${m.id === mode ? "sel" : ""}`} onClick={() => (setSettings({ permMode: m.id }), setPick(null))}>
              <span className="menu-text">
                <b>{m.label}</b>
                <span className="muted small">{m.desc}</span>
              </span>
              {m.id === mode && <Icon name="check" size={13} />}
            </button>
          ))}
        </div>
      )}
      <div className="sp-row">
        <Icon name="globe" size={15} />
        <span className="sp-label">Web search</span>
        <button className={`toggle ${web ? "on" : ""}`} role="switch" aria-checked={web} onClick={() => setSettings({ web: !web })} title={web ? "Turn web search off" : "Turn web search on"} />
      </div>
    </Section>
  );
}

const PARTS = [
  { key: "messages", label: "Messages", color: "var(--c-msgs)" },
  { key: "tool_results", label: "Tool results", color: "var(--c-results)" },
  { key: "tools", label: "System tools", color: "var(--c-tools)" },
  { key: "mcp", label: "MCP tools", color: "var(--c-mcp)" },
  { key: "system", label: "System prompt", color: "var(--c-system)" },
] as const;

function Context() {
  const c = useApp((s) => s.context);
  const ready = useApp((s) => s.status?.state === "ready");
  const running = useApp((s) => s.running);
  const auto = useApp((s) => s.settings.autocompact);
  const at = useApp((s) => s.settings.compactAt);
  if (!c || !ready) {
    return (
      <Section id="ctx" title="Context">
        <div className="sp-empty">Load a model to see what fills its context.</div>
      </Section>
    );
  }
  const buffer = auto ? Math.round(c.ctx * (1 - at)) : 0;
  const free = Math.max(0, c.ctx - c.total - buffer);
  const until = auto ? Math.max(0, Math.round(c.ctx * at) - c.total) : 0;
  return (
    <Section id="ctx" title="Context">
      <div className="sp-ctx-top">
        <b>{Math.round((c.total / c.ctx) * 100)}% used</b>
        <span className="muted">
          {fmt(c.total)} / {fmt(c.ctx)}
        </span>
      </div>
      <div className="sp-ctx-bar">
        {PARTS.map((p) => (
          <span key={p.key} style={{ width: `${(c[p.key] / c.ctx) * 100}%`, background: p.color }} />
        ))}
        {buffer > 0 && <span className="buffer" style={{ width: `${(buffer / c.ctx) * 100}%` }} />}
      </div>
      <div className="sp-ctx-rows">
        {PARTS.map((p) => (
          <div key={p.key} className="sp-ctx-row">
            <i style={{ background: p.color }} />
            <span>{p.label}</span>
            <span className="muted">{fmt(c[p.key])}</span>
            <b>{pct(c[p.key] / c.ctx)}</b>
          </div>
        ))}
        {buffer > 0 && (
          <div className="sp-ctx-row" title={`Compacting starts at ${Math.round(at * 100)}%, so this much is kept free for it`}>
            <i className="buffer" />
            <span>Autocompact buffer</span>
            <span className="muted">{fmt(buffer)}</span>
            <b>{pct(buffer / c.ctx)}</b>
          </div>
        )}
        <div className="sp-ctx-row">
          <i className="free" />
          <span>Free space</span>
          <span className="muted">{fmt(free)}</span>
          <b>{pct(free / c.ctx)}</b>
        </div>
      </div>
      <div className="sp-ctx-foot">
        <span className="muted small ellipsis">{auto ? `${fmt(until)} to compact` : "Auto-compact off"}</span>
        <button className="btn small" disabled={running} onClick={() => compactNow()}>
          Compact session
        </button>
      </div>
    </Section>
  );
}

function Connectors() {
  const cfgs = useApp((s) => s.settings.mcpServers);
  const live = useApp((s) => s.mcp);
  const [all, setAll] = useState(false);
  const manage = (
    <button className="sp-link" onClick={() => app.set({ settingsOpen: true, settingsTab: "Connectors" })} title="Manage connectors">
      <Icon name="settings" size={13} />
    </button>
  );
  const shown = all ? cfgs : cfgs.slice(0, 4);
  return (
    <Section id="mcp" title="Connectors" action={manage}>
      {!cfgs.length && (
        <button className="sp-row click" onClick={() => app.set({ settingsOpen: true, settingsTab: "Connectors" })}>
          <Icon name="plus" size={15} />
          <span className="sp-label muted">Add a connector</span>
        </button>
      )}
      {shown.map((c) => {
        const m = live.find((x) => x.name === c.name);
        const p = presetOf(c);
        const state = c.disabled ? "off" : m?.running ? "on" : m ? "error" : "off";
        return (
          <button key={c.name} className="sp-row click" onClick={() => app.set({ settingsOpen: true, settingsTab: "Connectors" })} title={state === "error" ? m?.error : `${c.command} ${(c.args || []).join(" ")}`}>
            <span className="sp-conn-ico">
              <Icon name={p?.icon || "plug"} size={13} />
            </span>
            <span className="sp-label ellipsis">{p?.name || c.name}</span>
            <span className="sp-meta">{state === "on" ? `${m!.tools.length} tools` : state === "off" ? "off" : ""}</span>
            <span className={`status-dot ${state === "on" ? "ready" : state === "error" ? "error" : "empty"}`} />
          </button>
        );
      })}
      {cfgs.length > 4 && (
        <button className="sp-more" onClick={() => setAll(!all)}>
          {all ? "Show less" : "Show more"}
        </button>
      )}
    </Section>
  );
}

function Outputs() {
  const msgs = useApp((s) => s.chat?.messages);
  const archive = useApp((s) => s.chat?.archive);
  const folder = useApp((s) => s.chat?.folder ?? null);
  const files = useMemo(() => {
    const m = new Map<string, { path: string; after: string; created: boolean }>();
    for (const x of [...(archive || []), ...(msgs || [])] as Msg[]) {
      if (x.role !== "tool" || (x.ui.kind !== "write" && x.ui.kind !== "edit") || !x.ui.path || x.ui.after === undefined) continue;
      const had = m.get(x.ui.path);
      m.delete(x.ui.path);
      m.set(x.ui.path, { path: x.ui.path, after: x.ui.after, created: had ? had.created : x.ui.kind === "write" && x.ui.before === null });
    }
    return [...m.values()].reverse();
  }, [msgs, archive]);
  return (
    <Section id="out" title="Outputs">
      {!files.length && <div className="sp-empty">Files Pelica creates or edits in this chat show up here.</div>}
      {files.map((f) => {
        const name = f.path.split("/").pop() || f.path;
        return (
          <div key={f.path} className="sp-row click sp-file" title={f.path}>
            <button className="sp-file-open" onClick={() => openArtifact(name, f.after, (name.split(".").pop() || "").toLowerCase())}>
              <Icon name="code" size={14} />
              <span className="sp-label ellipsis">{name}</span>
              {f.created && <span className="sp-new">new</span>}
            </button>
            {folder && bridge && (
              <button className="sp-link" title="Show in Explorer" onClick={() => bridge!.fs.reveal(folder, f.path)}>
                <Icon name="external" size={12} />
              </button>
            )}
          </div>
        );
      })}
    </Section>
  );
}

export function SidePanel() {
  return (
    <aside className="side-panel">
      <Environment />
      <Context />
      <Connectors />
      <Outputs />
    </aside>
  );
}
