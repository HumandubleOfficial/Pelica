// Connectors (MCP servers) without writing JSON: a gallery of well-known
// servers to add in one click (with the one or two fields they need), a form
// for any other, and each server's state, tools and log. The JSON is still
// there under Advanced.
import { useEffect, useState } from "react";
import { bridge, type McpServer } from "../lib/bridge";
import { app, setSettings, toast, useApp, type Settings } from "../state";
import { Icon } from "./Icons";

type Cfg = Settings["mcpServers"][number];

interface Preset {
  id: string;
  name: string;
  desc: string;
  icon: string;
  launcher: "npx" | "uvx";
  args: string[];
  /** a field the user fills in: an argument (folders, a file) or an env variable (a key) */
  field?: { label: string; kind: "folders" | "file" | "env"; env?: string; placeholder?: string; help?: string };
}

const PRESETS: Preset[] = [
  { id: "filesystem", name: "Files", desc: "Read, write and search files in folders you pick", icon: "folder", launcher: "npx", args: ["-y", "@modelcontextprotocol/server-filesystem"], field: { label: "Folders", kind: "folders", help: "The server can only touch these folders." } },
  { id: "memory", name: "Memory", desc: "A knowledge graph the model can remember things in across chats", icon: "brain", launcher: "npx", args: ["-y", "@modelcontextprotocol/server-memory"] },
  { id: "thinking", name: "Sequential thinking", desc: "Step-by-step problem solving with revisions and branches", icon: "list", launcher: "npx", args: ["-y", "@modelcontextprotocol/server-sequential-thinking"] },
  { id: "fetch", name: "Fetch", desc: "Fetch web pages and turn them into Markdown", icon: "globe", launcher: "uvx", args: ["mcp-server-fetch"] },
  { id: "git", name: "Git", desc: "Status, diffs, logs and commits for a git repository", icon: "code", launcher: "uvx", args: ["mcp-server-git", "--repository"], field: { label: "Repository folder", kind: "folders", help: "One git repository." } },
  { id: "time", name: "Time", desc: "Current time and time-zone conversions", icon: "gauge", launcher: "uvx", args: ["mcp-server-time"] },
  { id: "sqlite", name: "SQLite", desc: "Query and change a SQLite database", icon: "layers", launcher: "uvx", args: ["mcp-server-sqlite", "--db-path"], field: { label: "Database file", kind: "file", placeholder: "C:\\data\\app.db" } },
  { id: "playwright", name: "Playwright browser", desc: "Drive a real browser: open pages, click, fill forms, screenshots", icon: "window", launcher: "npx", args: ["-y", "@playwright/mcp@latest"] },
  { id: "github", name: "GitHub", desc: "Issues, pull requests, files and searches on GitHub", icon: "code", launcher: "npx", args: ["-y", "@modelcontextprotocol/server-github"], field: { label: "Personal access token", kind: "env", env: "GITHUB_PERSONAL_ACCESS_TOKEN", placeholder: "github_pat_…", help: "Create one at github.com → Settings → Developer settings. It stays on this PC." } },
  { id: "brave", name: "Brave Search", desc: "Web and local search through the Brave Search API", icon: "search", launcher: "npx", args: ["-y", "@modelcontextprotocol/server-brave-search"], field: { label: "API key", kind: "env", env: "BRAVE_API_KEY", placeholder: "BSA…", help: "From brave.com/search/api (has a free tier)." } },
];

const LAUNCHER_HELP: Record<string, string> = {
  npx: "needs Node.js (nodejs.org)",
  uvx: "needs uv (docs.astral.sh/uv)",
};

export const presetOf = (c: Cfg) => PRESETS.find((p) => c.command === p.launcher && p.args.every((a, i) => c.args?.[i] === a));

function save(list: Cfg[]) {
  setSettings({ mcpServers: list });
}

async function restart(name: string) {
  if (!bridge) return;
  app.set({ mcp: await bridge.mcp.restart(name) });
}

function AddPreset({ p, onDone }: { p: Preset; onDone: () => void }) {
  const servers = useApp((s) => s.settings.mcpServers);
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  async function pick() {
    const d = await bridge?.folder.pick();
    if (d) setValue((v) => (p.field?.kind === "folders" && p.id === "filesystem" && v ? `${v}; ${d}` : d));
  }
  async function add() {
    const f = p.field;
    if (f && !value.trim()) return toast(`${p.name} needs: ${f.label.toLowerCase()}.`);
    let name = p.id;
    for (let i = 2; servers.some((s) => s.name === name); i++) name = `${p.id}-${i}`;
    const extra = f && f.kind !== "env" ? value.split(";").map((x) => x.trim()).filter(Boolean) : [];
    const cfg: Cfg = { name, command: p.launcher, args: [...p.args, ...extra], env: f?.kind === "env" ? { [f.env!]: value.trim() } : undefined };
    save([...servers, cfg]);
    setBusy(true);
    await restart(name);
    setBusy(false);
    toast(`${p.name} added`);
    onDone();
  }
  return (
    <div className="conn-add">
      {p.field && (
        <label className="conn-field">
          <span>{p.field.label}</span>
          <div className="inline">
            <input className="text-input wide" type={p.field.kind === "env" ? "password" : "text"} value={value} placeholder={p.field.placeholder || (p.field.kind === "folders" ? "C:\\Projects" : "")} onChange={(e) => setValue(e.target.value)} autoFocus />
            {p.field.kind === "folders" && (
              <button className="btn small" onClick={pick}>
                Choose…
              </button>
            )}
          </div>
          {p.field.help && <small className="muted">{p.field.help}</small>}
        </label>
      )}
      <div className="row-end">
        <button className="btn ghost small" onClick={onDone}>
          Cancel
        </button>
        <button className="btn primary small" onClick={add} disabled={busy}>
          {busy ? "Starting… (the first start downloads it)" : `Add ${p.name}`}
        </button>
      </div>
    </div>
  );
}

function CustomForm({ init, onDone }: { init?: Cfg; onDone: () => void }) {
  const servers = useApp((s) => s.settings.mcpServers);
  const [name, setName] = useState(init?.name || "");
  const [command, setCommand] = useState(init?.command || "");
  const [args, setArgs] = useState((init?.args || []).join("\n"));
  const [env, setEnv] = useState(Object.entries(init?.env || {}).map(([k, v]) => `${k}=${v}`).join("\n"));
  async function ok() {
    const n = name.trim().replace(/\s+/g, "-");
    if (!n || !command.trim()) return toast("A server needs a name and a command.");
    if (servers.some((s) => s.name === n && s !== init)) return toast(`There's already a server called ${n}.`);
    const e: Record<string, string> = {};
    for (const l of env.split("\n")) {
      const i = l.indexOf("=");
      if (i > 0) e[l.slice(0, i).trim()] = l.slice(i + 1).trim();
    }
    const cfg: Cfg = { name: n, command: command.trim(), args: args.split("\n").map((x) => x.trim()).filter(Boolean), env: Object.keys(e).length ? e : undefined, disabled: init?.disabled };
    save(init ? servers.map((s) => (s === init ? cfg : s)) : [...servers, cfg]);
    if (init && init.name !== n) await bridge?.mcp.restart(init.name);
    await restart(n);
    onDone();
  }
  return (
    <div className="conn-form">
      <div className="conn-grid">
        <label>
          Name
          <input className="text-input" value={name} onChange={(e) => setName(e.target.value)} placeholder="my-server" />
        </label>
        <label>
          Command
          <input className="text-input mono" value={command} onChange={(e) => setCommand(e.target.value)} placeholder="npx, uvx, node, python, C:\tools\server.exe" />
        </label>
        <label>
          Arguments <small className="muted">one per line</small>
          <textarea className="text-input mono" rows={3} value={args} onChange={(e) => setArgs(e.target.value)} placeholder={"-y\n@scope/some-mcp-server"} spellCheck={false} />
        </label>
        <label>
          Environment <small className="muted">NAME=value, one per line</small>
          <textarea className="text-input mono" rows={3} value={env} onChange={(e) => setEnv(e.target.value)} placeholder="API_KEY=…" spellCheck={false} />
        </label>
      </div>
      <div className="row-end">
        <button className="btn ghost small" onClick={onDone}>
          Cancel
        </button>
        <button className="btn primary small" onClick={ok}>
          {init ? "Save and restart" : "Add server"}
        </button>
      </div>
    </div>
  );
}

function ServerCard({ c, live }: { c: Cfg; live?: McpServer }) {
  const servers = useApp((s) => s.settings.mcpServers);
  const [open, setOpen] = useState(false);
  const [edit, setEdit] = useState(false);
  const [busy, setBusy] = useState(false);
  const p = presetOf(c);
  const state = c.disabled ? "off" : busy ? "starting" : live?.running ? "on" : live ? "error" : "off";
  async function again() {
    setBusy(true);
    await restart(c.name);
    setBusy(false);
  }
  async function toggle() {
    const n = { ...c, disabled: !c.disabled };
    save(servers.map((s) => (s === c ? n : s)));
    setBusy(!n.disabled);
    await restart(c.name);
    setBusy(false);
  }
  function remove() {
    save(servers.filter((s) => s !== c));
    restart(c.name);
  }
  return (
    <div className={`conn-card ${state}`}>
      <div className="conn-head">
        <span className="conn-ico">
          <Icon name={p?.icon || "plug"} size={16} />
        </span>
        <div className="conn-title">
          <b>{p?.name || c.name}</b>
          <span className="muted small">
            {state === "on" ? `${live!.tools.length} tool${live!.tools.length === 1 ? "" : "s"}` : state === "starting" ? "Starting…" : state === "off" ? "Off" : live?.error || "Not running"}
          </span>
        </div>
        <span className={`status-dot ${state === "on" ? "ready" : state === "starting" ? "loading" : state === "error" ? "error" : "empty"}`} />
        <button className={`toggle ${c.disabled ? "" : "on"}`} onClick={toggle} role="switch" aria-checked={!c.disabled} title={c.disabled ? "Turn on" : "Turn off"} />
        <button className="ghost-btn" title="Details" onClick={() => setOpen(!open)}>
          <Icon name="chevron" size={14} className={open ? "rot90" : ""} />
        </button>
      </div>
      {open && !edit && (
        <div className="conn-body">
          <code className="conn-cmd">
            {c.command} {(c.args || []).join(" ")}
          </code>
          {live?.tools.length ? (
            <div className="conn-tools">
              {live.tools.map((t) => (
                <span key={t.name} className="pill" title={t.description || ""}>
                  {t.name}
                </span>
              ))}
            </div>
          ) : null}
          {live?.log && <pre className="log-pre small">{live.log.slice(-1500)}</pre>}
          <div className="row-end">
            <button className="btn ghost small danger" onClick={remove}>
              <Icon name="trash" size={13} /> Remove
            </button>
            <button className="btn ghost small" onClick={() => setEdit(true)}>
              <Icon name="edit" size={13} /> Edit
            </button>
            <button className="btn small" onClick={again} disabled={busy || c.disabled}>
              <Icon name="reload" size={13} /> Restart
            </button>
          </div>
        </div>
      )}
      {open && edit && <CustomForm init={c} onDone={() => setEdit(false)} />}
    </div>
  );
}

export function ConnectorsTab() {
  const s = useApp((x) => x.settings);
  const live = useApp((x) => x.mcp);
  const [adding, setAdding] = useState<string | null>(null);
  const [custom, setCustom] = useState(false);
  const [has, setHas] = useState<Record<string, boolean> | null>(null);
  const [json, setJson] = useState<string | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    bridge?.mcp.launchers().then(setHas).catch(() => {});
    bridge?.mcp.list().then((mcp) => app.set({ mcp })).catch(() => {});
  }, []);
  const added = new Set(s.mcpServers.map((c) => presetOf(c)?.id).filter(Boolean));
  async function saveJson() {
    try {
      const v = JSON.parse(json || "[]");
      if (!Array.isArray(v)) throw new Error("a list of servers: [{ name, command, args }]");
      save(v);
      setErr("");
      if (bridge) app.set({ mcp: await bridge.mcp.start() });
      setJson(null);
      toast("Connectors restarted");
    } catch (e: any) {
      setErr(e.message);
    }
  }
  return (
    <>
      <p className="set-desc">Connectors (MCP servers) give every model more tools: files, memory, a browser, GitHub and more. They run on this PC and their tools show up in chats right away.</p>
      {s.mcpServers.length > 0 && (
        <>
          <div className="set-title">Your connectors</div>
          <div className="conn-list">
            {s.mcpServers.map((c) => (
              <ServerCard key={c.name} c={c} live={live.find((m) => m.name === c.name)} />
            ))}
          </div>
        </>
      )}
      <div className="set-title">Add a connector</div>
      <div className="conn-gallery">
        {PRESETS.map((p) => {
          const missing = has && !has[p.launcher];
          return (
            <div key={p.id} className={`conn-tile ${adding === p.id ? "open" : ""}`}>
              <button className="conn-tile-main" onClick={() => setAdding(adding === p.id ? null : p.id)} disabled={!!missing}>
                <span className="conn-ico">
                  <Icon name={p.icon} size={16} />
                </span>
                <span className="conn-tile-text">
                  <b>
                    {p.name} {added.has(p.id) && <span className="pill tiny">added</span>}
                  </b>
                  <span className="muted small">{missing ? `Can't add yet: ${LAUNCHER_HELP[p.launcher]}` : p.desc}</span>
                </span>
                {!missing && <Icon name="plus" size={14} />}
              </button>
              {adding === p.id && <AddPreset p={p} onDone={() => setAdding(null)} />}
            </div>
          );
        })}
      </div>
      {custom ? (
        <CustomForm onDone={() => setCustom(false)} />
      ) : (
        <div className="row-end">
          <button className="btn ghost small" onClick={() => setJson(json === null ? JSON.stringify(s.mcpServers, null, 2) : null)}>
            <Icon name="code" size={13} /> Advanced (JSON)
          </button>
          <button className="btn small" onClick={() => setCustom(true)}>
            <Icon name="plus" size={13} /> Custom server…
          </button>
        </div>
      )}
      {json !== null && (
        <>
          <textarea className="code-input" rows={9} value={json} onChange={(e) => setJson(e.target.value)} spellCheck={false} />
          {err && <div className="err-text">{err}</div>}
          <div className="row-end">
            <button className="btn" onClick={saveJson}>
              Save and restart all
            </button>
          </div>
        </>
      )}
    </>
  );
}
