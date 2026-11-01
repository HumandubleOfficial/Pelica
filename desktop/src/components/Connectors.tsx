// Connectors (MCP servers) without writing JSON: a gallery of well-known
// servers to add in one click (with the one or two fields they need), remote
// servers by address, the servers other apps on this PC already have
// (Claude Desktop, Cursor…), a form for any other, and each server's state,
// log and tools, each tool allowed, asked for, or off. The JSON is still
// there under Advanced.
import { useEffect, useState } from "react";
import { toolPerm } from "../agent/tools";
import { bridge, type McpConfig, type McpServer, type ToolPerm } from "../lib/bridge";
import { app, setSettings, toast, useApp } from "../state";
import { Icon } from "./Icons";

type Cfg = McpConfig;

interface Preset {
  id: string;
  name: string;
  desc: string;
  icon: string;
  launcher: "npx" | "uvx" | "remote";
  args?: string[];
  /** a remote server's address */
  url?: string;
  /** a field the user fills in: an argument (folders, a file) or an env variable (a key) */
  field?: { label: string; kind: "folders" | "file" | "env"; env?: string; placeholder?: string; help?: string };
}

const PRESETS: Preset[] = [
  { id: "filesystem", name: "Files", desc: "Read, write and search files in folders you pick", icon: "folder", launcher: "npx", args: ["-y", "@modelcontextprotocol/server-filesystem"], field: { label: "Folders", kind: "folders", help: "The server can only touch these folders." } },
  { id: "memory", name: "Memory", desc: "A knowledge graph the model can remember things in across chats", icon: "brain", launcher: "npx", args: ["-y", "@modelcontextprotocol/server-memory"] },
  { id: "thinking", name: "Sequential thinking", desc: "Step-by-step problem solving with revisions and branches", icon: "list", launcher: "npx", args: ["-y", "@modelcontextprotocol/server-sequential-thinking"] },
  { id: "fetch", name: "Fetch", desc: "Fetch web pages and turn them into Markdown", icon: "globe", launcher: "uvx", args: ["mcp-server-fetch"] },
  { id: "git", name: "Git", desc: "Status, diffs, logs and commits for a git repository", icon: "branch", launcher: "uvx", args: ["mcp-server-git", "--repository"], field: { label: "Repository folder", kind: "folders", help: "One git repository." } },
  { id: "time", name: "Time", desc: "Current time and time-zone conversions", icon: "clock", launcher: "uvx", args: ["mcp-server-time"] },
  { id: "sqlite", name: "SQLite", desc: "Query and change a SQLite database", icon: "layers", launcher: "uvx", args: ["mcp-server-sqlite", "--db-path"], field: { label: "Database file", kind: "file", placeholder: "C:\\data\\app.db" } },
  { id: "playwright", name: "Playwright browser", desc: "Drive a real browser: open pages, click, fill forms, screenshots", icon: "window", launcher: "npx", args: ["-y", "@playwright/mcp@latest"] },
  { id: "github", name: "GitHub", desc: "Issues, pull requests, files and searches on GitHub", icon: "code", launcher: "npx", args: ["-y", "@modelcontextprotocol/server-github"], field: { label: "Personal access token", kind: "env", env: "GITHUB_PERSONAL_ACCESS_TOKEN", placeholder: "github_pat_…", help: "Create one at github.com → Settings → Developer settings. It stays on this PC." } },
  { id: "brave", name: "Brave Search", desc: "Web and local search through the Brave Search API", icon: "search", launcher: "npx", args: ["-y", "@modelcontextprotocol/server-brave-search"], field: { label: "API key", kind: "env", env: "BRAVE_API_KEY", placeholder: "BSA…", help: "From brave.com/search/api (has a free tier)." } },
  { id: "deepwiki", name: "DeepWiki", desc: "Ask about any public GitHub repository: its code, docs and design (online, no account)", icon: "library", launcher: "remote", url: "https://mcp.deepwiki.com/mcp" },
  { id: "context7", name: "Context7", desc: "Current docs and code examples for thousands of libraries (online, no account)", icon: "file", launcher: "remote", url: "https://mcp.context7.com/mcp" },
];

const LAUNCHER_HELP: Record<string, string> = {
  npx: "needs Node.js (nodejs.org)",
  uvx: "needs uv (docs.astral.sh/uv)",
};

export const presetOf = (c: Cfg) => PRESETS.find((p) => (p.url ? c.url === p.url : c.command === p.launcher && p.args!.every((a, i) => c.args?.[i] === a)));

function save(list: Cfg[]) {
  setSettings({ mcpServers: list });
}

async function restart(name: string) {
  if (!bridge) return;
  app.set({ mcp: await bridge.mcp.restart(name) });
}

const freeName = (servers: Cfg[], base: string) => {
  let name = base;
  for (let i = 2; servers.some((s) => s.name === name); i++) name = `${base}-${i}`;
  return name;
};

/** "NAME=value" / "Name: value" lines as a map */
function pairs(text: string, sep: "=" | ":"): Record<string, string> | undefined {
  const out: Record<string, string> = {};
  for (const l of text.split("\n")) {
    const i = l.indexOf(sep);
    if (i > 0) out[l.slice(0, i).trim()] = l.slice(i + 1).trim();
  }
  return Object.keys(out).length ? out : undefined;
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
    const name = freeName(servers, p.id);
    const extra = f && f.kind !== "env" ? value.split(";").map((x) => x.trim()).filter(Boolean) : [];
    const cfg: Cfg = p.url ? { name, url: p.url } : { name, command: p.launcher, args: [...p.args!, ...extra], env: f?.kind === "env" ? { [f.env!]: value.trim() } : undefined };
    save([...servers, cfg]);
    setBusy(true);
    await restart(name);
    setBusy(false);
    const live = app.get().mcp.find((m) => m.name === name);
    toast(live?.running ? `${p.name} added` : `${p.name} added, but it didn't start: ${live?.error || "see its log"}`);
    onDone();
  }
  return (
    <div className="conn-add">
      {p.url && <small className="muted">Connects to {new URL(p.url).host}. What the model asks it is sent there.</small>}
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
          {busy ? (p.url ? "Connecting…" : "Starting… (the first start downloads it)") : `Add ${p.name}`}
        </button>
      </div>
    </div>
  );
}

function CustomForm({ init, onDone }: { init?: Cfg; onDone: () => void }) {
  const servers = useApp((s) => s.settings.mcpServers);
  const [kind, setKind] = useState<"local" | "remote">(init?.url ? "remote" : "local");
  const [name, setName] = useState(init?.name || "");
  const [command, setCommand] = useState(init?.command || "");
  const [args, setArgs] = useState((init?.args || []).join("\n"));
  const [env, setEnv] = useState(Object.entries(init?.env || {}).map(([k, v]) => `${k}=${v}`).join("\n"));
  const [url, setUrl] = useState(init?.url || "");
  const [headers, setHeaders] = useState(Object.entries(init?.headers || {}).map(([k, v]) => `${k}: ${v}`).join("\n"));
  const [busy, setBusy] = useState(false);
  async function ok() {
    const n = name.trim().replace(/\s+/g, "-");
    if (!n) return toast("A connector needs a name.");
    if (servers.some((s) => s.name === n && s !== init)) return toast(`There's already a connector called ${n}.`);
    let cfg: Cfg;
    if (kind === "remote") {
      if (!/^https?:\/\/\S+$/i.test(url.trim())) return toast("A remote connector needs its address: https://…");
      cfg = { name: n, url: url.trim(), headers: pairs(headers, ":"), disabled: init?.disabled, tools: init?.tools };
    } else {
      if (!command.trim()) return toast("A connector on this PC needs a command.");
      cfg = { name: n, command: command.trim(), args: args.split("\n").map((x) => x.trim()).filter(Boolean), env: pairs(env, "="), disabled: init?.disabled, tools: init?.tools };
    }
    save(init ? servers.map((s) => (s === init ? cfg : s)) : [...servers, cfg]);
    setBusy(true);
    if (init && init.name !== n) await bridge?.mcp.restart(init.name);
    await restart(n);
    setBusy(false);
    onDone();
  }
  return (
    <div className="conn-form">
      <div className="seg conn-kind">
        <button className={kind === "local" ? "sel" : ""} onClick={() => setKind("local")}>
          On this PC
        </button>
        <button className={kind === "remote" ? "sel" : ""} onClick={() => setKind("remote")}>
          Remote (web address)
        </button>
      </div>
      <div className="conn-grid">
        <label>
          Name
          <input className="text-input" value={name} onChange={(e) => setName(e.target.value)} placeholder="my-server" />
        </label>
        {kind === "local" ? (
          <>
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
          </>
        ) : (
          <>
            <label>
              Address
              <input className="text-input mono" value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://example.com/mcp" />
            </label>
            <label className="conn-span">
              Headers <small className="muted">Name: value, one per line (for a token: Authorization: Bearer …)</small>
              <textarea className="text-input mono" rows={2} value={headers} onChange={(e) => setHeaders(e.target.value)} placeholder="Authorization: Bearer …" spellCheck={false} />
            </label>
          </>
        )}
      </div>
      <div className="row-end">
        <button className="btn ghost small" onClick={onDone}>
          Cancel
        </button>
        <button className="btn primary small" onClick={ok} disabled={busy}>
          {busy ? "Connecting…" : init ? "Save and restart" : "Add connector"}
        </button>
      </div>
    </div>
  );
}

const PERMS: [ToolPerm, string, string][] = [
  ["allow", "Allow", "Runs without asking"],
  ["ask", "Ask", "Asks you first (unless the permission mode is Bypass)"],
  ["off", "Off", "Not offered to the model"],
];

/** A server's tools, each allowed, asked for, or off. */
function ToolList({ c, live }: { c: Cfg; live: McpServer }) {
  const servers = useApp((s) => s.settings.mcpServers);
  const set = (tool: string | null, v: ToolPerm | null) => {
    let tools: Cfg["tools"];
    if (tool === null) tools = v ? Object.fromEntries(live.tools.map((t) => [t.name, v])) : undefined;
    else {
      tools = { ...(c.tools || {}) };
      if (v) tools[tool] = v;
      else delete tools[tool];
    }
    save(servers.map((s) => (s === c ? { ...c, tools: tools && Object.keys(tools).length ? tools : undefined } : s)));
  };
  const off = live.tools.filter((t) => toolPerm(c, t) === "off").length;
  return (
    <div className="conn-tools-list">
      <div className="conn-tools-head">
        <span className="muted small">
          {live.tools.length} tool{live.tools.length === 1 ? "" : "s"}
          {off ? `, ${off} off` : ""}. Tools that say they only read are allowed by default; the rest ask first.
        </span>
        <div className="inline">
          <button className="btn ghost small" onClick={() => set(null, "allow")}>
            Allow all
          </button>
          <button className="btn ghost small" onClick={() => set(null, "off")}>
            All off
          </button>
          {c.tools && (
            <button className="btn ghost small" onClick={() => set(null, null)}>
              Reset
            </button>
          )}
        </div>
      </div>
      {live.tools.map((t) => {
        const p = toolPerm(c, t);
        return (
          <div key={t.name} className={`conn-tool ${p === "off" ? "off" : ""}`}>
            <div className="conn-tool-text">
              <span className="mono">{t.name}</span>
              {t.description && <span className="muted small">{t.description.split("\n")[0]}</span>}
            </div>
            <div className="seg conn-perm">
              {PERMS.map(([v, l, tip]) => (
                <button key={v} className={p === v ? "sel" : ""} title={tip} onClick={() => set(t.name, v)}>
                  {l}
                </button>
              ))}
            </div>
          </div>
        );
      })}
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
  const on = live?.running ? live.tools.filter((t) => toolPerm(c, t) !== "off").length : 0;
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
          <Icon name={p?.icon || (c.url ? "globe" : "plug")} size={16} />
        </span>
        <div className="conn-title">
          <b>{p?.name || c.name}</b>
          <span className="muted small">
            {state === "on" ? `${on} of ${live!.tools.length} tool${live!.tools.length === 1 ? "" : "s"} on${c.url ? ` · ${live!.remote || new URL(c.url).host}` : ""}` : state === "starting" ? "Starting…" : state === "off" ? "Off" : live?.error || "Not running"}
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
          <code className="conn-cmd">{c.url ? c.url : `${c.command} ${(c.args || []).join(" ")}`}</code>
          {live?.running && live.tools.length > 0 && <ToolList c={c} live={live} />}
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

/** The connectors other apps on this PC have, to bring over in one click. */
function FromOtherApps() {
  const servers = useApp((s) => s.settings.mcpServers);
  const [found, setFound] = useState<{ app: string; file: string; servers: Cfg[] }[]>([]);
  const [busy, setBusy] = useState("");
  useEffect(() => {
    bridge?.mcp.others().then(setFound).catch(() => {});
  }, []);
  if (!found.length) return null;
  const isNew = (x: Cfg) => !servers.some((s) => s.name === x.name || (x.url ? s.url === x.url : s.command === x.command && (s.args || []).join(" ") === (x.args || []).join(" ")));
  async function bring(f: (typeof found)[number]) {
    const add = f.servers.filter(isNew);
    if (!add.length) return;
    setBusy(f.file);
    save([...servers, ...add]);
    if (bridge) app.set({ mcp: await bridge.mcp.start() });
    setBusy("");
    toast(`Brought over ${add.length} connector${add.length === 1 ? "" : "s"} from ${f.app}`);
  }
  return (
    <>
      <div className="set-title">From your other apps</div>
      <div className="conn-list">
        {found.map((f) => {
          const add = f.servers.filter(isNew);
          return (
            <div key={f.file} className="conn-card">
              <div className="conn-head">
                <span className="conn-ico">
                  <Icon name="download" size={16} />
                </span>
                <div className="conn-title">
                  <b>{f.app}</b>
                  <span className="muted small" title={f.file}>
                    {f.servers.map((x) => x.name).join(", ")}
                  </span>
                </div>
                <button className="btn small" disabled={!add.length || !!busy} onClick={() => bring(f)}>
                  {busy === f.file ? "Starting…" : add.length ? `Bring over ${add.length}` : "All here"}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </>
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
      // Pelica's list, or the { "mcpServers": { … } } other apps use
      const v = bridge ? await bridge.mcp.parse(json || "[]") : JSON.parse(json || "[]");
      if (!Array.isArray(v)) throw new Error("a list of servers: [{ name, command, args }] or { \"mcpServers\": { … } }");
      // keep each tool's setting for servers that stay
      save(v.map((c: Cfg) => ({ ...c, tools: c.tools ?? s.mcpServers.find((o) => o.name === c.name)?.tools })));
      setErr("");
      if (bridge) app.set({ mcp: await bridge.mcp.start() });
      setJson(null);
      toast("Connectors restarted");
    } catch (e: any) {
      setErr(String(e.message || e).replace(/^Error invoking remote method '[^']+': (\w*Error: )?/, ""));
    }
  }
  return (
    <>
      <p className="set-desc">Connectors (MCP servers) give every model more tools: files, memory, a browser, GitHub, docs and more. They run on this PC or online, and their tools show up in chats right away. Tools that can change things ask you first.</p>
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
      <FromOtherApps />
      <div className="set-title">Add a connector</div>
      <div className="conn-gallery">
        {PRESETS.map((p) => {
          const missing = p.launcher !== "remote" && has && !has[p.launcher];
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
            <Icon name="plus" size={13} /> Custom connector…
          </button>
        </div>
      )}
      {json !== null && (
        <>
          <p className="set-note">Pelica's list, or a config from another app ({`{ "mcpServers": { … } }`}) pasted as it is.</p>
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
