"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.watch = watch;
exports.startOne = startOne;
exports.launchers = launchers;
exports.startAll = startAll;
exports.stopOne = stopOne;
exports.stopAll = stopAll;
exports.list = list;
exports.call = call;
exports.fromConfig = fromConfig;
exports.otherApps = otherApps;
// A small Model Context Protocol client for the servers the user lists in
// Settings → Connectors: programs on this PC spoken to over stdio in
// newline-delimited JSON-RPC, and remote servers over HTTP (Streamable HTTP,
// with the older HTTP+SSE transport as a fallback). initialize, tools/list,
// tools/call, and notifications/tools/list_changed (the tool list refreshes).
const node_child_process_1 = require("node:child_process");
const node_fs_1 = __importDefault(require("node:fs"));
const node_os_1 = __importDefault(require("node:os"));
const node_path_1 = __importDefault(require("node:path"));
// cmd.exe joins the arguments as they are: quote the ones with spaces
const arg = (a) => (process.platform === "win32" && /[\s&|<>^()]/.test(a) && !/^".*"$/.test(a) ? `"${a.replace(/"/g, '\\"')}"` : a);
const PROTOCOL = "2025-06-18";
const CLIENT = { name: "pelica", version: "0.14.0" };
/** What both transports share: ids, waiting requests, and what the server tells us. */
class Server {
    cfg;
    changed;
    tools = [];
    error = "";
    log = "";
    remote = "";
    id = 0;
    waiting = new Map();
    constructor(cfg, changed) {
        this.cfg = cfg;
        this.changed = changed;
    }
    addLog(s) {
        this.log = (this.log + s).slice(-4000);
    }
    /** a message from the server: a reply, or a notification */
    receive(m) {
        if (m.id !== undefined && (m.result !== undefined || m.error !== undefined)) {
            const w = this.waiting.get(m.id);
            if (!w)
                return;
            this.waiting.delete(m.id);
            m.error ? w.bad(new Error(m.error.message || "MCP error")) : w.ok(m.result);
        }
        else if (m.method === "notifications/tools/list_changed") {
            this.listTools().then(this.changed, () => { });
        }
        else if (m.method === "ping" && m.id !== undefined) {
            this.send({ jsonrpc: "2.0", id: m.id, result: {} }).catch(() => { });
        }
        else if (m.method && m.id !== undefined) {
            // sampling, roots, elicitation: not offered (no capabilities declared)
            this.send({ jsonrpc: "2.0", id: m.id, error: { code: -32601, message: "not supported by Pelica" } }).catch(() => { });
        }
    }
    failAll(why) {
        for (const w of this.waiting.values())
            w.bad(new Error(why));
        this.waiting.clear();
    }
    async start() {
        await this.open();
        // (the first start of an npx/uvx server downloads it: give it time)
        try {
            await this.request("initialize", { protocolVersion: PROTOCOL, capabilities: {}, clientInfo: CLIENT }, this.cfg.url ? 30000 : 150000);
        }
        catch (e) {
            const last = this.log.trim().split(/\r?\n/).filter(Boolean).pop();
            throw new Error(last && !e.message.includes(last) ? `${e.message}: ${last.slice(0, 240)}` : e.message);
        }
        await this.notify("notifications/initialized", {});
        await this.listTools();
        this.error = "";
    }
    async listTools() {
        const all = [];
        let cursor;
        do {
            const r = await this.request("tools/list", cursor ? { cursor } : {}, 20000);
            all.push(...(r?.tools || []));
            cursor = r?.nextCursor;
        } while (cursor && all.length < 500);
        this.tools = all;
    }
    request(method, params, timeout = 120000) {
        if (!this.running)
            return Promise.reject(new Error(`MCP server ${this.cfg.name} is not running`));
        const id = ++this.id;
        return new Promise((ok, bad) => {
            this.waiting.set(id, { ok, bad });
            setTimeout(() => {
                if (this.waiting.delete(id))
                    bad(new Error(`MCP ${method} timed out`));
            }, timeout);
            this.send({ jsonrpc: "2.0", id, method, params }, id).catch((e) => {
                if (this.waiting.delete(id))
                    bad(e);
            });
        });
    }
    notify(method, params) {
        return this.send({ jsonrpc: "2.0", method, params }).catch(() => { });
    }
}
/** A program on this PC, over stdin/stdout. */
class StdioServer extends Server {
    proc = null;
    buf = "";
    get running() {
        return !!this.proc;
    }
    async open() {
        this.proc = (0, node_child_process_1.spawn)(arg(this.cfg.command), (this.cfg.args || []).map(arg), { env: { ...process.env, ...(this.cfg.env || {}) }, stdio: ["pipe", "pipe", "pipe"], shell: process.platform === "win32", windowsHide: true });
        this.proc.stdout.setEncoding("utf8");
        this.proc.stdout.on("data", (d) => {
            this.buf += d;
            let i;
            while ((i = this.buf.indexOf("\n")) >= 0) {
                const line = this.buf.slice(0, i).trim();
                this.buf = this.buf.slice(i + 1);
                if (!line)
                    continue;
                try {
                    this.receive(JSON.parse(line));
                }
                catch { }
            }
        });
        this.proc.stderr.setEncoding("utf8");
        this.proc.stderr.on("data", (d) => this.addLog(d));
        this.proc.on("exit", (code) => {
            this.error = `exited (${code})`;
            this.failAll(`MCP server ${this.cfg.name} exited`);
            this.proc = null;
            this.changed();
        });
        this.proc.on("error", (e) => (this.error = e.message));
    }
    async send(msg) {
        if (!this.proc)
            throw new Error(`MCP server ${this.cfg.name} is not running`);
        this.proc.stdin.write(JSON.stringify(msg) + "\n");
    }
    stop() {
        this.proc?.kill();
        this.proc = null;
    }
}
/** Reads a text/event-stream body and hands over each event. */
async function events(body, on) {
    const reader = body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
        const { done, value } = await reader.read();
        if (done)
            break;
        buf += dec.decode(value, { stream: true }).replace(/\r\n?/g, "\n");
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
            const block = buf.slice(0, i);
            buf = buf.slice(i + 2);
            let event = "message";
            const data = [];
            for (const l of block.split("\n")) {
                if (l.startsWith("event:"))
                    event = l.slice(6).trim();
                else if (l.startsWith("data:"))
                    data.push(l.slice(5).replace(/^ /, ""));
            }
            if (data.length)
                on(event, data.join("\n"));
        }
    }
}
/** A remote server: Streamable HTTP (one POST per message; replies as JSON or
 *  an event stream), or, when the server answers that with 4xx, the older
 *  transport (a GET event stream that names where to POST). */
class HttpServer extends Server {
    live = false;
    session = "";
    legacy = null;
    aborts = new Set();
    version = "";
    get running() {
        return this.live;
    }
    headers(extra = {}) {
        const h = { ...(this.cfg.headers || {}), ...extra };
        if (this.session)
            h["Mcp-Session-Id"] = this.session;
        if (this.version)
            h["MCP-Protocol-Version"] = this.version;
        return h;
    }
    async open() {
        this.live = true;
        this.remote = new URL(this.cfg.url).host;
    }
    async start() {
        try {
            await super.start();
        }
        catch (e) {
            // an older server: a 4xx to the POST means "use the event stream"
            if (!this.legacy && /^HTTP 4(04|05|00)\b/.test(e.message)) {
                await this.openLegacy();
                return super.start();
            }
            throw e;
        }
    }
    async request(method, params, timeout) {
        const r = await super.request(method, params, timeout);
        if (method === "initialize")
            this.version = r?.protocolVersion || PROTOCOL;
        return r;
    }
    async openLegacy() {
        const abort = new AbortController();
        const res = await fetch(this.cfg.url, { headers: this.headers({ Accept: "text/event-stream" }), signal: abort.signal });
        if (!res.ok || !res.body)
            throw new Error(`HTTP ${res.status} ${res.statusText}`.trim());
        let found = () => { };
        const endpoint = new Promise((ok, bad) => {
            found = ok;
            setTimeout(() => bad(new Error("the server sent no endpoint")), 15000);
        });
        events(res.body, (ev, data) => {
            if (ev === "endpoint")
                found(new URL(data, this.cfg.url).href);
            else
                try {
                    this.receive(JSON.parse(data));
                }
                catch { }
        })
            .catch(() => { })
            .finally(() => {
            if (this.legacy?.abort === abort) {
                this.live = false;
                this.error = "the connection closed";
                this.failAll(`MCP server ${this.cfg.name} closed the connection`);
                this.changed();
            }
        });
        this.legacy = { post: await endpoint, abort };
    }
    async send(msg, id) {
        if (this.legacy) {
            const res = await fetch(this.legacy.post, { method: "POST", headers: this.headers({ "Content-Type": "application/json" }), body: JSON.stringify(msg) });
            if (!res.ok)
                throw new Error(`HTTP ${res.status} ${res.statusText}`.trim());
            return;
        }
        const abort = new AbortController();
        this.aborts.add(abort);
        try {
            const res = await fetch(this.cfg.url, { method: "POST", headers: this.headers({ "Content-Type": "application/json", Accept: "application/json, text/event-stream" }), body: JSON.stringify(msg), signal: abort.signal });
            const sid = res.headers.get("mcp-session-id");
            if (sid)
                this.session = sid;
            if (res.status === 404 && this.session && msg.method !== "initialize") {
                this.session = "";
                throw new Error("the server ended the session; restart the connector");
            }
            if (!res.ok) {
                const text = (await res.text().catch(() => "")).slice(0, 200).replace(/\s+/g, " ");
                const hint = res.status === 401 || res.status === 403 ? " (the server wants a sign-in: add its token under Headers, e.g. Authorization: Bearer …)" : "";
                throw new Error(`HTTP ${res.status} ${res.statusText}${hint}${text && !hint ? `: ${text}` : ""}`.trim());
            }
            if (id === undefined || res.status === 202)
                return;
            const type = res.headers.get("content-type") || "";
            if (type.includes("text/event-stream") && res.body) {
                await events(res.body, (_ev, data) => {
                    try {
                        this.receive(JSON.parse(data));
                    }
                    catch { }
                });
            }
            else {
                const j = await res.json();
                for (const m of Array.isArray(j) ? j : [j])
                    this.receive(m);
            }
        }
        finally {
            this.aborts.delete(abort);
        }
    }
    stop() {
        this.live = false;
        const l = this.legacy;
        this.legacy = null;
        l?.abort.abort();
        for (const a of this.aborts)
            a.abort();
        if (this.session && !l)
            fetch(this.cfg.url, { method: "DELETE", headers: this.headers() }).catch(() => { });
        this.failAll(`MCP server ${this.cfg.name} stopped`);
    }
}
const servers = new Map();
let onChange = () => { };
/** Called when a server's tools change by themselves (or it exits). */
function watch(fn) {
    onChange = fn;
}
const usable = (c) => !c.disabled && !!c.name && !!(c.command || c.url);
async function launch(c) {
    const s = c.url ? new HttpServer(c, () => onChange()) : new StdioServer(c, () => onChange());
    servers.set(c.name, s);
    try {
        await s.start();
    }
    catch (e) {
        s.error = e.message;
        s.stop();
    }
    onChange();
}
/** Starts (or restarts) one server. */
async function startOne(c) {
    servers.get(c.name)?.stop();
    servers.delete(c.name);
    if (usable(c))
        await launch(c);
}
/** Which launchers are installed (npx comes with Node.js, uvx with uv). */
function launchers() {
    const has = (cmd) => new Promise((ok) => (0, node_child_process_1.execFile)("where", [cmd], { windowsHide: true }, (err) => ok(!err)));
    return Promise.all([has("npx"), has("uvx"), has("docker")]).then(([npx, uvx, docker]) => ({ npx, uvx, docker }));
}
async function startAll(cfgs) {
    stopAll();
    await Promise.all(cfgs.filter(usable).map(launch));
}
function stopOne(name) {
    servers.get(name)?.stop();
    servers.delete(name);
}
function stopAll() {
    for (const s of servers.values())
        s.stop();
    servers.clear();
}
function list() {
    return [...servers.values()].map((s) => ({ name: s.cfg.name, running: s.running, error: s.error, log: s.log, tools: s.tools, remote: s.remote }));
}
async function call(server, tool, args) {
    const s = servers.get(server);
    if (!s)
        throw new Error(`no MCP server called ${server}`);
    const r = await s.request("tools/call", { name: tool, arguments: args || {} });
    const parts = [];
    for (const c of r?.content || []) {
        if (c.type === "text")
            parts.push(c.text);
        else if (c.type === "resource")
            parts.push(c.resource?.text || `[resource ${c.resource?.uri}]`);
        else if (c.type === "resource_link")
            parts.push(`[${c.name || "resource"}: ${c.uri}]`);
        else
            parts.push(`[${c.type} content]`);
    }
    const text = parts.join("\n") || JSON.stringify(r?.structuredContent ?? r ?? null);
    return r?.isError ? `Error: ${text}` : text;
}
// ---------- importing from other apps ----------
/** The usual `{ "mcpServers": { name: { command, args, env } | { url, headers } } }`
 *  (Claude Desktop, Claude Code, Cursor, VS Code's `servers`) as Pelica's list. */
function fromConfig(j) {
    const m = j?.mcpServers ?? j?.servers ?? j?.mcp?.servers ?? (Array.isArray(j) ? null : j);
    if (Array.isArray(j))
        return j.filter((c) => c && c.name && (c.command || c.url));
    if (!m || typeof m !== "object")
        return [];
    const out = [];
    for (const [name, v] of Object.entries(m)) {
        if (!v || typeof v !== "object")
            continue;
        const n = name.trim().replace(/\s+/g, "-");
        if (typeof v.url === "string" || typeof v.serverUrl === "string")
            out.push({ name: n, url: v.url || v.serverUrl, headers: v.headers && typeof v.headers === "object" ? v.headers : undefined, disabled: v.disabled || undefined });
        else if (typeof v.command === "string")
            out.push({ name: n, command: v.command, args: Array.isArray(v.args) ? v.args.map(String) : undefined, env: v.env && typeof v.env === "object" ? v.env : undefined, disabled: v.disabled || undefined });
    }
    return out;
}
/** Other apps on this PC whose MCP servers can be brought over. */
function otherApps() {
    const home = node_os_1.default.homedir();
    const roaming = process.env.APPDATA || node_path_1.default.join(home, "AppData", "Roaming");
    const local = process.env.LOCALAPPDATA || node_path_1.default.join(home, "AppData", "Local");
    const places = [
        ["Claude Desktop", node_path_1.default.join(roaming, "Claude", "claude_desktop_config.json")],
        ["Claude Code", node_path_1.default.join(home, ".claude.json")],
        ["Cursor", node_path_1.default.join(home, ".cursor", "mcp.json")],
        ["Windsurf", node_path_1.default.join(home, ".codeium", "windsurf", "mcp_config.json")],
        ["VS Code", node_path_1.default.join(roaming, "Code", "User", "mcp.json")],
    ];
    // the Microsoft Store Claude keeps its files in its package folder
    try {
        for (const d of node_fs_1.default.readdirSync(node_path_1.default.join(local, "Packages")))
            if (/^Claude_/i.test(d))
                places.splice(1, 0, ["Claude Desktop", node_path_1.default.join(local, "Packages", d, "LocalCache", "Roaming", "Claude", "claude_desktop_config.json")]);
    }
    catch { }
    const out = [];
    for (const [app, file] of places) {
        try {
            const servers = fromConfig(JSON.parse(node_fs_1.default.readFileSync(file, "utf8")));
            if (servers.length)
                out.push({ app, file, servers });
        }
        catch { }
    }
    return out;
}
