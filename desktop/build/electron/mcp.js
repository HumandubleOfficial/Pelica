"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.startOne = startOne;
exports.launchers = launchers;
exports.startAll = startAll;
exports.stopOne = stopOne;
exports.stopAll = stopAll;
exports.list = list;
exports.call = call;
// A small Model Context Protocol client: stdio servers the user lists in
// Settings, spoken to in newline-delimited JSON-RPC (initialize, tools/list,
// tools/call).
const node_child_process_1 = require("node:child_process");
// cmd.exe joins the arguments as they are: quote the ones with spaces
const arg = (a) => (process.platform === "win32" && /[\s&|<>^()]/.test(a) && !/^".*"$/.test(a) ? `"${a.replace(/"/g, '\\"')}"` : a);
class Server {
    cfg;
    proc = null;
    tools = [];
    error = "";
    log = "";
    id = 0;
    buf = "";
    waiting = new Map();
    constructor(cfg) {
        this.cfg = cfg;
    }
    async start() {
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
                    const m = JSON.parse(line);
                    const w = m.id !== undefined ? this.waiting.get(m.id) : undefined;
                    if (w) {
                        this.waiting.delete(m.id);
                        m.error ? w.bad(new Error(m.error.message || "MCP error")) : w.ok(m.result);
                    }
                }
                catch { }
            }
        });
        this.proc.stderr.setEncoding("utf8");
        this.proc.stderr.on("data", (d) => (this.log = (this.log + d).slice(-4000)));
        this.proc.on("exit", (code) => {
            this.error = `exited (${code})`;
            for (const w of this.waiting.values())
                w.bad(new Error(`MCP server ${this.cfg.name} exited`));
            this.waiting.clear();
            this.proc = null;
        });
        this.proc.on("error", (e) => (this.error = e.message));
        // (the first start of an npx/uvx server downloads it: give it time)
        try {
            await this.request("initialize", { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "pelica", version: "0.2.0" } }, 150000);
        }
        catch (e) {
            const last = this.log.trim().split(/\r?\n/).filter(Boolean).pop();
            throw new Error(last ? `${e.message}: ${last.slice(0, 240)}` : e.message);
        }
        this.notify("notifications/initialized", {});
        const r = await this.request("tools/list", {}, 20000);
        this.tools = r?.tools || [];
        this.error = "";
    }
    request(method, params, timeout = 120000) {
        if (!this.proc)
            return Promise.reject(new Error(`MCP server ${this.cfg.name} is not running`));
        const id = ++this.id;
        this.proc.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
        return new Promise((ok, bad) => {
            this.waiting.set(id, { ok, bad });
            setTimeout(() => {
                if (this.waiting.delete(id))
                    bad(new Error(`MCP ${method} timed out`));
            }, timeout);
        });
    }
    notify(method, params) {
        this.proc?.stdin.write(JSON.stringify({ jsonrpc: "2.0", method, params }) + "\n");
    }
    stop() {
        this.proc?.kill();
        this.proc = null;
    }
}
const servers = new Map();
/** Starts (or restarts) one server. */
async function startOne(c) {
    servers.get(c.name)?.stop();
    servers.delete(c.name);
    if (c.disabled || !c.name || !c.command)
        return;
    const s = new Server(c);
    servers.set(c.name, s);
    try {
        await s.start();
    }
    catch (e) {
        s.error = e.message;
    }
}
/** Which launchers are installed (npx comes with Node.js, uvx with uv). */
function launchers() {
    const has = (cmd) => new Promise((ok) => (0, node_child_process_1.execFile)("where", [cmd], { windowsHide: true }, (err) => ok(!err)));
    return Promise.all([has("npx"), has("uvx"), has("docker")]).then(([npx, uvx, docker]) => ({ npx, uvx, docker }));
}
async function startAll(cfgs) {
    stopAll();
    await Promise.all(cfgs
        .filter((c) => !c.disabled && c.name && c.command)
        .map(async (c) => {
        const s = new Server(c);
        servers.set(c.name, s);
        try {
            await s.start();
        }
        catch (e) {
            s.error = e.message;
        }
    }));
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
    return [...servers.values()].map((s) => ({ name: s.cfg.name, running: !!s.proc, error: s.error, log: s.log, tools: s.tools }));
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
        else
            parts.push(`[${c.type} content]`);
    }
    const text = parts.join("\n") || JSON.stringify(r?.structuredContent ?? r ?? null);
    return r?.isError ? `Error: ${text}` : text;
}
