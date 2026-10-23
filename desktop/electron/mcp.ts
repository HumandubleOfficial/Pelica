// A small Model Context Protocol client: stdio servers the user lists in
// Settings, spoken to in newline-delimited JSON-RPC (initialize, tools/list,
// tools/call).
import { spawn, execFile, ChildProcess } from "node:child_process";

// cmd.exe joins the arguments as they are: quote the ones with spaces
const arg = (a: string) => (process.platform === "win32" && /[\s&|<>^()]/.test(a) && !/^".*"$/.test(a) ? `"${a.replace(/"/g, '\\"')}"` : a);

export interface McpServerConfig {
  name: string;
  command: string;
  args?: string[];
  env?: Record<string, string>;
  disabled?: boolean;
}

export interface McpTool {
  name: string;
  description?: string;
  inputSchema?: unknown;
}

class Server {
  proc: ChildProcess | null = null;
  tools: McpTool[] = [];
  error = "";
  log = "";
  private id = 0;
  private buf = "";
  private waiting = new Map<number, { ok: (v: any) => void; bad: (e: Error) => void }>();

  constructor(public cfg: McpServerConfig) {}

  async start(): Promise<void> {
    this.proc = spawn(arg(this.cfg.command), (this.cfg.args || []).map(arg), { env: { ...process.env, ...(this.cfg.env || {}) }, stdio: ["pipe", "pipe", "pipe"], shell: process.platform === "win32", windowsHide: true });
    this.proc.stdout!.setEncoding("utf8");
    this.proc.stdout!.on("data", (d: string) => {
      this.buf += d;
      let i;
      while ((i = this.buf.indexOf("\n")) >= 0) {
        const line = this.buf.slice(0, i).trim();
        this.buf = this.buf.slice(i + 1);
        if (!line) continue;
        try {
          const m = JSON.parse(line);
          const w = m.id !== undefined ? this.waiting.get(m.id) : undefined;
          if (w) {
            this.waiting.delete(m.id);
            m.error ? w.bad(new Error(m.error.message || "MCP error")) : w.ok(m.result);
          }
        } catch {}
      }
    });
    this.proc.stderr!.setEncoding("utf8");
    this.proc.stderr!.on("data", (d: string) => (this.log = (this.log + d).slice(-4000)));
    this.proc.on("exit", (code) => {
      this.error = `exited (${code})`;
      for (const w of this.waiting.values()) w.bad(new Error(`MCP server ${this.cfg.name} exited`));
      this.waiting.clear();
      this.proc = null;
    });
    this.proc.on("error", (e) => (this.error = e.message));
    // (the first start of an npx/uvx server downloads it: give it time)
    try {
      await this.request("initialize", { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "pelica", version: "0.2.0" } }, 150000);
    } catch (e: any) {
      const last = this.log.trim().split(/\r?\n/).filter(Boolean).pop();
      throw new Error(last ? `${e.message}: ${last.slice(0, 240)}` : e.message);
    }
    this.notify("notifications/initialized", {});
    const r = await this.request("tools/list", {}, 20000);
    this.tools = r?.tools || [];
    this.error = "";
  }

  request(method: string, params: unknown, timeout = 120000): Promise<any> {
    if (!this.proc) return Promise.reject(new Error(`MCP server ${this.cfg.name} is not running`));
    const id = ++this.id;
    this.proc.stdin!.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
    return new Promise((ok, bad) => {
      this.waiting.set(id, { ok, bad });
      setTimeout(() => {
        if (this.waiting.delete(id)) bad(new Error(`MCP ${method} timed out`));
      }, timeout);
    });
  }

  notify(method: string, params: unknown) {
    this.proc?.stdin!.write(JSON.stringify({ jsonrpc: "2.0", method, params }) + "\n");
  }

  stop() {
    this.proc?.kill();
    this.proc = null;
  }
}

const servers = new Map<string, Server>();

/** Starts (or restarts) one server. */
export async function startOne(c: McpServerConfig): Promise<void> {
  servers.get(c.name)?.stop();
  servers.delete(c.name);
  if (c.disabled || !c.name || !c.command) return;
  const s = new Server(c);
  servers.set(c.name, s);
  try {
    await s.start();
  } catch (e: any) {
    s.error = e.message;
  }
}

/** Which launchers are installed (npx comes with Node.js, uvx with uv). */
export function launchers(): Promise<Record<string, boolean>> {
  const has = (cmd: string) => new Promise<boolean>((ok) => execFile("where", [cmd], { windowsHide: true }, (err) => ok(!err)));
  return Promise.all([has("npx"), has("uvx"), has("docker")]).then(([npx, uvx, docker]) => ({ npx, uvx, docker }));
}

export async function startAll(cfgs: McpServerConfig[]): Promise<void> {
  stopAll();
  await Promise.all(
    cfgs
      .filter((c) => !c.disabled && c.name && c.command)
      .map(async (c) => {
        const s = new Server(c);
        servers.set(c.name, s);
        try {
          await s.start();
        } catch (e: any) {
          s.error = e.message;
        }
      }),
  );
}

export function stopOne(name: string) {
  servers.get(name)?.stop();
  servers.delete(name);
}

export function stopAll() {
  for (const s of servers.values()) s.stop();
  servers.clear();
}

export function list() {
  return [...servers.values()].map((s) => ({ name: s.cfg.name, running: !!s.proc, error: s.error, log: s.log, tools: s.tools }));
}

export async function call(server: string, tool: string, args: unknown): Promise<string> {
  const s = servers.get(server);
  if (!s) throw new Error(`no MCP server called ${server}`);
  const r = await s.request("tools/call", { name: tool, arguments: args || {} });
  const parts: string[] = [];
  for (const c of r?.content || []) {
    if (c.type === "text") parts.push(c.text);
    else if (c.type === "resource") parts.push(c.resource?.text || `[resource ${c.resource?.uri}]`);
    else parts.push(`[${c.type} content]`);
  }
  const text = parts.join("\n") || JSON.stringify(r?.structuredContent ?? r ?? null);
  return r?.isError ? `Error: ${text}` : text;
}
