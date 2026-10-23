// The tools the models get, and running them: web (any chat), files and
// commands (chats with a workspace folder; commands ask the user first),
// a todo list for long tasks, MCP servers.
import { bridge, need } from "../lib/bridge";
import type { ToolSpec } from "../lib/engine";
import { app, uid, type Chat, type Todo, type ToolUI } from "../state";

const obj = (properties: Record<string, any>, required: string[]) => ({ type: "object", properties, required });
const str = (description: string) => ({ type: "string", description });

const WEB: ToolSpec[] = [
  { name: "web_search", description: "Search the web. Only for things that need current or outside information (news, prices, recent releases, facts you don't know). Never for greetings or small talk.", parameters: obj({ query: str("what to search for") }, ["query"]) },
  { name: "open_page", description: "Read a web page as text (after web_search, to get the details).", parameters: obj({ url: str("the page's http(s) URL") }, ["url"]) },
];

const FILES: ToolSpec[] = [
  { name: "list_files", description: "List the files and folders in the workspace folder (or a subfolder).", parameters: obj({ path: str("folder, relative to the workspace (default: the workspace)"), depth: { type: "integer", description: "levels to show (default 2)" } }, []) },
  { name: "read_file", description: "Read a text file, with line numbers.", parameters: obj({ path: str("file path, relative to the workspace"), offset: { type: "integer", description: "first line (default 1)" }, limit: { type: "integer", description: "lines to read (default 500)" } }, ["path"]) },
  { name: "write_file", description: "Create a file, or replace a file's whole content. Write complete, working content (no placeholders).", parameters: obj({ path: str("file path, relative to the workspace"), content: str("the complete file content") }, ["path", "content"]) },
  { name: "edit_file", description: "Change part of a file: old_string is replaced by new_string. old_string must match the file exactly (indentation too) and be unique, unless replace_all is true. Read the file first.", parameters: obj({ path: str("file path"), old_string: str("exact text to replace"), new_string: str("the new text"), replace_all: { type: "boolean", description: "replace every occurrence" } }, ["path", "old_string", "new_string"]) },
  { name: "search_files", description: "Search the workspace's files for a regular expression; returns path:line: text.", parameters: obj({ pattern: str("regular expression"), path: str("subfolder (optional)"), glob: str("only files matching this, e.g. *.ts (optional)") }, ["pattern"]) },
  { name: "run_command", description: "Run a command in the workspace folder (Windows cmd.exe) to build, test or run code; returns its output. The user approves each command. Set background to true for servers and watchers that keep running.", parameters: obj({ command: str("the command line"), timeout_seconds: { type: "integer", description: "stop it after this long (default 120)" }, background: { type: "boolean", description: "leave it running and return its first output" } }, ["command"]) },
  { name: "delete_file", description: "Move a file or folder in the workspace to the Recycle Bin.", parameters: obj({ path: str("path, relative to the workspace") }, ["path"]) },
  {
    name: "todo_write",
    description: "Plan and track multi-step work: send the whole list every time, each item pending, in_progress (one at a time) or completed. Use it for any task with 3 or more steps, and keep working until every item is completed.",
    parameters: obj({ todos: { type: "array", items: obj({ content: str("the step"), status: { type: "string", enum: ["pending", "in_progress", "completed"] } }, ["content", "status"]) } }, ["todos"]),
  },
];

// the computer itself (Settings → Capabilities → Computer tools)
const COMPUTER: ToolSpec[] = [
  { name: "system_info", description: "Facts about the user's computer: Windows version, CPU, GPU, memory, displays, free disk space, local time, battery.", parameters: obj({}, []) },
  { name: "clipboard_read", description: "Read the text the user has copied (their clipboard), when they refer to it. The user approves each read.", parameters: obj({}, []) },
  { name: "clipboard_write", description: "Put text on the user's clipboard, ready for them to paste (when they ask you to copy something).", parameters: obj({ text: str("the text to copy") }, ["text"]) },
];
const OPEN: ToolSpec = { name: "open_file", description: "Open a workspace file in the app Windows uses for it (a web page in the browser, a document in its editor, an image in the viewer), e.g. to show the user what you built. The user approves it.", parameters: obj({ path: str("file path, relative to the workspace") }, ["path"]) };


const READ_ONLY = new Set(["list_files", "read_file", "search_files", "todo_write"]);

const GREETING = /^\s*(hi|hello|hey|yo|hiya|howdy|sup|hola|thanks|thank you|thx|ty|ok|okay|cool|nice|great|bye|good (morning|afternoon|evening|night)|what'?s up|how are you|who are you|what are you)\b[\s!.?,]*([a-z]+[\s!.?,]*){0,4}$/i;

/** Small talk gets no tools at all: small models otherwise search for anything. */
export function isSmallTalk(text: string): boolean {
  const t = text.trim();
  return t.length < 60 && GREETING.test(t);
}

export function toolsFor(chat: Chat, opts: { web: boolean; lastUser: string }): { tools: ToolSpec[]; mcp: ToolSpec[] } {
  if (isSmallTalk(opts.lastUser) && !chat.messages.some((m) => m.role === "tool")) return { tools: [], mcp: [] };
  const tools: ToolSpec[] = [];
  if (opts.web) tools.push(...WEB);
  const mode = app.get().settings.permMode;
  // plan mode: looking only
  if (chat.folder && bridge) tools.push(...(mode === "plan" ? FILES.filter((t) => READ_ONLY.has(t.name)) : FILES));
  if (bridge && app.get().settings.computerTools) {
    tools.push(...COMPUTER);
    if (chat.folder && mode !== "plan") tools.push(OPEN);
  }
  const mcp: ToolSpec[] = [];
  for (const s of app.get().mcp) {
    if (!s.running) continue;
    for (const t of s.tools) mcp.push({ name: `mcp__${s.name}__${t.name}`.replace(/[^\w]/g, "_").slice(0, 64), description: (t.description || "").slice(0, 600), parameters: t.inputSchema || { type: "object", properties: {} } });
  }
  return { tools, mcp };
}

// ---------- running ----------

/** Commands the user allowed "for this chat": chat id -> command lines */
const allowed = new Map<string, Set<string>>();

export function askApproval(chat: Chat, command: string, title = "Run this command?", why?: string, kind: "command" | "file" | "other" = "command"): Promise<"allow" | "always" | "deny"> {
  const mode = app.get().settings.permMode;
  // Full auto: commands and file changes go ahead (the clipboard still asks)
  if (mode === "auto" && kind !== "other") return Promise.resolve("allow");
  if (allowed.get(chat.id)?.has(command)) return Promise.resolve("allow");
  return new Promise((resolve) => {
    app.set({
      approval: {
        id: uid(),
        title,
        command,
        folder: chat.folder || "",
        why,
        chatTitle: chat.title,
        resolve: (r) => {
          app.set({ approval: null });
          if (r === "always") {
            if (!allowed.has(chat.id)) allowed.set(chat.id, new Set());
            allowed.get(chat.id)!.add(command);
          }
          resolve(r);
        },
      },
    });
  });
}

export interface ToolResult {
  content: string;
  ui: ToolUI;
  todos?: Todo[];
}

/** In Ask mode a file change waits for the user's OK; null when it may go ahead. */
async function fileOk(chat: Chat, what: string, detail: string, signal: { stopped: boolean }): Promise<ToolResult | null> {
  if (app.get().settings.permMode !== "ask") return null;
  const ok = await askApproval(chat, what, "Let Pelica change this file?", detail, "file");
  if (ok === "deny" || signal.stopped) return { content: `The user declined: ${what}. Ask them how to proceed.`, ui: { kind: "denied", title: what, detail: "Declined" } };
  return null;
}

const clip = (s: string, n: number) => (s.length > n ? s.slice(0, n / 2) + `\n…[${s.length - n} characters cut]…\n` + s.slice(-n / 2) : s);

/** Runs one call; `live` gets the card's state while it runs (command output). */
export async function runTool(chat: Chat, name: string, args: any, live: (ui: Partial<ToolUI>) => void, signal: { stopped: boolean }): Promise<ToolResult> {
  const a = args && typeof args === "object" ? args : {};
  const root = chat.folder || "";
  const p = String(a.path ?? "");
  try {
    switch (name) {
      case "web_search": {
        const q = String(a.query || "").trim();
        if (!q) throw new Error('web_search needs a "query"');
        const hits = await need().web.search(q);
        const text = hits.length ? hits.map((h, i) => `${i + 1}. ${h.title}\n   ${h.url}\n   ${h.snippet}`).join("\n") : "No results.";
        return { content: text, ui: { kind: "search", title: q, results: hits.map((h) => ({ title: h.title, url: h.url })) } };
      }
      case "open_page": {
        const page = await need().web.open(String(a.url || ""), 14000);
        return { content: `Title: ${page.title}\nURL: ${page.url}\n\n${page.text}`, ui: { kind: "page", title: page.title || page.url, url: page.url } };
      }
      case "list_files": {
        const t = await need().fs.list(root, p || ".", Math.min(5, Number(a.depth) || 2));
        return { content: t, ui: { kind: "list", title: p || ".", output: t } };
      }
      case "read_file": {
        const r = await need().fs.read(root, p, Number(a.offset) || 1, Math.min(2000, Number(a.limit) || 500));
        return { content: r.text, ui: { kind: "read", title: r.path, path: r.path, detail: `${r.lines} lines` } };
      }
      case "write_file": {
        const content = String(a.content ?? "");
        const denied = await fileOk(chat, `Write ${p}`, `${content.split("\n").length} lines`, signal);
        if (denied) return denied;
        const r = await need().fs.write(root, p, content);
        return { content: `${r.before === null ? "Created" : "Wrote"} ${r.path} (${r.after.split("\n").length} lines).`, ui: { kind: "write", title: r.path, path: r.path, before: r.before, after: r.after } };
      }
      case "edit_file": {
        const denied = await fileOk(chat, `Edit ${p}`, `Replace:\n${String(a.old_string ?? "").slice(0, 600)}\n\nwith:\n${String(a.new_string ?? "").slice(0, 600)}`, signal);
        if (denied) return denied;
        const r = await need().fs.edit(root, p, String(a.old_string ?? ""), String(a.new_string ?? ""), !!a.replace_all);
        return { content: `Edited ${r.path} (${r.count} replacement${r.count > 1 ? "s" : ""}).`, ui: { kind: "edit", title: r.path, path: r.path, before: r.before, after: r.after } };
      }
      case "search_files": {
        const t = await need().fs.search(root, String(a.pattern || ""), p || ".", a.glob ? String(a.glob) : undefined);
        return { content: t, ui: { kind: "grep", title: String(a.pattern || ""), output: t } };
      }
      case "delete_file": {
        const denied = await fileOk(chat, `Move ${p} to the Recycle Bin`, "It can be restored from the Recycle Bin.", signal);
        if (denied) return denied;
        await need().fs.trash(root, p);
        return { content: `Moved ${p} to the Recycle Bin.`, ui: { kind: "trash", title: p, path: p } };
      }
      case "todo_write": {
        const todos: Todo[] = (Array.isArray(a.todos) ? a.todos : []).map((t: any) => ({ content: String(t.content ?? t.task ?? ""), status: ["pending", "in_progress", "completed"].includes(t.status) ? t.status : "pending" })).filter((t: Todo) => t.content);
        const left = todos.filter((t) => t.status !== "completed").length;
        return { content: left ? `Todo list updated: ${left} of ${todos.length} left. Continue with the next item.` : "All items completed.", ui: { kind: "todo", title: `${todos.length - left}/${todos.length} done`, todos }, todos };
      }
      case "system_info": {
        const info = await need().sys.info();
        const st = app.get().status;
        if (st?.gpu) info.gpu = st.gpu;
        if (st?.model) info.model_loaded = st.model;
        const text = Object.entries(info)
          .map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`)
          .join("\n");
        return { content: text, ui: { kind: "system", title: "this computer", output: text } };
      }
      case "clipboard_read": {
        const ok = await askApproval(chat, "Read the clipboard", "Let Pelica read your clipboard?", "The model asked to see the text you copied.", "other");
        if (ok === "deny" || signal.stopped) return { content: "The user declined to share the clipboard.", ui: { kind: "denied", title: "reading the clipboard", detail: "Declined" } };
        const t = await need().sys.clipRead();
        return { content: t ? clip(t, 20000) : "(the clipboard has no text)", ui: { kind: "clipboard", title: `Read the clipboard (${t.length.toLocaleString()} characters)`, output: t.slice(0, 4000) } };
      }
      case "clipboard_write": {
        const t = String(a.text ?? "");
        if (!t) throw new Error('clipboard_write needs a "text"');
        await need().sys.clipWrite(t);
        return { content: `Copied ${t.length} characters to the clipboard.`, ui: { kind: "clipboard", title: `Copied to the clipboard (${t.length.toLocaleString()} characters)`, output: t.slice(0, 4000) } };
      }
      case "open_file": {
        if (!p) throw new Error('open_file needs a "path"');
        const ok = await askApproval(chat, `Open ${p}`, "Open this file?", "It opens in the app Windows uses for this kind of file.");
        if (ok === "deny" || signal.stopped) return { content: "The user declined to open the file.", ui: { kind: "denied", title: `opening ${p}`, detail: "Declined" } };
        await need().sys.open(root, p);
        return { content: `Opened ${p} in its default app.`, ui: { kind: "open", title: p, path: p } };
      }
      case "run_command": {
        const command = String(a.command || "").trim();
        if (!command) throw new Error('run_command needs a "command"');
        live({ kind: "command", title: command, running: false, detail: "Waiting for your approval…" });
        const ok = await askApproval(chat, command);
        if (ok === "deny" || signal.stopped) return { content: "The user declined to run this command. Ask them, or try another way.", ui: { kind: "denied", title: command, detail: "Declined" } };
        const id = uid();
        let out = "";
        live({ kind: "command", title: command, running: true, detail: "", output: "" });
        const off = need().cmd.onData((v) => {
          if (v.id === id) {
            out += v.data;
            live({ output: out.slice(-20000) });
          }
        });
        const stopWatch = setInterval(() => signal.stopped && need().cmd.kill(id), 300);
        try {
          const r = await need().cmd.run(id, root, command, Math.min(1800, Number(a.timeout_seconds) || 120) * 1000, !!a.background);
          const status = r.background ? "still running in the background" : r.timedOut ? "timed out" : `exit code ${r.code}`;
          return { content: `${status}\n${clip(r.output.trim(), 16000) || "(no output)"}`, ui: { kind: "command", title: command, output: r.output.slice(-20000), code: r.code, detail: status, running: false } };
        } finally {
          off();
          clearInterval(stopWatch);
        }
      }
      default: {
        const m = name.match(/^mcp__(.+?)__(.+)$/);
        if (m) {
          const s = app.get().mcp.find((x) => x.name.replace(/[^\w]/g, "_") === m[1]);
          const t = s?.tools.find((x) => x.name.replace(/[^\w]/g, "_") === m[2]);
          if (!s || !t) throw new Error(`no MCP tool ${name}`);
          const r = await need().mcp.call(s.name, t.name, a);
          return { content: clip(r, 20000), ui: { kind: "mcp", title: `${s.name} · ${t.name}`, output: r.slice(0, 20000) } };
        }
        throw new Error(`there is no tool called "${name}"`);
      }
    }
  } catch (e: any) {
    const msg = String(e?.message || e).replace(/^Error invoking remote method '[^']+': (Error: )?/, "");
    return { content: `Error: ${msg}`, ui: { kind: "error", title: name, error: msg, path: p || undefined } };
  }
}
