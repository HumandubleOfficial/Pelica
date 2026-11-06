// The tools the models get, and running them: web (any chat), files and
// commands (chats with a workspace folder; commands ask the user first),
// a todo list for long tasks, MCP servers (connectors; each tool allowed,
// asked for, or off).
import { bridge, need, type McpConfig, type McpToolInfo, type RepoReport, type ToolPerm } from "../lib/bridge";
import type { ToolSpec } from "../lib/engine";
import { app, setSettings, uid, type Chat, type Photo, type Todo, type ToolUI } from "../state";
import { shrink } from "../components/Attachments";

/** The reply being written: its file changes share one restore point. */
export const turnNow = { id: "" };

const obj = (properties: Record<string, any>, required: string[]) => ({ type: "object", properties, required });
const str = (description: string) => ({ type: "string", description });

const WEB: ToolSpec[] = [
  { name: "web_search", description: "Search the web (DuckDuckGo); returns titles, links and snippets. Use it for anything current or outside your knowledge: news, prices, recent releases and versions, docs for a library you aren't sure of. Search before answering rather than guessing; then open_page the best results. Never for greetings or small talk.", parameters: obj({ query: str("what to search for") }, ["query"]) },
  { name: "open_page", description: "Read a web page as plain text. Use it on web_search results (snippets are too short to answer from) or a URL the user gives. Cite the pages you used.", parameters: obj({ url: str("the page's http(s) URL") }, ["url"]) },
];

const FILES: ToolSpec[] = [
  { name: "list_files", description: "List the files and folders in the workspace folder or a subfolder (folders end with /). Use it first to get your bearings in a project.", parameters: obj({ path: str("folder, relative to the workspace (default: the workspace)"), depth: { type: "integer", description: "levels to show (default 2)" } }, []) },
  { name: "read_file", description: "Read a text file; each line comes prefixed with its line number and two spaces (the prefix is not part of the file). Read a file before editing it. For long files, read in parts with offset and limit.", parameters: obj({ path: str("file path, relative to the workspace"), offset: { type: "integer", description: "first line (default 1)" }, limit: { type: "integer", description: "lines to read (default 500)" } }, ["path"]) },
  { name: "write_file", description: "Create a new file, or replace a file's whole content; missing folders are created. Always the complete, working content: no placeholders, no \"...\" or \"rest unchanged\". For a small change to an existing file use edit_file instead.", parameters: obj({ path: str("file path, relative to the workspace"), content: str("the complete file content") }, ["path", "content"]) },
  { name: "edit_file", description: "Change part of an existing file: old_string is replaced by new_string. Read the file first. old_string must be copied exactly from it (same spaces, tabs and indentation, without read_file's line-number prefix) and occur only once, so include enough surrounding lines to make it unique, unless replace_all is true. On a mismatch, read the file again and retry.", parameters: obj({ path: str("file path"), old_string: str("exact text to replace"), new_string: str("the new text"), replace_all: { type: "boolean", description: "replace every occurrence" } }, ["path", "old_string", "new_string"]) },
  { name: "search_files", description: "Search the workspace's files for a regular expression (JavaScript syntax, case-insensitive); returns path:line: text for each match. Use it to find where something is defined or used before changing it.", parameters: obj({ pattern: str("regular expression"), path: str("subfolder (optional)"), glob: str("only files matching this, e.g. *.ts (optional)") }, ["pattern"]) },
  { name: "run_command", description: "Run a command in the workspace folder with Windows cmd.exe and get its output and exit code: install packages, build, run tests, run scripts, use git. Use Windows syntax (dir, type, copy, del, where; chain with &&), not Unix tools. Set background to true for anything that keeps running (dev servers, watchers), otherwise it is stopped at the timeout. Depending on the permission mode the user may approve it first.", parameters: obj({ command: str("the command line"), description: str("a few words on what it does, shown to the user (e.g. \"Build the app\")"), timeout_seconds: { type: "integer", description: "stop it after this long (default 120)" }, background: { type: "boolean", description: "leave it running and return its first output" } }, ["command"]) },
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

/** Memory across chats (Settings → Memory): what's kept shows in every chat's system prompt. */
const MEMORY: ToolSpec[] = [
  { name: "remember", description: "Save one short fact to keep across all chats: something the user asked you to remember, or a lasting fact you'll need again (their name, preferences, setup, ongoing projects). One fact per call, in plain words, about the user or their work, not the conversation. Don't save passwords, keys or other secrets. The user sees what you saved.", parameters: obj({ text: str("the fact, e.g. \"Prefers answers in British English\"") }, ["text"]) },
  { name: "forget", description: "Remove a saved memory, when the user asks you to forget something or a fact has changed (then remember the new one).", parameters: obj({ memory: str("the saved memory's text, or enough of it to tell which") }, ["memory"]) },
];

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
  if (app.get().settings.memoryOn) tools.push(...MEMORY);
  if (bridge && app.get().settings.computerTools) {
    tools.push(...COMPUTER);
    if (chat.folder && mode !== "plan") tools.push(OPEN);
  }
  const mcp: ToolSpec[] = [];
  for (const s of app.get().mcp) {
    if (!s.running) continue;
    const cfg = app.get().settings.mcpServers.find((c) => c.name === s.name);
    for (const t of s.tools) {
      if (toolPerm(cfg, t) === "off") continue;
      // plan mode: only the tools that say they just look
      if (mode === "plan" && !t.annotations?.readOnlyHint) continue;
      mcp.push({ name: `mcp__${s.name}__${t.name}`.replace(/[^\w]/g, "_").slice(0, 64), description: (t.description || "").slice(0, 600), parameters: t.inputSchema || { type: "object", properties: {} } });
    }
  }
  return { tools, mcp };
}

/** A connector tool's setting: the user's choice, else "allow" for tools that
 *  say they only read and "ask" for the rest. */
export function toolPerm(cfg: McpConfig | undefined, t: McpToolInfo): ToolPerm {
  return cfg?.tools?.[t.name] || (t.annotations?.readOnlyHint ? "allow" : "ask");
}

// ---------- running ----------

/** Commands the user allowed "for this chat": chat id -> command lines */
const allowed = new Map<string, Set<string>>();

/** `force`: ask even in Bypass permissions mode or when allowed for this chat
 *  (a command the GitHub safety check flagged). */
export function askApproval(chat: Chat, command: string, title = "Run this command?", why?: string, kind: "command" | "file" | "other" = "command", force = false, args?: string): Promise<"allow" | "always" | "deny"> {
  const mode = app.get().settings.permMode;
  // Bypass permissions: commands and file changes go ahead (the clipboard still asks)
  if (mode === "auto" && kind !== "other" && !force) return Promise.resolve("allow");
  if (allowed.get(chat.id)?.has(command) && !force) return Promise.resolve("allow");
  return new Promise((resolve) => {
    app.set({
      approval: {
        id: uid(),
        title,
        command,
        folder: chat.folder || "",
        why,
        args,
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
  /** pictures it answered with (a connector's screenshot or chart) */
  images?: Photo[];
}

/** A connector's pictures, resized like a sent photo (JPEG data: URLs). */
async function toolPictures(list: { data: string; mimeType: string }[], tool: string): Promise<Photo[]> {
  const out: Photo[] = [];
  for (const [i, im] of list.entries()) {
    try {
      const blob = await (await fetch(`data:${im.mimeType};base64,${im.data}`)).blob();
      out.push({ name: `${tool} ${i + 1}`, data: await shrink(blob) });
    } catch {}
  }
  return out;
}

/** In Ask mode a file change waits for the user's OK; null when it may go ahead. */
async function fileOk(chat: Chat, what: string, detail: string, signal: { stopped: boolean }): Promise<ToolResult | null> {
  if (app.get().settings.permMode !== "ask") return null;
  const ok = await askApproval(chat, what, "Let Pelica change this file?", detail, "file");
  if (ok === "deny" || signal.stopped) return { content: `The user declined: ${what}. Ask them how to proceed.`, ui: { kind: "denied", title: what, detail: "Declined" } };
  return null;
}

/** Commands that fetch or install code from somewhere. */
const FETCHES = /\b(git\s+(clone|submodule\s+add|pull|fetch|remote\s+add)|gh\s+repo\s+(clone|fork)|degit|pip3?\s+install|uv\s+(pip\s+install|add)|npm\s+(i|install|add)|npx|pnpm\s+(add|dlx|i|install)|yarn\s+(add|dlx)|bunx?\s|cargo\s+install|go\s+(get|install)|curl|wget|iwr|irm|invoke-webrequest|invoke-restmethod|start-bitstransfer)\b/i;

/** The GitHub safety check for repositories named in `text`: a line for the
 *  model, the reports for the card, and whether any looks unsafe. */
async function repoSafety(text: string): Promise<{ note: string; reports: RepoReport[]; risky: boolean } | null> {
  if (!/github(usercontent)?\.com|github:|gh\s+repo|degit/i.test(text)) return null;
  const reports = await need().web.repos(text).catch(() => [] as RepoReport[]);
  if (!reports.length) return null;
  const word = { ok: "looks fine", caution: "use with care", risky: "LOOKS UNSAFE" };
  const note = reports.map((r) => `GitHub safety check, ${r.repo}: ${word[r.verdict]}. ${r.reasons.join(" ")}`).join("\n");
  return { note, reports, risky: reports.some((r) => r.verdict === "risky") };
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
        const url = String(a.url || "");
        const [page, safe] = await Promise.all([need().web.open(url, 14000), repoSafety(url)]);
        const head = safe ? `${safe.note}\n${safe.risky ? "Don't clone, install or run anything from it unless the user says so after seeing this.\n" : ""}\n` : "";
        return { content: `${head}Title: ${page.title}\nURL: ${page.url}\n\n${page.text}`, ui: { kind: "page", title: page.title || page.url, url: page.url, safety: safe?.reports } };
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
        const r = await need().fs.write(root, p, content, turnNow.id);
        return { content: `${r.before === null ? "Created" : "Wrote"} ${r.path} (${r.after.split("\n").length} lines).`, ui: { kind: "write", undo: turnNow.id, title: r.path, path: r.path, before: r.before, after: r.after } };
      }
      case "edit_file": {
        const denied = await fileOk(chat, `Edit ${p}`, `Replace:\n${String(a.old_string ?? "").slice(0, 600)}\n\nwith:\n${String(a.new_string ?? "").slice(0, 600)}`, signal);
        if (denied) return denied;
        const r = await need().fs.edit(root, p, String(a.old_string ?? ""), String(a.new_string ?? ""), !!a.replace_all, turnNow.id);
        return { content: `Edited ${r.path} (${r.count} replacement${r.count > 1 ? "s" : ""}).`, ui: { kind: "edit", undo: turnNow.id, title: r.path, path: r.path, before: r.before, after: r.after } };
      }
      case "search_files": {
        const t = await need().fs.search(root, String(a.pattern || ""), p || ".", a.glob ? String(a.glob) : undefined);
        return { content: t, ui: { kind: "grep", title: String(a.pattern || ""), output: t } };
      }
      case "delete_file": {
        const denied = await fileOk(chat, `Move ${p} to the Recycle Bin`, "It can be restored from the Recycle Bin.", signal);
        if (denied) return denied;
        await need().fs.trash(root, p, turnNow.id);
        return { content: `Moved ${p} to the Recycle Bin.`, ui: { kind: "trash", title: p, path: p, undo: turnNow.id } };
      }
      case "todo_write": {
        const todos: Todo[] = (Array.isArray(a.todos) ? a.todos : []).map((t: any) => ({ content: String(t.content ?? t.task ?? ""), status: ["pending", "in_progress", "completed"].includes(t.status) ? t.status : "pending" })).filter((t: Todo) => t.content);
        const left = todos.filter((t) => t.status !== "completed").length;
        return { content: left ? `Todo list updated: ${left} of ${todos.length} left. Continue with the next item.` : "All items completed.", ui: { kind: "todo", title: `${todos.length - left}/${todos.length} done`, todos }, todos };
      }
      case "remember": {
        const text = String(a.text || "").trim().replace(/\s+/g, " ").slice(0, 400);
        if (!text) throw new Error("nothing to remember");
        const list = app.get().settings.memories;
        if (list.some((m) => m.text.toLowerCase() === text.toLowerCase())) return { content: `Already remembered: ${text}`, ui: { kind: "memory", title: text, detail: "Already remembered" } };
        setSettings({ memories: [...list, { id: uid(), text, created: Date.now(), by: "pelica" }] });
        return { content: `Saved to memory: ${text}`, ui: { kind: "memory", title: text, detail: "Remembered" } };
      }
      case "forget": {
        const q = String(a.memory || "").trim().toLowerCase();
        const list = app.get().settings.memories;
        const hit = list.find((m) => m.text.toLowerCase() === q) || list.find((m) => m.text.toLowerCase().includes(q) || (q.length > 8 && q.includes(m.text.toLowerCase())));
        if (!q || !hit) return { content: `No saved memory matches "${a.memory}". Saved: ${list.map((m) => m.text).join(" | ") || "nothing"}`, ui: { kind: "memory", title: String(a.memory || ""), detail: "Not found" } };
        setSettings({ memories: list.filter((m) => m !== hit) });
        return { content: `Forgot: ${hit.text}`, ui: { kind: "memory", title: hit.text, detail: "Forgot" } };
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
        const about = String(a.description || "").trim().slice(0, 120) || undefined;
        const safe = FETCHES.test(command) ? await repoSafety(command) : null;
        live({ kind: "command", title: command, about, running: false, detail: "Waiting for your approval…", safety: safe?.reports });
        const ok = safe?.risky
          ? await askApproval(chat, command, "This repository looks unsafe. Run anyway?", safe.note, "command", true)
          : await askApproval(chat, command, undefined, safe ? safe.note : undefined);
        if (ok === "deny" || signal.stopped) return { content: `${safe ? safe.note + "\n" : ""}The user declined to run this command. Ask them, or try another way.`, ui: { kind: "denied", title: command, detail: "Declined", safety: safe?.reports } };
        const id = uid();
        let out = "";
        live({ kind: "command", title: command, about, running: true, detail: "", output: "" });
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
          return { content: `${safe ? safe.note + "\n" : ""}${status}\n${clip(r.output.trim(), 16000) || "(no output)"}`, ui: { kind: "command", title: command, about, output: r.output.slice(-20000), code: r.code, detail: status, running: false, safety: safe?.reports } };
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
          const perm = toolPerm(app.get().settings.mcpServers.find((c) => c.name === s.name), t);
          const what = `${s.name} · ${t.title || t.annotations?.title || t.name}`;
          if (perm === "off") throw new Error(`the user turned off ${what}`);
          if (perm === "ask") {
            const args = JSON.stringify(a, null, 1);
            const about = t.description ? t.description.split("\n")[0].slice(0, 240) : undefined;
            const ok = await askApproval(chat, what, "Let Pelica use this connector?", about, "command", false, args.length > 1500 ? args.slice(0, 1500) + "…" : args);
            if (ok === "deny" || signal.stopped) return { content: `The user declined: ${what}. Ask them how to proceed.`, ui: { kind: "denied", title: what, detail: "Declined" } };
          }
          const r = await need().mcp.call(s.name, t.name, a);
          const images = await toolPictures(r.images, t.name);
          let content = clip(r.text, 20000);
          if (images.length) {
            const n = images.length === 1 ? "a picture" : `${images.length} pictures`;
            if (app.get().status?.vision) content += `\n[The tool answered with ${n}, attached as image${images.length === 1 ? "" : "s"}.]`;
            else {
              // (a model that can't see: the text Windows reads in them, said plainly)
              const read = await Promise.all(images.map((x) => need().media.ocr(x.data).catch(() => "")));
              content += `\n[The tool answered with ${n}. The loaded model can't see images${read.some(Boolean) ? `; the text Windows read in ${images.length === 1 ? "it" : "them"} (OCR):\n${read.map((t, i) => (images.length > 1 ? `Picture ${i + 1}: ` : "") + (t || "(no text)")).join("\n")}` : ", and Windows found no text in it"}]`;
            }
          }
          return { content, images: images.length ? images : undefined, ui: { kind: "mcp", title: `${s.name} · ${t.name}`, output: r.text.slice(0, 20000) } };
        }
        throw new Error(`there is no tool called "${name}"`);
      }
    }
  } catch (e: any) {
    const msg = String(e?.message || e).replace(/^Error invoking remote method '[^']+': (Error: )?/, "");
    return { content: `Error: ${msg}`, ui: { kind: "error", title: name, error: msg, path: p || undefined } };
  }
}
