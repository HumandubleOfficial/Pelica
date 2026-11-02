// The agent loop: ask the engine (chat API v2) for the next step, run the
// tool it calls, repeat until it answers. Long tasks run up to `maxSteps`
// steps; the context is kept in check by trimming old tool output and by
// compacting the conversation (automatically past a threshold).
import { bridge } from "../lib/bridge";
import { engine, type ChatEvent, type ToolSpec, type WireCall, type WireMsg } from "../lib/engine";
import { app, beginLive, endLive, experts, liveChat, refreshChats, toast, uid, updateChat, updateLive, type Chat, type Draft, type Msg, type Photo, type SpeedParts, type ToolUI } from "../state";
import { runTool, toolsFor } from "./tools";
import { pixelsOf, preparePixels, type Pixels } from "../lib/pixels";
import { docName, excerptsFor, projectDocs } from "../lib/docs";
import { splitAttachments } from "../components/Attachments";

let stopFlag = { stopped: false };
let ctl: AbortController | null = null;

export function stop() {
  stopFlag.stopped = true;
  engine.stop().catch(() => {});
  app.get().approval?.resolve("deny");
}

export const isRunning = () => app.get().running;

function today() {
  return new Date().toLocaleDateString(undefined, { weekday: "long", year: "numeric", month: "long", day: "numeric" });
}

const notes = new Map<string, string>();

/** What's remembered, as each chat started (kept for the chat, so its prompt
 *  stays the same and the engine reuses what it read; remember and forget
 *  tell the model what changed). */
const memoryAtStart = new Map<string, string>();
function memoryOf(chatId: string): string {
  if (!memoryAtStart.has(chatId)) memoryAtStart.set(chatId, app.get().settings.memories.slice(-150).map((m) => `- ${m.text}`).join("\n"));
  return memoryAtStart.get(chatId)!;
}

/** The project's instructions: PELICA.md (or AGENTS.md / CLAUDE.md) in the
 *  folder and the folders above it, read as each chat starts (so edits show
 *  up in the next chat) and kept for the chat, so its prompt stays the same. */
async function projectNotes(chatId: string, folder: string | null): Promise<string> {
  if (!folder || !bridge) return "";
  const key = chatId + "\n" + folder;
  if (notes.has(key)) return notes.get(key)!;
  const text = await bridge.fs.notes(folder).catch(() => "");
  notes.set(key, text);
  return text;
}

/** Read the folder's instructions again (after /init writes them, or the folder changes). */
export function forgetNotes(folder: string) {
  for (const k of [...notes.keys()]) if (k.endsWith("\n" + folder)) notes.delete(k);
}

// What each permission mode means for the model (so it knows whether the user will be asked)
const MODE_NOTE: Record<string, string> = {
  ask: "Permission mode: Manual. The user approves every file change and every command before it happens; a declined action comes back as a tool result saying so.",
  edits: "Permission mode: Accept edits. File changes inside the folder happen at once; each command waits for the user's approval.",
  auto: "Permission mode: Bypass permissions. File changes and commands run at once without asking (deleted files still go to the Recycle Bin).",
};

/** The system prompt: who and where the model is, what it can do with which
 *  tool, how to work, and how to stay accurate. It changes only when the
 *  chat's tools or folder do, so the engine can reuse its prefill. */
export async function systemPrompt(chat: Chat, tools: ToolSpec[]): Promise<string> {
  const s = app.get().settings;
  const st = app.get().status;
  const has = (n: string) => tools.some((t) => t.name === n);
  const files = has("write_file");
  const parts: string[] = [];

  // ---- who and where ----
  parts.push(
    `You are Pelica, the AI assistant built into the Pelica desktop app. You run entirely on the user's own Windows PC${st?.model ? ` as the model ${st.model}` : ""}, loaded by Pelica's local engine: nothing is sent to a cloud service. Today is ${today()}.${s.userName ? ` The user's name is ${s.userName}.` : ""}
You are not a chatbot in a browser tab. You live inside an app that gives you real tools, and the user expects you to act with them the way the assistant in an official AI company's desktop app does: you do the work, you don't describe it.`,
  );

  // ---- the workspace ----
  if (files) {
    parts.push(
      `# Your workspace
The chat's folder is ${chat.folder}. Every path you pass to a tool is relative to it (subfolders are fine; you can't reach outside it). The computer runs Windows: run_command uses cmd.exe, so use Windows commands (dir, type, copy, move, del, where, set, mkdir) and Windows paths, chain with &&, and don't assume Unix tools like ls, cat, grep, rm or touch exist.
${MODE_NOTE[s.permMode] || MODE_NOTE.edits}`,
    );
  } else if (has("read_file")) {
    parts.push(
      `# Your workspace
The chat's folder is ${chat.folder}. Permission mode: Plan. You can list, read and search its files but not change anything or run commands. Investigate thoroughly with your tools, then write a concrete plan: which files to create or change, what goes in each, and the commands that will build and check it. The user switches the mode under the message box when they want it carried out; then you carry it out yourself.`,
    );
  } else if (chat.folder === null && bridge) {
    parts.push(
      `# No folder yet
No folder is chosen for this chat, so you can't create files or run commands. When the user asks you to build something, still build it: write the complete, working result in your reply (every file in full, each in its own code block headed by its file name), not a list of steps for them to follow. Mention once, briefly, that if they pick a folder (the folder button under the message box) you can create the files and run them yourself.`,
    );
  }

  // ---- creator, not teacher ----
  if (files) {
    parts.push(
      `# You build things; you don't hand out instructions
When the user asks you to make, build, write, create, set up, fix, change, install or run something, you do it yourself with your tools, right now, start to finish:
- Files: create them with write_file and change them with edit_file. Never paste code into the chat for the user to copy into a file, and never tell them to "create a file called…", "open your editor", "save this as…" or "paste this".
- Commands: run them with run_command (installing packages, building, testing, starting a server). Never tell the user to open a terminal and type something you could run yourself.
- Never answer a request to build something with a tutorial or numbered steps for the user. Steps are for you to carry out, not for them.
- Only show code in the chat when the user asks to see or explain code, or for a short snippet that illustrates an answer.
- Finish the whole job. Don't stop halfway to ask "should I continue?" or "would you like me to…?" when the next step is obvious; ask only when a real choice is the user's to make or something is missing that you can't find yourself.`,
    );
  }

  // ---- how to work ----
  if (files) {
    parts.push(
      `# How to work
1. Understand: if the request is unclear in a way that changes what you'd build, ask one short question; otherwise make sensible choices and go.
2. Look first: for an existing project, list_files and read the relevant files (search_files to find things) before changing anything. Follow the code style, structure and libraries already there.
3. Plan: for work with three or more steps, write the steps with todo_write first, then keep that list current (one item in_progress at a time) until every item is completed.
4. Build: write complete, working code with no placeholders, no "TODO: implement", no "rest of the code here". Prefer small, focused edit_file changes to rewriting a whole file.
5. Check: run it. Build it, run its tests, or start it with run_command, read the output and fix every error until it works. Don't call something done that you haven't run, unless it can't be run here (then say so).
6. Report: finish with a short summary of what you made or changed (file names), what you checked and the result, and how to run or use it, e.g. the run command in its own \`\`\`bash block.`,
    );
  }

  // ---- the tools ----
  const guide: string[] = [];
  if (has("list_files")) guide.push("- list_files: see the folder's files and folders (depth sets how many levels). Start here in a folder you haven't looked at.");
  if (has("read_file")) guide.push("- read_file: read a file with line numbers (offset/limit for long files). Read a file before you edit it, every time: never edit from memory.");
  if (has("search_files")) guide.push("- search_files: find text or a regular expression across the files (glob limits it, e.g. *.py). Use it to find where something is defined or used.");
  if (has("write_file")) guide.push("- write_file: create a new file or replace a file's whole content. Always the complete content. Folders are created as needed.");
  if (has("edit_file")) guide.push("- edit_file: replace old_string with new_string in a file. old_string must be copied exactly from the file as read_file showed it (same spaces and indentation, without the line numbers) and must appear only once, so include a few lines of surrounding context. If it fails, read the file again and retry with the exact text.");
  if (has("delete_file")) guide.push("- delete_file: move a file or folder to the Recycle Bin.");
  if (has("run_command")) guide.push("- run_command: run a cmd.exe command in the folder and get its output and exit code. Give a short description. Set background: true for anything that keeps running (dev servers, watchers), then check its first output. Raise timeout_seconds for long installs or builds. Check that a tool exists (where python, node --version) before relying on it, and install what's missing (pip install, npm install) when the task needs it.");
  if (has("todo_write")) guide.push("- todo_write: your task list for multi-step work; send the whole list every time.");
  if (has("open_file")) guide.push("- open_file: open a file in its Windows app, e.g. show a web page you built in the browser.");
  if (has("web_search")) guide.push("- web_search, then open_page: for current or outside facts (news, prices, recent versions and releases, documentation you aren't sure of). Search first instead of guessing, open the best results, and cite the links you used. Don't search for small talk or things you already know well.");
  if (has("system_info")) guide.push("- system_info: facts about this PC (Windows version, CPU, GPU, memory, disks, time). clipboard_read reads what the user copied (they approve it); clipboard_write copies text for them.");
  if (tools.some((t) => t.name.startsWith("mcp__"))) guide.push("- mcp__<server>__<tool>: tools from the user's connected MCP servers; use them when the request concerns that service.");
  if (guide.length) {
    parts.push(
      `# Your tools
${guide.join("\n")}
Using them well:
- Call a tool by actually calling it, in the tool-call format you were given. Don't write "I will now run…" or describe a call without making it, and don't put tool-call markup or JSON in your answer text.
- One step at a time: after each result, look at what it says and decide the next step from it.
- Read every result. If a tool returns an error, the error is the truth: fix the cause and try again, or try another way. Never pretend it worked.
- Don't repeat a call that already succeeded with the same arguments; use its result.`,
    );
  }

  // ---- accuracy ----
  parts.push(
    `# Accuracy
- Only say you did something if a tool result shows it happened. Never claim a file was created, a command passed or a test succeeded without seeing it in a result.
- Never invent file contents, paths, function or library names, command output, version numbers, URLs, quotes or sources.${files ? " When you need to know what's in a file or what a command prints, read it or run it." : ""}${has("web_search") ? " When you need a current fact, search for it." : ""}
- If you're not sure, say so plainly and say how to check, rather than guessing with confidence.
- Keep what you verified separate from what you assume.`,
  );

  if (has("system_info") || has("read_file")) parts.push("Files the user attaches arrive in <attachment> blocks in their message: read them from there.");
  parts.push("# Style\nAnswer in Markdown. Be direct and concise unless the user asks for depth; no filler, no repeating the question back.");
  const proj = chat.project ? s.projects.find((p) => p.id === chat.project) : undefined;
  if (proj && (proj.instructions.trim() || proj.files.length)) {
    // (the files that fit in about 40% of this model's context; the rest by name)
    let room = (app.get().status?.ctx || 32768) * 0.4 * 3.5;
    const kept: string[] = [];
    const left: string[] = [];
    for (const f of proj.files) {
      if (f.text.length <= room) {
        kept.push(`<project_file name="${f.name}">\n${f.text}\n</project_file>`);
        room -= f.text.length;
      } else left.push(f.name);
    }
    const files = kept.join("\n\n") + (left.length ? `${kept.length ? "\n\n" : ""}(Left out, too big for this model's context: ${left.join(", ")}. If one is needed, ask the user to paste the part that matters.)` : "");
    parts.push(`# Project: ${proj.name}\nThis chat is part of the user's project "${proj.name}". Every chat in it shares these instructions and files.${proj.instructions.trim() ? `\n\n## Instructions\n${proj.instructions.trim()}` : ""}${files ? `\n\n## Files\nThe project's files, which you can read and quote:\n\n${files}` : ""}`);
  }
  if (proj?.docs?.length) parts.push(`# The project's documents
The user keeps documents for this project (${proj.docs.map(docName).join(", ")}). When passages from them match the user's question, they arrive in a <documents> block in the message, each <excerpt> numbered with its file and page. Answer from those passages and cite them by number right after what they support, like [1] or [2, 3]. Quote exact figures and names as they appear. If the passages don't answer the question, say that the documents don't seem to cover it rather than guessing; the user can rephrase or add documents.`);
  const pn = await projectNotes(chat.id, chat.folder);
  if (pn) parts.push(`# Project instructions\nFrom the project's PELICA.md (or AGENTS.md / CLAUDE.md). Follow them when you work in this folder; the user's own instructions below come first.\n\n${pn}`);
  if (s.memoryOn) {
    const mem = memoryOf(chat.id);
    parts.push(`# Memory\nWhat you remember about the user from earlier chats (they can see and edit it in Settings → Memory)${mem ? `:\n${mem}` : ": nothing yet."}${has("remember") ? "\nWhen the user asks you to remember something, call remember. You may also remember a lasting fact on your own when you'll clearly need it again; mention it briefly when you do. When they ask you to forget something, or a fact changes, call forget." : ""}`);
  }
  if (app.get().voice) parts.push("# Voice\nThe user is talking to you by voice and hears your reply read aloud. Answer in a few short, plain spoken sentences, the way you'd say it out loud. No tables, lists, code or Markdown unless they ask for them.");
  if (s.customInstructions.trim()) parts.push(`# The user's instructions\n${s.customInstructions.trim()}`);
  return parts.join("\n\n");
}

const clip = (s: string, n: number) => (s.length > n ? s.slice(0, n) + `\n[…${s.length - n} more characters of older output trimmed]` : s);

/** The conversation as the engine takes it. Old tool output is trimmed. */
export function wire(chat: Chat, trim: boolean): WireMsg[] {
  const msgs = chat.messages.filter((m) => m.role !== "event") as Exclude<Msg, { role: "event" }>[];
  const users = msgs.map((m, i) => (m.role === "user" ? i : -1)).filter((i) => i >= 0);
  const keepFrom = users.length >= 2 ? users[users.length - 2] : 0;
  const lastUser = users.length ? users[users.length - 1] : 0;
  const out: WireMsg[] = [];
  msgs.forEach((m, i) => {
    if (m.role === "user") {
      const px = m.images?.length && app.get().status?.vision ? m.images.map((x) => pixelsOf(x.data)).filter((x): x is Pixels => !!x) : [];
      out.push({ role: "user", content: m.content, images: px.length ? px : undefined });
    }
    else if (m.role === "assistant") {
      out.push({ role: "assistant", content: m.content, reasoning: i > lastUser ? m.reasoning : undefined, tool_calls: m.tool_calls?.length ? m.tool_calls : undefined });
      // a call left without a result (stopped): say so
      for (const c of m.tool_calls || []) {
        if (!msgs.slice(i + 1).some((x) => x.role === "tool" && x.tool_call_id === c.id)) out.push({ role: "tool", tool_call_id: c.id, name: c.name, content: "Error: interrupted by the user." });
      }
    } else {
      // (a tool's pictures, for a model that sees; the old ones go when old results are trimmed)
      const px = m.images?.length && app.get().status?.vision && !(trim && i < keepFrom) ? m.images.map((x) => pixelsOf(x.data)).filter((x): x is Pixels => !!x) : [];
      out.push({ role: "tool", tool_call_id: m.tool_call_id, name: m.name, content: trim && i < keepFrom ? clip(m.content, 600) : m.content, images: px.length ? px : undefined });
    }
  });
  return out;
}

// ---------- the loop ----------

export interface LoopHooks {
  get(): Chat;
  set(fn: (c: Chat) => Chat): void;
  system(chat: Chat, tools: ToolSpec[]): Promise<string>;
  tools(chat: Chat): { tools: ToolSpec[]; mcp: ToolSpec[] };
  maxSteps: number;
  /** may compact before a step */
  compact?: boolean;
  label?: Draft["phase"];
  /** the chat whose view shows the draft (default: this one) */
  view?: string;
}

export interface LoopResult {
  changed: Set<string>;
  steps: number;
  finish: string;
}

function effortNow(): string | undefined {
  const st = app.get().status;
  const want = app.get().settings.effort;
  if (!st?.efforts?.length) return undefined;
  return st.efforts.includes(want) ? want : st.effort || undefined;
}

/** Small models sometimes echo the tool format tags into their answer. */
const untag = (s: string) => s.replace(/<\/?(tool_response|tool_call|tool_result)>/g, "");

/** A reply's speed readings, kept small: 48 points at most, and the fastest. */
function speedStats(s: number[]): { speed?: number[]; peak?: number } {
  if (s.length < 2) return {};
  const n = Math.min(48, s.length);
  const out = Array.from({ length: n }, (_, i) => {
    const a = Math.floor((i * s.length) / n), b = Math.max(a + 1, Math.floor(((i + 1) * s.length) / n));
    const part = s.slice(a, b);
    return Math.round((part.reduce((x, y) => x + y, 0) / part.length) * 100) / 100;
  });
  return { speed: out, peak: Math.max(...s) };
}

/** Streams engine events into the draft, a frame at a time. */
function drafter(chatId: string, phase: Draft["phase"]) {
  const d: Draft = { chatId, content: "", reasoning: "", phase: "waiting", tps: 0, generated: 0, started: Date.now() };
  let call: WireCall | null = null;
  let cut: Extract<ChatEvent, { t: "cut" }> | null = null;
  let parts: SpeedParts | undefined;
  // tok/s moment to moment: tokens between two stats events (every 250 ms), smoothed
  const speed: number[] = [];
  let lastS: { n: number; t: number } | null = null;
  let ema = 0;
  let raf = 0;
  const flush = () => {
    raf = 0;
    app.set({ draft: { ...d, content: untag(d.content) } });
  };
  const kick = () => {
    if (!raf) raf = requestAnimationFrame(flush);
  };
  app.set({ draft: { ...d, phase: phase === "cqs" ? "cqs" : "waiting" } });
  const on = (e: ChatEvent) => {
    switch (e.t) {
      case "start":
        d.phase = "prefill";
        d.prefill = { done: e.reused, total: e.prompt };
        break;
      case "prefill":
        d.prefill = { done: e.done, total: e.total };
        break;
      case "think":
        d.phase = phase === "compacting" || phase === "cqs" ? phase : "thinking";
        d.reasoning += e.d;
        d.generated++;
        break;
      case "text":
        d.phase = phase === "compacting" || phase === "cqs" ? phase : "writing";
        d.content += e.d;
        break;
      case "stats":
        if (lastS && e.elapsed_s > lastS.t + 0.05) {
          const now = (e.generated - lastS.n) / (e.elapsed_s - lastS.t);
          ema = ema ? ema * 0.6 + now * 0.4 : now;
          speed.push(Math.round(ema * 100) / 100);
        }
        lastS = { n: e.generated, t: e.elapsed_s };
        d.tps = e.tps;
        d.generated = e.generated;
        d.speed = speed.slice(-48);
        break;
      case "tool_call":
        call = { id: e.id, name: e.name || "unknown", arguments: e.arguments ?? {} };
        // a call the model wrote as plain JSON: keep only the text before it
        // (and drop a last line that only introduces it, like "Here is the call:")
        if (e.before !== undefined) d.content = e.before.replace(/\n*[^\n]*:\s*$/, "");
        d.phase = "tool";
        d.toolName = call.name;
        break;
      case "cut":
        cut = e;
        return;
      case "speed": {
        const { t: _t, ...rest } = e;
        parts = rest;
        return;
      }
      case "experts":
        if (e.layers && e.experts) experts.grid(e.layers, e.experts);
        if (e.hits?.length) experts.hit(e.hits);
        return;
      default:
        break;
    }
    kick();
  };
  return { d, on, call: () => call, cut: () => cut, speed: () => speed, parts: () => parts, end: () => raf && cancelAnimationFrame(raf) };
}

const callKey = (c: WireCall) => c.name + " " + JSON.stringify(c.arguments ?? {});
const LOOKUPS = new Set(["web_search", "open_page"]);

/** Why a call shouldn't run: it repeats one made since the user's last message
 *  (small models can get stuck calling the same thing over and over). */
function repeated(chat: Chat, call: WireCall): string | null {
  const msgs = chat.messages;
  let i = msgs.length - 1;
  while (i >= 0 && msgs[i].role !== "user") i--;
  const calls = msgs.slice(i + 1).flatMap((m) => (m.role === "assistant" ? m.tool_calls || [] : []));
  const prior = calls.slice(0, -1);
  const key = callKey(call);
  if (LOOKUPS.has(call.name) && prior.some((c) => callKey(c) === key)) return `You already called ${call.name} with exactly these arguments; its result is above. Don't repeat it: use what you found, try a different query, or answer now.`;
  if (prior.length && callKey(prior[prior.length - 1]) === key) return `You just made this exact call and got its result above. Don't repeat it: do something different or answer now.`;
  if (call.name === "web_search" && prior.filter((c) => c.name === "web_search").length >= 8) return "That's enough searching for this request. Answer from what you have found so far.";
  return null;
}

// a reply that hands the work back to the user ("open an editor, paste this…")
const MANUAL = /\b(open (a|your|the) (text |code )?editor|open (vs ?code|notepad)|create a (new )?(file|folder|directory)( (named|called))?|make a (new )?(file|folder) (named|called)|copy (and paste )?(the|this|that|the following) code|paste (it|the|this|that) (code )?(in|into)|save (the|this|it) (file )?as|save the file|open (a|your|the) (terminal|command prompt|powershell|cmd)|in (your|a|the) (terminal|command prompt|cmd)|run the following (command|commands)|type the following|follow these steps|here('s| is| are) (how|the steps)|you(?:'ll| will)? need to (create|install|run|open|save)|step \d+\s*[:.)-]\s*(create|open|install|save|copy|paste|run))/i;
// a reply that ends by promising a step it never took ("Now I'll create the file.")
const PROMISE = /\b(let me|let's|i(?:'ll| will| am going to|'m going to)|now i(?:'ll| will)|next,? i(?:'ll| will))\b[^\n]{0,80}\b(create|write|run|build|edit|update|make|install|start|read|look|check|open|fix|add|search|test)\b[^\n]{0,120}[.:…]?\s*$/i;

/** Why to send a tool-capable reply back: it handed the work to the user, or
 *  promised a step and stopped. "" when it's fine. */
function handBack(text: string, tools: ToolSpec[]): string {
  if (!tools.some((t) => t.name === "write_file")) return "";
  const t = text.trim();
  if (MANUAL.test(t) && (t.includes("```") || /^\s*\d+[.)]\s/m.test(t)))
    return "Your previous reply told the user how to do the work by hand (create files, paste code, run commands). That's not how you work here: you have the tools, so do it yourself now. Call write_file for each file with its full content, edit_file for changes, and run_command to install, build and run it. Then report what you did.";
  if (PROMISE.test(t.slice(-300)))
    return "Your previous reply said you would take a step but stopped without calling a tool. Don't announce it: make the tool call now and carry on until the task is finished.";
  return "";
}

export async function loop(h: LoopHooks): Promise<LoopResult> {
  const changed = new Set<string>();
  let steps = 0;
  let finish = "stop";
  let skipped = 0;
  let nudged = 0;
  let nudge = "";
  const s = app.get().settings;
  for (; steps < h.maxSteps; steps++) {
    if (stopFlag.stopped) {
      finish = "stopped";
      break;
    }
    let chat = h.get();
    // after two refused repeats the model answers without tools
    const stuck = skipped >= 2;
    const { tools, mcp } = stuck ? { tools: [], mcp: [] } : h.tools(chat);
    const all = [...tools, ...mcp];
    let system = await h.system(chat, all);
    if (stuck) system += "\n\nYou kept repeating the same tool call. Tools are now off: answer the user from what you already have.";
    if (nudge) system += "\n\n" + nudge;
    nudge = "";
    // the photos resized for a model that sees them (once each)
    if (app.get().status?.vision) await preparePixels(chat.messages.flatMap((m) => ((m.role === "user" || m.role === "tool") && m.images ? m.images.map((x) => x.data) : [])));
    let msgs = wire(chat, s.trimToolResults);
    // the context: count it, compact when it's nearly full
    const ctx = app.get().status?.ctx || 32768;
    try {
      const c = await engine.count({ system, messages: msgs, tools, mcp_tools: mcp });
      if (!c.error && h.compact) app.set({ context: { ...c, ctx } });
      if (h.compact && s.autocompact && c.total > s.compactAt * ctx && chat.messages.filter((m) => m.role !== "event").length > 4) {
        await compact("auto", "", h);
        chat = h.get();
        msgs = wire(chat, s.trimToolResults);
      }
    } catch {}
    const dr = drafter(h.view || chat.id, h.label || "writing");
    ctl = new AbortController();
    let done;
    try {
      done = await engine.chat({ messages: [{ role: "system", content: system }, ...msgs], tools: all, effort: effortNow(), max_tokens: s.maxTokens, experts: true, save: s.saveChats }, dr.on, ctl.signal);
    } finally {
      dr.end();
    }
    const call = done.finish === "tool_call" ? dr.call() : null;
    // up to twice per run: it told the user to do by hand what it could have
    // done itself, or promised a step and stopped; the reply is dropped and redone
    const why = !call && nudged < 2 && !stopFlag.stopped ? handBack(dr.d.content, all) : "";
    if (why) {
      nudged++;
      nudge = why;
      app.set({ draft: null });
      continue;
    }
    const st = app.get().status;
    const amsg: Msg = {
      id: uid(),
      role: "assistant",
      content: untag(dr.d.content).trim(),
      reasoning: dr.d.reasoning.trim() || undefined,
      tool_calls: call ? [call] : undefined,
      stats: { tps: done.tps, generated: done.generated, prompt: done.prompt, prefill_s: done.prefill_s, decode_s: done.decode_s, think_s: done.think_s, ctx_used: done.ctx_used, ctx: done.ctx, first_s: done.first_s, ...speedStats(dr.speed()), parts: dr.parts() },
      model: st?.model,
      effort: effortNow(),
      ts: Date.now(),
    };
    h.set((c) => ({ ...c, messages: [...c.messages, amsg] }));
    app.set({ draft: null });
    finish = done.finish;
    // a reply that ran out of room: say where, and how to get past it
    const cut = dr.cut();
    if (cut && h.compact && !stopFlag.stopped)
      pushEvent(
        "info",
        cut.while === "thinking"
          ? `The reply filled the context window (${Math.round((done.ctx || 0) / 1024)}K) while still thinking. Load the model with a larger context, use a lower effort, or compact the chat, then send "continue".`
          : `The answer reached the length limit (Settings → Max tokens). Send "continue" for the rest, or raise the limit.`,
      );
    if (!call || stopFlag.stopped) break;
    // run the call, showing it as it goes
    const tid = uid();
    const placeholder: Msg = { id: tid, role: "tool", tool_call_id: call.id, name: call.name, content: "", ui: { kind: "command", title: call.name, running: true }, ts: Date.now() };
    h.set((c) => ({ ...c, messages: [...c.messages, placeholder] }));
    const live = (ui: Partial<ToolUI>) => h.set((c) => ({ ...c, messages: c.messages.map((m) => (m.id === tid && m.role === "tool" ? { ...m, ui: { ...m.ui, ...ui } } : m)) }));
    app.set({ draft: { chatId: h.view || chat.id, content: "", reasoning: "", phase: h.label === "cqs" ? "cqs" : "tool", tps: 0, generated: 0, toolName: call.name, started: Date.now() } });
    const again = repeated(h.get(), call);
    if (again) skipped++;
    const r = again ? { content: again, ui: { kind: "error" as const, title: call.name, error: "Skipped a repeated call" } } : await runTool(h.get(), call.name, call.arguments, live, stopFlag);
    app.set({ draft: null });
    if (r.ui.path && (r.ui.kind === "write" || r.ui.kind === "edit")) changed.add(r.ui.path);
    h.set((c) => ({ ...c, todos: r.todos ?? c.todos, messages: c.messages.map((m) => (m.id === tid && m.role === "tool" ? { ...m, content: r.content, ui: r.ui, ...("images" in r && r.images ? { images: r.images } : {}) } : m)) }));
  }
  if (steps >= h.maxSteps) finish = "steps";
  return { changed, steps, finish };
}

function mainHooks(): LoopHooks {
  const s = app.get().settings;
  return {
    get: () => liveChat() || app.get().chat!,
    set: updateLive,
    system: systemPrompt,
    tools: (chat) => {
      const lastUser = [...chat.messages].reverse().find((m) => m.role === "user")?.content || "";
      return toolsFor(chat, { web: s.web, lastUser });
    },
    maxSteps: s.maxSteps,
    compact: true,
  };
}

function titleFrom(text: string): string {
  const t = text.replace(/\s+/g, " ").trim();
  return t.length > 48 ? t.slice(0, 46).replace(/\s\S*$/, "") + "…" : t || "New chat";
}

/** Sends a user message and runs the agent until it answers. */
export async function send(text: string, images?: Photo[]) {
  const st = app.get();
  if (st.running) return toast(st.liveId && st.liveId !== st.chat?.id ? "Pelica is replying in another chat: wait for it, or stop it." : "Wait for the reply to finish, or stop it first.");
  if (st.status?.state !== "ready") return toast(st.status?.state === "loading" ? "The model is still loading." : "Load a model first (model menu under the message box).");
  if (!st.chat) return;
  // a project with documents: the passages that match the question go with it
  const folders = projectDocs(st.chat, st.settings.projects);
  if (folders.length && !text.includes("<documents>")) {
    const before = [...st.chat.messages].reverse().find((x) => x.role === "user");
    text += await excerptsFor(folders, splitAttachments(text).text, before ? splitAttachments(before.content).text : "");
  }
  const m: Msg = { id: uid(), role: "user", content: text, ts: Date.now(), images: images?.length ? images : undefined };
  updateChat((c) => ({ ...c, title: c.messages.length ? c.title : titleFrom(text), messages: [...c.messages, m] }));
  await run();
}

/** Runs the agent on the open chat as it stands (send, retry, continue), or
 *  on `target` without opening it (background and scheduled tasks). */
export async function run(target?: Chat) {
  const chat = target || app.get().chat;
  if (app.get().running || !chat) return;
  stopFlag = { stopped: false };
  const id = chat.id;
  let failed = false;
  beginLive(chat);
  app.set({ running: true });
  try {
    const r = await loop(mainHooks());
    if (r.finish === "steps") pushEvent("info", `Stopped after ${r.steps} steps (the limit in Settings → Capabilities). Send "continue" to keep going.`);
    if (r.changed.size && app.get().settings.cqsAuto && !stopFlag.stopped && liveChat()?.folder) {
      const { runCQS } = await import("./cqs");
      await runCQS([...r.changed], "check");
    }
  } catch (e: any) {
    failed = true;
    if (!stopFlag.stopped) pushEvent("error", String(e?.message || e));
  } finally {
    app.set({ running: false, draft: null });
    endLive();
    refreshChats();
  }
  // a message queued while this ran goes now, if the reply ended by itself
  // and its chat is still open (stopped or failed: it waits in the box)
  const q = app.get().queued;
  if (q && q.chatId === id && !stopFlag.stopped && !failed && app.get().chat?.id === id) {
    app.set({ queued: null });
    await send(q.text, q.images);
  }
}

/** A note in the conversation (the one being replied in, else the open one). */
export function pushEvent(kind: "compact" | "cqs" | "error" | "info", text: string, detail?: string) {
  updateLive((c) => ({ ...c, messages: [...c.messages, { id: uid(), role: "event", kind, text, detail, ts: Date.now() }] }));
}

/** Runs a fresh, separate agent (CQS's fixer): its own conversation, the same model. */
export async function subagent(opts: { folder: string; system: string; task: string; tools: string[]; maxSteps: number; onStep?: (m: Msg) => void }): Promise<{ summary: string; changed: string[] }> {
  let chat: Chat = { id: uid(), title: "subagent", created: Date.now(), updated: Date.now(), folder: opts.folder, messages: [{ id: uid(), role: "user", content: opts.task, ts: Date.now() }] };
  const s = app.get().settings;
  const r = await loop({
    get: () => chat,
    set: (fn) => {
      const before = chat.messages.length;
      chat = fn(chat);
      if (opts.onStep) for (const m of chat.messages.slice(before)) opts.onStep(m);
    },
    system: async () => opts.system,
    tools: (c) => ({ tools: toolsFor(c, { web: s.web, lastUser: "work" }).tools.filter((t) => opts.tools.includes(t.name)), mcp: [] }),
    maxSteps: opts.maxSteps,
    label: "cqs",
    view: liveChat()?.id || app.get().chat?.id,
  });
  const last = [...chat.messages].reverse().find((m) => m.role === "assistant" && m.content) as Extract<Msg, { role: "assistant" }> | undefined;
  return { summary: last?.content || "", changed: [...r.changed] };
}

// ---------- compaction ----------

function transcript(msgs: Msg[], maxChars: number): string {
  const lines: string[] = [];
  for (const m of msgs) {
    if (m.role === "user") lines.push(`User: ${m.content}`);
    else if (m.role === "assistant") {
      if (m.content) lines.push(`Assistant: ${m.content}`);
      for (const c of m.tool_calls || []) lines.push(`Assistant called ${c.name} ${JSON.stringify(c.arguments).slice(0, 400)}`);
    } else if (m.role === "tool") lines.push(`Result of ${m.name}: ${m.content.slice(0, 700)}${m.content.length > 700 ? " …" : ""}`);
  }
  let t = lines.join("\n\n");
  if (t.length > maxChars) t = "[…earlier part cut]\n" + t.slice(-maxChars);
  return t;
}

/** Replaces the conversation so far with a summary the model writes. */
export async function compact(reason: "manual" | "auto", focus = "", h?: Pick<LoopHooks, "get" | "set">) {
  const chat = h ? h.get() : app.get().chat;
  const set = h ? h.set : updateChat;
  if (!chat) return;
  const live = chat.messages.filter((m) => m.role !== "event");
  if (live.length < 3) return toast("Nothing to compact yet.");
  // mid-task (auto): keep the current turn as it is
  const lastUserIdx = reason === "auto" ? chat.messages.map((m) => m.role).lastIndexOf("user") : chat.messages.length;
  const head = chat.messages.slice(0, lastUserIdx);
  const tail = chat.messages.slice(lastUserIdx);
  if (head.filter((m) => m.role !== "event").length < 2) return;
  const ctx = app.get().status?.ctx || 32768;
  const text = transcript(head, Math.max(4000, Math.floor(ctx * 0.55 * 3)));
  const st = app.get().status;
  const effort = st?.efforts?.includes("off") ? "off" : st?.efforts?.includes("low") ? "low" : undefined;
  const dr = drafter(chat.id, "compacting");
  dr.d.phase = "compacting";
  let summary = "";
  try {
    await engine.chat(
      {
        messages: [
          { role: "system", content: "You write complete, compact summaries of conversations, so the work can continue with only the summary." },
          {
            role: "user",
            content: `Summarize the conversation below so it can be continued from your summary alone. Keep: what the user wants and asked for; decisions and preferences; files created or changed (paths) and what they contain; commands run and their results; facts and data found; the current state; unfinished work and next steps. Use short headings and bullet points.${focus ? ` Pay special attention to: ${focus}.` : ""}\n\n<conversation>\n${text}\n</conversation>`,
          },
        ],
        tools: [],
        effort,
        max_tokens: 2000,
      },
      dr.on,
    );
    summary = dr.d.content.trim();
  } finally {
    dr.end();
    app.set({ draft: null });
  }
  if (!summary) throw new Error("the model wrote an empty summary");
  const t = Date.now();
  set((c) => ({
    ...c,
    archive: [...(c.archive || []), ...head],
    messages: [
      { id: uid(), role: "event", kind: "compact", text: reason === "auto" ? "Context nearly full: conversation compacted" : "Conversation compacted", detail: summary, ts: t },
      { id: uid(), role: "user", content: `Here is a summary of our conversation so far:\n\n${summary}`, summary: true, ts: t },
      { id: uid(), role: "assistant", content: "Got it. I'll continue from this summary.", ack: true, ts: t },
      ...tail,
    ],
  }));
  app.set({ context: null });
}

/** /compact as a command: runs with the busy flag. */
export async function compactNow(focus = "") {
  if (app.get().running) return;
  app.set({ running: true });
  stopFlag = { stopped: false };
  try {
    await compact("manual", focus);
    toast("Conversation compacted.");
  } catch (e: any) {
    pushEvent("error", `Compacting failed: ${e?.message || e}`);
  } finally {
    app.set({ running: false, draft: null });
    refreshContext();
  }
}

/** Counts the open chat's context now (for the meter). */
export async function refreshContext() {
  const chat = app.get().chat;
  const st = app.get().status;
  if (!chat || st?.state !== "ready") return;
  const h = mainHooks();
  const { tools, mcp } = h.tools(chat);
  const system = await systemPrompt(chat, [...tools, ...mcp]);
  try {
    const c = await engine.count({ system, messages: wire(chat, app.get().settings.trimToolResults), tools, mcp_tools: mcp });
    if (!c.error) app.set({ context: { ...c, ctx: st.ctx || 32768 } });
  } catch {}
}
