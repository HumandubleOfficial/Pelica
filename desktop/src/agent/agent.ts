// The agent loop: ask the engine (chat API v2) for the next step, run the
// tool it calls, repeat until it answers. Long tasks run up to `maxSteps`
// steps; the context is kept in check by trimming old tool output and by
// compacting the conversation (automatically past a threshold).
import { bridge } from "../lib/bridge";
import { engine, type ChatEvent, type ToolSpec, type WireCall, type WireMsg } from "../lib/engine";
import { app, beginLive, endLive, experts, liveChat, refreshChats, toast, uid, updateChat, updateLive, type Chat, type Draft, type Msg, type Photo, type ToolUI } from "../state";
import { runTool, toolsFor } from "./tools";

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

/** Project notes the model should know: PELICA.md (or AGENTS.md / CLAUDE.md) in the folder. */
async function projectNotes(folder: string | null): Promise<string> {
  if (!folder || !bridge) return "";
  if (notes.has(folder)) return notes.get(folder)!;
  const text = await bridge.fs.notes(folder).catch(() => "");
  notes.set(folder, text);
  return text;
}

export function forgetNotes(folder: string) {
  notes.delete(folder);
}

export async function systemPrompt(chat: Chat, tools: ToolSpec[]): Promise<string> {
  const s = app.get().settings;
  const st = app.get().status;
  const has = (n: string) => tools.some((t) => t.name === n);
  const parts: string[] = [];
  parts.push(`You are Pelica, a capable assistant running locally on the user's Windows PC${st?.model ? ` (model: ${st.model})` : ""}. Today is ${today()}.${s.userName ? ` The user's name is ${s.userName}.` : ""}`);
  if (has("write_file")) {
    parts.push(
      `You are working in the folder ${chat.folder}. You can list, read, create and edit files there and run commands (the user approves each command). Paths are relative to that folder.
You do the work yourself with your tools: create and change files with write_file and edit_file, and run programs with run_command. Never tell the user to open an editor, copy code into a file or type commands themselves.
How to work:
- Look before you change: list_files and read_file to understand the project first.
- For any task with several steps, write a plan with todo_write, then work through it item by item, updating it as you go, until everything is done. Don't stop halfway or ask "should I continue?" — finish the job.
- Write complete, working code: no placeholders like "rest of the code here". Use edit_file for focused changes and write_file for new files or full rewrites.
- Test what you build: run it, its tests or a build with run_command, read the errors, and fix them.
- When you've built something the user can run, finish with the command to run it in its own \`\`\`bash block (one command per block).`,
    );
  } else if (has("read_file")) {
    parts.push(`You are working in the folder ${chat.folder}, in plan mode: you can list, read and search files, but not change anything or run commands. Investigate what the user asks about, then write a clear, step-by-step plan (files to create or change, and how). The user switches to another mode under the message box when they want you to carry it out.`);
  } else if (chat.folder === null && bridge) {
    parts.push("No workspace folder is chosen for this chat, so you can't touch files or run commands. If the user wants files created or code run, tell them to pick a folder with the folder button under the message box.");
  }
  if (has("web_search")) {
    parts.push("You can search the web (web_search) and read pages (open_page). Search only when the request needs current or outside information: news, prices, recent releases or versions, or facts you aren't sure of. Never search for greetings, small talk, opinions, math, writing, coding help or anything you already know. When a request does need a search, call web_search first, before writing any answer: don't guess and then search. After searching, open the best results and answer from what you read, citing the links.");
  } else {
    parts.push("If you are not sure of a fact, or it may have changed since your training, say so instead of guessing. Never invent names, numbers, quotes, citations or links.");
  }
  if (has("system_info")) parts.push("You can see facts about this computer (system_info), read the clipboard when the user mentions something they copied (clipboard_read, they approve it), and copy text for them (clipboard_write)." + (has("open_file") ? " open_file opens a workspace file in its app, e.g. a web page you built in the browser." : "") + " Files the user attaches arrive in <attachment> blocks in their message.");
  if (tools.some((t) => t.name.startsWith("mcp__"))) parts.push("Tools named mcp__<server>__<tool> come from the user's MCP servers.");
  parts.push("Answer in Markdown. Be direct and concise unless the user asks for depth.");
  const pn = await projectNotes(chat.folder);
  if (pn) parts.push(`Project notes (from the folder):\n${pn}`);
  if (s.customInstructions.trim()) parts.push(`The user's instructions:\n${s.customInstructions.trim()}`);
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
    if (m.role === "user") out.push({ role: "user", content: m.content, images: m.images?.length && app.get().status?.vision ? m.images.map((x) => x.data.replace(/^data:[^,]*,/, "")) : undefined });
    else if (m.role === "assistant") {
      out.push({ role: "assistant", content: m.content, reasoning: i > lastUser ? m.reasoning : undefined, tool_calls: m.tool_calls?.length ? m.tool_calls : undefined });
      // a call left without a result (stopped): say so
      for (const c of m.tool_calls || []) {
        if (!msgs.slice(i + 1).some((x) => x.role === "tool" && x.tool_call_id === c.id)) out.push({ role: "tool", tool_call_id: c.id, name: c.name, content: "Error: interrupted by the user." });
      }
    } else out.push({ role: "tool", tool_call_id: m.tool_call_id, name: m.name, content: trim && i < keepFrom ? clip(m.content, 600) : m.content });
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

/** Streams engine events into the draft, a frame at a time. */
function drafter(chatId: string, phase: Draft["phase"]) {
  const d: Draft = { chatId, content: "", reasoning: "", phase: "waiting", tps: 0, generated: 0, started: Date.now() };
  let call: WireCall | null = null;
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
        d.tps = e.tps;
        d.generated = e.generated;
        break;
      case "tool_call":
        call = { id: e.id, name: e.name || "unknown", arguments: e.arguments ?? {} };
        // a call the model wrote as plain JSON: keep only the text before it
        // (and drop a last line that only introduces it, like "Here is the call:")
        if (e.before !== undefined) d.content = e.before.replace(/\n*[^\n]*:\s*$/, "");
        d.phase = "tool";
        d.toolName = call.name;
        break;
      case "experts":
        if (e.layers && e.experts) experts.grid(e.layers, e.experts);
        if (e.hits?.length) experts.hit(e.hits);
        return;
      default:
        break;
    }
    kick();
  };
  return { d, on, call: () => call, end: () => raf && cancelAnimationFrame(raf) };
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
const MANUAL = /\b(open (a|your) (text )?editor|create a (new )?file (named|called)|copy (the|this) code|paste (the|this) code|save the file|open a (terminal|command prompt)|in your terminal|run the following command)/i;

export async function loop(h: LoopHooks): Promise<LoopResult> {
  const changed = new Set<string>();
  let steps = 0;
  let finish = "stop";
  let skipped = 0;
  let nudged = false;
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
      done = await engine.chat({ messages: [{ role: "system", content: system }, ...msgs], tools: all, effort: effortNow(), max_tokens: s.maxTokens, experts: true }, dr.on, ctl.signal);
    } finally {
      dr.end();
    }
    const call = done.finish === "tool_call" ? dr.call() : null;
    // once per run: it told the user to do by hand what it could have done itself
    if (!call && !nudged && !stopFlag.stopped && all.some((t) => t.name === "write_file") && dr.d.content.includes("```") && MANUAL.test(dr.d.content)) {
      nudged = true;
      nudge = "Your previous attempt told the user to create the files and run the commands by hand. Don't do that: do it yourself now. Call write_file to create each file with its full content, then run_command to run it.";
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
      stats: { tps: done.tps, generated: done.generated, prompt: done.prompt, prefill_s: done.prefill_s, decode_s: done.decode_s, think_s: done.think_s, ctx_used: done.ctx_used, ctx: done.ctx },
      model: st?.model,
      effort: effortNow(),
      ts: Date.now(),
    };
    h.set((c) => ({ ...c, messages: [...c.messages, amsg] }));
    app.set({ draft: null });
    finish = done.finish;
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
    h.set((c) => ({ ...c, todos: r.todos ?? c.todos, messages: c.messages.map((m) => (m.id === tid && m.role === "tool" ? { ...m, content: r.content, ui: r.ui } : m)) }));
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
  const m: Msg = { id: uid(), role: "user", content: text, ts: Date.now(), images: images?.length ? images : undefined };
  updateChat((c) => ({ ...c, title: c.messages.length ? c.title : titleFrom(text), messages: [...c.messages, m] }));
  await run();
}

/** Runs the agent on the open chat as it stands (send, retry, continue). */
export async function run() {
  if (app.get().running || !app.get().chat) return;
  stopFlag = { stopped: false };
  beginLive(app.get().chat!);
  app.set({ running: true });
  try {
    const r = await loop(mainHooks());
    if (r.finish === "steps") pushEvent("info", `Stopped after ${r.steps} steps (the limit in Settings → Capabilities). Send "continue" to keep going.`);
    if (r.changed.size && app.get().settings.cqsAuto && !stopFlag.stopped && liveChat()?.folder) {
      const { runCQS } = await import("./cqs");
      await runCQS([...r.changed], "check");
    }
  } catch (e: any) {
    if (!stopFlag.stopped) pushEvent("error", String(e?.message || e));
  } finally {
    app.set({ running: false, draft: null });
    endLive();
    refreshChats();
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
