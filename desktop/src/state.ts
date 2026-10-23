// App state: one small store (useSyncExternalStore), the settings, chats and
// their persistence.
import { useSyncExternalStore } from "react";
import { bridge, type ChatMeta, type McpServer } from "./lib/bridge";
import type { Status, WireCall } from "./lib/engine";

type Listener = () => void;

export function createStore<T extends object>(init: T) {
  let s = init;
  const ls = new Set<Listener>();
  return {
    get: () => s,
    set(p: Partial<T> | ((s: T) => Partial<T>)) {
      s = { ...s, ...(typeof p === "function" ? p(s) : p) };
      ls.forEach((l) => l());
    },
    sub(l: Listener) {
      ls.add(l);
      return () => {
        ls.delete(l);
      };
    },
  };
}

export function useStore<T extends object, U>(store: ReturnType<typeof createStore<T>>, sel: (s: T) => U): U {
  return useSyncExternalStore(store.sub, () => sel(store.get()));
}

// ---------- chats ----------

export interface ToolUI {
  kind: "search" | "page" | "read" | "write" | "edit" | "list" | "grep" | "command" | "todo" | "browser" | "mcp" | "trash" | "error" | "denied" | "cqs" | "system" | "clipboard" | "open";
  title: string;
  detail?: string;
  path?: string;
  before?: string | null;
  after?: string;
  output?: string;
  code?: number | null;
  results?: { title: string; url: string }[];
  url?: string;
  todos?: Todo[];
  error?: string;
  running?: boolean;
}

export interface Todo {
  content: string;
  status: "pending" | "in_progress" | "completed";
}

export interface Stats {
  tps: number;
  generated: number;
  prompt: number;
  prefill_s: number;
  decode_s: number;
  think_s?: number;
  ctx_used: number;
  ctx: number;
}

export type Msg =
  | { id: string; role: "user"; content: string; ts: number; summary?: boolean; images?: Photo[] }
  | { id: string; role: "assistant"; content: string; reasoning?: string; tool_calls?: WireCall[]; stats?: Stats; model?: string; effort?: string; ts: number; ack?: boolean }
  | { id: string; role: "tool"; tool_call_id: string; name: string; content: string; ui: ToolUI; ts: number }
  | { id: string; role: "event"; kind: "compact" | "cqs" | "error" | "info"; text: string; detail?: string; ts: number };

/** A photo sent with a message (downscaled, as a data: URL). */
export interface Photo {
  name: string;
  data: string;
}

export interface Chat {
  id: string;
  title: string;
  created: number;
  updated: number;
  pinned?: boolean;
  folder: string | null;
  messages: Msg[];
  /** messages before the last compaction (shown, never sent) */
  archive?: Msg[];
  todos?: Todo[];
}

export const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);

export function newChat(folder: string | null): Chat {
  const t = Date.now();
  return { id: uid(), title: "New chat", created: t, updated: t, folder, messages: [] };
}

// ---------- settings ----------

export interface Settings {
  theme: "dark" | "warm" | "light" | "midnight" | "neural";
  background: "none" | "experts";
  userName: string;
  web: boolean;
  effort: string;
  autocompact: boolean;
  compactAt: number;
  trimToolResults: boolean;
  maxSteps: number;
  customInstructions: string;
  cqsAuto: boolean;
  cqsFix: boolean;
  shell: "cmd" | "powershell";
  downloadDir: string;
  modelDirs: string[];
  mcpServers: { name: string; command: string; args?: string[]; env?: Record<string, string>; disabled?: boolean }[];
  lastModel?: string;
  lastCtx: number;
  folder: string | null;
  fontSize: number;
  sendOnEnter: boolean;
  maxTokens: number;
  animations: boolean;
  /** Windows 11 backdrop behind the window */
  material: "none" | "mica" | "acrylic";
  /** closing the window keeps Pelica in the tray */
  closeToTray: boolean;
  /** a desktop notification when a reply finishes (or a command waits) while you're elsewhere */
  notify: boolean;
  /** system_info, clipboard and open_file tools */
  computerTools: boolean;
  /** what the agent may do without asking: ask (files and commands ask),
   *  edits (files automatic, commands ask), plan (read only), auto (nothing asks) */
  permMode: PermMode;
  /** folders picked before, newest first (each chat has its own) */
  recentFolders: string[];
  /** CQS runs the project's checkers without asking (except in Ask mode) */
  cqsQuiet: boolean;
  /** the panel on the right of a chat (environment, context, connectors, outputs) */
  sidePanel: boolean;
  /** load the last model when Pelica starts (off: start with none loaded) */
  loadOnStart: boolean;
  /** its sections that are folded */
  panelFolded: string[];
}

export type PermMode = "ask" | "edits" | "plan" | "auto";
export const PERM_MODES: { id: PermMode; label: string; desc: string }[] = [
  { id: "ask", label: "Ask first", desc: "File changes and commands each ask for your OK" },
  { id: "edits", label: "Auto-edit", desc: "Files change without asking; commands ask" },
  { id: "plan", label: "Plan", desc: "Read only: looks around and writes a plan, changes nothing" },
  { id: "auto", label: "Full auto", desc: "Files and commands run without asking" },
];

export const DEFAULTS: Settings = {
  theme: "dark",
  background: "none",
  userName: "",
  web: true,
  effort: "",
  autocompact: true,
  compactAt: 0.8,
  trimToolResults: true,
  maxSteps: 150,
  customInstructions: "",
  cqsAuto: true,
  cqsFix: true,
  shell: "cmd",
  downloadDir: "D:\\Models",
  modelDirs: [],
  mcpServers: [],
  lastCtx: 32768,
  folder: null,
  fontSize: 15,
  sendOnEnter: true,
  maxTokens: 8192,
  animations: true,
  material: "none",
  closeToTray: false,
  notify: true,
  computerTools: true,
  permMode: "edits",
  recentFolders: [],
  cqsQuiet: true,
  sidePanel: true,
  loadOnStart: false,
  panelFolded: [],
};

// ---------- the store ----------

export interface Approval {
  id: string;
  title: string;
  command: string;
  folder: string;
  why?: string;
  /** the chat it's for (it may not be the open one) */
  chatTitle?: string;
  resolve: (r: "allow" | "always" | "deny") => void;
}

export interface Draft {
  chatId: string;
  content: string;
  reasoning: string;
  phase: "waiting" | "prefill" | "thinking" | "writing" | "tool" | "compacting" | "cqs";
  prefill?: { done: number; total: number };
  tps: number;
  generated: number;
  toolName?: string;
  started: number;
}

export interface ContextInfo {
  system: number;
  messages: number;
  tool_results: number;
  tools: number;
  mcp: number;
  total: number;
  ctx: number;
}

export interface AppState {
  settings: Settings;
  status: Status | null;
  chats: ChatMeta[];
  chat: Chat | null;
  draft: Draft | null;
  running: boolean;
  view: "chat" | "library" | "brain";
  settingsOpen: boolean;
  settingsTab: string;
  approval: Approval | null;
  context: ContextInfo | null;
  mcp: McpServer[];
  sidebar: boolean;
  toast: string | null;
  search: string;
  palette: boolean;
  /** what the side panel previews (HTML, SVG, Markdown, or code) */
  artifact: Artifact | null;
  /** the chat a reply is being written in (it may not be the open one) */
  liveId: string | null;
  /** the prompt maker and the model comparison windows */
  promptMaker: boolean;
  compare: string | null;
  /** a photo shown full size */
  photo: string | null;
}

export interface Artifact {
  title: string;
  kind: "html" | "svg" | "markdown" | "code";
  code: string;
  lang?: string;
}

export const app = createStore<AppState>({
  settings: DEFAULTS,
  status: null,
  chats: [],
  chat: null,
  draft: null,
  running: false,
  view: "chat",
  settingsOpen: false,
  settingsTab: "General",
  approval: null,
  context: null,
  mcp: [],
  sidebar: true,
  toast: null,
  search: "",
  palette: false,
  artifact: null,
  liveId: null,
  promptMaker: false,
  compare: null,
  photo: null,
});

export const useApp = <U,>(sel: (s: AppState) => U) => useStore(app, sel);

export async function loadSettings() {
  let s: Partial<Settings> = {};
  try {
    s = bridge ? await bridge.settings.get() : JSON.parse(localStorage.getItem("pelica.settings") || "{}");
  } catch {}
  const settings = { ...DEFAULTS, ...s };
  if (settings.folder && !settings.recentFolders.includes(settings.folder)) settings.recentFolders = [settings.folder, ...settings.recentFolders].slice(0, 8);
  app.set({ settings });
}

export function setSettings(p: Partial<Settings>) {
  const settings = { ...app.get().settings, ...p };
  app.set({ settings });
  if (bridge) bridge.settings.set(p).catch(() => {});
  else
    try {
      localStorage.setItem("pelica.settings", JSON.stringify(settings));
    } catch {}
}

export function toast(msg: string) {
  app.set({ toast: msg });
  setTimeout(() => app.get().toast === msg && app.set({ toast: null }), 3500);
}

// ---------- chat persistence ----------

const memChats = new Map<string, Chat>();

export async function refreshChats() {
  const chats = bridge ? await bridge.chats.list().catch(() => []) : [...memChats.values()].map((c) => ({ id: c.id, title: c.title, updated: c.updated, pinned: !!c.pinned, folder: c.folder }));
  app.set({ chats: chats.sort((a, b) => b.updated - a.updated) });
}

let saveTimer: ReturnType<typeof setTimeout> | null = null;

/** Replaces the open chat (and saves it, debounced). */
export function updateChat(fn: (c: Chat) => Chat) {
  const c = app.get().chat;
  if (!c) return;
  const next = { ...fn(c), updated: Date.now() };
  if (live && next.id === live.id) live = next;
  app.set({ chat: next });
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(() => persist(next), 400);
}

export async function persist(c: Chat) {
  if (!c.messages.length && !c.archive?.length) return;
  if (bridge) await bridge.chats.save(c).catch(() => {});
  else memChats.set(c.id, c);
  const meta = { id: c.id, title: c.title, updated: c.updated, pinned: !!c.pinned, folder: c.folder };
  app.set((s) => ({ chats: [meta, ...s.chats.filter((x) => x.id !== c.id)].sort((a, b) => b.updated - a.updated) }));
}

// ---------- the chat a reply is being written in ----------
// The agent works on its own copy, so the user can open other chats (or
// start new ones) meanwhile; the copy shows when its chat is the open one.

let live: Chat | null = null;
let liveTimer: ReturnType<typeof setTimeout> | null = null;
let finished: Chat | null = null;

export const liveChat = () => live;
/** The chat the last reply was written in (for the notification). */
export const lastFinished = () => finished;

export function beginLive(c: Chat) {
  live = c;
  app.set({ liveId: c.id });
  // (a new chat gets its sidebar row now, so the live dot has somewhere to show)
  if (!app.get().chats.some((x) => x.id === c.id)) persist(c);
}

export function endLive() {
  if (!live) return;
  if (liveTimer) clearTimeout(liveTimer);
  finished = live;
  persist(live);
  live = null;
  app.set({ liveId: null });
}

/** Replaces the live chat (and the open one, if it's the same). */
export function updateLive(fn: (c: Chat) => Chat) {
  if (!live) return updateChat(fn);
  live = { ...fn(live), updated: Date.now() };
  if (app.get().chat?.id === live.id) app.set({ chat: live });
  if (liveTimer) clearTimeout(liveTimer);
  const c = live;
  liveTimer = setTimeout(() => persist(c), 400);
}

export async function openChat(id: string) {
  if (live?.id === id) return app.set({ chat: live, view: "chat", context: null });
  const c: Chat | undefined = bridge ? await bridge.chats.load(id).catch(() => undefined) : memChats.get(id);
  if (c) app.set({ chat: c, view: "chat", context: null });
}

export function startNewChat() {
  app.set({ chat: newChat(null), view: "chat", context: null });
}

export async function deleteChat(id: string) {
  if (live?.id === id) return toast("Pelica is still replying in that chat: stop it first.");
  if (bridge) await bridge.chats.remove(id).catch(() => {});
  memChats.delete(id);
  app.set((s) => ({ chats: s.chats.filter((c) => c.id !== id), chat: s.chat?.id === id ? newChat(null) : s.chat }));
}

/** Sets the open chat's folder, remembering it among the recent ones. */
export function setChatFolder(p: string | null) {
  updateChat((c) => ({ ...c, folder: p }));
  if (p) setSettings({ recentFolders: [p, ...app.get().settings.recentFolders.filter((x) => x !== p)].slice(0, 8) });
}

/** A web link, in the default browser. */
export function openLink(url: string) {
  if (!/^https?:\/\//i.test(url)) return;
  if (bridge) bridge.sys.external(url).catch(() => {});
  else window.open(url, "_blank", "noopener");
}

// ---------- expert activity (outside React: the background canvas reads it) ----------

export const experts = {
  layers: 0,
  experts: 0,
  /** per (layer, expert): brightness at its last hit (0..1) and when that was */
  peak: new Float32Array(0),
  at: new Float64Array(0),
  /** picks seen this session, per (layer, expert) and in all */
  count: new Float64Array(0),
  picks: 0,
  last: 0,
  grid(layers: number, n: number) {
    if (layers !== this.layers || n !== this.experts) {
      this.layers = layers;
      this.experts = n;
      this.peak = new Float32Array(layers * n);
      this.at = new Float64Array(layers * n);
      this.count = new Float64Array(layers * n);
    }
  },
  /** How lit (layer, expert) `k` is at time `now` (performance.now()): fades by 4× a second. */
  heat(k: number, now: number): number {
    const p = this.peak[k];
    return p ? p * Math.pow(0.25, (now - this.at[k]) / 1000) : 0;
  },
  hit(h: number[]) {
    // brightness relative to the busiest expert of this update, so the
    // favourites stand out even when a burst of tokens touches nearly all
    let max = 1;
    for (let i = 2; i < h.length; i += 3) max = Math.max(max, h[i]);
    const now = performance.now();
    for (let i = 0; i + 2 < h.length; i += 3) {
      const k = h[i] * this.experts + h[i + 1];
      if (k >= this.peak.length) continue;
      this.peak[k] = Math.min(1, Math.max(this.heat(k, now), Math.pow(h[i + 2] / max, 1.4)));
      this.at[k] = now;
      this.count[k] += h[i + 2];
      this.picks += h[i + 2];
    }
    this.last = now;
  },
};
