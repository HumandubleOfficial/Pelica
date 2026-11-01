// App state: one small store (useSyncExternalStore), the settings, chats and
// their persistence.
import { useSyncExternalStore } from "react";
import { chatUsage, searchChats, type SearchHit, type Usage } from "../electron/chatdata";
import { bridge, type ChatMeta, type McpServer } from "./lib/bridge";
import type { Status, WireCall } from "./lib/engine";

/** Below this width the app is in compact mode (like Claude's small window). */
export const COMPACT = "(max-width: 820px)";

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
  /** a command's few words on what it does (the row shows them) */
  about?: string;
  /** GitHub safety check of the repositories it fetches or reads */
  safety?: { repo: string; verdict: "ok" | "caution" | "risky"; reasons: string[] }[];
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
  /** seconds to the first token */
  first_s?: number;
  /** tok/s through the reply (up to 48 points) and its fastest moment */
  speed?: number[];
  peak?: number;
  /** where the reply's time went (the speed panel; Qwen3.8 for now) */
  parts?: SpeedParts;
}

export interface SpeedParts {
  gpu_s: number;
  attention_s: number;
  experts_vram_s: number;
  experts_cpu_s: number;
  drive_s: number;
  other_s: number;
  in_vram: number;
  on_cpu: number;
  from_drive: number;
  drive_reads: number;
  ram_hits: number;
  drive_gb: number;
  passes: number;
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
  /** speed up (pelica repack) DeepSeek V4 files as soon as they finish downloading */
  autoRepack: boolean;
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
  /** its sections turned off (right-click the panel) */
  panelHidden: string[];
  /** its sections' order (dragged by their grips) */
  panelOrder: string[];
  /** replies in a serif face (as in Claude) or the interface's sans */
  replyFont: "serif" | "sans";
  /** a monospace font for code, diffs and the terminal ("" = Cascadia Code) */
  codeFont: string;
  /** folder groups folded in the sidebar (their paths) */
  sideFolded: string[];
  /** the sidebar's width in px (dragged at its edge) */
  sideWidth: number;
  /** other computers lending their RAM through `pelica bridge` */
  devices: { addr: string; code: string }[];
  /** the OpenAI-compatible local API on a fixed port, for other programs */
  apiServer: boolean;
  apiPort: number;
  /** where to look for new versions (a latest.json: web address or path; empty: Pelica's GitHub releases) */
  updateFeed: string;
  /** check the feed at start */
  updateAuto: boolean;
  /** scheduled tasks (agent/tasks.ts) */
  tasks: import("./agent/tasks").Task[];
  /** the version whose release notes were last shown (What's new) */
  lastSeenVersion: string;
  /** saved prompts (the + menu, the palette, and /name in the message box) */
  prompts: SavedPrompt[];
}

export interface SavedPrompt {
  name: string;
  text: string;
}

export type PermMode = "ask" | "edits" | "plan" | "auto";
export const PERM_MODES: { id: PermMode; label: string; desc: string }[] = [
  { id: "ask", label: "Manual", desc: "Always ask before making changes" },
  { id: "edits", label: "Accept edits", desc: "Automatically accept file edits; commands ask" },
  { id: "plan", label: "Plan", desc: "Create a plan before making changes" },
  { id: "auto", label: "Bypass permissions", desc: "Accepts all permissions" },
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
  autoRepack: false,
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
  panelHidden: [],
  panelOrder: [],
  replyFont: "serif",
  codeFont: "",
  sideFolded: [],
  sideWidth: 272,
  devices: [],
  apiServer: false,
  apiPort: 11435,
  updateFeed: "",
  updateAuto: true,
  tasks: [],
  lastSeenVersion: "",
  prompts: [],
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
  /** tok/s moment to moment, the last 48 readings */
  speed?: number[];
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
  /** What's new: the versions since the last one seen, or every release */
  whatsNew: "new" | "all" | null;
  /** a newer version the update feed offers (Settings → About) */
  update: import("./lib/bridge").UpdateCheck | null;
  /** the Scheduled tasks window */
  tasksOpen: boolean;
  approval: Approval | null;
  context: ContextInfo | null;
  mcp: McpServer[];
  sidebar: boolean;
  /** the window is narrow (under 820 px): the sidebar becomes a drawer */
  compact: boolean;
  /** in compact mode: the sidebar drawer is open */
  drawer: boolean;
  toast: string | null;
  /** a button on the toast (Undo) */
  toastAction: { label: string; run: () => void } | null;
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
  /** the tool pane on the right (title bar buttons) */
  dock: Dock | null;
  /** the computer is kept awake (this session only) */
  awake: boolean;
  /** a message typed while a reply runs, sent when it finishes */
  queued: Queued | null;
  /** the keyboard shortcuts sheet (Ctrl+/) */
  keys: boolean;
  /** the saved-prompt editor: a new prompt (index null) or a saved one */
  promptEdit: { index: number | null; name: string; text: string } | null;
}

export interface Queued {
  chatId: string;
  text: string;
  images?: Photo[];
}

export type Dock = "terminal" | "changes" | "browser" | "files";

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
  whatsNew: null,
  update: null,
  tasksOpen: false,
  approval: null,
  context: null,
  mcp: [],
  sidebar: true,
  compact: typeof window !== "undefined" && window.matchMedia?.(COMPACT).matches,
  drawer: false,
  toast: null,
  toastAction: null,
  search: "",
  palette: false,
  artifact: null,
  liveId: null,
  promptMaker: false,
  compare: null,
  photo: null,
  dock: null,
  awake: false,
  queued: null,
  keys: false,
  promptEdit: null,
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

export function toast(msg: string, action?: { label: string; run: () => void }) {
  app.set({ toast: msg, toastAction: action || null });
  setTimeout(() => app.get().toast === msg && app.set({ toast: null, toastAction: null }), action ? 7000 : 3500);
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
  // another chat's change still waiting goes now (one timer for all chats
  // used to drop it when you switched chats within 0.4 s)
  if (savePending && savePending.id !== next.id) persist(savePending);
  savePending = next;
  saveTimer = setTimeout(() => ((savePending = null), persist(next)), 400);
}
let savePending: Chat | null = null;

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

/** Chats with `q` in their messages (the sidebar search). */
export async function searchMessages(q: string): Promise<SearchHit[]> {
  return bridge ? bridge.chats.search(q).catch(() => []) : searchChats([...memChats.values()], q);
}

/** Totals over every saved chat (Settings → Usage). */
export async function loadUsage(): Promise<Usage> {
  return bridge ? bridge.chats.usage() : chatUsage([...memChats.values()]);
}

export async function openChat(id: string) {
  if (live?.id === id) return app.set({ chat: live, view: "chat", context: null });
  const c: Chat | undefined = bridge ? await bridge.chats.load(id).catch(() => undefined) : memChats.get(id);
  if (c) app.set({ chat: c, view: "chat", context: null });
}

/** Shows or hides the sidebar: the drawer in compact mode, the column otherwise. */
export function toggleSidebar(show?: boolean) {
  app.set((s) => (s.compact ? { drawer: show ?? !s.drawer } : { sidebar: show ?? !s.sidebar }));
}

export function startNewChat(folder: string | null = null) {
  app.set({ chat: newChat(folder), view: "chat", context: null });
}

export async function deleteChat(id: string) {
  if (live?.id === id) return toast("Pelica is still replying in that chat: stop it first.");
  // kept for Undo (the file itself goes to the Recycle Bin)
  const open = app.get().chat?.id === id;
  const was: Chat | null = open ? app.get().chat : bridge ? await bridge.chats.load(id).catch(() => null) : memChats.get(id) || null;
  if (bridge) await bridge.chats.remove(id).catch(() => {});
  memChats.delete(id);
  app.set((s) => ({ chats: s.chats.filter((c) => c.id !== id), chat: s.chat?.id === id ? newChat(null) : s.chat }));
  if (was)
    toast(`Deleted “${was.title}”`, {
      label: "Undo",
      run: async () => {
        await persist(was);
        // (back to it if it was open and nothing else has been started since)
        const c = app.get().chat;
        if (open && c && !c.messages.length) app.set({ chat: was, view: "chat" });
      },
    });
}

/** Changes a saved chat (open or not) and saves it. */
export async function patchChat(id: string, fn: (c: Chat) => Chat) {
  if (live?.id === id) return toast("Pelica is still replying in that chat: wait for it to finish.");
  const open = app.get().chat?.id === id ? app.get().chat : null;
  const c = open || (bridge ? await bridge.chats.load(id).catch(() => null) : memChats.get(id));
  if (!c) return;
  const n = fn(c);
  if (app.get().chat?.id === id) app.set({ chat: n });
  await persist(n);
}

/** A copy of a chat to go on from separately; it opens. */
export async function forkChat(id: string) {
  const c = app.get().chat?.id === id ? app.get().chat : live?.id === id ? live : bridge ? await bridge.chats.load(id).catch(() => null) : memChats.get(id);
  if (!c) return;
  const now = Date.now();
  const f: Chat = { ...c, id: uid(), title: `${c.title} (fork)`, created: now, updated: now, pinned: false, messages: [...c.messages] };
  await persist(f);
  app.set({ chat: f, view: "chat", context: null });
  toast("Forked into a new chat");
}

/** Sets the open chat's folder, remembering it among the recent ones. */
export function setChatFolder(p: string | null) {
  updateChat((c) => ({ ...c, folder: p }));
  if (p) setSettings({ recentFolders: [p, ...app.get().settings.recentFolders.filter((x) => x !== p)].slice(0, 8) });
  if (p) restoreFolder(p);
}

/** Brings back the chats saved in the folder's proj.pel that were deleted here
 *  (or come from another PC); an empty chat opens the newest of them. */
export async function restoreFolder(p: string) {
  const back = bridge ? await bridge.folder.restore(p).catch(() => [] as string[]) : [];
  if (!back.length) return;
  await refreshChats();
  const c = app.get().chat;
  if (c && !c.messages.length && !c.archive?.length) await openChat(back[0]);
  toast(back.length === 1 ? "Restored a chat from this folder's proj.pel" : `Restored ${back.length} chats from this folder's proj.pel`);
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
