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
  kind: "search" | "page" | "read" | "write" | "edit" | "list" | "grep" | "command" | "todo" | "browser" | "mcp" | "trash" | "error" | "denied" | "cqs" | "system" | "clipboard" | "open" | "memory";
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
  /** the restore point that can undo this change */
  undo?: string;
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
  // tails: what came after this message, one list per Retry (tailAt is the
  // one showing, and its own list is empty: it's in messages)
  | { id: string; role: "user"; content: string; ts: number; summary?: boolean; images?: Photo[]; tails?: Msg[][]; tailAt?: number }
  | { id: string; role: "assistant"; content: string; reasoning?: string; tool_calls?: WireCall[]; stats?: Stats; model?: string; effort?: string; ts: number; ack?: boolean }
  | { id: string; role: "tool"; tool_call_id: string; name: string; content: string; ui: ToolUI; ts: number; images?: Photo[] }
  // forModel: put in front of your next message, so the model knows too
  | { id: string; role: "event"; kind: "compact" | "cqs" | "error" | "info"; text: string; detail?: string; ts: number; forModel?: string };

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
  /** the project it belongs to (Settings' projects) */
  project?: string;
  /** the user's own labels, to find it by */
  tags?: string[];
  /** brought in from another app's export: where from, and its id there */
  imported?: { from: "chatgpt" | "claude"; id: string };
  messages: Msg[];
  /** messages before the last compaction (shown, never sent) */
  archive?: Msg[];
  todos?: Todo[];
}

export const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);

export function newChat(folder: string | null, project?: string): Chat {
  const t = Date.now();
  return { id: uid(), title: "New chat", created: t, updated: t, folder, messages: [], ...(project ? { project } : {}) };
}

/** A project: chats that share instructions, files and (optionally) a folder. */
export interface Project {
  id: string;
  name: string;
  /** what every chat in it is told first */
  instructions: string;
  /** text files every chat in it can read (kept in the settings) */
  files: { name: string; text: string }[];
  /** the folder its new chats work in */
  folder: string | null;
  /** folders of documents its chats search (PDFs, Word, notes…; electron/docs.ts) */
  docs?: string[];
  created: number;
}

/** A thing Pelica remembers about the user, across chats. */
export interface Memory {
  id: string;
  text: string;
  created: number;
  /** who saved it: the user (Settings) or the model (the remember tool) */
  by: "you" | "pelica";
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
  /** keep long chats' model memory on the drive, so reopening one answers at once (engine kvsave) */
  saveChats: boolean;
  modelDirs: string[];
  projects: Project[];
  /** memory across chats: on, and what's remembered */
  memoryOn: boolean;
  memories: Memory[];
  mcpServers: import("./lib/bridge").McpConfig[];
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
  /** Pelica on your phone (Settings → Phone): the server on this PC's network, and its port */
  phoneOn: boolean;
  phonePort: number;
  /** Settings → Experimental: the Brain page (the model's experts in 3D) in the sidebar and menus */
  brainMenu: boolean;
  /** Settings → Account: the Google OAuth client (the user's), and syncing through its Drive */
  googleClientId: string;
  googleClientSecret: string;
  syncOn: boolean;
  /** the first start's "Sign up or log in" was answered (or the user was already signed in) */
  accountAsked: boolean;
  /** first-run setup: seen, or not needed */
  setupDone: boolean;
  /** a tag's own colour, when it isn't the one from its name */
  tagColors: Record<string, string>;
  /** the pinned chats in the order they were dragged into */
  pinOrder: string[];
  /** the voice replies are read in ("" = Windows' default), and how fast */
  voiceName: string;
  voiceRate: number;
  /** voice mode keeps listening while Pelica talks, so talking stops it */
  voiceTalkOver: boolean;
  /** where to look for new versions (a latest.json: web address or path; empty: Pelica's GitHub releases) */
  updateFeed: string;
  /** when to check for a new version by itself */
  updateCheck: "start" | "daily" | "off";
  /** also offer previews (GitHub pre-releases) */
  updatePreview: boolean;
  /** a version the user chose to skip: not announced at start */
  updateSkip: string;
  /** when the feed was last asked (ms) */
  updateLast: number;
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
  downloadDir: "",
  autoRepack: false,
  saveChats: true,
  modelDirs: [],
  projects: [],
  memoryOn: true,
  memories: [],
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
  phoneOn: false,
  phonePort: 8737,
  brainMenu: false,
  googleClientId: "",
  googleClientSecret: "",
  syncOn: true,
  accountAsked: false,
  setupDone: false,
  tagColors: {},
  pinOrder: [],
  voiceName: "",
  voiceRate: 1,
  voiceTalkOver: false,
  updateFeed: "",
  updateCheck: "start",
  updatePreview: false,
  updateSkip: "",
  updateLast: 0,
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
  /** a connector tool: what it's called with */
  args?: string;
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
  view: "chat" | "library" | "brain" | "projects" | "project";
  /** the project page's project */
  projectId: string | null;
  /** the model library's tab */
  libTab: "local" | "discover";
  settingsOpen: boolean;
  settingsTab: string;
  /** What's new: the versions since the last one seen, or every release */
  whatsNew: "new" | "all" | null;
  /** a newer version the update feed offers (Settings → Updates) */
  update: import("./lib/bridge").UpdateCheck | null;
  /** the Scheduled tasks window */
  tasksOpen: boolean;
  approval: Approval | null;
  /** the Pelica account (Google) and its sync (Settings → Account) */
  account: import("./lib/bridge").AccountStatus | null;
  /** the first start's "Sign up or log in" (SignUp.tsx) */
  signUp: boolean;
  /** the first-run setup is open */
  setup: boolean;
  /** chats ticked in the sidebar (null: not selecting) */
  picked: string[] | null;
  /** the tag the sidebar's list is narrowed to */
  sideTag: string | null;
  /** voice mode (Voice.tsx): what it's doing, and what it last heard */
  voice: { phase: "starting" | "listening" | "thinking" | "speaking"; heard: string; line: string; full: boolean } | null;
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
  libTab: "local",
  projectId: null,
  settingsOpen: false,
  settingsTab: "General",
  whatsNew: null,
  update: null,
  tasksOpen: false,
  approval: null,
  account: null,
  signUp: false,
  setup: false,
  picked: null,
  sideTag: null,
  voice: null,
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
  const chats = bridge ? await bridge.chats.list().catch(() => []) : [...memChats.values()].map((c) => ({ id: c.id, title: c.title, updated: c.updated, pinned: !!c.pinned, folder: c.folder, project: c.project, tags: c.tags }));
  app.set({ chats: chats.sort((a, b) => b.updated - a.updated) });
}

let saveTimer: ReturnType<typeof setTimeout> | null = null;

/** Replaces the open chat (and saves it, debounced). */
/** Retry: the reply after your last message stays as a version, and the next
 *  one is a new version. False if there's no message to answer again. */
export function newVersion(): boolean {
  const c = app.get().chat;
  const i = c ? c.messages.map((m) => m.role).lastIndexOf("user") : -1;
  if (i < 0) return false;
  updateChat((x) => {
    const u = x.messages[i] as Extract<Msg, { role: "user" }>;
    const tail = x.messages.slice(i + 1);
    const head = x.messages.slice(0, i);
    // (no reply came, so there's nothing to keep: this try is used again)
    if (!tail.some((m) => m.role === "assistant")) return { ...x, messages: [...head, u] };
    const tails = [...(u.tails || [[]])];
    tails[u.tailAt ?? 0] = tail;
    return { ...x, messages: [...head, { ...u, tails: [...tails, []], tailAt: tails.length }] };
  });
  return true;
}

/** Shows another version of what came after message `id`. */
export function showVersion(id: string, to: number) {
  updateChat((x) => {
    const i = x.messages.findIndex((m) => m.id === id);
    const u = x.messages[i];
    if (u?.role !== "user" || !u.tails?.[to]) return x;
    const tails = [...u.tails];
    const next = tails[to];
    tails[to] = [];
    tails[u.tailAt ?? 0] = x.messages.slice(i + 1);
    return { ...x, messages: [...x.messages.slice(0, i), { ...u, tails, tailAt: to }, ...next] };
  });
}

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
  const meta = { id: c.id, title: c.title, updated: c.updated, pinned: !!c.pinned, folder: c.folder, project: c.project, tags: c.tags };
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

export function startNewChat(folder: string | null = null, project?: string) {
  app.set({ chat: newChat(folder, project), view: "chat", context: null });
}

// ---------- projects ----------

export const projectOf = (id?: string) => (id ? app.get().settings.projects.find((p) => p.id === id) : undefined);

export function saveProject(p: Project) {
  const list = app.get().settings.projects;
  setSettings({ projects: list.some((x) => x.id === p.id) ? list.map((x) => (x.id === p.id ? p : x)) : [...list, p] });
}

export function newProject(name = "New project"): Project {
  const p: Project = { id: uid(), name, instructions: "", files: [], folder: null, created: Date.now() };
  saveProject(p);
  return p;
}

/** Deletes a project; its chats stay, outside any project. */
export async function deleteProject(id: string) {
  const p = projectOf(id);
  if (!p) return;
  setSettings({ projects: app.get().settings.projects.filter((x) => x.id !== id) });
  for (const c of app.get().chats.filter((c) => c.project === id)) await patchChat(c.id, (x) => ({ ...x, project: undefined }));
  if (app.get().projectId === id) app.set({ view: "projects", projectId: null });
  toast(`Deleted the project “${p.name}”. Its chats are still in your chat list.`);
}

/** Opens Brain, if it's turned on in Settings → Experimental. */
export function openBrain() {
  if (app.get().settings.brainMenu) return app.set({ view: "brain", settingsOpen: false });
  toast("Brain is experimental: turn it on in Settings → Advanced.", { label: "Open", run: () => app.set({ settingsOpen: true, settingsTab: "Experimental" }) });
}

export function openProject(id: string) {
  app.set({ view: "project", projectId: id, drawer: false });
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

/** Deletes several chats, with one Undo (the files go to the Recycle Bin). */
export async function deleteChats(ids: string[]) {
  const mine = ids.filter((id) => live?.id !== id);
  if (mine.length < ids.length) toast("Pelica is still replying in one of those chats: it was left alone.");
  const was: Chat[] = [];
  for (const id of mine) {
    const c = app.get().chat?.id === id ? app.get().chat : bridge ? await bridge.chats.load(id).catch(() => null) : memChats.get(id) || null;
    if (c) was.push(c);
    if (bridge) await bridge.chats.remove(id).catch(() => {});
    memChats.delete(id);
  }
  app.set((s) => ({ chats: s.chats.filter((c) => !mine.includes(c.id)), chat: s.chat && mine.includes(s.chat.id) ? newChat(null) : s.chat }));
  if (was.length)
    toast(`Deleted ${was.length} ${was.length === 1 ? "chat" : "chats"}`, {
      label: "Undo",
      run: async () => {
        for (const c of was) await persist(c);
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
