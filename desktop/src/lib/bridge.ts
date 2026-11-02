// window.pelica (the Electron preload), typed; absent when the UI is opened
// in a plain browser, where the features that need it say so.
import type { SearchHit, Usage } from "../../electron/chatdata";

/** What the GitHub safety check found (electron/repocheck.ts). */
export interface RepoReport {
  repo: string;
  verdict: "ok" | "caution" | "risky";
  reasons: string[];
  stars?: number;
}

export interface RunResult {
  code: number | null;
  output: string;
  timedOut: boolean;
  background: boolean;
}

export interface Bridge {
  win: { minimize(): Promise<void>; maximize(): Promise<void>; close(): Promise<void>; isMaximized(): Promise<boolean>; onState(fn: (s: { maximized: boolean }) => void): () => void; material(m: Material): Promise<Material> };
  settings: { get(): Promise<Record<string, any>>; set(p: Record<string, any>): Promise<Record<string, any>>; onPushed(fn: (patch: Record<string, unknown>) => void): () => void };
  account: {
    status(): Promise<AccountStatus>;
    signIn(): Promise<AccountStatus>;
    signOut(): Promise<AccountStatus>;
    syncNow(): Promise<AccountStatus>;
    onChanged(fn: () => void): () => void;
    onSynced(fn: (c: { chats: string[]; settings: boolean }) => void): () => void;
  };
  folder: { pick(): Promise<string | null>; open(root: string): Promise<string>; restore(root: string): Promise<string[]> };
  chats: { list(): Promise<ChatMeta[]>; load(id: string): Promise<any>; save(c: any): Promise<boolean>; remove(id: string): Promise<void>; search(q: string): Promise<SearchHit[]>; usage(): Promise<Usage>; backup(): Promise<{ path: string; count: number } | null>; restore(): Promise<{ added: number; copies: number; same: number } | null>; import(): Promise<{ from: "chatgpt" | "claude"; added: number; same: number; empty: number } | null> };
  fs: {
    list(root: string, rel: string, depth: number): Promise<string>;
    read(root: string, rel: string, offset: number, limit: number): Promise<{ text: string; lines: number; path: string }>;
    write(root: string, rel: string, content: string): Promise<{ path: string; before: string | null; after: string }>;
    edit(root: string, rel: string, o: string, n: string, all: boolean): Promise<{ path: string; before: string; after: string; count: number }>;
    search(root: string, pattern: string, rel: string, glob?: string): Promise<string>;
    find(root: string, glob: string): Promise<string[]>;
    raw(root: string, rel: string): Promise<string>;
    notes(root: string): Promise<string>;
    trash(root: string, rel: string): Promise<void>;
    reveal(root: string, rel: string): Promise<void>;
  };
  cmd: { run(id: string, root: string, command: string, timeoutMs: number, background: boolean): Promise<RunResult>; kill(id: string): Promise<void>; onData(fn: (v: { id: string; data: string }) => void): () => void; onExit(fn: (v: { id: string; code: number | null }) => void): () => void };
  web: {
    search(q: string): Promise<{ title: string; url: string; snippet: string }[]>;
    open(url: string, max: number): Promise<{ title: string; url: string; text: string }>;
    /** safety reports for the GitHub repositories named in a command or URL */
    repos(text: string): Promise<RepoReport[]>;
  };
  mcp: { start(): Promise<McpServer[]>; list(): Promise<McpServer[]>; call(server: string, tool: string, args: unknown): Promise<{ text: string; images: { data: string; mimeType: string }[] }>; restart(name: string): Promise<McpServer[]>; launchers(): Promise<Record<string, boolean>>; others(): Promise<{ app: string; file: string; servers: McpConfig[] }[]>; parse(text: string): Promise<McpConfig[]>; signIn(name: string): Promise<McpServer[]>; signOut(name: string): Promise<McpServer[]>; onChanged(fn: (v: McpServer[]) => void): () => void };
  lib: {
    search(q: string): Promise<{ id: string; downloads: number; likes: number; updated: string; tags: string[] }[]>;
    files(id: string): Promise<{ path: string; size: number }[]>;
    download(repo: string, file: string, dir: string): Promise<string>;
    downloads(dir?: string): Promise<Download[]>;
    cancel(id: string): Promise<void>;
    pause(id: string): Promise<void>;
    resume(id: string): Promise<void>;
    /** asks first; false when the user said no, else whether it went to the
     *  Recycle Bin or (too big for it, or no bin on that drive) was deleted for good */
    uninstall(path: string): Promise<false | "recycled" | "deleted">;
    /** speeds a DeepSeek V4 file up (pelica repack), or puts it back; asks
     *  first and false when the user said no. It goes on in the background. */
    repack(path: string, undo: boolean): Promise<boolean>;
    repacks(): Promise<Repack[]>;
    repackStop(path: string): Promise<void>;
    onRepack(fn: (r: Repack[]) => void): () => void;
    pickDir(): Promise<string | null>;
    onProgress(fn: (d: Download[]) => void): () => void;
    /** the GPUs (dedicated memory), RAM, where downloads go, and every folder the engine looks in */
    hardware(): Promise<Hardware>;
    drives(): Promise<DriveInfo[]>;
    /** free space on the drive of `dir` (default: the download folder); null: no such drive */
    space(dir?: string): Promise<DriveInfo | null>;
    /** moves a model into `target`; its new path (null: cancelled or failed, see moves) */
    move(path: string, target: string): Promise<string | null>;
    moves(): Promise<ModelMove[]>;
    moveStop(path: string): Promise<void>;
    onMove(fn: (m: ModelMove[]) => void): () => void;
  };
  engine: { log(): Promise<string>; restart(): Promise<void>; api(): Promise<{ on: boolean; port: number; error: string }> };
  phone: {
    status(): Promise<PhoneStatus>;
    push(snap: unknown): Promise<void>;
    newKey(): Promise<PhoneStatus>;
    onCmd(fn: (c: PhoneCmd) => void): () => void;
  };
  app: {
    info(): Promise<{ version: string; userDir: string; pelica: string; electron: string; hotkey: string; hotkeyError: string }>;
    busy(b: { label: string; progress: number } | null): Promise<void>;
    notify(n: { title: string; body: string; chatId?: string }): Promise<boolean>;
    /** sets the quick-ask shortcut; returns why it couldn't be used ("" when fine) */
    hotkey(k: string): Promise<string>;
    login(on?: boolean): Promise<boolean>;
    onAction(fn: (a: AppAction) => void): () => void;
  };
  update: {
    check(): Promise<UpdateCheck>;
    install(f: UpdateFeed): Promise<boolean>;
    cancel(): Promise<void>;
    onProgress(fn: (p: { got: number; total: number }) => void): () => void;
  };
  quick: { submit(text: string, reply?: string): Promise<void>; hide(): Promise<void>; resize(h: number): Promise<void>; clip(): Promise<string>; onShown(fn: () => void): () => void };
  sys: { info(): Promise<Record<string, string>>; open(root: string, rel: string): Promise<void>; clipRead(): Promise<string>; clipWrite(t: string): Promise<void>; external(url: string): Promise<void>; saveAs(name: string, text: string, dir?: string): Promise<string | null>; reveal(path: string): Promise<void>; awake(on: boolean): Promise<boolean> };
  /** a project's documents (electron/docs.ts) */
  docs: { index(folder: string): Promise<DocsInfo>; info(folder: string): Promise<DocsInfo>; search(folders: string[], query: string, k?: number): Promise<DocPassage[]>; forget(folder: string): Promise<void>; onProgress(fn: (p: { folder: string; done: number; total: number; file: string }) => void): () => void };
  media: { ocr(dataUrl: string): Promise<string>; dictateStart(): Promise<void>; dictateStop(): Promise<void>; onDictate(fn: (e: DictateEvent) => void): () => void };
  pathFor(f: File): string;
}

export type DictateEvent = { kind: "ready"; lang: string } | { kind: "text"; text: string } | { kind: "error"; error: string } | { kind: "end" };

export type Material = "none" | "mica" | "acrylic" | "tabbed";

/** What the tray, jump list, notifications and quick bar ask the window to do. */
export type AppAction = { kind: "new-chat" } | { kind: "ask"; text: string; reply?: string } | { kind: "open-chat"; id: string } | { kind: "view"; view: "brain" | "library" | "chat" } | { kind: "settings" };

export interface ChatMeta {
  id: string;
  title: string;
  updated: number;
  pinned: boolean;
  folder: string | null;
  project?: string;
}

export interface McpServer {
  name: string;
  running: boolean;
  error: string;
  log: string;
  /** a remote server's host */
  remote?: string;
  /** an online connector's sign-in: "needed", "in" (signed in), or "" */
  auth?: "" | "needed" | "in";
  tools: McpToolInfo[];
}

export interface McpToolInfo {
  name: string;
  title?: string;
  description?: string;
  inputSchema?: any;
  annotations?: { readOnlyHint?: boolean; destructiveHint?: boolean; title?: string };
}

/** a connector in the settings: a program on this PC (command) or a remote server (url) */
export interface McpConfig {
  name: string;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  url?: string;
  headers?: Record<string, string>;
  disabled?: boolean;
  /** per tool: "allow" (no asking), "ask", "off" (not offered); unset: from the tool's hints */
  tools?: Record<string, ToolPerm>;
}
export type ToolPerm = "allow" | "ask" | "off";

export interface Hardware {
  gpus: { name: string; vram: number }[];
  ram: number;
  downloadDir: string;
  dirs: string[];
}

export interface DriveInfo {
  /** "C:\" */
  root: string;
  free: number;
  total: number;
}

export interface ModelMove {
  path: string;
  name: string;
  to: string;
  got: number;
  total: number;
  state: "running" | "done" | "error" | "cancelled";
  error?: string;
  moved?: string;
}

export interface Download {
  id: string;
  repo: string;
  file: string;
  dest: string;
  got: number;
  total: number;
  state: "running" | "paused" | "done" | "error" | "cancelled";
  error?: string;
}

export interface Repack {
  path: string;
  undo: boolean;
  state: "running" | "done" | "stopped" | "error";
  pct: number;
  /** seconds, once the first layer is done */
  left: number | null;
  error?: string;
}

export const bridge: Bridge | null = (window as any).pelica ?? null;

export function need(): Bridge {
  if (!bridge) throw new Error("This needs the Pelica desktop app (it isn't available in a plain browser).");
  return bridge;
}

/** an update feed's newest version (electron/updates.ts) */
export interface UpdateFeed {
  version: string;
  url: string;
  sha256: string;
  notes?: string[];
  title?: string;
}

export interface UpdateCheck {
  current: string;
  latest?: UpdateFeed;
  newer: boolean;
  /** the offered version is a preview (a GitHub pre-release) */
  preview?: boolean;
  error?: string;
}

/** Settings → Phone: the server on this PC's network. */
export interface PhoneStatus {
  on: boolean;
  port: number;
  error: string;
  lan: string[];
  tailscale: string | null;
  phones: number;
  key: string;
}

/** What the phone asks the window to do. */
export interface PhoneCmd {
  type: "send" | "open" | "new" | "approve" | "stop";
  id?: string;
  text?: string;
  answer: "allow" | "always" | "deny";
}

/** Settings → Account: the Google account, and the sync through its Drive. */
/** A folder of documents a project searches: how much of it is read. */
export interface DocsInfo {
  folder: string;
  /** when it was last read (ms; 0: never) */
  at: number;
  files: number;
  passages: number;
  unreadable: { file: string; error: string }[];
  busy: boolean;
}

/** A passage of a document that matches a question. */
export interface DocPassage {
  file: string;
  /** PDF page or slide, from 1 */
  page?: number;
  text: string;
  score: number;
}

export interface AccountStatus {
  profile: { name: string; email: string; picture: string } | null;
  sync: { busy: boolean; lastSync: number; error: string };
  clientId: string;
  hasSecret: boolean;
  /** Pelica has its own Google client (used unless one is set in Settings) */
  builtIn: boolean;
}
