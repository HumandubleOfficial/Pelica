// window.pelica (the Electron preload), typed; absent when the UI is opened
// in a plain browser, where the features that need it say so.

export interface RunResult {
  code: number | null;
  output: string;
  timedOut: boolean;
  background: boolean;
}

export interface Bridge {
  win: { minimize(): Promise<void>; maximize(): Promise<void>; close(): Promise<void>; isMaximized(): Promise<boolean>; onState(fn: (s: { maximized: boolean }) => void): () => void; material(m: Material): Promise<Material> };
  settings: { get(): Promise<Record<string, any>>; set(p: Record<string, any>): Promise<Record<string, any>> };
  folder: { pick(): Promise<string | null>; open(root: string): Promise<string> };
  chats: { list(): Promise<ChatMeta[]>; load(id: string): Promise<any>; save(c: any): Promise<boolean>; remove(id: string): Promise<void> };
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
  web: { search(q: string): Promise<{ title: string; url: string; snippet: string }[]>; open(url: string, max: number): Promise<{ title: string; url: string; text: string }> };
  mcp: { start(): Promise<McpServer[]>; list(): Promise<McpServer[]>; call(server: string, tool: string, args: unknown): Promise<string>; restart(name: string): Promise<McpServer[]>; launchers(): Promise<Record<string, boolean>> };
  lib: {
    search(q: string): Promise<{ id: string; downloads: number; likes: number; updated: string; tags: string[] }[]>;
    files(id: string): Promise<{ path: string; size: number }[]>;
    download(repo: string, file: string, dir: string): Promise<string>;
    downloads(dir?: string): Promise<Download[]>;
    cancel(id: string): Promise<void>;
    pause(id: string): Promise<void>;
    resume(id: string): Promise<void>;
    pickDir(): Promise<string | null>;
    onProgress(fn: (d: Download[]) => void): () => void;
  };
  engine: { log(): Promise<string>; restart(): Promise<void> };
  app: {
    info(): Promise<{ version: string; userDir: string; pelica: string; electron: string; hotkey: string; hotkeyError: string }>;
    busy(b: { label: string; progress: number } | null): Promise<void>;
    notify(n: { title: string; body: string; chatId?: string }): Promise<boolean>;
    /** sets the quick-ask shortcut; returns why it couldn't be used ("" when fine) */
    hotkey(k: string): Promise<string>;
    login(on?: boolean): Promise<boolean>;
    onAction(fn: (a: AppAction) => void): () => void;
  };
  quick: { submit(text: string, reply?: string): Promise<void>; hide(): Promise<void>; resize(h: number): Promise<void>; clip(): Promise<string>; onShown(fn: () => void): () => void };
  sys: { info(): Promise<Record<string, string>>; open(root: string, rel: string): Promise<void>; clipRead(): Promise<string>; clipWrite(t: string): Promise<void>; external(url: string): Promise<void>; saveAs(name: string, text: string, dir?: string): Promise<string | null>; reveal(path: string): Promise<void> };
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
}

export interface McpServer {
  name: string;
  running: boolean;
  error: string;
  log: string;
  tools: { name: string; description?: string; inputSchema?: any }[];
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

export const bridge: Bridge | null = (window as any).pelica ?? null;

export function need(): Bridge {
  if (!bridge) throw new Error("This needs the Pelica desktop app (it isn't available in a plain browser).");
  return bridge;
}
