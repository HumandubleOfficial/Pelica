// Pelica Desktop's main process: starts pelica.exe (the inference server,
// serving this app's built UI), opens a frameless window on it, and gives
// the UI what a page can't do itself: the workspace folder's files,
// commands, web requests, MCP servers, downloads.
// It also ties Pelica into Windows: a tray icon, a global quick-ask hotkey,
// notifications, taskbar progress and jump list, Mica.
import { app, BrowserWindow, dialog, ipcMain, shell, Menu, nativeImage, nativeTheme, Tray, globalShortcut, Notification, clipboard, screen, powerMonitor, powerSaveBlocker } from "electron";
import { spawn, ChildProcess, execFile } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import * as web from "./web";
import * as ws from "./workspace";
import * as mcp from "./mcp";
import * as lib from "./library";
import * as media from "./media";
import { chatUsage, searchChats } from "./chatdata";
import { checkRepo, reposIn } from "./repocheck";

const userDir = app.getPath("userData");
const settingsFile = path.join(userDir, "settings.json");
const chatsDir = path.join(userDir, "chats");

type Settings = Record<string, any>;

function loadSettings(): Settings {
  try {
    return JSON.parse(fs.readFileSync(settingsFile, "utf8"));
  } catch {
    return {};
  }
}

let settings: Settings = loadSettings();

function saveSettings() {
  fs.mkdirSync(userDir, { recursive: true });
  fs.writeFileSync(settingsFile, JSON.stringify(settings, null, 2));
}

/** Folders the user picked: the only roots file tools and commands may use. */
const roots = (): string[] => settings.roots || [];
let awakeId: number | null = null;

function checkRoot(root: string): string {
  const r = path.resolve(root || "");
  if (!roots().some((x) => path.resolve(x).toLowerCase() === r.toLowerCase())) throw new Error("no workspace folder chosen (pick one with the folder button)");
  return r;
}

// ---------- the engine ----------

let engine: ChildProcess | null = null;
let port = 0;
let engineLog = "";

function pelicaExe(): string {
  const cands = [settings.pelicaPath, path.resolve(__dirname, "../../../target/release/pelica.exe"), path.resolve(process.resourcesPath || "", "pelica.exe")].filter(Boolean) as string[];
  return cands.find((c) => fs.existsSync(c)) || cands[1];
}

function freePort(): Promise<number> {
  return new Promise((ok, bad) => {
    const s = net.createServer();
    s.listen(0, "127.0.0.1", () => {
      const p = (s.address() as net.AddressInfo).port;
      s.close(() => ok(p));
    });
    s.on("error", bad);
  });
}

async function startEngine(): Promise<void> {
  stopEngine();
  port = await freePort();
  const ui = path.resolve(__dirname, "../../dist");
  // no model until one is picked, unless "Load the last model at start" is on
  const args = ["app", "--no-open", "--no-autoload", "--port", String(port), "--ui", ui];
  if (settings.loadOnStart && settings.lastModel && fs.existsSync(settings.lastModel)) args.push("--model", settings.lastModel, "--ctx", String(settings.lastCtx || 32768));
  for (const d of settings.modelDirs || []) args.push("--models-dir", d);
  engineLog = `$ ${pelicaExe()} ${args.join(" ")}\n`;
  // computers lending their RAM (Settings → Devices); the app also sends changes live
  const devices = ((settings.devices || []) as { addr: string; code: string }[]).map((d) => `${d.addr}/${d.code}`).join(",");
  engine = spawn(pelicaExe(), args, { windowsHide: true, stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, PELICA_DEVICES: devices } });
  const add = (d: Buffer) => (engineLog = (engineLog + d.toString()).slice(-200000));
  engine.stdout!.on("data", add);
  engine.stderr!.on("data", add);
  engine.on("exit", (code) => {
    engineLog += `\n[pelica exited: ${code}]\n`;
    engine = null;
  });
  engine.on("error", (e) => (engineLog += `\n[could not start pelica: ${e.message}]\n`));
  // ready when the status answers
  for (let i = 0; i < 300; i++) {
    if (!engine) throw new Error("pelica stopped:\n" + engineLog.slice(-2000));
    try {
      const r = await fetch(`http://127.0.0.1:${port}/api/status`);
      if (r.ok) return announceEngine();
    } catch {}
    await new Promise((ok) => setTimeout(ok, 100));
  }
  throw new Error("pelica didn't answer:\n" + engineLog.slice(-2000));
}

// ---------- the local API (Settings → Models → Local API) ----------
// other programs on this computer use the loaded model through the engine's
// OpenAI-compatible endpoints, on a port that stays the same between starts
// (the engine's own changes); only /v1/ passes, and only from this computer
let apiServer: http.Server | null = null;
let apiPort = 0;
let apiError = "";
function syncApi() {
  const want = settings.apiServer ? Number(settings.apiPort) || 11435 : 0;
  if (want === apiPort && (apiServer || !want)) return;
  apiServer?.close();
  apiServer = null;
  apiPort = want;
  apiError = "";
  if (!want) return;
  const srv = http.createServer((req, res) => {
    if (!req.url?.startsWith("/v1/") || !port) {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: { message: port ? "not found" : "the engine isn't running", type: "not_found" } }));
      return;
    }
    // (the engine accepts its own host only, and turns away web pages)
    const up = http.request({ host: "127.0.0.1", port, path: req.url, method: req.method, headers: { ...req.headers, host: `127.0.0.1:${port}` } }, (r) => {
      res.writeHead(r.statusCode || 502, r.headers);
      r.pipe(res);
    });
    up.on("error", (e) => {
      if (!res.headersSent) res.writeHead(502, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: { message: e.message, type: "unavailable" } }));
    });
    // the caller going away stops the reply
    res.on("close", () => up.destroy());
    req.pipe(up);
  });
  srv.on("error", (e: any) => {
    apiError = e.code === "EADDRINUSE" ? `Port ${want} is in use by another program. Pick another.` : e.message;
    if (apiServer === srv) apiServer = null;
  });
  srv.listen(want, "127.0.0.1");
  apiServer = srv;
}

// where `pelica chat` in a terminal finds this engine, to use the loaded
// model instead of loading a second copy
const engineFile = path.join(userDir, "engine.json");
function announceEngine() {
  try {
    fs.writeFileSync(engineFile, JSON.stringify({ port, pid: engine?.pid, app: process.pid }));
  } catch {}
}

function stopEngine() {
  try {
    if (JSON.parse(fs.readFileSync(engineFile, "utf8")).port === port) fs.unlinkSync(engineFile);
  } catch {}
  if (engine?.pid) {
    execFile("taskkill", ["/pid", String(engine.pid), "/T", "/F"], () => {});
  }
  engine = null;
}

type EngineModel = { path: string; name: string; gb: number; repack: "yes" | "no" | "partial" | "unfinished" | null };
async function engineModel(p: string): Promise<EngineModel | undefined> {
  const r = await fetch(`http://127.0.0.1:${port}/api/models`).then((x) => x.json());
  const key = (x: string) => path.resolve(x).toLowerCase();
  return (r.models as EngineModel[]).find((x) => key(x.path) === key(p));
}

/** Unloads `p` if it's the loaded model; `what` finishes "Stop it, then …". */
async function unloadModel(p: string, what: string): Promise<void> {
  const status = () => fetch(`http://127.0.0.1:${port}/api/status`).then((x) => x.json());
  let st = await status();
  if (st.path === p && st.state === "loading") throw new Error(`It's still loading. Wait for it to finish, then ${what}.`);
  if (st.path === p && st.state === "ready") {
    const u = await fetch(`http://127.0.0.1:${port}/api/unload`, { method: "POST", body: "{}" });
    if (!u.ok) throw new Error(`Pelica is busy with a reply. Stop it, then ${what}.`);
    for (let i = 0; i < 120 && st.path === p && st.state === "ready"; i++) {
      await new Promise((res) => setTimeout(res, 250));
      st = await status();
    }
  }
}

// ---------- speeding up DeepSeek V4 and Qwen3.8 files (pelica repack) ----------

type Repack = { path: string; undo: boolean; state: "running" | "done" | "stopped" | "error"; pct: number; left: number | null; error?: string; child?: ChildProcess };
const repacks = new Map<string, Repack>();
const sendRepacks = () => win?.webContents.send("lib:repack", [...repacks.values()].map(({ child, ...r }) => r));

async function startRepack(p: string, undo: boolean): Promise<void> {
  if (repacks.get(p)?.state === "running") return;
  // (taken before anything waits: a second click meanwhile finds it running
  // instead of starting a repack that would fail on the locked file)
  const r: Repack = { path: p, undo, state: "running", pct: 0, left: null };
  repacks.set(p, r);
  sendRepacks();
  try {
    await unloadModel(p, undo ? "undo the speed-up" : "speed it up");
  } catch (e) {
    repacks.delete(p);
    sendRepacks();
    throw e;
  }
  const child = spawn(pelicaExe(), ["repack", p, ...(undo ? ["--undo"] : [])], { windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
  r.child = child;
  let tail = "";
  // its lines: "layer 12 done in 14.2s (28%, about 380s left)"
  child.stderr!.on("data", (d: Buffer) => {
    tail = (tail + d.toString()).slice(-4000);
    const all = [...tail.matchAll(/\((\d+)%, about (\d+)s left\)/g)];
    const last = all[all.length - 1];
    if (last) {
      r.pct = Number(last[1]);
      r.left = Number(last[2]);
      sendRepacks();
    }
  });
  child.on("error", (e) => ((r.state = "error"), (r.error = e.message), sendRepacks()));
  child.on("exit", (code, signal) => {
    r.child = undefined;
    if (r.state !== "running") return;
    if (code === 0) (r.state = "done"), (r.pct = 100), (r.left = 0);
    else (r.state = "error"), (r.error = (tail.match(/Error: ([\s\S]*)$/)?.[1] || tail || `pelica stopped (${code ?? signal})`).trim().split("\n").slice(0, 3).join(" "));
    sendRepacks();
    if (r.state === "done") new Notification({ title: undo ? "Speed-up undone" : "Model sped up", body: `${path.basename(p)} is ready to load.` }).show();
  });
  sendRepacks();
}

// downloads: a finished DeepSeek V4 or Qwen3.8 file is sped up on its own when settings.autoRepack is on
const dlRunning = new Set<string>();
function dlProgress(jobs: ReturnType<typeof lib.downloads>) {
  win?.webContents.send("lib:progress", jobs);
  for (const j of jobs) {
    if (j.state === "running") dlRunning.add(j.id);
    // (only a download that finished just now)
    if (j.state !== "done" || !dlRunning.delete(j.id) || !settings.autoRepack) continue;
    engineModel(j.dest)
      .then((m) => (m?.repack === "no" ? startRepack(m.path, false) : undefined))
      .catch(() => {});
  }
}

// ---------- the window ----------

let win: BrowserWindow | null = null;

// the first thing on screen while pelica.exe starts: the mark, breathing, over a soft glow
const SPLASH = `data:text/html;charset=utf-8,${encodeURIComponent(`<!doctype html><html><head><style>
html,body{margin:0;height:100%;background:#1f1e1d;color:#c9c5bd;font:14px "Segoe UI Variable Text","Segoe UI",system-ui;-webkit-app-region:drag;overflow:hidden}
body{display:grid;place-items:center;background:radial-gradient(600px 400px at 50% 46%,rgba(229,83,61,.16),transparent 70%),#1f1e1d}
.w{text-align:center;animation:in .5s cubic-bezier(.2,.8,.2,1)}
.m{width:72px;height:72px;margin:auto;border-radius:22px;animation:b 2.4s ease-in-out infinite;box-shadow:0 10px 40px rgba(229,83,61,.35)}
.t{margin-top:20px;letter-spacing:.4px;font-size:15px;color:#ecebe6}
.s{margin-top:6px;font-size:12px;color:#8f8b83}
.bar{width:160px;height:3px;margin:18px auto 0;border-radius:3px;background:rgba(255,255,255,.07);overflow:hidden}
.bar i{display:block;width:40%;height:100%;border-radius:3px;background:linear-gradient(90deg,#ffb26b,#e5533d);animation:r 1.3s ease-in-out infinite}
@keyframes b{50%{transform:scale(1.05);box-shadow:0 14px 60px rgba(229,83,61,.5)}}
@keyframes r{0%{transform:translateX(-110%)}100%{transform:translateX(260%)}}
@keyframes in{from{opacity:0;transform:translateY(8px)}}
</style></head><body><div class="w"><svg class="m" viewBox="0 0 64 64"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ffb26b"/><stop offset="1" stop-color="#e5533d"/></linearGradient></defs><rect width="64" height="64" rx="18" fill="url(#g)"/><path d="M17 47c0-15 5-27 15-27 6 0 9 4 9.5 8L55 31.5c1 .4 1 1.6 0 2L41.5 36C40 44 33 48 26 51z" fill="#fff"/><circle cx="33" cy="27" r="2.3" fill="#e5533d"/></svg>
<div class="t">Pelica</div><div class="s">Starting the engine…</div><div class="bar"><i></i></div></div></body></html>`)}`;

type Material = "none" | "mica" | "acrylic" | "tabbed";
/** Windows 11's backdrop behind the window (the page lets it show through). */
const material = (): Material => (["mica", "acrylic", "tabbed"].includes(settings.material) ? settings.material : "none");
const SOLID = "#1f1e1d";
/** Mica and Acrylic take their tint from the app's light or dark theme. */
const syncTheme = () => (nativeTheme.themeSource = settings.theme === "light" ? "light" : "dark");

async function createWindow() {
  syncTheme();
  const m = material();
  win = new BrowserWindow({
    width: 1320,
    height: 880,
    // (small enough for compact mode, like Claude's small window)
    minWidth: 460,
    minHeight: 420,
    frame: false,
    backgroundColor: m === "none" ? SOLID : "#00000000",
    backgroundMaterial: m,
    show: false,
    title: "Pelica",
    icon: icon ?? undefined,
    webPreferences: { preload: path.join(__dirname, "preload.js"), contextIsolation: true, nodeIntegration: false, sandbox: true, spellcheck: true, webviewTag: true },
  });
  Menu.setApplicationMenu(null);
  win.once("ready-to-show", () => !startHidden && win?.show());
  win.on("maximize", () => win?.webContents.send("win:state", { maximized: true }));
  win.on("unmaximize", () => win?.webContents.send("win:state", { maximized: false }));
  win.on("focus", () => win?.flashFrame(false));
  // with "keep running in the tray", closing only hides the window
  win.on("close", (e) => {
    if (settings.closeToTray && tray && !quitting) {
      e.preventDefault();
      win?.hide();
    }
  });
  // (the hidden quick-ask window mustn't keep Pelica running)
  win.on("closed", () => {
    win = null;
    app.quit();
  });
  // the UI never navigates away; links open in the default browser
  const external = (url: string) => /^https?:\/\//i.test(url) && shell.openExternal(url).catch(() => {});
  win.webContents.on("will-navigate", (e, url) => {
    if (!url.startsWith(`http://127.0.0.1:${port}/`)) {
      e.preventDefault();
      external(url);
    }
  });
  win.webContents.setWindowOpenHandler(({ url }) => {
    external(url);
    return { action: "deny" };
  });
  await win.loadURL(SPLASH);
  try {
    await startEngine();
    await win.loadURL(`http://127.0.0.1:${port}/`);
    setIcon();
  } catch (e: any) {
    const msg = `data:text/html;charset=utf-8,${encodeURIComponent(`<body style="background:#1f1e1d;color:#e8e4dc;font:14px system-ui;padding:40px"><h2>Pelica couldn't start</h2><pre style="white-space:pre-wrap;color:#c9c5bd">${String(e.message).replace(/</g, "&lt;")}</pre><p>Set the path to pelica.exe in settings.json (${settingsFile.replace(/\\/g, "\\\\")}) as "pelicaPath", or build it with cargo build --release.</p></body>`)}`;
    await win.loadURL(msg);
  }
}

/** The window, taskbar and tray icon: the logo, drawn from its SVG by the
 *  page and kept in the user folder so the next start has it at once. */
const iconFile = path.join(userDir, "icon.png");
let icon: Electron.NativeImage | null = fs.existsSync(iconFile) ? nativeImage.createFromPath(iconFile) : null;

async function setIcon() {
  try {
    const url: string = await win!.webContents.executeJavaScript(`(async () => {
      const svg = await (await fetch('./logo.svg')).text();
      const img = new Image(); img.src = 'data:image/svg+xml;base64,' + btoa(svg); await img.decode();
      const c = document.createElement('canvas'); c.width = c.height = 256;
      c.getContext('2d').drawImage(img, 0, 0, 256, 256); return c.toDataURL('image/png');
    })()`);
    icon = nativeImage.createFromDataURL(url);
    win?.setIcon(icon);
    fs.writeFileSync(iconFile, icon.toPNG());
    makeTray();
  } catch {}
}

// ---------- the desktop: tray, quick ask, notifications, taskbar ----------

let tray: Tray | null = null;
let quitting = false;
const startHidden = process.argv.includes("--hidden");

function showMain() {
  if (!win) return;
  if (win.isMinimized()) win.restore();
  win.show();
  win.focus();
}

/** Asks the UI to do something: new chat, ask, open a chat, show a page. */
function action(a: Record<string, unknown>) {
  showMain();
  win?.webContents.send("app:action", a);
}

function makeTray() {
  if (!icon) return;
  if (tray) return tray.setImage(icon.resize({ width: 16, height: 16, quality: "best" }));
  tray = new Tray(icon.resize({ width: 16, height: 16, quality: "best" }));
  tray.setToolTip("Pelica");
  tray.on("click", showMain);
  trayMenu();
}

function trayMenu() {
  if (!tray) return;
  const k = hotkey();
  tray.setContextMenu(
    Menu.buildFromTemplate([
      { label: "Open Pelica", click: showMain },
      { label: "New chat", click: () => action({ kind: "new-chat" }) },
      { label: "Quick ask…", click: openQuick, accelerator: k || undefined, registerAccelerator: false },
      { type: "separator" },
      { label: "Brain", click: () => action({ kind: "view", view: "brain" }) },
      { label: "Models", click: () => action({ kind: "view", view: "library" }) },
      { label: "Settings", click: () => action({ kind: "settings" }) },
      { type: "separator" },
      {
        label: "Quit Pelica",
        click: () => {
          quitting = true;
          app.quit();
        },
      },
    ]),
  );
}

/** The global quick-ask shortcut ("" turns it off). */
const hotkey = (): string => (typeof settings.hotkey === "string" ? settings.hotkey : "Alt+Shift+Space");
let hotkeyError = "";

function registerHotkey(): string {
  globalShortcut.unregisterAll();
  hotkeyError = "";
  const k = hotkey();
  if (k) {
    try {
      if (!globalShortcut.register(k, openQuick)) hotkeyError = `${k} is already used by another app`;
    } catch (e: any) {
      hotkeyError = `${k} isn't a shortcut Pelica understands (${e.message})`;
    }
  }
  trayMenu();
  return hotkeyError;
}

// the quick-ask window: a small bar over everything, wherever the mouse is
let quick: BrowserWindow | null = null;
const QW = 720;

function placeQuick(h: number) {
  if (!quick) return;
  const d = screen.getDisplayNearestPoint(screen.getCursorScreenPoint()).workArea;
  quick.setBounds({ x: Math.round(d.x + (d.width - QW) / 2), y: Math.round(d.y + d.height * 0.2), width: QW, height: Math.round(h) });
}

function openQuick() {
  if (!port || !engine) return showMain();
  if (quick && !quick.isDestroyed()) {
    if (quick.isVisible()) return quick.hide();
    placeQuick(quick.getBounds().height);
    quick.show();
    quick.focus();
    quick.webContents.send("quick:shown");
    return;
  }
  quick = new BrowserWindow({
    width: QW,
    height: 168,
    frame: false,
    transparent: true,
    resizable: false,
    maximizable: false,
    minimizable: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    show: false,
    backgroundColor: "#00000000",
    webPreferences: { preload: path.join(__dirname, "preload.js"), contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  quick.setAlwaysOnTop(true, "pop-up-menu");
  placeQuick(168);
  quick.loadURL(`http://127.0.0.1:${port}/#quick`);
  quick.once("ready-to-show", () => {
    quick?.show();
    quick?.focus();
  });
  quick.on("blur", () => quick?.hide());
  quick.on("closed", () => (quick = null));
}

/** The taskbar's jump list: New chat and Quick ask start (or reach) Pelica. */
function jumpList() {
  const base = app.isPackaged ? [] : [`"${app.getAppPath()}"`];
  const task = (flag: string, title: string, description: string) => ({ program: process.execPath, arguments: [...base, flag].join(" "), title, description, iconPath: process.execPath, iconIndex: 0 });
  try {
    app.setUserTasks([task("--new-chat", "New chat", "Start a new chat in Pelica"), task("--quick", "Quick ask", "Ask Pelica something from anywhere")]);
  } catch {}
}

/** What a second start (jump list, shortcut) asked for. */
function handleArgs(argv: string[]) {
  if (argv.includes("--quick")) openQuick();
  else if (argv.includes("--new-chat")) action({ kind: "new-chat" });
  else showMain();
}

const gb = (n: number) => `${(n / 1024 ** 3).toFixed(1)} GB`;

/** The computer, for the system_info tool (nothing personal: no user name, no paths). */
async function systemInfo() {
  const cpus = os.cpus();
  const info: Record<string, string> = {
    os: `${os.version()} ${os.release()} (${os.arch()})`,
    computer: os.hostname(),
    cpu: `${(cpus[0]?.model || "?").trim()}, ${cpus.length} threads`,
    memory: `${gb(os.freemem())} free of ${gb(os.totalmem())}`,
    uptime: `${(os.uptime() / 3600).toFixed(1)} hours`,
    local_time: new Date().toString(),
    locale: app.getLocale(),
    displays: screen
      .getAllDisplays()
      .map((d) => `${d.size.width}×${d.size.height} at ${Math.round(d.scaleFactor * 100)}%`)
      .join(", "),
    power: powerMonitor.isOnBatteryPower() ? "on battery" : "plugged in",
  };
  for (const r of roots().slice(0, 3)) {
    try {
      const s = await fs.promises.statfs(r);
      info[`disk (${path.parse(r).root})`] = `${gb(s.bavail * s.bsize)} free of ${gb(s.blocks * s.bsize)}`;
    } catch {}
  }
  return info;
}

// ---------- commands ----------

const procs = new Map<string, ChildProcess>();

function killTree(pid?: number) {
  if (pid) execFile("taskkill", ["/pid", String(pid), "/T", "/F"], () => {});
}

interface RunResult {
  code: number | null;
  output: string;
  timedOut: boolean;
  background: boolean;
}

function runCommand(id: string, root: string, command: string, timeoutMs: number, background: boolean, shellKind: string): Promise<RunResult> {
  const cwd = checkRoot(root);
  const [exe, args] = shellKind === "powershell" ? ["powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", command]] : ["cmd.exe", ["/d", "/s", "/c", `"${command}"`]];
  const p = spawn(exe, args, { cwd, windowsHide: true, env: { ...process.env, FORCE_COLOR: "0", CI: "1" }, windowsVerbatimArguments: shellKind !== "powershell" });
  procs.set(id, p);
  let output = "";
  const add = (d: Buffer) => {
    const s = d.toString();
    output = (output + s).slice(-400000);
    win?.webContents.send("cmd:data", { id, data: s });
  };
  p.stdout!.on("data", add);
  p.stderr!.on("data", add);
  return new Promise((ok) => {
    let done = false;
    const finish = (r: RunResult) => {
      if (!done) {
        done = true;
        ok(r);
      }
    };
    p.on("exit", (code) => {
      procs.delete(id);
      win?.webContents.send("cmd:exit", { id, code });
      finish({ code, output, timedOut: false, background: false });
    });
    p.on("error", (e) => {
      add(Buffer.from(`\n${e.message}\n`));
      finish({ code: -1, output, timedOut: false, background: false });
    });
    if (background) {
      // a server or watcher: report its first output and leave it running
      setTimeout(() => finish({ code: null, output, timedOut: false, background: true }), Math.min(timeoutMs, 8000));
    } else {
      setTimeout(() => {
        if (!done) {
          killTree(p.pid);
          finish({ code: null, output: output + `\n[stopped after ${Math.round(timeoutMs / 1000)} s]`, timedOut: true, background: false });
        }
      }, timeoutMs);
    }
  });
}

// ---------- IPC ----------

function handle(name: string, fn: (...a: any[]) => any) {
  ipcMain.handle(name, async (_e, ...a) => fn(...a));
}

// ---------- proj.pel: each folder keeps its own chats ----------
//
// Every save of a chat that works in a folder also updates <folder>/proj.pel
// (JSON: { pelica: "project", version: 1, updated, chats: [...] }). Deleting
// a chat in the app leaves it there, so choosing the folder again (or moving
// the folder and choosing it where it is now) brings its chats back.

/** each saved chat's folder, so a save knows when it moved without reading the old file */
const chatFolder = new Map<string, string | null>();
const projPending = new Map<string, Map<string, any>>(); // folder -> chat id -> chat (null = take out)
let projTimer: ReturnType<typeof setTimeout> | null = null;

function projQueue(folder: string, id: string, chat: any) {
  // only folders the user picked
  if (!roots().some((x) => path.resolve(x).toLowerCase() === path.resolve(folder).toLowerCase())) return;
  const m = projPending.get(folder) || new Map();
  m.set(id, chat);
  projPending.set(folder, m);
  if (!projTimer) projTimer = setTimeout(projFlush, 1500);
}

function readProject(folder: string): any[] {
  try {
    const v = JSON.parse(fs.readFileSync(path.join(folder, ws.PROJ_FILE), "utf8"));
    return Array.isArray(v?.chats) ? v.chats.filter((c: any) => c && typeof c.id === "string" && /^[\w-]+$/.test(c.id) && Array.isArray(c.messages)) : [];
  } catch {
    return [];
  }
}

function projFlush() {
  if (projTimer) clearTimeout(projTimer);
  projTimer = null;
  for (const [folder, m] of projPending) {
    try {
      if (!fs.statSync(folder).isDirectory()) continue;
      const chats = new Map(readProject(folder).map((c) => [c.id, c]));
      for (const [id, c] of m) {
        if (c) chats.set(id, c);
        else chats.delete(id);
      }
      const file = path.join(folder, ws.PROJ_FILE);
      if (!chats.size && !fs.existsSync(file)) continue;
      const list = [...chats.values()].sort((a, b) => (b.updated || 0) - (a.updated || 0));
      // written whole to a temporary file first, so a crash never leaves half a file
      const tmp = file + ".tmp";
      fs.writeFileSync(tmp, JSON.stringify({ pelica: "project", version: 1, updated: Date.now(), chats: list }));
      fs.renameSync(tmp, file);
    } catch {}
  }
  projPending.clear();
}

/** Brings back the folder's chats this PC doesn't have; returns their ids, newest first. */
function restoreProject(folder: string): string[] {
  projFlush();
  fs.mkdirSync(chatsDir, { recursive: true });
  const back: string[] = [];
  for (const c of readProject(folder)) {
    const p = path.join(chatsDir, c.id + ".json");
    if (fs.existsSync(p)) continue;
    // the folder may have moved: the chat works where it is now
    fs.writeFileSync(p, JSON.stringify({ ...c, folder }));
    back.push(c.id);
  }
  return back;
}

function registerIpc() {
  handle("win:minimize", () => win?.minimize());
  handle("win:maximize", () => (win?.isMaximized() ? win.unmaximize() : win?.maximize()));
  handle("win:close", () => win?.close());
  handle("win:isMaximized", () => win?.isMaximized() ?? false);

  handle("settings:get", () => settings);
  handle("settings:set", (patch: Settings) => {
    // (roots only grow through the folder picker)
    const { roots: _r, ...rest } = patch || {};
    settings = { ...settings, ...rest };
    saveSettings();
    if ("theme" in rest) syncTheme();
    if ("apiServer" in rest || "apiPort" in rest) syncApi();
    return settings;
  });
  handle("api:status", () => ({ on: !!apiServer && !apiError, port: apiPort, error: apiError }));

  handle("folder:pick", async () => {
    const r = await dialog.showOpenDialog(win!, { properties: ["openDirectory", "createDirectory"], title: "Choose the folder Pelica works in" });
    if (r.canceled || !r.filePaths[0]) return null;
    const p = r.filePaths[0];
    settings.roots = [p, ...roots().filter((x) => x.toLowerCase() !== p.toLowerCase())].slice(0, 20);
    saveSettings();
    return p;
  });
  // chats kept in the folder's proj.pel that this PC no longer has (deleted, or another PC's)
  handle("folder:restore", (root: string) => restoreProject(checkRoot(root)));
  handle("folder:open", (root: string) => shell.openPath(checkRoot(root)));

  handle("chats:list", () => {
    fs.mkdirSync(chatsDir, { recursive: true });
    return fs
      .readdirSync(chatsDir)
      .filter((f) => f.endsWith(".json"))
      .map((f) => {
        try {
          const c = JSON.parse(fs.readFileSync(path.join(chatsDir, f), "utf8"));
          return { id: c.id, title: c.title, updated: c.updated, pinned: !!c.pinned, folder: c.folder || null };
        } catch {
          return null;
        }
      })
      .filter(Boolean)
      .sort((a: any, b: any) => b.updated - a.updated);
  });
  // every saved chat, parsed (search and usage read them all)
  const allChats = (): any[] => {
    fs.mkdirSync(chatsDir, { recursive: true });
    const out: any[] = [];
    for (const f of fs.readdirSync(chatsDir)) {
      if (!f.endsWith(".json")) continue;
      try {
        out.push(JSON.parse(fs.readFileSync(path.join(chatsDir, f), "utf8")));
      } catch {}
    }
    return out;
  };
  handle("chats:search", (q: string) => searchChats(allChats(), q));
  handle("chats:usage", () => chatUsage(allChats()));
  // every chat in one JSON file, to keep or move to another PC
  handle("chats:backup", async () => {
    const d = new Date();
    const day = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    const r = await dialog.showSaveDialog(win!, { title: "Back up all chats", defaultPath: path.join(app.getPath("documents"), `Pelica chats ${day}.json`), filters: [{ name: "Pelica backup", extensions: ["json"] }] });
    if (r.canceled || !r.filePath) return null;
    const chats = allChats();
    fs.writeFileSync(r.filePath, JSON.stringify({ pelica: "backup", version: 1, created: Date.now(), chats }));
    return { path: r.filePath, count: chats.length };
  });
  // chats from a backup (or a folder's proj.pel): new ones are added; one
  // that differs from the chat of the same id here comes back as a copy, so
  // nothing here is overwritten
  handle("chats:restore", async () => {
    const r = await dialog.showOpenDialog(win!, { title: "Restore chats from a backup", properties: ["openFile"], filters: [{ name: "Pelica backup or proj.pel", extensions: ["json", "pel"] }] });
    if (r.canceled || !r.filePaths[0]) return null;
    let data: any;
    try {
      data = JSON.parse(fs.readFileSync(r.filePaths[0], "utf8"));
    } catch {
      throw new Error("That file isn't a Pelica backup.");
    }
    if (!data || (data.pelica !== "backup" && data.pelica !== "project") || !Array.isArray(data.chats)) throw new Error("That file isn't a Pelica backup.");
    fs.mkdirSync(chatsDir, { recursive: true });
    const out = { added: 0, copies: 0, same: 0 };
    for (const c of data.chats) {
      if (!c || typeof c.id !== "string" || !/^[\w-]+$/.test(c.id) || !Array.isArray(c.messages)) continue;
      const p = path.join(chatsDir, c.id + ".json");
      if (!fs.existsSync(p)) {
        fs.writeFileSync(p, JSON.stringify(c));
        out.added++;
        continue;
      }
      let here: any = null;
      try {
        here = JSON.parse(fs.readFileSync(p, "utf8"));
      } catch {}
      const same = (x: any) => x && x.updated === c.updated && (x.messages || []).length === c.messages.length;
      // (or a copy restored before)
      const copied = () =>
        fs
          .readdirSync(chatsDir)
          .filter((n) => n.startsWith(c.id + "-r") && n.endsWith(".json"))
          .some((n) => {
            try {
              return same(JSON.parse(fs.readFileSync(path.join(chatsDir, n), "utf8")));
            } catch {
              return false;
            }
          });
      if (same(here) || copied()) {
        out.same++;
        continue;
      }
      const id = c.id + "-r" + Date.now().toString(36);
      fs.writeFileSync(path.join(chatsDir, id + ".json"), JSON.stringify({ ...c, id, title: `${c.title || "Chat"} (restored)` }));
      out.copies++;
    }
    return out;
  });
  const chatPath = (id: string) => {
    if (!/^[\w-]+$/.test(id)) throw new Error("bad chat id");
    return path.join(chatsDir, id + ".json");
  };
  handle("chats:load", (id: string) => JSON.parse(fs.readFileSync(chatPath(id), "utf8")));
  handle("chats:save", (chat: any) => {
    fs.mkdirSync(chatsDir, { recursive: true });
    // the folder it had (known after the first save; read once otherwise)
    let was: string | null = null;
    if (chatFolder.has(chat.id)) was = chatFolder.get(chat.id)!;
    else
      try {
        was = JSON.parse(fs.readFileSync(chatPath(chat.id), "utf8")).folder || null;
      } catch {}
    chatFolder.set(chat.id, chat.folder || null);
    // (whole to a temporary file, then renamed: a crash mid-save never leaves
    // a broken file, which the list would skip, hiding the chat)
    const file = chatPath(chat.id);
    fs.writeFileSync(file + ".tmp", JSON.stringify(chat));
    fs.renameSync(file + ".tmp", file);
    if (was && was !== chat.folder) projQueue(was, chat.id, null);
    if (chat.folder) projQueue(chat.folder, chat.id, chat);
    return true;
  });
  // (to the Recycle Bin, not gone)
  handle("chats:delete", (id: string) => {
    chatFolder.delete(id);
    return shell.trashItem(chatPath(id));
  });

  handle("fs:list", (root: string, rel: string, depth: number) => ws.listFiles(checkRoot(root), rel, depth));
  handle("fs:read", (root: string, rel: string, offset: number, limit: number) => ws.readFile(checkRoot(root), rel, offset, limit));
  handle("fs:write", (root: string, rel: string, content: string) => ws.writeFile(checkRoot(root), rel, content));
  handle("fs:edit", (root: string, rel: string, o: string, n: string, all: boolean) => ws.editFile(checkRoot(root), rel, o, n, all));
  handle("fs:search", (root: string, pattern: string, rel: string, glob: string) => ws.searchFiles(checkRoot(root), pattern, rel, glob));
  // the project's instructions: PELICA.md (or AGENTS.md / CLAUDE.md) in the
  // folder and in the folders above it up to the repository's root (the one
  // with .git) when it is in one, the outermost first; "" when there are none
  handle("fs:notes", (root: string) => {
    const found: { dir: string; name: string; text: string }[] = [];
    let dir = checkRoot(root);
    // (above the folder only inside a repository)
    let levels = 0;
    for (let d = dir, i = 0; i < 8; i++) {
      if (fs.existsSync(path.join(d, ".git"))) {
        levels = i;
        break;
      }
      const parent = path.dirname(d);
      if (parent === d) break;
      d = parent;
    }
    for (let up = 0; up <= levels; up++) {
      for (const f of ["PELICA.md", "AGENTS.md", "CLAUDE.md"]) {
        const p = path.join(dir, f);
        try {
          if (fs.statSync(p).isFile()) {
            found.unshift({ dir, name: f, text: fs.readFileSync(p, "utf8").trim() });
            break;
          }
        } catch {}
      }
      dir = path.dirname(dir);
    }
    const max = 20000;
    let out = found.length === 1 ? found[0].text : found.map((f) => `From ${path.join(f.dir, f.name)}:\n${f.text}`).join("\n\n");
    if (out.length > max) out = out.slice(0, max) + "\n[… the rest of the instructions file is cut off]";
    return out;
  });
  handle("fs:find", (root: string, glob: string) => ws.findFiles(checkRoot(root), glob));
  // a workspace file as it is (for @-mentions in the message box)
  handle("fs:raw", (root: string, rel: string) => {
    const p = ws.inside(checkRoot(root), rel);
    if (fs.statSync(p).size > 4 * 1024 * 1024) throw new Error(rel + " is too big to attach");
    return fs.readFileSync(p, "utf8");
  });
  handle("fs:trash", (root: string, rel: string) => shell.trashItem(ws.inside(checkRoot(root), rel)));
  handle("fs:reveal", (root: string, rel: string) => shell.showItemInFolder(ws.inside(checkRoot(root), rel)));

  handle("cmd:run", (id: string, root: string, command: string, timeoutMs: number, background: boolean) => runCommand(id, root, command, timeoutMs || 120000, !!background, settings.shell || "cmd"));
  handle("cmd:kill", (id: string) => {
    killTree(procs.get(id)?.pid);
    procs.delete(id);
  });

  handle("web:search", (q: string) => web.search(q));
  handle("web:open", (url: string, max: number) => web.openPage(url, max));
  handle("web:repos", (text: string) => Promise.all(reposIn(text).map((r) => checkRepo(r.owner, r.repo))));

  handle("sys:external", (url: string) => /^https?:\/\//i.test(url) && shell.openExternal(url));

  handle("mcp:start", async () => {
    await mcp.startAll(settings.mcpServers || []);
    return mcp.list();
  });
  handle("mcp:list", () => mcp.list());
  handle("mcp:restart", async (name: string) => {
    const c = (settings.mcpServers || []).find((x: any) => x.name === name);
    if (c) await mcp.startOne(c);
    else mcp.stopOne(name);
    return mcp.list();
  });
  handle("mcp:launchers", () => mcp.launchers());
  handle("mcp:call", (server: string, tool: string, args: unknown) => mcp.call(server, tool, args));

  handle("lib:search", (q: string) => lib.searchRepos(q));
  handle("lib:files", (id: string) => lib.repoFiles(id));
  handle("lib:download", (repo: string, file: string, dir: string) => lib.download(repo, file, dir, dlProgress));
  handle("lib:downloads", (dir?: string) => lib.downloads(dir));
  handle("lib:pause", (id: string) => lib.pause(id));
  handle("lib:resume", (id: string) => lib.resume(id, dlProgress));
  // a cancelled download's part goes to the Recycle Bin
  handle("lib:cancel", async (id: string) => {
    for (const f of lib.cancel(id)) await shell.trashItem(f).catch(() => {});
    win?.webContents.send("lib:progress", lib.downloads());
  });
  // uninstalling a model: only one the engine lists, after asking; its files go to the Recycle Bin
  handle("lib:uninstall", async (p: string) => {
    const r = await fetch(`http://127.0.0.1:${port}/api/models`).then((x) => x.json());
    const m = (r.models as { path: string; name: string; gb: number }[]).find((x) => x.path === p);
    if (!m) throw new Error("That model isn't in your models folders.");
    const files = lib.modelFiles(p);
    const removable = await new Promise<boolean>((res) => execFile("powershell", ["-NoProfile", "-Command", `[System.IO.DriveInfo]::new('${path.parse(p).root.replace(/'/g, "")}').DriveType`], (_e, out) => res(String(out).trim() === "Removable")));
    const what = files.length > 1 ? `Its ${files.length} files` : "Its file";
    const { response } = await dialog.showMessageBox(win!, {
      type: "warning",
      title: "Uninstall model",
      message: `Uninstall ${m.name}?`,
      detail: removable
        ? `${what} (${m.gb.toFixed(1)} GB) are on a removable drive, which has no Recycle Bin: Windows may delete them for good.`
        : `${what} (${m.gb.toFixed(1)} GB) will go to the Recycle Bin. Empty it to free the space, or restore the model from there.`,
      buttons: ["Uninstall", "Cancel"],
      defaultId: 1,
      cancelId: 1,
      noLink: true,
    });
    if (response !== 0) return false;
    // (Windows can't move a file that's open)
    await unloadModel(p, "uninstall it");
    for (const f of files) await shell.trashItem(f);
    return true;
  });
  // speeding up a DeepSeek V4 file (pelica repack): after asking, or on its own
  // after a download when settings.autoRepack is on
  handle("lib:repacks", () => [...repacks.values()].map(({ child, ...r }) => r));
  handle("lib:repack", async (p: string, undo: boolean) => {
    const m = await engineModel(p);
    if (!m || !m.repack) throw new Error("Only DeepSeek V4 and Qwen3.8 models can be sped up this way.");
    if (repacks.get(p)?.state === "running") return true;
    // (the 164 GB file took about 10 minutes here; a layer, 1/43 of the file,
    // is held three times over in RAM)
    const size = m.gb >= 10 ? `${m.gb.toFixed(0)} GB` : `${m.gb.toFixed(1)} GB`;
    const takes = `about ${Math.max(1, Math.round(m.gb / 16))} minute${Math.round(m.gb / 16) > 1 ? "s" : ""}`;
    const ram = Math.max(1, Math.ceil((m.gb * 3) / 43));
    const { response } = await dialog.showMessageBox(win!, {
      type: undo ? "question" : "info",
      title: undo ? "Undo speed-up" : "Speed up model",
      message: undo ? `Put ${m.name} back the way it was?` : `Speed up ${m.name}?`,
      detail: undo
        ? `Pelica rewrites the file (${size}) back into the usual GGUF layout, so other programs can open it again. Answers get about 20% slower. It takes ${takes} and ${ram} GB of free RAM, and the model can't be used until it's done.`
        : `Pelica rearranges the file (${size}) so each expert loads in one read: answers come about 20% faster. It rewrites the file in place, which takes ${takes} and ${ram} GB of free RAM, and the model can't be used until it's done. Stopping partway is safe: start it again to finish.\n\nOther programs (llama.cpp, LM Studio) can't open the file afterwards. Undo puts it back.`,
      buttons: [undo ? "Undo speed-up" : "Speed up", "Cancel"],
      defaultId: 0,
      cancelId: 1,
      noLink: true,
    });
    if (response !== 0) return false;
    await startRepack(p, undo);
    return true;
  });
  // (safe at any point: the journal puts the layer it was on right the next time it runs)
  handle("lib:repackStop", (p: string) => {
    const r = repacks.get(p);
    if (r?.state !== "running") return;
    r.state = "stopped";
    r.child?.kill();
    sendRepacks();
  });
  handle("lib:pickDir", async () => {
    const r = await dialog.showOpenDialog(win!, { properties: ["openDirectory", "createDirectory"], title: "Where models are kept" });
    return r.canceled ? null : r.filePaths[0];
  });

  handle("engine:log", () => engineLog);
  handle("engine:restart", async () => {
    // (the quick bar points at the old port)
    quick?.destroy();
    await startEngine();
    await win?.loadURL(`http://127.0.0.1:${port}/`);
  });
  handle("app:info", () => ({ version: app.getVersion(), userDir, pelica: pelicaExe(), electron: process.versions.electron, hotkey: hotkey(), hotkeyError }));

  // the desktop
  handle("app:hotkey", (k: string) => {
    settings.hotkey = String(k ?? "");
    saveSettings();
    return registerHotkey();
  });
  // what the window is doing: the taskbar button fills, the tray says it
  handle("app:busy", (b: { label: string; progress: number } | null) => {
    if (!win) return;
    if (!b) {
      win.setProgressBar(-1);
      tray?.setToolTip("Pelica");
      return;
    }
    win.setProgressBar(b.progress >= 0 ? Math.min(1, b.progress) : 2, { mode: b.progress >= 0 ? "normal" : "indeterminate" });
    tray?.setToolTip(`Pelica · ${b.label}`.slice(0, 120));
  });
  // a desktop notification, only when the user is looking elsewhere
  handle("app:notify", (n: { title: string; body: string; chatId?: string }) => {
    if (!win || win.isFocused()) return false;
    win.flashFrame(true);
    if (!Notification.isSupported()) return false;
    const note = new Notification({ title: String(n.title).slice(0, 80), body: String(n.body).slice(0, 240), icon: icon ?? undefined });
    note.on("click", () => (n.chatId ? action({ kind: "open-chat", id: n.chatId }) : showMain()));
    note.show();
    return true;
  });
  handle("app:login", (on?: boolean) => {
    const opts = { path: process.execPath, args: [...(app.isPackaged ? [] : [app.getAppPath()]), "--hidden"] };
    if (typeof on === "boolean") app.setLoginItemSettings({ ...opts, openAtLogin: on });
    return app.getLoginItemSettings(opts).openAtLogin;
  });
  handle("win:material", (m: Material) => {
    settings.material = m;
    saveSettings();
    const mm = material();
    win?.setBackgroundColor(mm === "none" ? SOLID : "#00000000");
    win?.setBackgroundMaterial(mm);
    return mm;
  });

  handle("quick:submit", (text: string, reply?: string) => {
    quick?.hide();
    action({ kind: "ask", text: String(text), reply: typeof reply === "string" ? reply : undefined });
  });
  handle("quick:hide", () => quick?.hide());
  handle("quick:resize", (h: number) => quick && placeQuick(Math.max(80, Math.min(560, h))));
  // the clipboard, for the quick bar's "with clipboard" chip (the user's click)
  handle("quick:clip", () => clipboard.readText().slice(0, 20000));

  // the computer, for the tools (reading the clipboard and opening files ask first, in the UI)
  handle("sys:info", () => systemInfo());
  handle("sys:open", async (root: string, rel: string) => {
    const err = await shell.openPath(ws.inside(checkRoot(root), rel));
    if (err) throw new Error(err);
  });
  handle("sys:clipRead", () => clipboard.readText().slice(0, 50000));
  handle("sys:clipWrite", (t: string) => clipboard.writeText(String(t ?? "")));
  // a chat exported: the user picks where (nothing is written without the dialog)
  handle("sys:saveAs", async (name: string, text: string, dir?: string) => {
    const safe = String(name || "chat").replace(/[<>:"/\\|?*\x00-\x1f]+/g, " ").trim().slice(0, 80) || "chat";
    const base = dir && fs.existsSync(dir) ? dir : app.getPath("documents");
    const r = await dialog.showSaveDialog(win!, { title: "Export the chat", defaultPath: path.join(base, safe + ".md"), filters: [{ name: "Markdown", extensions: ["md"] }, { name: "Text", extensions: ["txt"] }] });
    if (r.canceled || !r.filePath) return null;
    fs.writeFileSync(r.filePath, String(text), "utf8");
    return r.filePath;
  });
  handle("sys:reveal", (p: string) => shell.showItemInFolder(String(p)));
  // "Keep computer awake" in the view options: until it's turned off or Pelica quits
  handle("sys:awake", (on: boolean) => {
    if (on && awakeId === null) awakeId = powerSaveBlocker.start("prevent-app-suspension");
    if (!on && awakeId !== null) {
      powerSaveBlocker.stop(awakeId);
      awakeId = null;
    }
    return awakeId !== null;
  });
  // photos for models that read text, and voice for the message box
  handle("media:ocr", (dataUrl: string) => media.ocr(dataUrl));
  handle("media:dictateStart", () => media.dictateStart((e) => win?.webContents.send("media:dictate", e)));
  handle("media:dictateStop", () => media.dictateStop());
}

// one Pelica at a time: a second start (jump list, a shortcut) reaches this one
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", (_e, argv) => handleArgs(argv));
  app.setAppUserModelId("Pelica.Desktop");

  // the browser pane's <webview>: web pages only, never with Pelica's preload;
  // links that open a new window load in the same pane
  app.on("web-contents-created", (_e, contents) => {
    contents.on("will-attach-webview", (e, wp, params) => {
      delete wp.preload;
      wp.nodeIntegration = false;
      wp.contextIsolation = true;
      wp.sandbox = true;
      if (!/^(https?:|about:blank)/i.test(params.src || "about:blank")) e.preventDefault();
    });
    if (contents.getType() === "webview") {
      contents.setWindowOpenHandler(({ url }) => {
        if (/^https?:\/\//i.test(url)) contents.loadURL(url);
        return { action: "deny" };
      });
    }
  });

  app.whenReady().then(async () => {
    registerIpc();
    makeTray();
    registerHotkey();
    jumpList();
    await createWindow();
    syncApi();
    mcp.startAll(settings.mcpServers || []).catch(() => {});
    if (process.argv.includes("--quick")) openQuick();
  });

  app.on("window-all-closed", () => {
    stopEngine();
    mcp.stopAll();
    for (const p of procs.values()) killTree(p.pid);
    app.quit();
  });

  app.on("before-quit", () => {
    projFlush();
    media.dictateStop();
    quitting = true;
    stopEngine();
    // (a stopped repack finishes the next time it runs)
    for (const r of repacks.values()) r.child?.kill();
  });
  app.on("will-quit", () => globalShortcut.unregisterAll());
}
