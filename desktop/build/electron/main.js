"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
// Pelica Desktop's main process: starts pelica.exe (the inference server,
// serving this app's built UI), opens a frameless window on it, and gives
// the UI what a page can't do itself: the workspace folder's files,
// commands, web requests, MCP servers, downloads.
// It also ties Pelica into Windows: a tray icon, a global quick-ask hotkey,
// notifications, taskbar progress and jump list, Mica.
const electron_1 = require("electron");
const node_child_process_1 = require("node:child_process");
const node_fs_1 = __importDefault(require("node:fs"));
const node_http_1 = __importDefault(require("node:http"));
const node_net_1 = __importDefault(require("node:net"));
const node_os_1 = __importDefault(require("node:os"));
const node_path_1 = __importDefault(require("node:path"));
const web = __importStar(require("./web"));
const ws = __importStar(require("./workspace"));
const mcp = __importStar(require("./mcp"));
const lib = __importStar(require("./library"));
const media = __importStar(require("./media"));
const chatdata_1 = require("./chatdata");
const repocheck_1 = require("./repocheck");
const updates = __importStar(require("./updates"));
const importer_1 = require("./importer");
const phone = __importStar(require("./phone"));
const oauth = __importStar(require("./oauth"));
const google = __importStar(require("./google"));
const sync = __importStar(require("./sync"));
const userDir = electron_1.app.getPath("userData");
const settingsFile = node_path_1.default.join(userDir, "settings.json");
const chatsDir = node_path_1.default.join(userDir, "chats");
function loadSettings() {
    try {
        return JSON.parse(node_fs_1.default.readFileSync(settingsFile, "utf8"));
    }
    catch {
        return {};
    }
}
let settings = loadSettings();
function saveSettings() {
    node_fs_1.default.mkdirSync(userDir, { recursive: true });
    node_fs_1.default.writeFileSync(settingsFile, JSON.stringify(settings, null, 2));
}
/** Folders the user picked: the only roots file tools and commands may use. */
const roots = () => settings.roots || [];
let awakeId = null;
function checkRoot(root) {
    const r = node_path_1.default.resolve(root || "");
    if (!roots().some((x) => node_path_1.default.resolve(x).toLowerCase() === r.toLowerCase()))
        throw new Error("no workspace folder chosen (pick one with the folder button)");
    return r;
}
// ---------- the engine ----------
let engine = null;
let port = 0;
let engineLog = "";
function pelicaExe() {
    const cands = [settings.pelicaPath, node_path_1.default.resolve(__dirname, "../../../target/release/pelica.exe"), node_path_1.default.resolve(process.resourcesPath || "", "pelica.exe")].filter(Boolean);
    return cands.find((c) => node_fs_1.default.existsSync(c)) || cands[1];
}
function freePort() {
    return new Promise((ok, bad) => {
        const s = node_net_1.default.createServer();
        s.listen(0, "127.0.0.1", () => {
            const p = s.address().port;
            s.close(() => ok(p));
        });
        s.on("error", bad);
    });
}
// ---------- models folders ----------
/** The engine's own default folders (when none are set). */
const DEFAULT_DIRS = ["C:\\models", "D:\\Models"];
const sameDir = (a, b) => node_path_1.default.resolve(a).toLowerCase() === node_path_1.default.resolve(b).toLowerCase();
const driveOk = (d) => node_fs_1.default.existsSync(node_path_1.default.parse(node_path_1.default.resolve(d)).root);
/** Where downloads go: the folder set in Settings when its drive is there,
 *  else whichever default folder's drive has the most free space. */
function downloadDir() {
    const set = String(settings.downloadDir || "");
    if (set && driveOk(set))
        return set;
    const free = (d) => {
        try {
            const st = node_fs_1.default.statfsSync(node_path_1.default.parse(d).root);
            return st.bavail * st.bsize;
        }
        catch {
            return -1;
        }
    };
    return DEFAULT_DIRS.filter(driveOk).sort((a, b) => free(b) - free(a))[0] || DEFAULT_DIRS[0];
}
/** Every folder the engine looks in: the ones set (or the defaults), and the
 *  download folder, so downloads always show up. */
function engineDirs() {
    const dirs = [...(settings.modelDirs?.length ? settings.modelDirs : DEFAULT_DIRS)];
    const dl = downloadDir();
    if (!dirs.some((d) => sameDir(d, dl)))
        dirs.push(dl);
    return dirs;
}
/** Tells the running engine about changed folders (no restart). */
function syncDirs() {
    if (!engine)
        return;
    fetch(`http://127.0.0.1:${port}/api/dirs`, { method: "POST", body: JSON.stringify({ dirs: engineDirs() }) }).catch(() => { });
}
async function startEngine() {
    stopEngine();
    port = await freePort();
    const ui = node_path_1.default.resolve(__dirname, "../../dist");
    // no model until one is picked, unless "Load the last model at start" is on
    const args = ["app", "--no-open", "--no-autoload", "--port", String(port), "--ui", ui];
    if (settings.loadOnStart && settings.lastModel && node_fs_1.default.existsSync(settings.lastModel))
        args.push("--model", settings.lastModel, "--ctx", String(settings.lastCtx || 32768));
    for (const d of engineDirs())
        args.push("--models-dir", d);
    engineLog = `$ ${pelicaExe()} ${args.join(" ")}\n`;
    // computers lending their RAM (Settings → Devices); the app also sends changes live
    const devices = (settings.devices || []).map((d) => `${d.addr}/${d.code}`).join(",");
    engine = (0, node_child_process_1.spawn)(pelicaExe(), args, { windowsHide: true, stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, PELICA_DEVICES: devices } });
    const add = (d) => (engineLog = (engineLog + d.toString()).slice(-200000));
    engine.stdout.on("data", add);
    engine.stderr.on("data", add);
    engine.on("exit", (code) => {
        engineLog += `\n[pelica exited: ${code}]\n`;
        // STATUS_ILLEGAL_INSTRUCTION: a processor without AVX2 (the engine is built for x86-64-v3)
        if (code === 0xc000001d || code === -1073741795)
            engineLog += "\nThis PC's processor is missing AVX2, which Pelica's engine needs (Intel processors from 2013 on and AMD from 2015 on have it).\n";
        engine = null;
    });
    engine.on("error", (e) => (engineLog += `\n[could not start pelica: ${e.message}]\n`));
    // ready when the status answers
    for (let i = 0; i < 300; i++) {
        if (!engine)
            throw new Error("pelica stopped:\n" + engineLog.slice(-2000));
        try {
            const r = await fetch(`http://127.0.0.1:${port}/api/status`);
            if (r.ok)
                return announceEngine();
        }
        catch { }
        await new Promise((ok) => setTimeout(ok, 100));
    }
    throw new Error("pelica didn't answer:\n" + engineLog.slice(-2000));
}
// ---------- the local API (Settings → Models → Local API) ----------
// other programs on this computer use the loaded model through the engine's
// OpenAI-compatible endpoints, on a port that stays the same between starts
// (the engine's own changes); only /v1/ passes, and only from this computer
let apiServer = null;
let apiPort = 0;
let apiError = "";
function syncApi() {
    const want = settings.apiServer ? Number(settings.apiPort) || 11435 : 0;
    if (want === apiPort && (apiServer || !want))
        return;
    apiServer?.close();
    apiServer = null;
    apiPort = want;
    apiError = "";
    if (!want)
        return;
    const srv = node_http_1.default.createServer((req, res) => {
        if (!req.url?.startsWith("/v1/") || !port) {
            res.writeHead(404, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: { message: port ? "not found" : "the engine isn't running", type: "not_found" } }));
            return;
        }
        // (the engine accepts its own host only, and turns away web pages)
        const up = node_http_1.default.request({ host: "127.0.0.1", port, path: req.url, method: req.method, headers: { ...req.headers, host: `127.0.0.1:${port}` } }, (r) => {
            res.writeHead(r.statusCode || 502, r.headers);
            r.pipe(res);
        });
        up.on("error", (e) => {
            if (!res.headersSent)
                res.writeHead(502, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: { message: e.message, type: "unavailable" } }));
        });
        // the caller going away stops the reply
        res.on("close", () => up.destroy());
        req.pipe(up);
    });
    srv.on("error", (e) => {
        apiError = e.code === "EADDRINUSE" ? `Port ${want} is in use by another program. Pick another.` : e.message;
        if (apiServer === srv)
            apiServer = null;
    });
    srv.listen(want, "127.0.0.1");
    apiServer = srv;
}
// Pelica on your phone (Settings → Phone): the page and its pairing key
function syncPhone() {
    if (settings.phoneOn && !settings.phoneKey) {
        settings.phoneKey = phone.newKey();
        saveSettings();
    }
    // (the phone follows replies as they stream, also while the window is hidden in the tray)
    win?.webContents.setBackgroundThrottling(!settings.phoneOn);
    phone.sync(settings.phoneOn ? Number(settings.phonePort) || 8737 : 0, settings.phoneKey || "", node_path_1.default.resolve(__dirname, "../../dist"), (c) => win?.webContents.send("phone:cmd", c));
}
// where `pelica chat` in a terminal finds this engine, to use the loaded
// model instead of loading a second copy
const engineFile = node_path_1.default.join(userDir, "engine.json");
function announceEngine() {
    try {
        node_fs_1.default.writeFileSync(engineFile, JSON.stringify({ port, pid: engine?.pid, app: process.pid }));
    }
    catch { }
}
function stopEngine() {
    try {
        if (JSON.parse(node_fs_1.default.readFileSync(engineFile, "utf8")).port === port)
            node_fs_1.default.unlinkSync(engineFile);
    }
    catch { }
    if (engine?.pid) {
        (0, node_child_process_1.execFile)("taskkill", ["/pid", String(engine.pid), "/T", "/F"], () => { });
    }
    engine = null;
}
async function engineModel(p) {
    const r = await fetch(`http://127.0.0.1:${port}/api/models`).then((x) => x.json());
    const key = (x) => node_path_1.default.resolve(x).toLowerCase();
    return r.models.find((x) => key(x.path) === key(p));
}
/** Unloads `p` if it's the loaded model; `what` finishes "Stop it, then …". */
async function unloadModel(p, what) {
    const status = () => fetch(`http://127.0.0.1:${port}/api/status`).then((x) => x.json());
    let st = await status();
    if (st.path === p && st.state === "loading")
        throw new Error(`It's still loading. Wait for it to finish, then ${what}.`);
    if (st.path === p && st.state === "ready") {
        const u = await fetch(`http://127.0.0.1:${port}/api/unload`, { method: "POST", body: "{}" });
        if (!u.ok)
            throw new Error(`Pelica is busy with a reply. Stop it, then ${what}.`);
        for (let i = 0; i < 120 && st.path === p && st.state === "ready"; i++) {
            await new Promise((res) => setTimeout(res, 250));
            st = await status();
        }
    }
}
const repacks = new Map();
/** Models being moved (Settings → Models → Storage), by their old path. */
const moves = new Map();
const sendRepacks = () => win?.webContents.send("lib:repack", [...repacks.values()].map(({ child, ...r }) => r));
async function startRepack(p, undo) {
    if (repacks.get(p)?.state === "running")
        return;
    // (taken before anything waits: a second click meanwhile finds it running
    // instead of starting a repack that would fail on the locked file)
    const r = { path: p, undo, state: "running", pct: 0, left: null };
    repacks.set(p, r);
    sendRepacks();
    try {
        await unloadModel(p, undo ? "undo the speed-up" : "speed it up");
    }
    catch (e) {
        repacks.delete(p);
        sendRepacks();
        throw e;
    }
    const child = (0, node_child_process_1.spawn)(pelicaExe(), ["repack", p, ...(undo ? ["--undo"] : [])], { windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
    r.child = child;
    let tail = "";
    // its lines: "layer 12 done in 14.2s (28%, about 380s left)"
    child.stderr.on("data", (d) => {
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
        if (r.state !== "running")
            return;
        if (code === 0)
            (r.state = "done"), (r.pct = 100), (r.left = 0);
        else
            (r.state = "error"), (r.error = (tail.match(/Error: ([\s\S]*)$/)?.[1] || tail || `pelica stopped (${code ?? signal})`).trim().split("\n").slice(0, 3).join(" "));
        sendRepacks();
        if (r.state === "done")
            new electron_1.Notification({ title: undo ? "Speed-up undone" : "Model sped up", body: `${node_path_1.default.basename(p)} is ready to load.` }).show();
    });
    sendRepacks();
}
// downloads: a finished DeepSeek V4 or Qwen3.8 file is sped up on its own when settings.autoRepack is on
const dlRunning = new Set();
function dlProgress(jobs) {
    win?.webContents.send("lib:progress", jobs);
    for (const j of jobs) {
        if (j.state === "running")
            dlRunning.add(j.id);
        // (only a download that finished just now)
        if (j.state !== "done" || !dlRunning.delete(j.id) || !settings.autoRepack)
            continue;
        engineModel(j.dest)
            .then((m) => (m?.repack === "no" ? startRepack(m.path, false) : undefined))
            .catch(() => { });
    }
}
// ---------- the window ----------
let win = null;
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
/** Windows 11's backdrop behind the window (the page lets it show through). */
const material = () => (["mica", "acrylic", "tabbed"].includes(settings.material) ? settings.material : "none");
const SOLID = "#1f1e1d";
/** Mica and Acrylic take their tint from the app's light or dark theme. */
const syncTheme = () => (electron_1.nativeTheme.themeSource = settings.theme === "light" ? "light" : "dark");
async function createWindow() {
    syncTheme();
    const m = material();
    win = new electron_1.BrowserWindow({
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
        webPreferences: { preload: node_path_1.default.join(__dirname, "preload.js"), contextIsolation: true, nodeIntegration: false, sandbox: true, spellcheck: true, webviewTag: true },
    });
    electron_1.Menu.setApplicationMenu(null);
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
        electron_1.app.quit();
    });
    // the UI never navigates away; links open in the default browser
    const external = (url) => /^https?:\/\//i.test(url) && electron_1.shell.openExternal(url).catch(() => { });
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
    }
    catch (e) {
        const msg = `data:text/html;charset=utf-8,${encodeURIComponent(`<body style="background:#1f1e1d;color:#e8e4dc;font:14px system-ui;padding:40px"><h2>Pelica couldn't start</h2><pre style="white-space:pre-wrap;color:#c9c5bd">${String(e.message).replace(/</g, "&lt;")}</pre><p>Set the path to pelica.exe in settings.json (${settingsFile.replace(/\\/g, "\\\\")}) as "pelicaPath", or build it with cargo build --release.</p></body>`)}`;
        await win.loadURL(msg);
    }
}
/** The window, taskbar and tray icon: the logo, drawn from its SVG by the
 *  page and kept in the user folder so the next start has it at once. */
const iconFile = node_path_1.default.join(userDir, "icon.png");
let icon = node_fs_1.default.existsSync(iconFile) ? electron_1.nativeImage.createFromPath(iconFile) : null;
async function setIcon() {
    try {
        const url = await win.webContents.executeJavaScript(`(async () => {
      const svg = await (await fetch('./logo.svg')).text();
      const img = new Image(); img.src = 'data:image/svg+xml;base64,' + btoa(svg); await img.decode();
      const c = document.createElement('canvas'); c.width = c.height = 256;
      c.getContext('2d').drawImage(img, 0, 0, 256, 256); return c.toDataURL('image/png');
    })()`);
        icon = electron_1.nativeImage.createFromDataURL(url);
        win?.setIcon(icon);
        node_fs_1.default.writeFileSync(iconFile, icon.toPNG());
        makeTray();
    }
    catch { }
}
// ---------- the desktop: tray, quick ask, notifications, taskbar ----------
let tray = null;
let quitting = false;
const startHidden = process.argv.includes("--hidden");
function showMain() {
    if (!win)
        return;
    if (win.isMinimized())
        win.restore();
    win.show();
    win.focus();
}
/** Asks the UI to do something: new chat, ask, open a chat, show a page. */
function action(a) {
    showMain();
    win?.webContents.send("app:action", a);
}
function makeTray() {
    if (!icon)
        return;
    if (tray)
        return tray.setImage(icon.resize({ width: 16, height: 16, quality: "best" }));
    tray = new electron_1.Tray(icon.resize({ width: 16, height: 16, quality: "best" }));
    tray.setToolTip("Pelica");
    tray.on("click", showMain);
    trayMenu();
}
function trayMenu() {
    if (!tray)
        return;
    const k = hotkey();
    tray.setContextMenu(electron_1.Menu.buildFromTemplate([
        { label: "Open Pelica", click: showMain },
        { label: "New chat", click: () => action({ kind: "new-chat" }) },
        { label: "Quick ask…", click: openQuick, accelerator: k || undefined, registerAccelerator: false },
        { type: "separator" },
        ...(settings.brainMenu ? [{ label: "Brain", click: () => action({ kind: "view", view: "brain" }) }] : []),
        { label: "Models", click: () => action({ kind: "view", view: "library" }) },
        { label: "Settings", click: () => action({ kind: "settings" }) },
        { type: "separator" },
        {
            label: "Quit Pelica",
            click: () => {
                quitting = true;
                electron_1.app.quit();
            },
        },
    ]));
}
/** The global quick-ask shortcut ("" turns it off). */
const hotkey = () => (typeof settings.hotkey === "string" ? settings.hotkey : "Alt+Shift+Space");
let hotkeyError = "";
function registerHotkey() {
    electron_1.globalShortcut.unregisterAll();
    hotkeyError = "";
    const k = hotkey();
    if (k) {
        try {
            if (!electron_1.globalShortcut.register(k, openQuick))
                hotkeyError = `${k} is already used by another app`;
        }
        catch (e) {
            hotkeyError = `${k} isn't a shortcut Pelica understands (${e.message})`;
        }
    }
    trayMenu();
    return hotkeyError;
}
// the quick-ask window: a small bar over everything, wherever the mouse is
let quick = null;
const QW = 720;
function placeQuick(h) {
    if (!quick)
        return;
    const d = electron_1.screen.getDisplayNearestPoint(electron_1.screen.getCursorScreenPoint()).workArea;
    quick.setBounds({ x: Math.round(d.x + (d.width - QW) / 2), y: Math.round(d.y + d.height * 0.2), width: QW, height: Math.round(h) });
}
function openQuick() {
    if (!port || !engine)
        return showMain();
    if (quick && !quick.isDestroyed()) {
        if (quick.isVisible())
            return quick.hide();
        placeQuick(quick.getBounds().height);
        quick.show();
        quick.focus();
        quick.webContents.send("quick:shown");
        return;
    }
    quick = new electron_1.BrowserWindow({
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
        webPreferences: { preload: node_path_1.default.join(__dirname, "preload.js"), contextIsolation: true, nodeIntegration: false, sandbox: true },
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
    const base = electron_1.app.isPackaged ? [] : [`"${electron_1.app.getAppPath()}"`];
    const task = (flag, title, description) => ({ program: process.execPath, arguments: [...base, flag].join(" "), title, description, iconPath: process.execPath, iconIndex: 0 });
    try {
        electron_1.app.setUserTasks([task("--new-chat", "New chat", "Start a new chat in Pelica"), task("--quick", "Quick ask", "Ask Pelica something from anywhere")]);
    }
    catch { }
}
/** What a second start (jump list, shortcut) asked for. */
function handleArgs(argv) {
    if (argv.includes("--quick"))
        openQuick();
    else if (argv.includes("--new-chat"))
        action({ kind: "new-chat" });
    else
        showMain();
}
const gb = (n) => `${(n / 1024 ** 3).toFixed(1)} GB`;
/** The computer, for the system_info tool (nothing personal: no user name, no paths). */
async function systemInfo() {
    const cpus = node_os_1.default.cpus();
    const info = {
        os: `${node_os_1.default.version()} ${node_os_1.default.release()} (${node_os_1.default.arch()})`,
        computer: node_os_1.default.hostname(),
        cpu: `${(cpus[0]?.model || "?").trim()}, ${cpus.length} threads`,
        memory: `${gb(node_os_1.default.freemem())} free of ${gb(node_os_1.default.totalmem())}`,
        uptime: `${(node_os_1.default.uptime() / 3600).toFixed(1)} hours`,
        local_time: new Date().toString(),
        locale: electron_1.app.getLocale(),
        displays: electron_1.screen
            .getAllDisplays()
            .map((d) => `${d.size.width}×${d.size.height} at ${Math.round(d.scaleFactor * 100)}%`)
            .join(", "),
        power: electron_1.powerMonitor.isOnBatteryPower() ? "on battery" : "plugged in",
    };
    for (const r of roots().slice(0, 3)) {
        try {
            const s = await node_fs_1.default.promises.statfs(r);
            info[`disk (${node_path_1.default.parse(r).root})`] = `${gb(s.bavail * s.bsize)} free of ${gb(s.blocks * s.bsize)}`;
        }
        catch { }
    }
    return info;
}
// ---------- commands ----------
const procs = new Map();
function killTree(pid) {
    if (pid)
        (0, node_child_process_1.execFile)("taskkill", ["/pid", String(pid), "/T", "/F"], () => { });
}
function runCommand(id, root, command, timeoutMs, background, shellKind) {
    const cwd = checkRoot(root);
    const [exe, args] = shellKind === "powershell" ? ["powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", command]] : ["cmd.exe", ["/d", "/s", "/c", `"${command}"`]];
    const p = (0, node_child_process_1.spawn)(exe, args, { cwd, windowsHide: true, env: { ...process.env, FORCE_COLOR: "0", CI: "1" }, windowsVerbatimArguments: shellKind !== "powershell" });
    procs.set(id, p);
    let output = "";
    const add = (d) => {
        const s = d.toString();
        output = (output + s).slice(-400000);
        win?.webContents.send("cmd:data", { id, data: s });
    };
    p.stdout.on("data", add);
    p.stderr.on("data", add);
    return new Promise((ok) => {
        let done = false;
        const finish = (r) => {
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
        }
        else {
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
function handle(name, fn) {
    electron_1.ipcMain.handle(name, async (_e, ...a) => fn(...a));
}
// ---------- proj.pel: each folder keeps its own chats ----------
//
// Every save of a chat that works in a folder also updates <folder>/proj.pel
// (JSON: { pelica: "project", version: 1, updated, chats: [...] }). Deleting
// a chat in the app leaves it there, so choosing the folder again (or moving
// the folder and choosing it where it is now) brings its chats back.
/** each saved chat's folder, so a save knows when it moved without reading the old file */
const chatFolder = new Map();
const projPending = new Map(); // folder -> chat id -> chat (null = take out)
let projTimer = null;
function projQueue(folder, id, chat) {
    // only folders the user picked
    if (!roots().some((x) => node_path_1.default.resolve(x).toLowerCase() === node_path_1.default.resolve(folder).toLowerCase()))
        return;
    const m = projPending.get(folder) || new Map();
    m.set(id, chat);
    projPending.set(folder, m);
    if (!projTimer)
        projTimer = setTimeout(projFlush, 1500);
}
function readProject(folder) {
    try {
        const v = JSON.parse(node_fs_1.default.readFileSync(node_path_1.default.join(folder, ws.PROJ_FILE), "utf8"));
        return Array.isArray(v?.chats) ? v.chats.filter((c) => c && typeof c.id === "string" && /^[\w-]+$/.test(c.id) && Array.isArray(c.messages)) : [];
    }
    catch {
        return [];
    }
}
function projFlush() {
    if (projTimer)
        clearTimeout(projTimer);
    projTimer = null;
    for (const [folder, m] of projPending) {
        try {
            if (!node_fs_1.default.statSync(folder).isDirectory())
                continue;
            const chats = new Map(readProject(folder).map((c) => [c.id, c]));
            for (const [id, c] of m) {
                if (c)
                    chats.set(id, c);
                else
                    chats.delete(id);
            }
            const file = node_path_1.default.join(folder, ws.PROJ_FILE);
            if (!chats.size && !node_fs_1.default.existsSync(file))
                continue;
            const list = [...chats.values()].sort((a, b) => (b.updated || 0) - (a.updated || 0));
            // written whole to a temporary file first, so a crash never leaves half a file
            const tmp = file + ".tmp";
            node_fs_1.default.writeFileSync(tmp, JSON.stringify({ pelica: "project", version: 1, updated: Date.now(), chats: list }));
            node_fs_1.default.renameSync(tmp, file);
        }
        catch { }
    }
    projPending.clear();
}
/** Brings back the folder's chats this PC doesn't have; returns their ids, newest first. */
function restoreProject(folder) {
    projFlush();
    node_fs_1.default.mkdirSync(chatsDir, { recursive: true });
    const back = [];
    for (const c of readProject(folder)) {
        const p = node_path_1.default.join(chatsDir, c.id + ".json");
        if (node_fs_1.default.existsSync(p))
            continue;
        // the folder may have moved: the chat works where it is now
        node_fs_1.default.writeFileSync(p, JSON.stringify({ ...c, folder }));
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
    handle("settings:set", (patch) => {
        // (roots only grow through the folder picker)
        const { roots: _r, ...rest } = patch || {};
        settings = { ...settings, ...rest };
        saveSettings();
        if ("theme" in rest)
            syncTheme();
        if ("apiServer" in rest || "apiPort" in rest)
            syncApi();
        if ("phoneOn" in rest || "phonePort" in rest)
            syncPhone();
        if ("brainMenu" in rest)
            trayMenu();
        if (sync.SYNCED.some((k) => k in rest))
            sync.noteProfile();
        if ("syncOn" in rest && rest.syncOn)
            sync.soon(300);
        if ("modelDirs" in rest || "downloadDir" in rest)
            syncDirs();
        return settings;
    });
    handle("api:status", () => ({ on: !!apiServer && !apiError, port: apiPort, error: apiError }));
    handle("phone:status", () => ({ ...phone.status(), key: settings.phoneKey || "" }));
    handle("phone:push", (snap) => phone.push(snap));
    // a new pairing key: phones paired before can't connect until they scan again
    handle("phone:newKey", () => {
        settings.phoneKey = phone.newKey();
        saveSettings();
        syncPhone();
        return { ...phone.status(), key: settings.phoneKey };
    });
    handle("folder:pick", async () => {
        const r = await electron_1.dialog.showOpenDialog(win, { properties: ["openDirectory", "createDirectory"], title: "Choose the folder Pelica works in" });
        if (r.canceled || !r.filePaths[0])
            return null;
        const p = r.filePaths[0];
        settings.roots = [p, ...roots().filter((x) => x.toLowerCase() !== p.toLowerCase())].slice(0, 20);
        saveSettings();
        return p;
    });
    // chats kept in the folder's proj.pel that this PC no longer has (deleted, or another PC's)
    handle("folder:restore", (root) => restoreProject(checkRoot(root)));
    handle("folder:open", (root) => electron_1.shell.openPath(checkRoot(root)));
    handle("chats:list", () => {
        node_fs_1.default.mkdirSync(chatsDir, { recursive: true });
        return node_fs_1.default
            .readdirSync(chatsDir)
            .filter((f) => f.endsWith(".json"))
            .map((f) => {
            try {
                const c = JSON.parse(node_fs_1.default.readFileSync(node_path_1.default.join(chatsDir, f), "utf8"));
                return { id: c.id, title: c.title, updated: c.updated, pinned: !!c.pinned, folder: c.folder || null, project: c.project || undefined };
            }
            catch {
                return null;
            }
        })
            .filter(Boolean)
            .sort((a, b) => b.updated - a.updated);
    });
    // every saved chat, parsed (search and usage read them all)
    const allChats = () => {
        node_fs_1.default.mkdirSync(chatsDir, { recursive: true });
        const out = [];
        for (const f of node_fs_1.default.readdirSync(chatsDir)) {
            if (!f.endsWith(".json"))
                continue;
            try {
                out.push(JSON.parse(node_fs_1.default.readFileSync(node_path_1.default.join(chatsDir, f), "utf8")));
            }
            catch { }
        }
        return out;
    };
    handle("chats:search", (q) => (0, chatdata_1.searchChats)(allChats(), q));
    handle("chats:usage", () => (0, chatdata_1.chatUsage)(allChats()));
    // every chat in one JSON file, to keep or move to another PC
    handle("chats:backup", async () => {
        const d = new Date();
        const day = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
        const r = await electron_1.dialog.showSaveDialog(win, { title: "Back up all chats", defaultPath: node_path_1.default.join(electron_1.app.getPath("documents"), `Pelica chats ${day}.json`), filters: [{ name: "Pelica backup", extensions: ["json"] }] });
        if (r.canceled || !r.filePath)
            return null;
        const chats = allChats();
        node_fs_1.default.writeFileSync(r.filePath, JSON.stringify({ pelica: "backup", version: 1, created: Date.now(), chats }));
        return { path: r.filePath, count: chats.length };
    });
    // chats from a backup (or a folder's proj.pel): new ones are added; one
    // that differs from the chat of the same id here comes back as a copy, so
    // nothing here is overwritten
    handle("chats:restore", async () => {
        const r = await electron_1.dialog.showOpenDialog(win, { title: "Restore chats from a backup", properties: ["openFile"], filters: [{ name: "Pelica backup or proj.pel", extensions: ["json", "pel"] }] });
        if (r.canceled || !r.filePaths[0])
            return null;
        let data;
        try {
            data = JSON.parse(node_fs_1.default.readFileSync(r.filePaths[0], "utf8"));
        }
        catch {
            throw new Error("That file isn't a Pelica backup.");
        }
        if (!data || (data.pelica !== "backup" && data.pelica !== "project") || !Array.isArray(data.chats))
            throw new Error("That file isn't a Pelica backup.");
        node_fs_1.default.mkdirSync(chatsDir, { recursive: true });
        const out = { added: 0, copies: 0, same: 0 };
        for (const c of data.chats) {
            if (!c || typeof c.id !== "string" || !/^[\w-]+$/.test(c.id) || !Array.isArray(c.messages))
                continue;
            const p = node_path_1.default.join(chatsDir, c.id + ".json");
            if (!node_fs_1.default.existsSync(p)) {
                node_fs_1.default.writeFileSync(p, JSON.stringify(c));
                out.added++;
                continue;
            }
            let here = null;
            try {
                here = JSON.parse(node_fs_1.default.readFileSync(p, "utf8"));
            }
            catch { }
            const same = (x) => x && x.updated === c.updated && (x.messages || []).length === c.messages.length;
            // (or a copy restored before)
            const copied = () => node_fs_1.default
                .readdirSync(chatsDir)
                .filter((n) => n.startsWith(c.id + "-r") && n.endsWith(".json"))
                .some((n) => {
                try {
                    return same(JSON.parse(node_fs_1.default.readFileSync(node_path_1.default.join(chatsDir, n), "utf8")));
                }
                catch {
                    return false;
                }
            });
            if (same(here) || copied()) {
                out.same++;
                continue;
            }
            const id = c.id + "-r" + Date.now().toString(36);
            node_fs_1.default.writeFileSync(node_path_1.default.join(chatsDir, id + ".json"), JSON.stringify({ ...c, id, title: `${c.title || "Chat"} (restored)` }));
            out.copies++;
        }
        return out;
    });
    // chats from ChatGPT's or Claude's data export (the zip, or its conversations.json)
    handle("chats:import", async () => {
        const r = await electron_1.dialog.showOpenDialog(win, { title: "Import chats from ChatGPT or Claude", properties: ["openFile"], filters: [{ name: "ChatGPT or Claude export", extensions: ["zip", "json"] }] });
        if (r.canceled || !r.filePaths[0])
            return null;
        const out = await (0, importer_1.importChats)(r.filePaths[0], chatsDir);
        sync.soon();
        return out;
    });
    const chatPath = (id) => {
        if (!/^[\w-]+$/.test(id))
            throw new Error("bad chat id");
        return node_path_1.default.join(chatsDir, id + ".json");
    };
    handle("chats:load", (id) => JSON.parse(node_fs_1.default.readFileSync(chatPath(id), "utf8")));
    handle("chats:save", (chat) => {
        node_fs_1.default.mkdirSync(chatsDir, { recursive: true });
        // the folder it had (known after the first save; read once otherwise)
        let was = null;
        if (chatFolder.has(chat.id))
            was = chatFolder.get(chat.id);
        else
            try {
                was = JSON.parse(node_fs_1.default.readFileSync(chatPath(chat.id), "utf8")).folder || null;
            }
            catch { }
        chatFolder.set(chat.id, chat.folder || null);
        // (whole to a temporary file, then renamed: a crash mid-save never leaves
        // a broken file, which the list would skip, hiding the chat)
        const file = chatPath(chat.id);
        node_fs_1.default.writeFileSync(file + ".tmp", JSON.stringify(chat));
        node_fs_1.default.renameSync(file + ".tmp", file);
        sync.soon();
        if (was && was !== chat.folder)
            projQueue(was, chat.id, null);
        if (chat.folder)
            projQueue(chat.folder, chat.id, chat);
        return true;
    });
    // (to the Recycle Bin, not gone)
    handle("chats:delete", (id) => {
        chatFolder.delete(id);
        sync.noteDelete(id);
        return electron_1.shell.trashItem(chatPath(id));
    });
    // the Pelica account (Settings → Account)
    // Pelica's own Google client (google-client.json, Google's download, kept out
    // of git and copied in by the installer); one set in Settings wins
    const builtIn = (() => {
        try {
            const j = JSON.parse(node_fs_1.default.readFileSync(node_path_1.default.join(__dirname, "../../google-client.json"), "utf8")).installed;
            return { id: String(j.client_id || ""), secret: String(j.client_secret || "") };
        }
        catch {
            return { id: "", secret: "" };
        }
    })();
    const client = () => (settings.googleClientId ? { id: settings.googleClientId, secret: settings.googleClientSecret || "" } : builtIn);
    const accountStatus = () => ({ profile: google.profile(), sync: sync.status(), clientId: client().id, hasSecret: !!client().secret, builtIn: !!builtIn.id });
    handle("account:status", accountStatus);
    handle("account:signIn", async () => {
        const c = client();
        const p = await google.signIn(c.id, c.secret, (u) => electron_1.shell.openExternal(u));
        // (another Google account than before: its Drive starts a fresh sync)
        if (settings.accountEmail !== p.email) {
            sync.reset();
            settings.accountEmail = p.email;
            saveSettings();
        }
        win?.show();
        win?.focus();
        sync.soon(300);
        return accountStatus();
    });
    handle("account:signOut", async () => {
        await google.signOut();
        return accountStatus();
    });
    handle("sync:now", async () => {
        await sync.syncNow();
        return accountStatus();
    });
    handle("fs:list", (root, rel, depth) => ws.listFiles(checkRoot(root), rel, depth));
    handle("fs:read", (root, rel, offset, limit) => ws.readFile(checkRoot(root), rel, offset, limit));
    handle("fs:write", (root, rel, content) => ws.writeFile(checkRoot(root), rel, content));
    handle("fs:edit", (root, rel, o, n, all) => ws.editFile(checkRoot(root), rel, o, n, all));
    handle("fs:search", (root, pattern, rel, glob) => ws.searchFiles(checkRoot(root), pattern, rel, glob));
    // the project's instructions: PELICA.md (or AGENTS.md / CLAUDE.md) in the
    // folder and in the folders above it up to the repository's root (the one
    // with .git) when it is in one, the outermost first; "" when there are none
    handle("fs:notes", (root) => {
        const found = [];
        let dir = checkRoot(root);
        // (above the folder only inside a repository)
        let levels = 0;
        for (let d = dir, i = 0; i < 8; i++) {
            if (node_fs_1.default.existsSync(node_path_1.default.join(d, ".git"))) {
                levels = i;
                break;
            }
            const parent = node_path_1.default.dirname(d);
            if (parent === d)
                break;
            d = parent;
        }
        for (let up = 0; up <= levels; up++) {
            for (const f of ["PELICA.md", "AGENTS.md", "CLAUDE.md"]) {
                const p = node_path_1.default.join(dir, f);
                try {
                    if (node_fs_1.default.statSync(p).isFile()) {
                        found.unshift({ dir, name: f, text: node_fs_1.default.readFileSync(p, "utf8").trim() });
                        break;
                    }
                }
                catch { }
            }
            dir = node_path_1.default.dirname(dir);
        }
        const max = 20000;
        let out = found.length === 1 ? found[0].text : found.map((f) => `From ${node_path_1.default.join(f.dir, f.name)}:\n${f.text}`).join("\n\n");
        if (out.length > max)
            out = out.slice(0, max) + "\n[… the rest of the instructions file is cut off]";
        return out;
    });
    handle("fs:find", (root, glob) => ws.findFiles(checkRoot(root), glob));
    // a workspace file as it is (for @-mentions in the message box)
    handle("fs:raw", (root, rel) => {
        const p = ws.inside(checkRoot(root), rel);
        if (node_fs_1.default.statSync(p).size > 4 * 1024 * 1024)
            throw new Error(rel + " is too big to attach");
        return node_fs_1.default.readFileSync(p, "utf8");
    });
    handle("fs:trash", (root, rel) => electron_1.shell.trashItem(ws.inside(checkRoot(root), rel)));
    handle("fs:reveal", (root, rel) => electron_1.shell.showItemInFolder(ws.inside(checkRoot(root), rel)));
    handle("cmd:run", (id, root, command, timeoutMs, background) => runCommand(id, root, command, timeoutMs || 120000, !!background, settings.shell || "cmd"));
    handle("cmd:kill", (id) => {
        killTree(procs.get(id)?.pid);
        procs.delete(id);
    });
    handle("web:search", (q) => web.search(q));
    handle("web:open", (url, max) => web.openPage(url, max));
    handle("web:repos", (text) => Promise.all((0, repocheck_1.reposIn)(text).map((r) => (0, repocheck_1.checkRepo)(r.owner, r.repo))));
    handle("sys:external", (url) => /^https?:\/\//i.test(url) && electron_1.shell.openExternal(url));
    handle("mcp:start", async () => {
        await mcp.startAll(settings.mcpServers || []);
        return mcp.list();
    });
    handle("mcp:list", () => mcp.list());
    handle("mcp:restart", async (name) => {
        const c = (settings.mcpServers || []).find((x) => x.name === name);
        if (c)
            await mcp.startOne(c);
        else
            mcp.stopOne(name);
        return mcp.list();
    });
    handle("mcp:launchers", () => mcp.launchers());
    handle("mcp:others", () => mcp.otherApps());
    handle("mcp:parse", (text) => mcp.fromConfig(JSON.parse(text)));
    // a server's tools changed by themselves (or it stopped): tell the window
    mcp.watch(() => win?.webContents.send("mcp:changed", mcp.list()));
    handle("mcp:call", (server, tool, args) => mcp.call(server, tool, args));
    // online connectors' sign-in (the browser opens the server's sign-in page)
    const mcpCfg = (name) => {
        const c = (settings.mcpServers || []).find((x) => x.name === name);
        if (!c)
            throw new Error(`no connector called ${name}`);
        return c;
    };
    handle("mcp:signIn", async (name) => {
        await mcp.signIn(mcpCfg(name), (u) => electron_1.shell.openExternal(u));
        win?.show();
        win?.focus();
        return mcp.list();
    });
    handle("mcp:signOut", async (name) => {
        await mcp.signOut(mcpCfg(name));
        return mcp.list();
    });
    handle("lib:search", (q) => lib.searchRepos(q));
    handle("lib:files", (id) => lib.repoFiles(id));
    handle("lib:download", (repo, file, dir) => lib.download(repo, file, dir, dlProgress));
    handle("lib:downloads", (dir) => lib.downloads(dir));
    handle("lib:pause", (id) => lib.pause(id));
    handle("lib:resume", (id) => lib.resume(id, dlProgress));
    // a cancelled download's part goes to the Recycle Bin
    handle("lib:cancel", async (id) => {
        for (const f of lib.cancel(id))
            await electron_1.shell.trashItem(f).catch(() => { });
        win?.webContents.send("lib:progress", lib.downloads());
    });
    // uninstalling a model: only one the engine lists, after asking; its files go to the Recycle Bin
    handle("lib:uninstall", async (p) => {
        const r = await fetch(`http://127.0.0.1:${port}/api/models`).then((x) => x.json());
        const m = r.models.find((x) => x.path === p);
        if (!m)
            throw new Error("That model isn't in your models folders.");
        const files = lib.modelFiles(p);
        const root = node_path_1.default.parse(p).root;
        const removable = await new Promise((res) => (0, node_child_process_1.execFile)("powershell", ["-NoProfile", "-Command", `[System.IO.DriveInfo]::new('${root.replace(/'/g, "")}').DriveType`], (_e, out) => res(String(out).trim() === "Removable")));
        // a file bigger than the Recycle Bin can hold isn't recycled: Windows deletes it for good
        const cap = removable ? 0 : await lib.recycleCap(root);
        const sizes = files.map((f) => lib.bytesOf([f]));
        const forGood = cap !== null && (cap === 0 || Math.max(...sizes) > cap || sizes.reduce((a, b) => a + b, 0) > cap * 0.9);
        const what = files.length > 1 ? `Its ${files.length} files` : "Its file";
        const gbs = (n) => (n < 1e9 ? `${Math.max(1, Math.round(n / 1e6))} MB` : `${(n / 1e9).toFixed(n < 10e9 ? 1 : 0)} GB`);
        const total = sizes.reduce((a, b) => a + b, 0);
        const { response } = await electron_1.dialog.showMessageBox(win, {
            type: "warning",
            title: "Uninstall model",
            message: forGood ? `Delete ${m.name} for good?` : `Uninstall ${m.name}?`,
            detail: forGood
                ? removable || cap === 0
                    ? `${what} (${gbs(total)}) are on a drive without a Recycle Bin, so they are deleted for good. This can't be undone.`
                    : `${what} (${gbs(total)}) are too big for the Recycle Bin on ${root.slice(0, 2)}, which holds ${gbs(cap)}, so they are deleted for good and the space is free right away. This can't be undone.`
                : `${what} (${gbs(total)}) will go to the Recycle Bin. Empty it to free the space, or restore the model from there.`,
            buttons: [forGood ? "Delete for good" : "Uninstall", "Cancel"],
            defaultId: 1,
            cancelId: 1,
            noLink: true,
        });
        if (response !== 0)
            return false;
        // (Windows can't move a file that's open)
        await unloadModel(p, "uninstall it");
        for (const f of files) {
            if (forGood)
                await node_fs_1.default.promises.rm(f, { recursive: true, force: true });
            else
                await electron_1.shell.trashItem(f);
        }
        // an emptied download folder goes too
        try {
            const dir = node_path_1.default.dirname(files[0]);
            if (!engineDirs().some((d) => sameDir(d, dir)) && !node_fs_1.default.readdirSync(dir).length)
                node_fs_1.default.rmdirSync(dir);
        }
        catch { }
        if (settings.lastModel && sameDir(settings.lastModel, p))
            settings.lastModel = "";
        saveSettings();
        return forGood ? "deleted" : "recycled";
    });
    // speeding up a DeepSeek V4 file (pelica repack): after asking, or on its own
    // after a download when settings.autoRepack is on
    handle("lib:repacks", () => [...repacks.values()].map(({ child, ...r }) => r));
    handle("lib:repack", async (p, undo) => {
        const m = await engineModel(p);
        if (!m || !m.repack)
            throw new Error("Only DeepSeek V4 and Qwen3.8 models can be sped up this way.");
        if (repacks.get(p)?.state === "running")
            return true;
        // (the 164 GB file took about 10 minutes here; a layer, 1/43 of the file,
        // is held three times over in RAM)
        const size = m.gb >= 10 ? `${m.gb.toFixed(0)} GB` : `${m.gb.toFixed(1)} GB`;
        const takes = `about ${Math.max(1, Math.round(m.gb / 16))} minute${Math.round(m.gb / 16) > 1 ? "s" : ""}`;
        const ram = Math.max(1, Math.ceil((m.gb * 3) / 43));
        const { response } = await electron_1.dialog.showMessageBox(win, {
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
        if (response !== 0)
            return false;
        await startRepack(p, undo);
        return true;
    });
    // (safe at any point: the journal puts the layer it was on right the next time it runs)
    handle("lib:repackStop", (p) => {
        const r = repacks.get(p);
        if (r?.state !== "running")
            return;
        r.state = "stopped";
        r.child?.kill();
        sendRepacks();
    });
    handle("lib:hardware", async () => ({ gpus: await lib.graphics(), ram: node_os_1.default.totalmem(), downloadDir: downloadDir(), dirs: engineDirs() }));
    handle("lib:drives", () => lib.drives());
    handle("lib:space", (dir) => lib.freeOn(dir || downloadDir()));
    handle("lib:moves", () => [...moves.values()].map(({ ctl, ...m }) => m));
    handle("lib:moveStop", (p) => moves.get(p)?.ctl.abort());
    // moving a model to another folder or drive: unloaded first; the folder is
    // added to the engine's folders if it isn't in one
    handle("lib:move", async (p, target) => {
        const m = await engineModel(p);
        if (!m)
            throw new Error("That model isn't in your models folders.");
        if (moves.get(p)?.state === "running")
            return null;
        if (!/^[a-z]:[\\/]/i.test(target || ""))
            throw new Error("Choose a folder by its full path (like D:\\models).");
        const files = lib.modelFiles(p);
        const dirs = engineDirs();
        const dest = lib.moveDest(p, files, dirs, target);
        const bytes = lib.bytesOf(files);
        const sameDrive = node_path_1.default.parse(node_path_1.default.resolve(p)).root.toLowerCase() === node_path_1.default.parse(node_path_1.default.resolve(dest)).root.toLowerCase();
        const free = await lib.freeOn(dest);
        if (!free)
            throw new Error(`There's no drive ${node_path_1.default.parse(dest).root}`);
        if (!sameDrive && free.free < bytes + 1e9)
            throw new Error(`Not enough space on ${free.root.slice(0, 2)}: the model needs ${(bytes / 1e9).toFixed(1)} GB and ${(free.free / 1e9).toFixed(1)} GB is free.`);
        if (files.some((f) => sameDir(node_path_1.default.dirname(f), dest)))
            throw new Error("It's already in that folder.");
        await unloadModel(p, "move it");
        const job = { path: p, name: m.name, to: dest, got: 0, total: bytes, state: "running", error: undefined, moved: undefined, ctl: new AbortController() };
        moves.set(p, job);
        let last = 0;
        const send = (force = false) => {
            if (!force && Date.now() - last < 300)
                return;
            last = Date.now();
            win?.webContents.send("lib:move", [...moves.values()].map(({ ctl, ...x }) => x));
        };
        send(true);
        try {
            job.moved = await lib.moveModel(p, files, dest, (got) => ((job.got = got), send()), job.ctl.signal);
            job.state = "done";
            // the engine has to see the new folder
            if (!dirs.some((d) => { const r = node_path_1.default.resolve(d).toLowerCase() + node_path_1.default.sep; return (node_path_1.default.resolve(dest).toLowerCase() + node_path_1.default.sep).startsWith(r); })) {
                const base = settings.modelDirs?.length ? settings.modelDirs : DEFAULT_DIRS;
                settings.modelDirs = [...base, target];
            }
            if (settings.lastModel && sameDir(settings.lastModel, p))
                settings.lastModel = job.moved;
            saveSettings();
            syncDirs();
        }
        catch (e) {
            job.state = job.ctl.signal.aborted ? "cancelled" : "error";
            job.error = job.ctl.signal.aborted ? undefined : e.message;
        }
        send(true);
        return job.moved || null;
    });
    handle("lib:pickDir", async () => {
        const r = await electron_1.dialog.showOpenDialog(win, { properties: ["openDirectory", "createDirectory"], title: "Where models are kept" });
        return r.canceled ? null : r.filePaths[0];
    });
    handle("engine:log", () => engineLog);
    handle("engine:restart", async () => {
        // (the quick bar points at the old port)
        quick?.destroy();
        await startEngine();
        await win?.loadURL(`http://127.0.0.1:${port}/`);
    });
    // updates (electron/updates.ts): only the installed app replaces itself
    let updating = null;
    handle("update:check", () => updates.check(settings.updateFeed || updates.DEFAULT_FEED, electron_1.app.getVersion(), { preview: !!settings.updatePreview }));
    handle("update:install", async (f) => {
        if (!electron_1.app.isPackaged)
            throw new Error("Only the installed Pelica can update itself (this one runs from the source folder).");
        if (updating)
            throw new Error("An update is already downloading.");
        updating = new AbortController();
        const dir = node_path_1.default.join(userDir, "updates");
        try {
            const exe = await updates.download(f, dir, process.execPath, (pr) => win?.webContents.send("update:progress", pr), updating.signal);
            updates.tidy(dir, exe);
            // the installer closes what's left of this app, installs, and starts Pelica again
            (0, node_child_process_1.spawn)(exe, ["/S", "/RELAUNCH", `/D=${node_path_1.default.dirname(process.execPath)}`], { detached: true, stdio: "ignore", windowsHide: true }).unref();
            setTimeout(() => electron_1.app.quit(), 300);
            return true;
        }
        finally {
            updating = null;
        }
    });
    handle("update:cancel", () => updating?.abort());
    handle("app:info", () => ({ version: electron_1.app.getVersion(), userDir, pelica: pelicaExe(), electron: process.versions.electron, hotkey: hotkey(), hotkeyError }));
    // the desktop
    handle("app:hotkey", (k) => {
        settings.hotkey = String(k ?? "");
        saveSettings();
        return registerHotkey();
    });
    // what the window is doing: the taskbar button fills, the tray says it
    handle("app:busy", (b) => {
        if (!win)
            return;
        if (!b) {
            win.setProgressBar(-1);
            tray?.setToolTip("Pelica");
            return;
        }
        win.setProgressBar(b.progress >= 0 ? Math.min(1, b.progress) : 2, { mode: b.progress >= 0 ? "normal" : "indeterminate" });
        tray?.setToolTip(`Pelica · ${b.label}`.slice(0, 120));
    });
    // a desktop notification, only when the user is looking elsewhere
    handle("app:notify", (n) => {
        if (!win || win.isFocused())
            return false;
        win.flashFrame(true);
        if (!electron_1.Notification.isSupported())
            return false;
        const note = new electron_1.Notification({ title: String(n.title).slice(0, 80), body: String(n.body).slice(0, 240), icon: icon ?? undefined });
        note.on("click", () => (n.chatId ? action({ kind: "open-chat", id: n.chatId }) : showMain()));
        note.show();
        return true;
    });
    handle("app:login", (on) => {
        const opts = { path: process.execPath, args: [...(electron_1.app.isPackaged ? [] : [electron_1.app.getAppPath()]), "--hidden"] };
        if (typeof on === "boolean")
            electron_1.app.setLoginItemSettings({ ...opts, openAtLogin: on });
        return electron_1.app.getLoginItemSettings(opts).openAtLogin;
    });
    handle("win:material", (m) => {
        settings.material = m;
        saveSettings();
        const mm = material();
        win?.setBackgroundColor(mm === "none" ? SOLID : "#00000000");
        win?.setBackgroundMaterial(mm);
        return mm;
    });
    handle("quick:submit", (text, reply) => {
        quick?.hide();
        action({ kind: "ask", text: String(text), reply: typeof reply === "string" ? reply : undefined });
    });
    handle("quick:hide", () => quick?.hide());
    handle("quick:resize", (h) => quick && placeQuick(Math.max(80, Math.min(560, h))));
    // the clipboard, for the quick bar's "with clipboard" chip (the user's click)
    handle("quick:clip", () => electron_1.clipboard.readText().slice(0, 20000));
    // the computer, for the tools (reading the clipboard and opening files ask first, in the UI)
    handle("sys:info", () => systemInfo());
    handle("sys:open", async (root, rel) => {
        const err = await electron_1.shell.openPath(ws.inside(checkRoot(root), rel));
        if (err)
            throw new Error(err);
    });
    handle("sys:clipRead", () => electron_1.clipboard.readText().slice(0, 50000));
    handle("sys:clipWrite", (t) => electron_1.clipboard.writeText(String(t ?? "")));
    // a chat exported: the user picks where (nothing is written without the dialog)
    handle("sys:saveAs", async (name, text, dir) => {
        const safe = String(name || "chat").replace(/[<>:"/\\|?*\x00-\x1f]+/g, " ").trim().slice(0, 80) || "chat";
        const base = dir && node_fs_1.default.existsSync(dir) ? dir : electron_1.app.getPath("documents");
        const r = await electron_1.dialog.showSaveDialog(win, { title: "Export the chat", defaultPath: node_path_1.default.join(base, safe + ".md"), filters: [{ name: "Markdown", extensions: ["md"] }, { name: "Text", extensions: ["txt"] }] });
        if (r.canceled || !r.filePath)
            return null;
        node_fs_1.default.writeFileSync(r.filePath, String(text), "utf8");
        return r.filePath;
    });
    handle("sys:reveal", (p) => electron_1.shell.showItemInFolder(String(p)));
    // "Keep computer awake" in the view options: until it's turned off or Pelica quits
    handle("sys:awake", (on) => {
        if (on && awakeId === null)
            awakeId = electron_1.powerSaveBlocker.start("prevent-app-suspension");
        if (!on && awakeId !== null) {
            electron_1.powerSaveBlocker.stop(awakeId);
            awakeId = null;
        }
        return awakeId !== null;
    });
    // photos for models that read text, and voice for the message box
    handle("media:ocr", (dataUrl) => media.ocr(dataUrl));
    handle("media:dictateStart", () => media.dictateStart((e) => win?.webContents.send("media:dictate", e)));
    handle("media:dictateStop", () => media.dictateStop());
}
// one Pelica at a time: a second start (jump list, a shortcut) reaches this one
if (!electron_1.app.requestSingleInstanceLock()) {
    electron_1.app.quit();
}
else {
    electron_1.app.on("second-instance", (_e, argv) => handleArgs(argv));
    electron_1.app.setAppUserModelId("Pelica.Desktop");
    // the browser pane's <webview>: web pages only, never with Pelica's preload;
    // links that open a new window load in the same pane
    electron_1.app.on("web-contents-created", (_e, contents) => {
        contents.on("will-attach-webview", (e, wp, params) => {
            delete wp.preload;
            wp.nodeIntegration = false;
            wp.contextIsolation = true;
            wp.sandbox = true;
            if (!/^(https?:|about:blank)/i.test(params.src || "about:blank"))
                e.preventDefault();
        });
        if (contents.getType() === "webview") {
            contents.setWindowOpenHandler(({ url }) => {
                if (/^https?:\/\//i.test(url))
                    contents.loadURL(url);
                return { action: "deny" };
            });
        }
    });
    electron_1.app.whenReady().then(async () => {
        // (connector sign-ins first: connectors may start as soon as the window asks)
        oauth.init(userDir);
        // the Pelica account (Google) and the sync through its Drive app folder
        google.init(userDir, () => win?.webContents.send("account:changed"));
        sync.init(userDir, chatsDir, {
            settings: () => settings,
            apply: (patch) => {
                settings = { ...settings, ...patch };
                saveSettings();
                win?.webContents.send("settings:pushed", patch);
            },
            changed: (c) => win?.webContents.send("sync:changed", c),
            on: () => settings.syncOn !== false,
        });
        registerIpc();
        makeTray();
        registerHotkey();
        jumpList();
        await createWindow();
        syncApi();
        syncPhone();
        sync.soon(3000);
        setInterval(() => sync.soon(0), 3 * 60_000);
        mcp.startAll(settings.mcpServers || []).catch(() => { });
        if (process.argv.includes("--quick"))
            openQuick();
    });
    electron_1.app.on("window-all-closed", () => {
        stopEngine();
        mcp.stopAll();
        for (const p of procs.values())
            killTree(p.pid);
        electron_1.app.quit();
    });
    electron_1.app.on("before-quit", () => {
        projFlush();
        media.dictateStop();
        quitting = true;
        stopEngine();
        // (a stopped repack finishes the next time it runs)
        for (const r of repacks.values())
            r.child?.kill();
    });
    electron_1.app.on("will-quit", () => electron_1.globalShortcut.unregisterAll());
}
