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
const node_net_1 = __importDefault(require("node:net"));
const node_os_1 = __importDefault(require("node:os"));
const node_path_1 = __importDefault(require("node:path"));
const web = __importStar(require("./web"));
const ws = __importStar(require("./workspace"));
const mcp = __importStar(require("./mcp"));
const lib = __importStar(require("./library"));
const media = __importStar(require("./media"));
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
async function startEngine() {
    stopEngine();
    port = await freePort();
    const ui = node_path_1.default.resolve(__dirname, "../../dist");
    // no model until one is picked, unless "Load the last model at start" is on
    const args = ["app", "--no-open", "--no-autoload", "--port", String(port), "--ui", ui];
    if (settings.loadOnStart && settings.lastModel && node_fs_1.default.existsSync(settings.lastModel))
        args.push("--model", settings.lastModel, "--ctx", String(settings.lastCtx || 32768));
    for (const d of settings.modelDirs || [])
        args.push("--models-dir", d);
    engineLog = `$ ${pelicaExe()} ${args.join(" ")}\n`;
    engine = (0, node_child_process_1.spawn)(pelicaExe(), args, { windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
    const add = (d) => (engineLog = (engineLog + d.toString()).slice(-200000));
    engine.stdout.on("data", add);
    engine.stderr.on("data", add);
    engine.on("exit", (code) => {
        engineLog += `\n[pelica exited: ${code}]\n`;
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
                return;
        }
        catch { }
        await new Promise((ok) => setTimeout(ok, 100));
    }
    throw new Error("pelica didn't answer:\n" + engineLog.slice(-2000));
}
function stopEngine() {
    if (engine?.pid) {
        (0, node_child_process_1.execFile)("taskkill", ["/pid", String(engine.pid), "/T", "/F"], () => { });
    }
    engine = null;
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
        minWidth: 760,
        minHeight: 520,
        frame: false,
        backgroundColor: m === "none" ? SOLID : "#00000000",
        backgroundMaterial: m,
        show: false,
        title: "Pelica",
        icon: icon ?? undefined,
        webPreferences: { preload: node_path_1.default.join(__dirname, "preload.js"), contextIsolation: true, nodeIntegration: false, sandbox: true, spellcheck: true },
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
        { label: "Brain", click: () => action({ kind: "view", view: "brain" }) },
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
        return settings;
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
    handle("folder:open", (root) => electron_1.shell.openPath(checkRoot(root)));
    handle("chats:list", () => {
        node_fs_1.default.mkdirSync(chatsDir, { recursive: true });
        return node_fs_1.default
            .readdirSync(chatsDir)
            .filter((f) => f.endsWith(".json"))
            .map((f) => {
            try {
                const c = JSON.parse(node_fs_1.default.readFileSync(node_path_1.default.join(chatsDir, f), "utf8"));
                return { id: c.id, title: c.title, updated: c.updated, pinned: !!c.pinned, folder: c.folder || null };
            }
            catch {
                return null;
            }
        })
            .filter(Boolean)
            .sort((a, b) => b.updated - a.updated);
    });
    const chatPath = (id) => {
        if (!/^[\w-]+$/.test(id))
            throw new Error("bad chat id");
        return node_path_1.default.join(chatsDir, id + ".json");
    };
    handle("chats:load", (id) => JSON.parse(node_fs_1.default.readFileSync(chatPath(id), "utf8")));
    handle("chats:save", (chat) => {
        node_fs_1.default.mkdirSync(chatsDir, { recursive: true });
        node_fs_1.default.writeFileSync(chatPath(chat.id), JSON.stringify(chat));
        return true;
    });
    // (to the Recycle Bin, not gone)
    handle("chats:delete", (id) => electron_1.shell.trashItem(chatPath(id)));
    handle("fs:list", (root, rel, depth) => ws.listFiles(checkRoot(root), rel, depth));
    handle("fs:read", (root, rel, offset, limit) => ws.readFile(checkRoot(root), rel, offset, limit));
    handle("fs:write", (root, rel, content) => ws.writeFile(checkRoot(root), rel, content));
    handle("fs:edit", (root, rel, o, n, all) => ws.editFile(checkRoot(root), rel, o, n, all));
    handle("fs:search", (root, pattern, rel, glob) => ws.searchFiles(checkRoot(root), pattern, rel, glob));
    // the project notes file, if there is one (no error when there is none)
    handle("fs:notes", (root) => {
        const r = checkRoot(root);
        for (const f of ["PELICA.md", "AGENTS.md", "CLAUDE.md"]) {
            const p = node_path_1.default.join(r, f);
            if (node_fs_1.default.existsSync(p))
                return node_fs_1.default.readFileSync(p, "utf8").slice(0, 6000);
        }
        return "";
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
    handle("mcp:call", (server, tool, args) => mcp.call(server, tool, args));
    handle("lib:search", (q) => lib.searchRepos(q));
    handle("lib:files", (id) => lib.repoFiles(id));
    handle("lib:download", (repo, file, dir) => lib.download(repo, file, dir, (j) => win?.webContents.send("lib:progress", j)));
    handle("lib:downloads", (dir) => lib.downloads(dir));
    handle("lib:pause", (id) => lib.pause(id));
    handle("lib:resume", (id) => lib.resume(id, (j) => win?.webContents.send("lib:progress", j)));
    // a cancelled download's part goes to the Recycle Bin
    handle("lib:cancel", async (id) => {
        for (const f of lib.cancel(id))
            await electron_1.shell.trashItem(f).catch(() => { });
        win?.webContents.send("lib:progress", lib.downloads());
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
    electron_1.app.whenReady().then(async () => {
        registerIpc();
        makeTray();
        registerHotkey();
        jumpList();
        await createWindow();
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
        media.dictateStop();
        quitting = true;
        stopEngine();
    });
    electron_1.app.on("will-quit", () => electron_1.globalShortcut.unregisterAll());
}
