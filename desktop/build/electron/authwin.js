"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.openSmall = openSmall;
// Sign-in pages (Google, online connectors) in a small window over Pelica
// instead of a full browser tab. Google only allows signing in in a real
// browser, never in a window inside an app, so this is Microsoft Edge (on
// every Windows 10 and 11), or Chrome, opened as an app window: no tabs or
// address bar. It has a profile of its own in Pelica's folder, so it keeps
// its own size even when the browser is already open, remembers the account
// for next time, and Pelica can close it once the sign-in is done. Without
// either browser, the page opens in the default browser as before.
const node_child_process_1 = require("node:child_process");
const node_fs_1 = __importDefault(require("node:fs"));
const node_path_1 = __importDefault(require("node:path"));
const electron_1 = require("electron");
const W = 480;
const H = 680;
function browser() {
    const roots = [process.env["ProgramFiles(x86)"], process.env.ProgramFiles, process.env.LOCALAPPDATA].filter((x) => !!x);
    const rel = [node_path_1.default.join("Microsoft", "Edge", "Application", "msedge.exe"), node_path_1.default.join("Google", "Chrome", "Application", "chrome.exe")];
    for (const r of rel)
        for (const root of roots)
            if (node_fs_1.default.existsSync(node_path_1.default.join(root, r)))
                return node_path_1.default.join(root, r);
    return null;
}
/** Opens `url` in a small browser window centred over `over` (Pelica's window). */
function openSmall(url, profileDir, over, onClosed) {
    const exe = browser();
    if (!exe) {
        electron_1.shell.openExternal(url).catch(() => { });
        return { close() { } };
    }
    const x = over ? Math.round(over.x + (over.width - W) / 2) : 200;
    const y = over ? Math.round(over.y + Math.max(0, (over.height - H) / 2)) : 120;
    let p = null;
    try {
        node_fs_1.default.mkdirSync(profileDir, { recursive: true });
        p = (0, node_child_process_1.spawn)(exe, [`--app=${url}`, `--user-data-dir=${profileDir}`, `--window-size=${W},${H}`, `--window-position=${x},${y}`, "--no-first-run", "--no-default-browser-check", "--disable-sync"], { detached: false, stdio: "ignore", windowsHide: false });
        p.on("error", () => electron_1.shell.openExternal(url).catch(() => { }));
        // (closed by the user before Pelica closed it: the sign-in is off)
        const mine = p;
        p.on("exit", () => p === mine && (p = null, onClosed?.()));
    }
    catch {
        electron_1.shell.openExternal(url).catch(() => { });
    }
    return {
        close(ms = 0) {
            const q = p;
            p = null;
            if (!q || q.exitCode !== null || !q.pid)
                return;
            // (the whole browser of that profile: its window and helpers)
            setTimeout(() => (0, node_child_process_1.spawn)("taskkill", ["/PID", String(q.pid), "/T", "/F"], { windowsHide: true, stdio: "ignore" }), ms);
        },
    };
}
