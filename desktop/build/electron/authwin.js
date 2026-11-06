"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.openSignIn = openSignIn;
// Sign-in pages (Google, online connectors) in a maximized window of their own
// instead of a tab in your browser. Google only allows signing in in a real
// browser, never in a window inside an app, so this is Microsoft Edge (on
// every Windows 10 and 11), or Chrome, opened as an app window: no tabs or
// address bar. It has a profile of its own in Pelica's folder, so it opens the
// same way even when the browser is already open, remembers the account for
// next time, and Pelica can close it once the sign-in is done. Without
// either browser, the page opens in the default browser as before.
const node_child_process_1 = require("node:child_process");
const node_fs_1 = __importDefault(require("node:fs"));
const node_path_1 = __importDefault(require("node:path"));
const electron_1 = require("electron");
function browser() {
    const roots = [process.env["ProgramFiles(x86)"], process.env.ProgramFiles, process.env.LOCALAPPDATA].filter((x) => !!x);
    const rel = [node_path_1.default.join("Microsoft", "Edge", "Application", "msedge.exe"), node_path_1.default.join("Google", "Chrome", "Application", "chrome.exe")];
    for (const r of rel)
        for (const root of roots)
            if (node_fs_1.default.existsSync(node_path_1.default.join(root, r)))
                return node_path_1.default.join(root, r);
    return null;
}
/** Fills the screen Pelica is on. The browser only maximizes on the main
 *  screen from its command line, so on another one it's sized to fit. */
function fill(over) {
    const d = over ? electron_1.screen.getDisplayMatching(over) : null;
    if (!d || d.id === electron_1.screen.getPrimaryDisplay().id)
        return ["--start-maximized"];
    const a = d.workArea;
    return [`--window-position=${a.x},${a.y}`, `--window-size=${a.width},${a.height}`];
}
/** Opens `url` in a browser window that fills the screen `over` (Pelica's window) is on. */
function openSignIn(url, profileDir, over, onClosed) {
    const exe = browser();
    if (!exe) {
        electron_1.shell.openExternal(url).catch(() => { });
        return { close() { } };
    }
    let p = null;
    try {
        node_fs_1.default.mkdirSync(profileDir, { recursive: true });
        p = (0, node_child_process_1.spawn)(exe, [`--app=${url}`, `--user-data-dir=${profileDir}`, ...fill(over), "--no-first-run", "--no-default-browser-check", "--disable-sync"], { detached: false, stdio: "ignore", windowsHide: false });
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
