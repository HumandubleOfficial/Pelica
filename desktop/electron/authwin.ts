// Sign-in pages (Google, online connectors) in a small window over Pelica
// instead of a full browser tab. Google only allows signing in in a real
// browser, never in a window inside an app, so this is Microsoft Edge (on
// every Windows 10 and 11), or Chrome, opened as an app window: no tabs or
// address bar. It has a profile of its own in Pelica's folder, so it keeps
// its own size even when the browser is already open, remembers the account
// for next time, and Pelica can close it once the sign-in is done. Without
// either browser, the page opens in the default browser as before.
import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { shell } from "electron";

const W = 480;
const H = 680;

function browser(): string | null {
  const roots = [process.env["ProgramFiles(x86)"], process.env.ProgramFiles, process.env.LOCALAPPDATA].filter((x): x is string => !!x);
  const rel = [path.join("Microsoft", "Edge", "Application", "msedge.exe"), path.join("Google", "Chrome", "Application", "chrome.exe")];
  for (const r of rel) for (const root of roots) if (fs.existsSync(path.join(root, r))) return path.join(root, r);
  return null;
}

export interface AuthWindow {
  /** closes it (after `ms`, so the "Signed in" page shows for a moment) */
  close(ms?: number): void;
}

/** Opens `url` in a small browser window centred over `over` (Pelica's window). */
export function openSmall(url: string, profileDir: string, over?: { x: number; y: number; width: number; height: number }, onClosed?: () => void): AuthWindow {
  const exe = browser();
  if (!exe) {
    shell.openExternal(url).catch(() => {});
    return { close() {} };
  }
  const x = over ? Math.round(over.x + (over.width - W) / 2) : 200;
  const y = over ? Math.round(over.y + Math.max(0, (over.height - H) / 2)) : 120;
  let p: ChildProcess | null = null;
  try {
    fs.mkdirSync(profileDir, { recursive: true });
    p = spawn(exe, [`--app=${url}`, `--user-data-dir=${profileDir}`, `--window-size=${W},${H}`, `--window-position=${x},${y}`, "--no-first-run", "--no-default-browser-check", "--disable-sync"], { detached: false, stdio: "ignore", windowsHide: false });
    p.on("error", () => shell.openExternal(url).catch(() => {}));
    // (closed by the user before Pelica closed it: the sign-in is off)
    const mine = p;
    p.on("exit", () => p === mine && (p = null, onClosed?.()));
  } catch {
    shell.openExternal(url).catch(() => {});
  }
  return {
    close(ms = 0) {
      const q = p;
      p = null;
      if (!q || q.exitCode !== null || !q.pid) return;
      // (the whole browser of that profile: its window and helpers)
      setTimeout(() => spawn("taskkill", ["/PID", String(q.pid), "/T", "/F"], { windowsHide: true, stdio: "ignore" }), ms);
    },
  };
}
