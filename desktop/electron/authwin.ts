// Sign-in pages (Google, online connectors) in a maximized window of their own
// instead of a tab in your browser. Google only allows signing in in a real
// browser, never in a window inside an app, so this is Microsoft Edge (on
// every Windows 10 and 11), or Chrome, opened as an app window: no tabs or
// address bar. It has a profile of its own in Pelica's folder, so it opens the
// same way even when the browser is already open, remembers the account for
// next time, and Pelica can close it once the sign-in is done. Without
// either browser, the page opens in the default browser as before.
import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { screen, shell } from "electron";

function browser(): string | null {
  const roots = [process.env["ProgramFiles(x86)"], process.env.ProgramFiles, process.env.LOCALAPPDATA].filter((x): x is string => !!x);
  const rel = [path.join("Microsoft", "Edge", "Application", "msedge.exe"), path.join("Google", "Chrome", "Application", "chrome.exe")];
  for (const r of rel) for (const root of roots) if (fs.existsSync(path.join(root, r))) return path.join(root, r);
  return null;
}

type Box = { x: number; y: number; width: number; height: number };

/** Fills the screen Pelica is on. The browser only maximizes on the main
 *  screen from its command line, so on another one it's sized to fit. */
function fill(over?: Box): string[] {
  const d = over ? screen.getDisplayMatching(over) : null;
  if (!d || d.id === screen.getPrimaryDisplay().id) return ["--start-maximized"];
  const a = d.workArea;
  return [`--window-position=${a.x},${a.y}`, `--window-size=${a.width},${a.height}`];
}

export interface AuthWindow {
  /** closes it (after `ms`, so the "Signed in" page shows for a moment) */
  close(ms?: number): void;
}

/** Opens `url` in a browser window that fills the screen `over` (Pelica's window) is on. */
export function openSignIn(url: string, profileDir: string, over?: Box, onClosed?: () => void): AuthWindow {
  const exe = browser();
  if (!exe) {
    shell.openExternal(url).catch(() => {});
    return { close() {} };
  }
  let p: ChildProcess | null = null;
  try {
    fs.mkdirSync(profileDir, { recursive: true });
    p = spawn(exe, [`--app=${url}`, `--user-data-dir=${profileDir}`, ...fill(over), "--no-first-run", "--no-default-browser-check", "--disable-sync"], { detached: false, stdio: "ignore", windowsHide: false });
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
