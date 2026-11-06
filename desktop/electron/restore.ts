// Restore points: before Pelica's file tools change a file, it keeps a copy
// here, so a reply's changes can be undone (and put back). One folder per
// reply, under restore/ in Pelica's data: index.json and the copies. Files a
// command changed aren't covered: nothing here sees them. A file Pelica
// created goes to the Recycle Bin on undo, never straight to deletion.
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { shell } from "electron";
import { inside } from "./workspace";

/** Bigger than this isn't copied (it can't be undone, and the page says so). */
const MAX_FILE = 50e6;
const MAX_FOLDER = 200e6;
/** Restore points older than this go, then the oldest until they fit. */
const KEEP_DAYS = 30;
const MAX_TOTAL = 2e9;

interface Entry {
  root: string;
  rel: string;
  /** it was there before Pelica changed it (its copy is <n>) */
  existed: boolean;
  /** too big to copy */
  cant?: boolean;
  /** what the file should be now: a hash; "" for not there */
  expect?: string;
  /** undo found a file there (its copy is <n>.redo) */
  redo?: boolean;
}

interface Point {
  created: number;
  undone: boolean;
  entries: Entry[];
}

let base = "";
let allowed: (root: string) => string = (r) => r;

export function init(userDir: string, checkRoot: (root: string) => string) {
  base = path.join(userDir, "restore");
  allowed = checkRoot;
  setTimeout(prune, 20000);
}

const okId = (turn?: string): turn is string => !!turn && /^[\w-]{1,64}$/.test(turn);
const dirOf = (turn: string) => path.join(base, turn);
const copyOf = (turn: string, n: number) => path.join(dirOf(turn), String(n));

function load(turn: string): Point | null {
  try {
    return JSON.parse(fs.readFileSync(path.join(dirOf(turn), "index.json"), "utf8"));
  } catch {
    return null;
  }
}

function save(turn: string, p: Point) {
  fs.mkdirSync(dirOf(turn), { recursive: true });
  fs.writeFileSync(path.join(dirOf(turn), "index.json"), JSON.stringify(p));
}

function hash(p: string): string {
  let st: fs.Stats;
  try {
    st = fs.statSync(p);
  } catch {
    return "";
  }
  if (st.isDirectory()) return "folder";
  if (st.size > MAX_FILE) return `big:${st.size}:${st.mtimeMs}`;
  return crypto.createHash("sha1").update(fs.readFileSync(p)).digest("hex");
}

/** The size of a folder, stopping once it's past `limit`. */
function sizeOf(p: string, limit: number): number {
  const st = fs.statSync(p);
  if (!st.isDirectory()) return st.size;
  let n = 0;
  for (const e of fs.readdirSync(p)) {
    n += sizeOf(path.join(p, e), limit - n);
    if (n > limit) break;
  }
  return n;
}

const copy = (from: string, to: string) => {
  fs.mkdirSync(path.dirname(to), { recursive: true });
  fs.cpSync(from, to, { recursive: true, force: true, dereference: true });
};

/** Before a file tool changes `rel`: keeps it, the first time in this reply. */
export function before(turn: string | undefined, root: string, name: string) {
  if (!okId(turn)) return;
  try {
    const rel = path.normalize(name);
    const p = load(turn) || { created: Date.now(), undone: false, entries: [] };
    if (p.entries.some((e) => e.root === root && e.rel === rel)) return;
    const file = inside(root, rel);
    const e: Entry = { root, rel, existed: fs.existsSync(file) };
    if (e.existed) {
      const dir = fs.statSync(file).isDirectory();
      if (sizeOf(file, MAX_FOLDER) > (dir ? MAX_FOLDER : MAX_FILE)) e.cant = true;
      else copy(file, copyOf(turn, p.entries.length));
    }
    p.entries.push(e);
    save(turn, p);
  } catch {
    // (a restore point that can't be made must not stop the tool)
  }
}

/** After it changed: what Pelica left there. */
export function after(turn: string | undefined, root: string, name: string) {
  if (!okId(turn)) return;
  try {
    const rel = path.normalize(name);
    const p = load(turn);
    const e = p?.entries.find((x) => x.root === root && x.rel === rel);
    if (!p || !e) return;
    e.expect = hash(inside(root, rel));
    p.undone = false;
    save(turn, p);
  } catch {}
}

const changedOnes = (p: Point) => p.entries.filter((e) => e.expect !== undefined);

export function info(turns: string[]) {
  const points = turns.filter(okId).map(load).filter((p): p is Point => !!p);
  const all = points.flatMap(changedOnes);
  return {
    files: [...new Set(all.filter((e) => !e.cant).map((e) => e.rel))],
    cant: [...new Set(all.filter((e) => e.cant).map((e) => e.rel))],
    undone: points.length > 0 && points.every((p) => p.undone),
  };
}

export interface Result {
  done: string[];
  /** changed since, so nothing was touched (ask, then `force`) */
  changed: string[];
  failed: string[];
}

function clashes(turns: string[]): string[] {
  const out = new Set<string>();
  // (a file two of these replies changed: only the newest says what it is now)
  const seen = new Set<string>();
  for (const t of [...turns].reverse()) {
    const p = load(t);
    for (const e of p ? changedOnes(p) : []) {
      const key = `${e.root}|${e.rel}`;
      if (e.cant || seen.has(key)) continue;
      seen.add(key);
      try {
        if (hash(inside(allowed(e.root), e.rel)) !== e.expect) out.add(e.rel);
      } catch {}
    }
  }
  return [...out];
}

/** Puts a reply's files back as they were (the newest reply first). */
export async function undo(list: string[], force = false): Promise<Result> {
  const turns = list.filter(okId);
  const r: Result = { done: [], changed: force ? [] : clashes(turns), failed: [] };
  if (r.changed.length) return r;
  for (const t of [...turns].reverse()) {
    const p = load(t);
    if (!p || p.undone) continue;
    for (let n = p.entries.length - 1; n >= 0; n--) {
      const e = p.entries[n];
      if (e.expect === undefined || e.cant) continue;
      try {
        const file = inside(allowed(e.root), e.rel);
        const redo = `${copyOf(t, n)}.redo`;
        fs.rmSync(redo, { recursive: true, force: true });
        e.redo = fs.existsSync(file);
        if (e.redo) copy(file, redo);
        if (e.existed) copy(copyOf(t, n), file);
        else if (e.redo) await shell.trashItem(file);
        e.expect = hash(file);
        r.done.push(e.rel);
      } catch (err: any) {
        r.failed.push(`${e.rel}: ${err?.message || err}`);
      }
    }
    p.undone = true;
    save(t, p);
  }
  return r;
}

/** Puts back what undo took away. */
export async function redo(list: string[], force = false): Promise<Result> {
  const turns = list.filter(okId);
  const r: Result = { done: [], changed: force ? [] : clashes(turns), failed: [] };
  if (r.changed.length) return r;
  for (const t of turns) {
    const p = load(t);
    if (!p || !p.undone) continue;
    for (let n = 0; n < p.entries.length; n++) {
      const e = p.entries[n];
      if (e.expect === undefined || e.cant) continue;
      try {
        const file = inside(allowed(e.root), e.rel);
        if (e.redo) copy(`${copyOf(t, n)}.redo`, file);
        else if (fs.existsSync(file)) await shell.trashItem(file);
        e.expect = hash(file);
        r.done.push(e.rel);
      } catch (err: any) {
        r.failed.push(`${e.rel}: ${err?.message || err}`);
      }
    }
    p.undone = false;
    save(t, p);
  }
  return r;
}

/** Old restore points go: past KEEP_DAYS, then the oldest over MAX_TOTAL. */
function prune() {
  try {
    const all = fs
      .readdirSync(base)
      .map((id) => ({ id, p: load(id) }))
      .map((x) => ({ id: x.id, created: x.p?.created || 0, size: sizeOf(dirOf(x.id), Infinity) }))
      .sort((a, b) => b.created - a.created);
    let total = 0;
    for (const x of all) {
      total += x.size;
      if (Date.now() - x.created > KEEP_DAYS * 864e5 || total > MAX_TOTAL) fs.rmSync(dirOf(x.id), { recursive: true, force: true });
    }
  } catch {}
}
