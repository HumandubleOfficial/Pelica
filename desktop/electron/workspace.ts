// The models' file tools, confined to a workspace folder the user picked:
// every path is resolved (symbolic links included) and refused if it lands
// outside the folder.
import fs from "node:fs";
import path from "node:path";

const SKIP = new Set(["node_modules", ".git", "target", "dist", "build", "__pycache__", ".venv", "venv", ".next", ".cache", ".idea", ".vs"]);

function realish(p: string): string {
  // the real path of the deepest part that exists, plus the rest
  let cur = p;
  const rest: string[] = [];
  for (;;) {
    try {
      return path.join(fs.realpathSync.native(cur), ...rest.reverse());
    } catch {
      const up = path.dirname(cur);
      if (up === cur) return p;
      rest.push(path.basename(cur));
      cur = up;
    }
  }
}

/** The absolute path of `rel` inside `root`; throws if it's outside. */
export function inside(root: string, rel: string): string {
  const r = realish(path.resolve(root));
  const p = realish(path.resolve(r, rel || "."));
  const a = p.toLowerCase();
  const b = r.toLowerCase();
  if (a !== b && !a.startsWith(b.endsWith(path.sep) ? b : b + path.sep)) throw new Error(`${rel} is outside the workspace folder (${root})`);
  return p;
}

const relOf = (root: string, p: string) => path.relative(realish(path.resolve(root)), p).split(path.sep).join("/") || ".";

export function listFiles(root: string, rel = ".", depth = 2): string {
  const base = inside(root, rel);
  const out: string[] = [];
  const walk = (d: string, lvl: number, pre: string) => {
    let ents: fs.Dirent[] = [];
    try {
      ents = fs.readdirSync(d, { withFileTypes: true });
    } catch {
      return;
    }
    ents.sort((x, y) => Number(y.isDirectory()) - Number(x.isDirectory()) || x.name.localeCompare(y.name));
    for (const e of ents) {
      if (out.length >= 500) return;
      if (e.isDirectory()) {
        if (SKIP.has(e.name)) {
          out.push(`${pre}${e.name}/ (skipped)`);
          continue;
        }
        out.push(`${pre}${e.name}/`);
        if (lvl < depth) walk(path.join(d, e.name), lvl + 1, pre + "  ");
      } else {
        let size = 0;
        try {
          size = fs.statSync(path.join(d, e.name)).size;
        } catch {}
        out.push(`${pre}${e.name} (${size < 1024 ? size + " B" : (size / 1024).toFixed(1) + " KB"})`);
      }
    }
  };
  walk(base, 1, "");
  if (out.length >= 500) out.push("... (more entries not shown)");
  return `${relOf(root, base)}/\n` + (out.length ? out.join("\n") : "(empty)");
}

function isBinary(b: Buffer): boolean {
  const n = Math.min(b.length, 8000);
  for (let i = 0; i < n; i++) if (b[i] === 0) return true;
  return false;
}

export function readFile(root: string, rel: string, offset = 1, limit = 500): { text: string; lines: number; path: string } {
  const p = inside(root, rel);
  const st = fs.statSync(p);
  if (st.isDirectory()) throw new Error(`${rel} is a folder; use list_files`);
  if (st.size > 5_000_000) throw new Error(`${rel} is ${(st.size / 1e6).toFixed(1)} MB, too large to read`);
  const b = fs.readFileSync(p);
  if (isBinary(b)) throw new Error(`${rel} is a binary file`);
  const all = b.toString("utf8").split(/\r?\n/);
  const start = Math.max(1, offset);
  const slice = all.slice(start - 1, start - 1 + Math.max(1, limit));
  const text = slice.map((l, i) => `${String(start + i).padStart(5)}  ${l.length > 2000 ? l.slice(0, 2000) + "…" : l}`).join("\n");
  const more = start - 1 + slice.length < all.length ? `\n[lines ${start}-${start - 1 + slice.length} of ${all.length}; read more with offset]` : "";
  return { text: text + more, lines: all.length, path: relOf(root, p) };
}

export function writeFile(root: string, rel: string, content: string): { path: string; before: string | null; after: string } {
  const p = inside(root, rel);
  let before: string | null = null;
  try {
    before = fs.readFileSync(p, "utf8");
  } catch {}
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, content, "utf8");
  return { path: relOf(root, p), before, after: content };
}

export function editFile(root: string, rel: string, oldS: string, newS: string, all = false): { path: string; before: string; after: string; count: number } {
  const p = inside(root, rel);
  const before = fs.readFileSync(p, "utf8");
  // the model may write \n where the file has \r\n
  const crlf = before.includes("\r\n") && !oldS.includes("\r\n");
  const o = crlf ? oldS.replace(/\n/g, "\r\n") : oldS;
  const n = crlf ? newS.replace(/\n/g, "\r\n") : newS;
  if (!o) throw new Error("old_string is empty; use write_file to create a file");
  const count = before.split(o).length - 1;
  if (count === 0) throw new Error(`old_string was not found in ${rel}. Read the file again and copy the text exactly, including indentation.`);
  if (count > 1 && !all) throw new Error(`old_string appears ${count} times in ${rel}; add more surrounding lines to make it unique, or set replace_all`);
  const after = all ? before.split(o).join(n) : before.replace(o, () => n);
  fs.writeFileSync(p, after, "utf8");
  return { path: relOf(root, p), before, after, count: all ? count : 1 };
}

function globRe(g: string): RegExp {
  const s = g
    .replace(/[.+^${}()|[\]\\]/g, "\\$&")
    .replace(/\*\*\//g, "\u0000")
    .replace(/\*/g, "[^/]*")
    .replace(/\?/g, "[^/]")
    .replace(/\u0000/g, "(?:.*/)?");
  return new RegExp(`(^|/)${s}$`, "i");
}

export function searchFiles(root: string, pattern: string, rel = ".", glob?: string): string {
  const base = inside(root, rel);
  let re: RegExp;
  try {
    re = new RegExp(pattern, "i");
  } catch {
    re = new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i");
  }
  const gre = glob ? globRe(glob) : null;
  const out: string[] = [];
  let files = 0;
  const walk = (d: string) => {
    let ents: fs.Dirent[] = [];
    try {
      ents = fs.readdirSync(d, { withFileTypes: true });
    } catch {
      return;
    }
    for (const e of ents) {
      if (out.length >= 200 || files > 5000) return;
      const p = path.join(d, e.name);
      if (e.isDirectory()) {
        if (!SKIP.has(e.name)) walk(p);
        continue;
      }
      const r = relOf(root, p);
      if (gre && !gre.test(r)) continue;
      files++;
      let b: Buffer;
      try {
        if (fs.statSync(p).size > 2_000_000) continue;
        b = fs.readFileSync(p);
      } catch {
        continue;
      }
      if (isBinary(b)) continue;
      const lines = b.toString("utf8").split(/\r?\n/);
      for (let i = 0; i < lines.length && out.length < 200; i++) {
        if (re.test(lines[i])) out.push(`${r}:${i + 1}: ${lines[i].trim().slice(0, 240)}`);
      }
    }
  };
  walk(base);
  return out.length ? out.join("\n") + (out.length >= 200 ? "\n[first 200 matches]" : "") : "No matches.";
}

/** Files under `root` whose path matches the glob (for CQS and the UI). */
export function findFiles(root: string, glob: string, max = 2000): string[] {
  const base = inside(root, ".");
  const gre = globRe(glob);
  const out: string[] = [];
  const walk = (d: string) => {
    let ents: fs.Dirent[] = [];
    try {
      ents = fs.readdirSync(d, { withFileTypes: true });
    } catch {
      return;
    }
    for (const e of ents) {
      if (out.length >= max) return;
      const p = path.join(d, e.name);
      if (e.isDirectory()) {
        if (!SKIP.has(e.name)) walk(p);
      } else if (gre.test(relOf(root, p))) out.push(relOf(root, p));
    }
  };
  walk(base);
  return out;
}
