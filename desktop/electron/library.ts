// The model library: GGUF models on Hugging Face (search, files, downloads
// with progress and resume) and the local models folders.
import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

export interface RepoHit {
  id: string;
  downloads: number;
  likes: number;
  updated: string;
  tags: string[];
}

export async function searchRepos(q: string): Promise<RepoHit[]> {
  const u = new URL("https://huggingface.co/api/models");
  if (q.trim()) u.searchParams.set("search", q.trim());
  u.searchParams.set("filter", "gguf");
  u.searchParams.set("sort", "downloads");
  u.searchParams.set("direction", "-1");
  u.searchParams.set("limit", "40");
  const r = await fetch(u, { headers: { Accept: "application/json" } });
  if (!r.ok) throw new Error(`Hugging Face: HTTP ${r.status}`);
  const v = (await r.json()) as any[];
  return v.map((m) => ({ id: m.id || m.modelId, downloads: m.downloads || 0, likes: m.likes || 0, updated: m.lastModified || "", tags: m.tags || [] }));
}

export async function repoFiles(id: string): Promise<{ path: string; size: number }[]> {
  if (!/^[\w.-]+\/[\w.-]+$/.test(id)) throw new Error("not a repository name");
  const r = await fetch(`https://huggingface.co/api/models/${id}/tree/main?recursive=1`);
  if (!r.ok) throw new Error(`Hugging Face: HTTP ${r.status}`);
  const v = (await r.json()) as any[];
  return v.filter((f) => f.type === "file" && /\.gguf$/i.test(f.path)).map((f) => ({ path: f.path, size: f.lfs?.size ?? f.size ?? 0 }));
}

interface Job {
  id: string;
  repo: string;
  file: string;
  /** the models folder it goes into */
  dir: string;
  dest: string;
  got: number;
  total: number;
  state: "running" | "paused" | "done" | "error" | "cancelled";
  error?: string;
  ctl: AbortController;
  /** settles once this run has stopped writing */
  idle?: Promise<void>;
}

const jobs = new Map<string, Job>();
type Progress = (j: ReturnType<typeof downloads>) => void;

/** A download's bytes so far, and beside them what they are (so a paused
 *  download is still there after Pelica restarts). */
const partOf = (dest: string) => dest + ".part";
const noteOf = (dest: string) => dest + ".part.json";

/** Every download; with `dir`, first the unfinished ones found there from
 *  earlier runs (as paused). */
export function downloads(dir?: string) {
  if (dir) restore(dir);
  return [...jobs.values()].map(({ ctl: _c, idle: _i, ...j }) => j);
}

function restore(dir: string) {
  let subs: string[] = [];
  try {
    subs = fs.readdirSync(dir, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name);
  } catch {
    return;
  }
  for (const sub of subs) {
    let names: string[] = [];
    try {
      names = fs.readdirSync(path.join(dir, sub)).filter((n) => n.endsWith(".gguf.part.json"));
    } catch {
      continue;
    }
    for (const n of names) {
      try {
        const note = JSON.parse(fs.readFileSync(path.join(dir, sub, n), "utf8"));
        const dest = path.join(dir, sub, n.slice(0, -".part.json".length));
        const id = `${note.repo}/${note.file}`;
        if (jobs.has(id) || typeof note.repo !== "string" || typeof note.file !== "string") continue;
        let got = 0;
        try {
          got = fs.statSync(partOf(dest)).size;
        } catch {
          continue;
        }
        jobs.set(id, { id, repo: note.repo, file: note.file, dir, dest, got, total: Number(note.total) || 0, state: "paused", ctl: new AbortController() });
      } catch {}
    }
  }
}

/** Stops a download, keeping what it has (resume picks up from there). */
export function pause(id: string) {
  const j = jobs.get(id);
  if (j && j.state === "running") {
    j.state = "paused";
    j.ctl.abort();
  }
}

/** Goes on with a paused (or failed) download. */
export function resume(id: string, onProgress: Progress) {
  const j = jobs.get(id);
  if (j && (j.state === "paused" || j.state === "error")) download(j.repo, j.file, j.dir, onProgress);
}

/** Stops a download for good: the files it leaves (its part, its note), for
 *  the caller to put in the Recycle Bin. */
export function cancel(id: string): string[] {
  const j = jobs.get(id);
  if (!j || j.state === "done") return [];
  j.ctl.abort();
  j.state = "cancelled";
  jobs.delete(id);
  return [partOf(j.dest), noteOf(j.dest)].filter((f) => fs.existsSync(f));
}

/** What uninstalling a model removes: a checkpoint folder, or a .gguf with
 *  its other parts (name-00001-of-00003.gguf) and the notes Pelica keeps
 *  beside it (.pelica-usage and the like). */
export function modelFiles(p: string): string[] {
  if (fs.statSync(p).isDirectory()) return [p];
  const dir = path.dirname(p);
  const base = path.basename(p);
  const split = base.match(/^(.*)-\d{5}-of-(\d{5})\.gguf$/i);
  const parts = split ? fs.readdirSync(dir).filter((f) => f.toLowerCase().startsWith(split[1].toLowerCase() + "-") && new RegExp(`-\\d{5}-of-${split[2]}\\.gguf$`, "i").test(f)) : [base];
  const all = fs.readdirSync(dir).filter((f) => parts.some((g) => f === g || f.startsWith(g + ".")));
  return all.map((f) => path.join(dir, f));
}

/** Starts a download into `dir`/<repo name>/<file>, or goes on with it from
 *  its part; `onProgress` gets every job's state. */
export function download(repo: string, file: string, dir: string, onProgress: Progress): string {
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo) || file.includes("..") || !/\.gguf$/i.test(file)) throw new Error("not a GGUF file of a repository");
  const id = `${repo}/${file}`;
  if (jobs.get(id)?.state === "running") return id;
  const dest = path.join(dir, repo.split("/")[1], path.basename(file));
  // (an earlier run of it may still be closing its file)
  const prev = jobs.get(id)?.idle;
  const j: Job = { id, repo, file, dir, dest, got: jobs.get(id)?.got || 0, total: jobs.get(id)?.total || 0, state: "running", ctl: new AbortController() };
  jobs.set(id, j);
  let last = 0;
  const tick = (force = false) => {
    const now = Date.now();
    if (force || now - last > 400) {
      last = now;
      onProgress(downloads());
    }
  };
  j.idle = (async () => {
    await prev;
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    const part = partOf(dest);
    let have = 0;
    try {
      have = fs.statSync(part).size;
    } catch {}
    const url = `https://huggingface.co/${repo}/resolve/main/${file.split("/").map(encodeURIComponent).join("/")}`;
    const r = await fetch(url, { signal: j.ctl.signal, headers: have ? { Range: `bytes=${have}-` } : {} });
    if (!r.ok && r.status !== 206) throw new Error(`HTTP ${r.status}`);
    if (r.status !== 206) have = 0;
    j.total = have + Number(r.headers.get("content-length") || 0);
    j.got = have;
    fs.writeFileSync(noteOf(dest), JSON.stringify({ repo, file, total: j.total }));
    const out = fs.createWriteStream(part, { flags: have ? "a" : "w" });
    const reader = r.body!.getReader();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!out.write(value)) await new Promise<void>((ok) => out.once("drain", () => ok()));
        j.got += value.length;
        tick();
      }
    } finally {
      await new Promise<void>((ok) => out.end(() => ok()));
    }
    if (j.total && j.got < j.total) throw new Error("the connection ended early: resume to go on");
    fs.renameSync(part, dest);
    try {
      fs.unlinkSync(noteOf(dest));
    } catch {}
    j.state = "done";
    tick(true);
  })().catch((e) => {
    if (j.state === "running") {
      j.state = "error";
      j.error = e.message;
    }
    tick(true);
  });
  tick(true);
  return id;
}

// ---------- this PC: hardware, drives, moving models ----------

export interface Gpu {
  name: string;
  /** dedicated memory, bytes */
  vram: number;
}

let gpus: Promise<Gpu[]> | null = null;

/** Each display adapter and its dedicated memory: Windows keeps the true size
 *  in the driver's registry key (WMI's AdapterRAM stops at 4 GB). Read once. */
export function graphics(): Promise<Gpu[]> {
  gpus ??= new Promise((ok) => {
    const ps =
      String.raw`Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}' -ErrorAction SilentlyContinue | ForEach-Object { $p = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue; if ($p.DriverDesc) { '{0}|{1}' -f $p.DriverDesc, [uint64]$p.'HardwareInformation.qwMemorySize' } }`;
    execFile("powershell", ["-NoProfile", "-Command", ps], { windowsHide: true }, (_e, out) => {
      const list = String(out || "")
        .split(/\r?\n/)
        .map((l) => l.trim().split("|"))
        .filter((x) => x.length === 2 && x[0])
        .map(([name, b]) => ({ name, vram: Number(b) || 0 }));
      ok(list);
    });
  });
  return gpus;
}

export interface Drive {
  /** "C:\" */
  root: string;
  free: number;
  total: number;
}

/** Every drive with a file system (C: to Z:). */
export async function drives(): Promise<Drive[]> {
  const out: Drive[] = [];
  for (let c = 67; c <= 90; c++) {
    const root = String.fromCharCode(c) + ":" + path.sep;
    try {
      const s = await fs.promises.statfs(root);
      if (s.blocks) out.push({ root, free: s.bavail * s.bsize, total: s.blocks * s.bsize });
    } catch {}
  }
  return out;
}

/** Free bytes on the drive that holds `dir` (null: no such drive). */
export async function freeOn(dir: string): Promise<Drive | null> {
  const root = path.parse(path.resolve(dir)).root;
  try {
    const s = await fs.promises.statfs(root);
    return { root, free: s.bavail * s.bsize, total: s.blocks * s.bsize };
  } catch {
    return null;
  }
}

/** How much the Recycle Bin on `root`'s drive holds, in bytes (null: unknown).
 *  A file bigger than that can't be recycled: Windows deletes it for good. */
export function recycleCap(root: string): Promise<number | null> {
  const letter = root[0];
  if (!/^[A-Z]$/i.test(letter)) return Promise.resolve(null);
  // the volume's GUID names its Recycle Bin settings
  const ps =
    `$v = (Get-Volume -DriveLetter ${letter} -ErrorAction Stop).Path; ` +
    String.raw`$g = [regex]::Match($v, '\{[0-9a-fA-F-]+\}').Value; $p = Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\BitBucket\Volume\$g" -ErrorAction Stop; if ($p.NukeOnDelete) { 0 } else { $p.MaxCapacity }`;
  return new Promise((ok) =>
    execFile("powershell", ["-NoProfile", "-Command", ps], { windowsHide: true }, (e, out) => {
      const mb = Number(String(out).trim());
      ok(e || !String(out).trim() || !isFinite(mb) ? null : mb * 1024 * 1024);
    }),
  );
}

/** Bytes of a model's files (`modelFiles`). */
export function bytesOf(files: string[]): number {
  let n = 0;
  const walk = (f: string) => {
    const s = fs.statSync(f);
    if (s.isDirectory()) for (const c of fs.readdirSync(f)) walk(path.join(f, c));
    else n += s.size;
  };
  for (const f of files) walk(f);
  return n;
}

/** The folder a model's files go into in `target`: a model in its own folder
 *  (most downloads: <models>/<repo name>/<file>) keeps that folder's name; a
 *  checkpoint folder goes in as it is. */
export function moveDest(p: string, files: string[], roots: string[], target: string): string {
  const key = (x: string) => path.resolve(x).toLowerCase();
  if (fs.statSync(p).isDirectory()) return target;
  const parent = path.dirname(files[0]);
  return roots.some((r) => key(r) === key(parent)) ? target : path.join(target, path.basename(parent));
}

export interface Move {
  path: string;
  name: string;
  to: string;
  got: number;
  total: number;
  state: "running" | "done" | "error" | "cancelled";
  error?: string;
  /** the model's new path, once done */
  moved?: string;
}

/** Moves a model's files into `destDir` (made if needed): a rename on the
 *  same drive, else a copy that is checked and only then removed from the old
 *  drive. A cancelled or failed copy takes away only its own unfinished copies. */
export async function moveModel(p: string, files: string[], destDir: string, onProgress: (got: number, total: number) => void, signal: AbortSignal): Promise<string> {
  const total = bytesOf(files);
  const same = path.parse(path.resolve(files[0])).root.toLowerCase() === path.parse(path.resolve(destDir)).root.toLowerCase();
  fs.mkdirSync(destDir, { recursive: true });
  const dests = files.map((f) => path.join(destDir, path.basename(f)));
  for (const d of dests) if (fs.existsSync(d)) throw new Error(`${d} is already there`);
  if (same) {
    files.forEach((f, i) => fs.renameSync(f, dests[i]));
  } else {
    // a folder model: its files one by one
    const pairs: [string, string][] = [];
    const add = (src: string, dst: string) => {
      if (fs.statSync(src).isDirectory()) {
        fs.mkdirSync(dst, { recursive: true });
        for (const c of fs.readdirSync(src)) add(path.join(src, c), path.join(dst, c));
      } else pairs.push([src, dst]);
    };
    files.forEach((f, i) => add(f, dests[i]));
    let got = 0;
    const made: string[] = [];
    try {
      for (const [src, dst] of pairs) {
        made.push(dst);
        await new Promise<void>((ok, bad) => {
          const r = fs.createReadStream(src, { highWaterMark: 4 << 20 });
          const w = fs.createWriteStream(dst);
          const stop = () => (r.destroy(new Error("cancelled")), w.destroy());
          signal.addEventListener("abort", stop, { once: true });
          r.on("data", (c) => {
            got += c.length;
            onProgress(got, total);
          });
          r.on("error", bad);
          w.on("error", bad);
          w.on("finish", () => (signal.removeEventListener("abort", stop), ok()));
          r.pipe(w);
        });
        if (fs.statSync(dst).size !== fs.statSync(src).size) throw new Error(`the copy of ${path.basename(src)} came out the wrong size`);
      }
    } catch (e) {
      for (const d of made) fs.rmSync(d, { force: true });
      for (const d of dests) if (fs.existsSync(d) && fs.statSync(d).isDirectory()) fs.rmSync(d, { recursive: true, force: true });
      // and the folder it made for them, if nothing else is in it
      try {
        if (!fs.readdirSync(destDir).length) fs.rmdirSync(destDir);
      } catch {}
      throw e;
    }
    // every copy checked: now the originals go
    for (const f of files) fs.rmSync(f, { recursive: true, force: true });
  }
  // an emptied download folder goes too
  const old = path.dirname(files[0]);
  try {
    if (!fs.readdirSync(old).length) fs.rmdirSync(old);
  } catch {}
  onProgress(total, total);
  return path.join(destDir, path.basename(p));
}
