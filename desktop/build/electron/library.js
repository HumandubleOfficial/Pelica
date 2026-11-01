"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.searchRepos = searchRepos;
exports.repoFiles = repoFiles;
exports.downloads = downloads;
exports.pause = pause;
exports.resume = resume;
exports.cancel = cancel;
exports.modelFiles = modelFiles;
exports.download = download;
exports.graphics = graphics;
exports.drives = drives;
exports.freeOn = freeOn;
exports.recycleCap = recycleCap;
exports.bytesOf = bytesOf;
exports.moveDest = moveDest;
exports.moveModel = moveModel;
// The model library: GGUF models on Hugging Face (search, files, downloads
// with progress and resume) and the local models folders.
const node_child_process_1 = require("node:child_process");
const node_fs_1 = __importDefault(require("node:fs"));
const node_path_1 = __importDefault(require("node:path"));
async function searchRepos(q) {
    const u = new URL("https://huggingface.co/api/models");
    if (q.trim())
        u.searchParams.set("search", q.trim());
    u.searchParams.set("filter", "gguf");
    u.searchParams.set("sort", "downloads");
    u.searchParams.set("direction", "-1");
    u.searchParams.set("limit", "40");
    const r = await fetch(u, { headers: { Accept: "application/json" } });
    if (!r.ok)
        throw new Error(`Hugging Face: HTTP ${r.status}`);
    const v = (await r.json());
    return v.map((m) => ({ id: m.id || m.modelId, downloads: m.downloads || 0, likes: m.likes || 0, updated: m.lastModified || "", tags: m.tags || [] }));
}
async function repoFiles(id) {
    if (!/^[\w.-]+\/[\w.-]+$/.test(id))
        throw new Error("not a repository name");
    const r = await fetch(`https://huggingface.co/api/models/${id}/tree/main?recursive=1`);
    if (!r.ok)
        throw new Error(`Hugging Face: HTTP ${r.status}`);
    const v = (await r.json());
    return v.filter((f) => f.type === "file" && /\.gguf$/i.test(f.path)).map((f) => ({ path: f.path, size: f.lfs?.size ?? f.size ?? 0 }));
}
const jobs = new Map();
/** A download's bytes so far, and beside them what they are (so a paused
 *  download is still there after Pelica restarts). */
const partOf = (dest) => dest + ".part";
const noteOf = (dest) => dest + ".part.json";
/** Every download; with `dir`, first the unfinished ones found there from
 *  earlier runs (as paused). */
function downloads(dir) {
    if (dir)
        restore(dir);
    return [...jobs.values()].map(({ ctl: _c, idle: _i, ...j }) => j);
}
function restore(dir) {
    let subs = [];
    try {
        subs = node_fs_1.default.readdirSync(dir, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name);
    }
    catch {
        return;
    }
    for (const sub of subs) {
        let names = [];
        try {
            names = node_fs_1.default.readdirSync(node_path_1.default.join(dir, sub)).filter((n) => n.endsWith(".gguf.part.json"));
        }
        catch {
            continue;
        }
        for (const n of names) {
            try {
                const note = JSON.parse(node_fs_1.default.readFileSync(node_path_1.default.join(dir, sub, n), "utf8"));
                const dest = node_path_1.default.join(dir, sub, n.slice(0, -".part.json".length));
                const id = `${note.repo}/${note.file}`;
                if (jobs.has(id) || typeof note.repo !== "string" || typeof note.file !== "string")
                    continue;
                let got = 0;
                try {
                    got = node_fs_1.default.statSync(partOf(dest)).size;
                }
                catch {
                    continue;
                }
                jobs.set(id, { id, repo: note.repo, file: note.file, dir, dest, got, total: Number(note.total) || 0, state: "paused", ctl: new AbortController() });
            }
            catch { }
        }
    }
}
/** Stops a download, keeping what it has (resume picks up from there). */
function pause(id) {
    const j = jobs.get(id);
    if (j && j.state === "running") {
        j.state = "paused";
        j.ctl.abort();
    }
}
/** Goes on with a paused (or failed) download. */
function resume(id, onProgress) {
    const j = jobs.get(id);
    if (j && (j.state === "paused" || j.state === "error"))
        download(j.repo, j.file, j.dir, onProgress);
}
/** Stops a download for good: the files it leaves (its part, its note), for
 *  the caller to put in the Recycle Bin. */
function cancel(id) {
    const j = jobs.get(id);
    if (!j || j.state === "done")
        return [];
    j.ctl.abort();
    j.state = "cancelled";
    jobs.delete(id);
    return [partOf(j.dest), noteOf(j.dest)].filter((f) => node_fs_1.default.existsSync(f));
}
/** What uninstalling a model removes: a checkpoint folder, or a .gguf with
 *  its other parts (name-00001-of-00003.gguf) and the notes Pelica keeps
 *  beside it (.pelica-usage and the like). */
function modelFiles(p) {
    if (node_fs_1.default.statSync(p).isDirectory())
        return [p];
    const dir = node_path_1.default.dirname(p);
    const base = node_path_1.default.basename(p);
    const split = base.match(/^(.*)-\d{5}-of-(\d{5})\.gguf$/i);
    const parts = split ? node_fs_1.default.readdirSync(dir).filter((f) => f.toLowerCase().startsWith(split[1].toLowerCase() + "-") && new RegExp(`-\\d{5}-of-${split[2]}\\.gguf$`, "i").test(f)) : [base];
    const all = node_fs_1.default.readdirSync(dir).filter((f) => parts.some((g) => f === g || f.startsWith(g + ".")));
    return all.map((f) => node_path_1.default.join(dir, f));
}
/** Starts a download into `dir`/<repo name>/<file>, or goes on with it from
 *  its part; `onProgress` gets every job's state. */
function download(repo, file, dir, onProgress) {
    if (!/^[\w.-]+\/[\w.-]+$/.test(repo) || file.includes("..") || !/\.gguf$/i.test(file))
        throw new Error("not a GGUF file of a repository");
    const id = `${repo}/${file}`;
    if (jobs.get(id)?.state === "running")
        return id;
    const dest = node_path_1.default.join(dir, repo.split("/")[1], node_path_1.default.basename(file));
    // (an earlier run of it may still be closing its file)
    const prev = jobs.get(id)?.idle;
    const j = { id, repo, file, dir, dest, got: jobs.get(id)?.got || 0, total: jobs.get(id)?.total || 0, state: "running", ctl: new AbortController() };
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
        node_fs_1.default.mkdirSync(node_path_1.default.dirname(dest), { recursive: true });
        const part = partOf(dest);
        let have = 0;
        try {
            have = node_fs_1.default.statSync(part).size;
        }
        catch { }
        const url = `https://huggingface.co/${repo}/resolve/main/${file.split("/").map(encodeURIComponent).join("/")}`;
        const r = await fetch(url, { signal: j.ctl.signal, headers: have ? { Range: `bytes=${have}-` } : {} });
        if (!r.ok && r.status !== 206)
            throw new Error(`HTTP ${r.status}`);
        if (r.status !== 206)
            have = 0;
        j.total = have + Number(r.headers.get("content-length") || 0);
        j.got = have;
        node_fs_1.default.writeFileSync(noteOf(dest), JSON.stringify({ repo, file, total: j.total }));
        const out = node_fs_1.default.createWriteStream(part, { flags: have ? "a" : "w" });
        const reader = r.body.getReader();
        try {
            for (;;) {
                const { done, value } = await reader.read();
                if (done)
                    break;
                if (!out.write(value))
                    await new Promise((ok) => out.once("drain", () => ok()));
                j.got += value.length;
                tick();
            }
        }
        finally {
            await new Promise((ok) => out.end(() => ok()));
        }
        if (j.total && j.got < j.total)
            throw new Error("the connection ended early: resume to go on");
        node_fs_1.default.renameSync(part, dest);
        try {
            node_fs_1.default.unlinkSync(noteOf(dest));
        }
        catch { }
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
let gpus = null;
/** Each display adapter and its dedicated memory: Windows keeps the true size
 *  in the driver's registry key (WMI's AdapterRAM stops at 4 GB). Read once. */
function graphics() {
    gpus ??= new Promise((ok) => {
        const ps = String.raw `Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}' -ErrorAction SilentlyContinue | ForEach-Object { $p = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue; if ($p.DriverDesc) { '{0}|{1}' -f $p.DriverDesc, [uint64]$p.'HardwareInformation.qwMemorySize' } }`;
        (0, node_child_process_1.execFile)("powershell", ["-NoProfile", "-Command", ps], { windowsHide: true }, (_e, out) => {
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
/** Every drive with a file system (C: to Z:). */
async function drives() {
    const out = [];
    for (let c = 67; c <= 90; c++) {
        const root = String.fromCharCode(c) + ":" + node_path_1.default.sep;
        try {
            const s = await node_fs_1.default.promises.statfs(root);
            if (s.blocks)
                out.push({ root, free: s.bavail * s.bsize, total: s.blocks * s.bsize });
        }
        catch { }
    }
    return out;
}
/** Free bytes on the drive that holds `dir` (null: no such drive). */
async function freeOn(dir) {
    const root = node_path_1.default.parse(node_path_1.default.resolve(dir)).root;
    try {
        const s = await node_fs_1.default.promises.statfs(root);
        return { root, free: s.bavail * s.bsize, total: s.blocks * s.bsize };
    }
    catch {
        return null;
    }
}
/** How much the Recycle Bin on `root`'s drive holds, in bytes (null: unknown).
 *  A file bigger than that can't be recycled: Windows deletes it for good. */
function recycleCap(root) {
    const letter = root[0];
    if (!/^[A-Z]$/i.test(letter))
        return Promise.resolve(null);
    // the volume's GUID names its Recycle Bin settings
    const ps = `$v = (Get-Volume -DriveLetter ${letter} -ErrorAction Stop).Path; ` +
        String.raw `$g = [regex]::Match($v, '\{[0-9a-fA-F-]+\}').Value; $p = Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\BitBucket\Volume\$g" -ErrorAction Stop; if ($p.NukeOnDelete) { 0 } else { $p.MaxCapacity }`;
    return new Promise((ok) => (0, node_child_process_1.execFile)("powershell", ["-NoProfile", "-Command", ps], { windowsHide: true }, (e, out) => {
        const mb = Number(String(out).trim());
        ok(e || !String(out).trim() || !isFinite(mb) ? null : mb * 1024 * 1024);
    }));
}
/** Bytes of a model's files (`modelFiles`). */
function bytesOf(files) {
    let n = 0;
    const walk = (f) => {
        const s = node_fs_1.default.statSync(f);
        if (s.isDirectory())
            for (const c of node_fs_1.default.readdirSync(f))
                walk(node_path_1.default.join(f, c));
        else
            n += s.size;
    };
    for (const f of files)
        walk(f);
    return n;
}
/** The folder a model's files go into in `target`: a model in its own folder
 *  (most downloads: <models>/<repo name>/<file>) keeps that folder's name; a
 *  checkpoint folder goes in as it is. */
function moveDest(p, files, roots, target) {
    const key = (x) => node_path_1.default.resolve(x).toLowerCase();
    if (node_fs_1.default.statSync(p).isDirectory())
        return target;
    const parent = node_path_1.default.dirname(files[0]);
    return roots.some((r) => key(r) === key(parent)) ? target : node_path_1.default.join(target, node_path_1.default.basename(parent));
}
/** Moves a model's files into `destDir` (made if needed): a rename on the
 *  same drive, else a copy that is checked and only then removed from the old
 *  drive. A cancelled or failed copy takes away only its own unfinished copies. */
async function moveModel(p, files, destDir, onProgress, signal) {
    const total = bytesOf(files);
    const same = node_path_1.default.parse(node_path_1.default.resolve(files[0])).root.toLowerCase() === node_path_1.default.parse(node_path_1.default.resolve(destDir)).root.toLowerCase();
    node_fs_1.default.mkdirSync(destDir, { recursive: true });
    const dests = files.map((f) => node_path_1.default.join(destDir, node_path_1.default.basename(f)));
    for (const d of dests)
        if (node_fs_1.default.existsSync(d))
            throw new Error(`${d} is already there`);
    if (same) {
        files.forEach((f, i) => node_fs_1.default.renameSync(f, dests[i]));
    }
    else {
        // a folder model: its files one by one
        const pairs = [];
        const add = (src, dst) => {
            if (node_fs_1.default.statSync(src).isDirectory()) {
                node_fs_1.default.mkdirSync(dst, { recursive: true });
                for (const c of node_fs_1.default.readdirSync(src))
                    add(node_path_1.default.join(src, c), node_path_1.default.join(dst, c));
            }
            else
                pairs.push([src, dst]);
        };
        files.forEach((f, i) => add(f, dests[i]));
        let got = 0;
        const made = [];
        try {
            for (const [src, dst] of pairs) {
                made.push(dst);
                await new Promise((ok, bad) => {
                    const r = node_fs_1.default.createReadStream(src, { highWaterMark: 4 << 20 });
                    const w = node_fs_1.default.createWriteStream(dst);
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
                if (node_fs_1.default.statSync(dst).size !== node_fs_1.default.statSync(src).size)
                    throw new Error(`the copy of ${node_path_1.default.basename(src)} came out the wrong size`);
            }
        }
        catch (e) {
            for (const d of made)
                node_fs_1.default.rmSync(d, { force: true });
            for (const d of dests)
                if (node_fs_1.default.existsSync(d) && node_fs_1.default.statSync(d).isDirectory())
                    node_fs_1.default.rmSync(d, { recursive: true, force: true });
            // and the folder it made for them, if nothing else is in it
            try {
                if (!node_fs_1.default.readdirSync(destDir).length)
                    node_fs_1.default.rmdirSync(destDir);
            }
            catch { }
            throw e;
        }
        // every copy checked: now the originals go
        for (const f of files)
            node_fs_1.default.rmSync(f, { recursive: true, force: true });
    }
    // an emptied download folder goes too
    const old = node_path_1.default.dirname(files[0]);
    try {
        if (!node_fs_1.default.readdirSync(old).length)
            node_fs_1.default.rmdirSync(old);
    }
    catch { }
    onProgress(total, total);
    return node_path_1.default.join(destDir, node_path_1.default.basename(p));
}
