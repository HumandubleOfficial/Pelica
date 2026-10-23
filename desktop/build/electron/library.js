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
exports.download = download;
// The model library: GGUF models on Hugging Face (search, files, downloads
// with progress and resume) and the local models folders.
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
