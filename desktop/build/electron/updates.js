"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.RELEASES_PAGE = exports.DEFAULT_FEED = void 0;
exports.older = older;
exports.check = check;
exports.signature = signature;
exports.download = download;
exports.launch = launch;
exports.tidy = tidy;
// Updates: Pelica asks an update feed what the newest version is: by default
// the latest.json of Pelica's newest GitHub release (DEFAULT_FEED), or the web
// address or file path set in Settings → Updates. The feed is the latest.json
// that installer/make.mjs writes beside each installer:
//   { "version": "0.13.0", "url": "Pelica-Setup-0.13.0.exe", "sha256": "…", "notes": [...] }
// `url` may be relative to the feed. An update is downloaded into the app's
// data folder, must match its sha256 (and, when this copy of Pelica is
// signed, carry a valid signature from the same publisher), and then runs
// (/S /RELAUNCH: no questions, a small progress window) after the app quits;
// the installer starts Pelica again when it's done. Nothing is downloaded
// until the user chooses Install.
const node_child_process_1 = require("node:child_process");
const node_crypto_1 = __importDefault(require("node:crypto"));
const node_fs_1 = __importDefault(require("node:fs"));
const node_path_1 = __importDefault(require("node:path"));
const node_url_1 = require("node:url");
/** the newest GitHub release's feed (each release carries its installer and latest.json) */
exports.DEFAULT_FEED = "https://github.com/HumandubleOfficial/Pelica/releases/latest/download/latest.json";
exports.RELEASES_PAGE = "https://github.com/HumandubleOfficial/Pelica/releases";
/** every release, previews (pre-releases) included, newest first */
const RELEASES_API = "https://api.github.com/repos/HumandubleOfficial/Pelica/releases?per_page=20";
/** a < b, for versions like 0.12.0 */
function older(a, b) {
    const pa = a.split(/[.-]/).map((x) => parseInt(x, 10) || 0);
    const pb = b.split(/[.-]/).map((x) => parseInt(x, 10) || 0);
    for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
        if ((pa[i] || 0) !== (pb[i] || 0))
            return (pa[i] || 0) < (pb[i] || 0);
    }
    return false;
}
const isWeb = (s) => /^https?:\/\//i.test(s);
/** the feed as a URL (a plain path becomes file:) */
function feedUrl(feed) {
    const f = feed.trim();
    if (isWeb(f) || /^file:/i.test(f))
        return new URL(f);
    return (0, node_url_1.pathToFileURL)(node_path_1.default.resolve(f));
}
async function read(u, signal) {
    if (u.protocol === "file:")
        return node_fs_1.default.promises.readFile((0, node_url_1.fileURLToPath)(u));
    if (u.protocol !== "https:" && u.protocol !== "http:")
        throw new Error(`can't read ${u.protocol} addresses`);
    const r = await fetch(u, { signal, headers: { "Cache-Control": "no-cache" } });
    if (!r.ok)
        throw new Error(`${u.href}: ${r.status} ${r.statusText}`);
    return Buffer.from(await r.arrayBuffer());
}
/** With previews: the newest release on GitHub that carries a latest.json, pre-releases included. */
async function previewFeed(signal) {
    const r = await fetch(RELEASES_API, { signal, headers: { Accept: "application/vnd.github+json" } });
    if (!r.ok)
        throw new Error(`GitHub: ${r.status} ${r.statusText}`);
    for (const rel of (await r.json())) {
        if (rel.draft)
            continue;
        const a = (rel.assets || []).find((x) => x.name === "latest.json");
        if (a)
            return { url: new URL(a.browser_download_url), preview: !!rel.prerelease };
    }
    throw new Error("no GitHub release has a latest.json");
}
/** What the feed offers. `preview`: with the GitHub feed, also pre-releases. */
async function check(feed, current, opts = {}) {
    if (!feed?.trim())
        return { current, newer: false, error: "No update feed is set." };
    try {
        let u = feedUrl(feed);
        let preview = false;
        if (opts.preview && feed === exports.DEFAULT_FEED)
            ({ url: u, preview } = await previewFeed(AbortSignal.timeout(15000)));
        const j = JSON.parse((await read(u, AbortSignal.timeout(15000))).toString("utf8"));
        if (typeof j.version !== "string" || typeof j.url !== "string" || !/^[0-9a-f]{64}$/i.test(j.sha256 || ""))
            throw new Error("the feed needs version, url and sha256");
        j.url = new URL(j.url, u).href;
        return { current, latest: j, newer: older(current, j.version), preview };
    }
    catch (e) {
        return { current, newer: false, error: `Couldn't read the update feed: ${e?.message || e}` };
    }
}
/** Windows' verdict on a file's signature: status and signer (PowerShell's Get-AuthenticodeSignature). */
function signature(file) {
    const ps = `$s = Get-AuthenticodeSignature -LiteralPath $env:PELICA_SIGFILE; Write-Output ([string]$s.Status); Write-Output ([string]$s.SignerCertificate.Subject)`;
    return new Promise((res) => {
        (0, node_child_process_1.execFile)("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", ps], { windowsHide: true, timeout: 30000, env: { ...process.env, PELICA_SIGFILE: file } }, (err, out) => {
            if (err)
                return res({ status: "Unknown", subject: "" });
            const [status = "", subject = ""] = String(out).split(/\r?\n/);
            res({ status: status.trim(), subject: subject.trim() });
        });
    });
}
async function sha256Of(file) {
    const hash = node_crypto_1.default.createHash("sha256");
    for await (const chunk of node_fs_1.default.createReadStream(file))
        hash.update(chunk);
    return hash.digest("hex");
}
/**
 * Downloads the update into `dir`, checks it, and returns the installer's path.
 * `self` is this app's exe: when it is signed, the update must be signed by the same publisher.
 */
async function download(f, dir, self, onProgress, signal) {
    node_fs_1.default.mkdirSync(dir, { recursive: true });
    const u = new URL(f.url);
    const name = node_path_1.default.basename(u.pathname).replace(/[^\w.-]/g, "_") || `Pelica-Setup-${f.version}.exe`;
    if (!/\.exe$/i.test(name))
        throw new Error("the update isn't an .exe");
    const out = node_path_1.default.join(dir, name);
    const part = out + ".part";
    const same = (sum) => sum.toLowerCase() === f.sha256.toLowerCase();
    // (already here from an earlier try: no need to fetch it again)
    if (node_fs_1.default.existsSync(out)) {
        onProgress({ got: 0, total: 0, phase: "check" });
        if (same(await sha256Of(out))) {
            await checkSigner(out, self);
            return out;
        }
        node_fs_1.default.rmSync(out, { force: true });
    }
    const hash = node_crypto_1.default.createHash("sha256");
    if (u.protocol === "file:") {
        const src = (0, node_url_1.fileURLToPath)(u);
        const total = node_fs_1.default.statSync(src).size;
        await node_fs_1.default.promises.copyFile(src, part);
        hash.update(await node_fs_1.default.promises.readFile(part));
        onProgress({ got: total, total, phase: "download" });
    }
    else {
        if (u.protocol !== "https:" && u.protocol !== "http:")
            throw new Error(`can't download ${u.protocol} addresses`);
        const r = await fetch(u, { signal });
        if (!r.ok || !r.body)
            throw new Error(`${u.href}: ${r.status} ${r.statusText}`);
        const total = Number(r.headers.get("content-length")) || 0;
        const w = node_fs_1.default.createWriteStream(part);
        let got = 0;
        let told = 0;
        try {
            for await (const chunk of r.body) {
                hash.update(chunk);
                got += chunk.length;
                if (!w.write(chunk))
                    await new Promise((r2) => w.once("drain", () => r2()));
                // (ten times a second is plenty for a progress bar)
                if (Date.now() - told >= 100)
                    (told = Date.now()), onProgress({ got, total, phase: "download" });
            }
            onProgress({ got, total, phase: "download" });
        }
        finally {
            await new Promise((r2) => w.end(() => r2()));
        }
    }
    onProgress({ got: 0, total: 0, phase: "check" });
    if (!same(hash.digest("hex"))) {
        node_fs_1.default.rmSync(part, { force: true });
        throw new Error("the download doesn't match the feed's sha256, so it wasn't run");
    }
    try {
        await checkSigner(part, self);
    }
    catch (e) {
        node_fs_1.default.rmSync(part, { force: true });
        throw e;
    }
    node_fs_1.default.rmSync(out, { force: true });
    node_fs_1.default.renameSync(part, out);
    return out;
}
/** When this copy of Pelica is signed, `file` must be signed by the same publisher. */
async function checkSigner(file, self) {
    const mine = await signature(self);
    if (mine.status !== "Valid")
        return;
    const theirs = await signature(file);
    if (theirs.status !== "Valid" || theirs.subject !== mine.subject) {
        throw new Error(`the update isn't signed by ${mine.subject} (${theirs.status}${theirs.subject ? ", " + theirs.subject : ""}), so it wasn't run`);
    }
}
/**
 * Starts the installer, and resolves once Windows has really started it. A file
 * just written can be held for a moment by antivirus, so it tries for a few
 * seconds. If it can't, this throws and Pelica stays open.
 */
async function launch(exe, args) {
    let last;
    for (let i = 0; i < 6; i++) {
        if (i)
            await new Promise((r) => setTimeout(r, 800));
        try {
            await new Promise((ok, no) => {
                const p = (0, node_child_process_1.spawn)(exe, args, { detached: true, stdio: "ignore" });
                p.once("error", no);
                p.once("spawn", () => (p.unref(), ok()));
            });
            return;
        }
        catch (e) {
            last = e;
        }
    }
    throw new Error(`Windows wouldn't start the update (${last?.code || last?.message || last}). Try again in a moment, or install it from ${exports.RELEASES_PAGE}`);
}
/** Old downloaded installers (not `keep`) go. */
function tidy(dir, keep) {
    try {
        for (const n of node_fs_1.default.readdirSync(dir)) {
            const p = node_path_1.default.join(dir, n);
            if (p !== keep && /^Pelica-Setup-.*\.exe(\.part)?$/i.test(n))
                node_fs_1.default.rmSync(p, { force: true });
        }
    }
    catch { }
}
